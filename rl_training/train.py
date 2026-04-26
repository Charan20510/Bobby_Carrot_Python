"""Training loop: stage iteration, checkpointing, heartbeat, and resume."""
from __future__ import annotations
import gc, glob, math, os, sys, threading, time
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from .config import (
    STAGE_ALL_LEVELS, STAGE_MAX_STEPS, STAGE_MAX_EPISODE_STEPS,
    STAGE_ENT_COEF, STAGE_CHECK_FREQ, STAGE_EVAL_EPS,
    RECURRENT_FROM_STAGE, PPO_BASE_KWARGS, RECURRENT_KWARGS, get_policy_kwargs,
)
from .wrappers import CurriculumEnv
from .callbacks import TabularLogCallback, WinRateCallback, StageProgressCallback

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT = True
except ImportError:
    _HAS_RECURRENT = False

def _use_recurrent(stage): return _HAS_RECURRENT and stage >= RECURRENT_FROM_STAGE
def _policy_name(stage): return "MultiInputLstmPolicy" if _use_recurrent(stage) else "MultiInputPolicy"

def make_env_fn(stage):
    horizon = STAGE_MAX_EPISODE_STEPS.get(stage, 500)
    def _init(): return CurriculumEnv(stage=stage, max_episode_steps=horizon)
    return _init

def _lr_schedule(stage):
    def schedule(progress_remaining):
        frac = 1.0 - progress_remaining
        return 1e-5 + 0.5 * (3e-4 - 1e-5) * (1 + math.cos(math.pi * frac))
    return schedule

def _find_latest_ckpt(model_dir):
    files = glob.glob(os.path.join(model_dir, "ckpt_*_steps.zip"))
    if not files: return None, 0
    def _parse(p):
        try:
            for tok in os.path.basename(p).replace(".zip","").split("_"):
                if tok.isdigit(): return int(tok)
            return 0
        except: return 0
    best = max(files, key=_parse)
    return best, _parse(best)

class _Heartbeat:
    def __init__(self, model, interval=30.0, stall_warn=120.0):
        self.model, self.interval, self.stall_warn = model, interval, stall_warn
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._t0 = time.time(); self._last_step = -1; self._last_change = self._t0
    def start(self): self._thread.start()
    def stop(self): self._stop.set(); self._thread.join(timeout=2.0)
    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                cur = int(getattr(self.model, "num_timesteps", 0)); now = time.time()
                if cur != self._last_step: self._last_step = cur; self._last_change = now
                stall = now - self._last_change; elapsed = now - self._t0
                tag = f"[FREEZE WARNING +{stall:.0f}s]" if stall > self.stall_warn else "[heartbeat]"
                msg = f"  {tag} elapsed={elapsed/60:.1f}min  num_timesteps={cur:,}  last_step_age={stall:.0f}s"
                try:
                    from tqdm.auto import tqdm; tqdm.write(msg, file=sys.__stdout__)
                except: sys.__stdout__.write(msg+"\n"); sys.__stdout__.flush()
            except: pass

def run_training(drive_dir, device, n_envs):
    n_envs_recurrent = n_envs
    policy_kwargs = get_policy_kwargs()
    for stage in range(1, 6):
        level_range = f"L{STAGE_ALL_LEVELS[stage][0]:02d}-L{STAGE_ALL_LEVELS[stage][-1]:02d}"
        use_recurrent = _use_recurrent(stage)
        algo_name = "RecurrentPPO" if use_recurrent else "PPO"
        n_env = n_envs_recurrent if use_recurrent else n_envs
        print(f"\n{'='*65}\n  STAGE {stage}/5   Levels {level_range}   [{algo_name}  x{n_env} envs]  [{device}]\n{'='*65}")
        final_path = f"{drive_dir}/models/stage_{stage}_final.zip"
        if os.path.exists(final_path):
            print(f"  Already complete ({final_path}). Skipping."); continue
        model_dir = f"{drive_dir}/models/stage_{stage}"
        os.makedirs(model_dir, exist_ok=True)
        vec_env = VecMonitor(DummyVecEnv([make_env_fn(stage)] * n_env))
        AlgoCls = RecurrentPPO if use_recurrent else PPO
        algo_kwargs = dict(**(RECURRENT_KWARGS if use_recurrent else PPO_BASE_KWARGS),
                           ent_coef=STAGE_ENT_COEF[stage], learning_rate=_lr_schedule(stage))
        ckpt_path, ckpt_steps = _find_latest_ckpt(model_dir)
        resume_steps = 0; reset_ts = False
        if ckpt_path:
            print(f"  [RESUME] {os.path.basename(ckpt_path)}  ({ckpt_steps:,} / {STAGE_MAX_STEPS[stage]:,} steps done)")
            model = AlgoCls.load(ckpt_path, env=vec_env, tensorboard_log=f"{drive_dir}/tb_logs/", device=device, verbose=0)
            model.ent_coef = STAGE_ENT_COEF[stage]; model.learning_rate = _lr_schedule(stage)
            resume_steps = ckpt_steps
        elif stage == 1:
            model = AlgoCls(_policy_name(stage), vec_env, **algo_kwargs, policy_kwargs=policy_kwargs,
                            tensorboard_log=f"{drive_dir}/tb_logs/", device=device, verbose=0)
            reset_ts = True
        else:
            prev_candidates = [f"{drive_dir}/models/stage_{stage-1}/best_model.zip",
                               f"{drive_dir}/models/stage_{stage-1}_final.zip"]
            load_path = next((p for p in prev_candidates if os.path.exists(p)), None)
            prev_recurrent = _use_recurrent(stage - 1)
            if load_path and (prev_recurrent == use_recurrent):
                print(f"  Warm-starting from {load_path}")
                model = AlgoCls.load(load_path, env=vec_env, tensorboard_log=f"{drive_dir}/tb_logs/", device=device, verbose=0)
                model.ent_coef = STAGE_ENT_COEF[stage]; model.learning_rate = _lr_schedule(stage)
            else:
                if load_path: print(f"  Algorithm switch at stage {stage}: starting fresh.")
                model = AlgoCls(_policy_name(stage), vec_env, **algo_kwargs, policy_kwargs=policy_kwargs,
                                tensorboard_log=f"{drive_dir}/tb_logs/", device=device, verbose=0)
        tab_cb = TabularLogCallback()
        win_cb = WinRateCallback(stage=stage, eval_levels=STAGE_ALL_LEVELS[stage],
                                 n_eval_episodes=STAGE_EVAL_EPS[stage], check_freq=STAGE_CHECK_FREQ[stage],
                                 max_eval_steps=250, verbose=1)
        win_cb._steps_at_last_check = resume_steps
        ckpt_cb = CheckpointCallback(save_freq=max(500_000 // n_env, 1), save_path=model_dir, name_prefix="ckpt")
        heartbeat = _Heartbeat(model, interval=30.0, stall_warn=120.0); heartbeat.start()
        if resume_steps > 0:
            print(f"  [resume] firing eval-on-resume (1x) before training...")
            win_cb.model = model; win_cb.num_timesteps = resume_steps; win_cb._stage_start = resume_steps - 1
            try:
                from sb3_contrib import RecurrentPPO as _RPPO; win_cb._is_recurrent = isinstance(model, _RPPO)
            except ImportError: win_cb._is_recurrent = False
            win_cb._run_eval()
        progress_cb = StageProgressCallback(stage=stage, resume_steps=resume_steps)
        try:
            model.learn(total_timesteps=STAGE_MAX_STEPS[stage], callback=[tab_cb, win_cb, ckpt_cb, progress_cb],
                        reset_num_timesteps=reset_ts, tb_log_name=f"{algo_name}_stage{stage}", progress_bar=False)
            model.save(f"{drive_dir}/models/stage_{stage}_final")
        except KeyboardInterrupt:
            rescue = os.path.join(model_dir, f"ckpt_interrupt_{model.num_timesteps}_steps.zip")
            print(f"\n  [INTERRUPT] saving {rescue}"); model.save(rescue); raise
        finally:
            heartbeat.stop(); vec_env.close(); gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("\n[OK] All training stages complete!")
