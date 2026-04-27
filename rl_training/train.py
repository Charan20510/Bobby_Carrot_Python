"""Training loop: stage iteration, checkpointing, heartbeat, and resume."""
from __future__ import annotations
import gc, glob, math, os, sys, threading, time, json
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from .config import (
    STAGE_ALL_LEVELS, STAGE_MAX_STEPS, STAGE_MAX_EPISODE_STEPS,
    STAGE_ENT_COEF, STAGE_CHECK_FREQ, STAGE_EVAL_EPS,
    RECURRENT_FROM_STAGE, PPO_BASE_KWARGS, RECURRENT_KWARGS,
    RECURRENT_POLICY_KWARGS, get_policy_kwargs,
)
from .wrappers import CurriculumEnv
from .callbacks import TabularLogCallback, WinRateCallback, StageProgressCallback, safe_print
from .potential import simulate_level, format_simulation

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

def _policy_kwargs_for(stage: int) -> dict:
    """policy_kwargs for the given stage.

    Stages 3-5 use RecurrentPPO; lstm_hidden_size/n_lstm_layers must be
    threaded through policy_kwargs (not algo kwargs) — wiring them as
    algo kwargs raises TypeError at model instantiation.
    """
    pk = dict(get_policy_kwargs())
    if _use_recurrent(stage):
        pk.update(RECURRENT_POLICY_KWARGS)
    return pk


def run_training(drive_dir, device, n_envs):
    n_envs_recurrent = n_envs
    for stage in range(1, 6):
        policy_kwargs = _policy_kwargs_for(stage)
        level_range = f"L{STAGE_ALL_LEVELS[stage][0]:02d}-L{STAGE_ALL_LEVELS[stage][-1]:02d}"
        use_recurrent = _use_recurrent(stage)
        algo_name = "RecurrentPPO" if use_recurrent else "PPO"
        n_env = n_envs_recurrent if use_recurrent else n_envs
        print(f"\n{'='*65}\n  STAGE {stage}/5   Levels {level_range}   [{algo_name}  x{n_env} envs]  [{device}]\n{'='*65}")
        final_path = f"{drive_dir}/models/stage_{stage}_final.zip"
        if os.path.exists(final_path):
            print(f"  Already complete ({final_path}). Skipping."); continue

        # Per-level static analysis (path length, crumble criticality, etc.)
        # Cheap (~1ms per level) and surfaces "this stage cannot be solved
        # because L0X has no path" failure modes before training starts.
        for lvl in STAGE_ALL_LEVELS[stage]:
            try:
                sim = simulate_level(lvl)
                safe_print(format_simulation(sim))
            except Exception as exc:
                safe_print(f"  [simulate_level] L{lvl:02d} failed: {exc!r}")

        # Config sanity asserts: catch silent drift before burning Colab hours.
        rollout_size = PPO_BASE_KWARGS["n_steps"] * n_env
        assert rollout_size % PPO_BASE_KWARGS["batch_size"] == 0, (
            f"rollout buffer ({rollout_size}) not divisible by batch_size "
            f"({PPO_BASE_KWARGS['batch_size']}); n_envs * n_steps must be a "
            f"multiple of batch_size"
        )
        assert PPO_BASE_KWARGS["n_epochs"] == 10, (
            f"n_epochs={PPO_BASE_KWARGS['n_epochs']}; PPO requires multiple "
            f"epochs per rollout — do not reduce."
        )
        safe_print(
            f"  [CONFIG] n_envs={n_env}  n_steps={PPO_BASE_KWARGS['n_steps']}  "
            f"batch={PPO_BASE_KWARGS['batch_size']}  rollout={rollout_size}  "
            f"ent_coef={STAGE_ENT_COEF[stage]}"
        )
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
                                 max_eval_steps=500, verbose=1)
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


def run_level_training(drive_dir: str, device: str, n_envs: int, level: int):
    """Train a single level independently (no curriculum pooling).

    Each level gets its own model, checkpoint directory, and step budget.
    Checkpoint layout:
        models/level_{NN}/ckpt_*_steps.zip
        models/level_{NN}/best_model.zip
        models/level_{NN}_final.zip
    """
    from .config import (
        LEVEL_MAX_STEPS, LEVEL_MAX_EPISODE_STEPS, LEVEL_ENT_COEF,
        LEVEL_CHECK_FREQ, LEVEL_EVAL_EPS, LEVEL_USE_RECURRENT,
        LEVEL_WIN_THRESHOLD, LEVEL_MIN_STEPS,
    )
    from .wrappers import SingleLevelEnv

    use_recurrent = LEVEL_USE_RECURRENT[level] and _HAS_RECURRENT
    AlgoCls = RecurrentPPO if use_recurrent else PPO
    algo_name = "RecurrentPPO" if use_recurrent else "PPO"
    horizon = LEVEL_MAX_EPISODE_STEPS[level]
    max_steps = LEVEL_MAX_STEPS[level]
    ent_coef = LEVEL_ENT_COEF[level]
    n_env = n_envs

    print(f"\n{'='*65}")
    print(f"  LEVEL {level:02d}  [{algo_name} x{n_env} envs]  [{device}]")
    print(f"  budget: {max_steps:,} steps  horizon: {horizon}  ent_coef: {ent_coef}")
    print(f"{'='*65}")

    final_path = f"{drive_dir}/models/level_{level:02d}_final.zip"
    if os.path.exists(final_path):
        print(f"  Already complete ({final_path}). Skipping.")
        return

    try:
        sim = simulate_level(level)
        safe_print(format_simulation(sim))
    except Exception as exc:
        safe_print(f"  [simulate_level] L{level:02d} failed: {exc!r}")

    rollout_size = PPO_BASE_KWARGS["n_steps"] * n_env
    assert rollout_size % PPO_BASE_KWARGS["batch_size"] == 0, (
        f"rollout buffer ({rollout_size}) not divisible by batch_size "
        f"({PPO_BASE_KWARGS['batch_size']})"
    )
    safe_print(
        f"  [CONFIG] n_envs={n_env}  n_steps={PPO_BASE_KWARGS['n_steps']}  "
        f"batch={PPO_BASE_KWARGS['batch_size']}  rollout={rollout_size}  "
        f"ent_coef={ent_coef}"
    )
    
    start_time = time.time()

    model_dir = f"{drive_dir}/models/level_{level:02d}"
    os.makedirs(model_dir, exist_ok=True)

    def _make_env(lvl=level, h=horizon):
        def _init():
            return SingleLevelEnv(level=lvl, max_episode_steps=h)
        return _init

    vec_env = VecMonitor(DummyVecEnv([_make_env()] * n_env))

    policy_name = "MultiInputLstmPolicy" if use_recurrent else "MultiInputPolicy"
    pk = dict(get_policy_kwargs())
    if use_recurrent:
        pk.update(RECURRENT_POLICY_KWARGS)

    algo_kwargs = dict(
        **(RECURRENT_KWARGS if use_recurrent else PPO_BASE_KWARGS),
        ent_coef=ent_coef,
        learning_rate=_lr_schedule(1),
    )

    ckpt_path, ckpt_steps = _find_latest_ckpt(model_dir)
    resume_steps = 0
    reset_ts = True

    if ckpt_path:
        print(f"  [RESUME] {os.path.basename(ckpt_path)}  ({ckpt_steps:,} / {max_steps:,} steps done)")
        model = AlgoCls.load(
            ckpt_path, env=vec_env,
            tensorboard_log=f"{drive_dir}/tb_logs/",
            device=device, verbose=0,
        )
        model.ent_coef = ent_coef
        model.learning_rate = _lr_schedule(1)
        resume_steps = ckpt_steps
        reset_ts = False
    else:
        model = AlgoCls(
            policy_name, vec_env, **algo_kwargs,
            policy_kwargs=pk,
            tensorboard_log=f"{drive_dir}/tb_logs/",
            device=device, verbose=0,
        )

    tab_cb = TabularLogCallback()
    win_cb = WinRateCallback(
        stage=level,
        eval_levels=[level],
        n_eval_episodes=LEVEL_EVAL_EPS[level],
        check_freq=LEVEL_CHECK_FREQ[level],
        max_eval_steps=500, verbose=1,
    )
    win_cb.hardest_level = level
    win_cb._steps_at_last_check = resume_steps
    ckpt_cb = CheckpointCallback(
        save_freq=max(500_000 // n_env, 1),
        save_path=model_dir, name_prefix="ckpt",
    )
    heartbeat = _Heartbeat(model, interval=30.0, stall_warn=120.0)
    heartbeat.start()

    if resume_steps > 0:
        print(f"  [resume] firing eval-on-resume (1x) before training...")
        win_cb.model = model
        win_cb.num_timesteps = resume_steps
        win_cb._stage_start = resume_steps - 1
        try:
            from sb3_contrib import RecurrentPPO as _RPPO
            win_cb._is_recurrent = isinstance(model, _RPPO)
        except ImportError:
            win_cb._is_recurrent = False
        win_cb._run_eval()

    progress_cb = StageProgressCallback(stage=level, resume_steps=resume_steps)

    try:
        model.learn(
            total_timesteps=max_steps,
            callback=[tab_cb, win_cb, ckpt_cb, progress_cb],
            reset_num_timesteps=reset_ts,
            tb_log_name=f"{algo_name}_L{level:02d}",
            progress_bar=False,
        )
        model.save(f"{drive_dir}/models/level_{level:02d}_final")
    except KeyboardInterrupt:
        rescue = os.path.join(
            model_dir, f"ckpt_interrupt_{model.num_timesteps}_steps.zip")
        print(f"\n  [INTERRUPT] saving {rescue}")
        model.save(rescue)
        raise
    finally:
        heartbeat.stop()
        vec_env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wall_time_min = (time.time() - start_time) / 60.0
    
    # Run post-training evaluation
    from .evaluate import evaluate_level_model
    print(f"\n  [EVAL] Running post-training evaluation on Level {level:02d}...")
    eval_result = evaluate_level_model(drive_dir, level, n_episodes=20, max_steps=500)
    
    # Save training log
    log_data = {
        "level": level,
        "algo": algo_name,
        "total_steps": model.num_timesteps,
        "wall_time_min": round(wall_time_min, 2),
        "config": {
            "ent_coef": ent_coef,
            "horizon": horizon,
            "max_steps": max_steps,
        },
        "eval": {
            "win_rate": eval_result["win_rate"],
            "mean_reward": eval_result["mean_reward"],
            "mean_steps": eval_result["mean_steps"],
            "carrot_pct": eval_result["carrot_pct"],
            "all_carrots_collected": eval_result["all_carrots_collected"],
            "failure_breakdown": eval_result["failure_breakdown"],
        }
    }
    
    log_path = f"{model_dir}/training_log.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
        
    print(f"\n╔{'═'*58}╗")
    print(f"║  LEVEL {level:02d} — TRAINING SUMMARY                           ║")
    print(f"╠{'═'*58}╣")
    print(f"║  Algorithm   : {algo_name:<13}Total Steps: {model.num_timesteps:<10,}║")
    print(f"║  Wall Time   : {wall_time_min:<5.1f} min{' '*33}║")
    print(f"║  Final Win%  : {eval_result['win_rate']:<4.0%}          Mean Steps : {eval_result['mean_steps']:<14.0f}║")
    print(f"║  Carrot%     : {eval_result['carrot_pct']:<4.0%}          Mean Reward: {eval_result['mean_reward']:<+14.1f}║")
    
    pass_status = "PASS" if eval_result['win_rate'] >= 0.70 and eval_result['all_carrots_collected'] else "FAIL"
    print(f"║  Status      : {pass_status:<44}║")
    print(f"╚{'═'*58}╝")

    print(f"\n[OK] Level {level:02d} training complete!")


def run_all_level_training(drive_dir: str, device: str, n_envs: int):
    """Train levels 1–15 sequentially, each independently."""
    from .config import INDIVIDUAL_LEVELS
    for level in INDIVIDUAL_LEVELS:
        run_level_training(drive_dir, device, n_envs, level)
        
    # Print combined summary table
    print(f"\n{'═'*75}")
    print("  PER-LEVEL TRAINING RESULTS (Levels 1–15)")
    print(f"{'═'*75}")
    print(f"  Level │ Win% │ Carrot% │  Steps │ Reward │ Status")
    print("  ──────┼──────┼─────────┼────────┼────────┼────────")
    
    passes = 0
    total = 0
    avg_win = avg_car = avg_steps = avg_rew = 0.0
    
    for l in INDIVIDUAL_LEVELS:
        log_path = f"{drive_dir}/models/level_{l:02d}/training_log.json"
        if os.path.exists(log_path):
            total += 1
            log = json.load(open(log_path))
            ev = log.get("eval", {})
            wr = ev.get("win_rate", 0)
            car = ev.get("carrot_pct", 0)
            steps = ev.get("mean_steps", 0)
            rew = ev.get("mean_reward", 0)
            
            avg_win += wr
            avg_car += car
            avg_steps += steps
            avg_rew += rew
            
            all_collected = ev.get("all_carrots_collected", car >= 0.999)
            status = "✓ PASS" if wr >= 0.70 and all_collected else "✗ FAIL"
            if "PASS" in status: passes += 1
            
            print(f"  L{l:02d}   │ {wr:>4.0%} │ {car:>7.0%} │ {steps:>6.0f} │ {rew:>+6.1f} │ {status}")
        else:
            print(f"  L{l:02d}   │ {'—':>4} │ {'—':>7} │ {'—':>6} │ {'—':>6} │ not trained")
            
    print("  ──────┼──────┼─────────┼────────┼────────┼────────")
    if total > 0:
        print(f"  AVG   │ {avg_win/total:>4.0%} │ {avg_car/total:>7.0%} │ {avg_steps/total:>6.0f} │ {avg_rew/total:>+6.1f} │ {passes}/{total} PASS")
    print(f"{'═'*75}\n")

    print("[OK] All 15 levels complete!")
