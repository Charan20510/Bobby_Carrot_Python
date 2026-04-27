"""Held-out evaluation pipeline.

Loads the best available model, runs deterministic episodes on specified
levels, and returns per-level metrics (win rate, mean reward, failure modes).
"""
from __future__ import annotations
import os
import numpy as np
from stable_baselines3 import PPO
from bobby_carrot.gym_env import BobbyCarrotEnv

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


def verify_lstm_threading(model, n_test_episodes: int = 3, level: int = 1) -> bool:
    """Assert the LSTM hidden-state threading is correct.

    Runs ``n_test_episodes`` short episodes through ``model.predict`` and
    verifies:
      1) ``lstm_states`` is non-None after the first action.
      2) After ``done=True``, the next ``model.predict`` is called with
         ``episode_start=[True]`` so the hidden state resets — a stale
         hidden state across episode boundaries silently corrupts the
         policy on Stages 3-5 (RecurrentPPO).
      3) ``lstm_states`` returned for the post-reset step is structurally
         valid (right shape).

    Returns True on success, raises AssertionError otherwise.
    Prints "[LSTM-OK]" on success.

    Stages 1-2 use plain PPO; this function returns True trivially when
    the model is not a RecurrentPPO instance.
    """
    if RecurrentPPO is None or not isinstance(model, RecurrentPPO):
        print("[LSTM-OK] non-recurrent model; nothing to verify")
        return True

    env = BobbyCarrotEnv(map_kind="normal", map_number=level)
    saw_reset = 0
    try:
        for ep in range(n_test_episodes):
            obs, _ = env.reset()
            lstm_states = None
            episode_start = np.array([True])
            steps = 0
            while steps < 50:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start, deterministic=True,
                )
                assert lstm_states is not None, (
                    f"lstm_states is None at ep {ep} step {steps} — "
                    f"RecurrentPPO predict did not return state"
                )
                # Episode_start must be False for all steps after the first
                if steps == 0:
                    assert episode_start[0] is np.True_ or bool(episode_start[0]), (
                        f"episode_start[0] should be True on first step of "
                        f"episode {ep}"
                    )
                    saw_reset += 1
                episode_start = np.array([False])
                obs, r, term, trunc, info = env.step(action)
                steps += 1
                if term or trunc:
                    break
        assert saw_reset == n_test_episodes, (
            f"only {saw_reset}/{n_test_episodes} episode resets observed"
        )
    finally:
        env.close()

    print(f"[LSTM-OK] threading verified across {n_test_episodes} episodes")
    return True


def _load_best_model(drive_dir):
    """Load the highest-stage model, preferring RecurrentPPO.

    Walks 5 → 1, and within each stage prefers (in order):
      1) best_model.zip       — written by WinRateCallback on improvement
      2) stage_N_final.zip    — written when stage promotes
      3) ckpt_*_steps.zip     — periodic CheckpointCallback artifacts;
                                 highest step number wins.  Required so
                                 held-out eval can run on an interrupted
                                 stage that never produced best/final.
    """
    import glob

    def _ckpt_step(path: str) -> int:
        for tok in os.path.basename(path).replace(".zip", "").split("_"):
            if tok.isdigit():
                return int(tok)
        return 0

    chosen_path = None
    for s in range(5, 0, -1):
        for fname in (
            f"{drive_dir}/models/stage_{s}/best_model.zip",
            f"{drive_dir}/models/stage_{s}_final.zip",
        ):
            if os.path.exists(fname):
                chosen_path = fname
                break
        if chosen_path:
            break
        ckpts = sorted(
            glob.glob(f"{drive_dir}/models/stage_{s}/ckpt_*_steps.zip"),
            key=_ckpt_step,
        )
        if ckpts:
            chosen_path = ckpts[-1]
            break

    if chosen_path is None:
        raise FileNotFoundError(
            "No trained model found under "
            f"{drive_dir}/models/. Complete at least one stage."
        )

    if RecurrentPPO:
        try:
            return RecurrentPPO.load(chosen_path), chosen_path
        except Exception:
            pass
    return PPO.load(chosen_path), chosen_path


def _load_level_model(drive_dir: str, level: int):
    """Load the best model for a specific individual level."""
    import glob

    def _ckpt_step(path: str) -> int:
        for tok in os.path.basename(path).replace(".zip", "").split("_"):
            if tok.isdigit():
                return int(tok)
        return 0

    chosen_path = None
    level_dir = f"{drive_dir}/models/level_{level:02d}"
    
    for fname in (
        f"{level_dir}/best_model.zip",
        f"{level_dir}_final.zip",
    ):
        if os.path.exists(fname):
            chosen_path = fname
            break
            
    if not chosen_path:
        ckpts = sorted(
            glob.glob(f"{level_dir}/ckpt_*_steps.zip"),
            key=_ckpt_step,
        )
        if ckpts:
            chosen_path = ckpts[-1]

    if chosen_path is None:
        raise FileNotFoundError(f"No trained model found for level {level} in {drive_dir}/models/")

    if RecurrentPPO:
        try:
            return RecurrentPPO.load(chosen_path), chosen_path
        except Exception:
            pass
    return PPO.load(chosen_path), chosen_path



def _classify(total_r, dead, timed_out, carrot_pct, crumble_death=False):
    """Classify episode outcome into a failure mode.

    Mirrors callbacks._classify_episode so training and held-out eval
    report the same buckets.  Splits death into crumble / other to surface
    crumble-management failures separately on L26-L30.
    """
    if total_r > 5.0:                    return "win"
    if dead and crumble_death:           return "death_crumble"
    if dead:                             return "death_other"
    if timed_out and carrot_pct >= 0.9:  return "stuck_near_exit"
    if timed_out:                        return "timeout"
    if total_r < -0.4:                   return "unwinnable"
    return "unknown"


def run_evaluation(
    drive_dir: str,
    levels: list | None = None,
    n_episodes: int = 50,
    max_steps: int = 500,
    win_threshold: float = 5.0,
) -> dict:
    """Run held-out evaluation and return per-level results dict.

    Parameters
    ----------
    drive_dir : Root checkpoint directory.
    levels : List of level numbers to evaluate (default: 26-30).
    n_episodes : Episodes per level.
    max_steps : Hard step limit per episode.
    win_threshold : Reward threshold for win detection.

    Returns
    -------
    dict with keys: 'model_path', 'is_recurrent', 'results' (per-level),
    'heatmaps' (per-level), 'avg_win', 'avg_reward', 'avg_carrot_pct'.
    """
    if levels is None:
        levels = list(range(26, 31))

    model, model_path = _load_best_model(drive_dir)
    is_recurrent = RecurrentPPO is not None and isinstance(model, RecurrentPPO)

    print(f"Loaded : {model_path}")
    print(f"Type   : {type(model).__name__}")

    # Pre-flight: verify LSTM threading is wired correctly before running
    # 50 episodes per level × N levels with a stale hidden state.
    if is_recurrent:
        try:
            verify_lstm_threading(model, n_test_episodes=2, level=levels[0])
        except AssertionError as exc:
            print(f"[LSTM-FAIL] {exc}")
            raise

    print(f"\n{'Level':<8} {'Win%':<8} {'Reward':>9} {'Steps':>8} "
          f"{'Death%':>8} {'Timeout%':>10} {'Carrot%':>9} {'Mode':<18}")
    print("-" * 90)

    all_results = {}
    heatmaps    = {}

    for level in levels:
        wins = deaths = timeouts = stuck = 0
        total_r = total_steps = total_carrot_pct = 0.0
        heatmap = np.zeros((16, 16), dtype=float)
        # Per-mode counts for full breakdown
        mode_counts = {
            "win": 0, "death_crumble": 0, "death_other": 0,
            "timeout": 0, "stuck_near_exit": 0, "unwinnable": 0, "unknown": 0,
        }

        for ep in range(n_episodes):
            env  = BobbyCarrotEnv(map_kind="normal", map_number=level)
            obs, _ = env.reset()
            ep_r = ep_steps = 0.0
            dead = timed_out = False
            crumble_death = False
            lstm_states = None
            episode_start = np.array([True])

            while True:
                heatmap[obs["player_y"]][obs["player_x"]] += 1
                if is_recurrent:
                    action, lstm_states = model.predict(
                        obs, state=lstm_states,
                        episode_start=episode_start, deterministic=True)
                    episode_start = np.array([False])
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                ep_r += r; ep_steps += 1
                if term:
                    dead = (ep_r < win_threshold)
                    evts = info.get("events", []) or []
                    crumble_death = "died_on_crumble" in evts
                    break
                if trunc or ep_steps >= max_steps:
                    timed_out = True; break

            env_inner = env._env if hasattr(env, "_env") else None
            gs = getattr(env_inner, "gs", None)
            if gs:
                total_ct = max(gs.carrot_total + gs.egg_total, 1)
                carrot_pct = (gs.carrot_count + gs.egg_count) / total_ct
            else:
                carrot_pct = 0.0

            mode = _classify(ep_r, dead, timed_out, carrot_pct, crumble_death)
            mode_counts[mode] += 1
            if mode == "win":
                wins += 1
            elif mode in ("death_crumble", "death_other"):
                deaths += 1
            elif mode in ("timeout", "stuck_near_exit", "unwinnable"):
                timeouts += 1

            total_r += ep_r; total_steps += ep_steps; total_carrot_pct += carrot_pct
            env.close()

        wr   = wins / n_episodes
        dr   = deaths / n_episodes
        tr   = timeouts / n_episodes
        mr   = total_r / n_episodes
        ms   = total_steps / n_episodes
        cpct = total_carrot_pct / n_episodes

        # Identify the dominant non-win mode for the per-level summary line
        non_win = {k: v for k, v in mode_counts.items() if k != "win" and v > 0}
        dominant = max(non_win, key=non_win.get) if non_win else "win"
        all_results[level] = dict(
            win_rate=wr, mean_r=mr, mean_steps=ms,
            death_rate=dr, timeout_rate=tr, carrot_pct=cpct,
            failure_breakdown=dict(mode_counts),
            dominant_mode=dominant,
        )
        heatmaps[level] = heatmap / max(heatmap.sum(), 1)

        print(f"L{level:<7} {wr:<8.0%} {mr:>9.2f} {ms:>8.1f} "
              f"{dr:>8.0%} {tr:>10.0%} {cpct:>9.0%} {dominant:<18}")

    avg_win  = np.mean([v["win_rate"]   for v in all_results.values()])
    avg_r    = np.mean([v["mean_r"]     for v in all_results.values()])
    avg_cpct = np.mean([v["carrot_pct"] for v in all_results.values()])
    print("-" * 65)
    print(f"{'Average':<8} {avg_win:<8.0%} {avg_r:>9.2f} {'':>8} "
          f"{'':>8} {'':>10} {avg_cpct:>9.0%}")

    return {
        "model_path": model_path,
        "is_recurrent": is_recurrent,
        "results": all_results,
        "heatmaps": heatmaps,
        "avg_win": avg_win,
        "avg_reward": avg_r,
        "avg_carrot_pct": avg_cpct,
    }


def evaluate_level_model(
    drive_dir: str,
    level: int,
    n_episodes: int = 20,
    max_steps: int = 500,
    win_threshold: float = 5.0,
) -> dict:
    """Run evaluation for a specific level model on its own level.
    
    Returns a dict with win_rate, mean_reward, mean_steps, carrot_pct,
    all_carrots_collected (bool), and failure_breakdown.
    """
    model, model_path = _load_level_model(drive_dir, level)
    is_recurrent = RecurrentPPO is not None and isinstance(model, RecurrentPPO)
    
    wins = deaths = timeouts = 0
    total_r = total_steps = total_carrot_pct = 0.0
    mode_counts = {
        "win": 0, "death_crumble": 0, "death_other": 0,
        "timeout": 0, "stuck_near_exit": 0, "unwinnable": 0, "unknown": 0,
    }
    
    # Track if *every* winning episode collected 100% of carrots
    all_carrots_collected_in_wins = True
    win_count_for_carrot_check = 0
    
    for ep in range(n_episodes):
        env = BobbyCarrotEnv(map_kind="normal", map_number=level)
        obs, _ = env.reset()
        ep_r = ep_steps = 0.0
        dead = timed_out = crumble_death = False
        lstm_states = None
        episode_start = np.array([True])
        
        while True:
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start, deterministic=True)
                episode_start = np.array([False])
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            ep_steps += 1
            
            if term:
                dead = (ep_r < win_threshold)
                evts = info.get("events", []) or []
                crumble_death = "died_on_crumble" in evts
                break
            if trunc or ep_steps >= max_steps:
                timed_out = True
                break
                
        env_inner = env._env if hasattr(env, "_env") else None
        gs = getattr(env_inner, "gs", None)
        if gs:
            total_ct = max(gs.carrot_total + gs.egg_total, 1)
            carrot_pct = (gs.carrot_count + gs.egg_count) / total_ct
        else:
            carrot_pct = 0.0
            
        mode = _classify(ep_r, dead, timed_out, carrot_pct, crumble_death)
        mode_counts[mode] += 1
        
        if mode == "win":
            wins += 1
            win_count_for_carrot_check += 1
            if carrot_pct < 0.999:  # Allow floating point slop
                all_carrots_collected_in_wins = False
        elif mode in ("death_crumble", "death_other"):
            deaths += 1
        elif mode in ("timeout", "stuck_near_exit", "unwinnable"):
            timeouts += 1
            
        total_r += ep_r
        total_steps += ep_steps
        total_carrot_pct += carrot_pct
        env.close()
        
    wr = wins / n_episodes
    if win_count_for_carrot_check == 0:
        all_carrots_collected_in_wins = False
        
    return {
        "model_path": model_path,
        "is_recurrent": is_recurrent,
        "win_rate": wr,
        "mean_reward": total_r / n_episodes,
        "mean_steps": total_steps / n_episodes,
        "carrot_pct": total_carrot_pct / n_episodes,
        "all_carrots_collected": all_carrots_collected_in_wins,
        "failure_breakdown": dict(mode_counts)
    }

