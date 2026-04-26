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


def _load_best_model(drive_dir):
    """Load the highest-stage model, preferring RecurrentPPO."""
    candidates = []
    for s in range(5, 0, -1):
        candidates += [
            f"{drive_dir}/models/stage_{s}/best_model.zip",
            f"{drive_dir}/models/stage_{s}_final.zip",
        ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError("No trained model found. Complete at least one stage.")
    if RecurrentPPO:
        try:
            return RecurrentPPO.load(path), path
        except Exception:
            pass
    return PPO.load(path), path


def _classify(total_r, dead, timed_out, carrot_pct):
    """Classify episode outcome into a failure mode."""
    if total_r > 5.0:          return "win"
    if dead:                   return "death"
    if timed_out and carrot_pct > 0.9: return "stuck_near_exit"
    if timed_out:              return "timeout"
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
    print(f"\n{'Level':<8} {'Win%':<8} {'Reward':>9} {'Steps':>8} "
          f"{'Death%':>8} {'Timeout%':>10} {'Carrot%':>9}")
    print("-" * 65)

    all_results = {}
    heatmaps    = {}

    for level in levels:
        wins = deaths = timeouts = stuck = 0
        total_r = total_steps = total_carrot_pct = 0.0
        heatmap = np.zeros((16, 16), dtype=float)

        for ep in range(n_episodes):
            env  = BobbyCarrotEnv(map_kind="normal", map_number=level)
            obs, _ = env.reset()
            ep_r = ep_steps = 0.0
            dead = timed_out = False
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
                    dead = (ep_r < win_threshold); break
                if trunc or ep_steps >= max_steps:
                    timed_out = True; break

            env_inner = env._env if hasattr(env, "_env") else None
            gs = getattr(env_inner, "gs", None)
            if gs:
                total_ct = max(gs.carrot_total + gs.egg_total, 1)
                carrot_pct = (gs.carrot_count + gs.egg_count) / total_ct
            else:
                carrot_pct = 0.0

            mode = _classify(ep_r, dead, timed_out, carrot_pct)
            if mode == "win":      wins += 1
            elif mode == "death":  deaths += 1
            elif mode in ("timeout", "stuck_near_exit"): timeouts += 1

            total_r += ep_r; total_steps += ep_steps; total_carrot_pct += carrot_pct
            env.close()

        wr   = wins / n_episodes
        dr   = deaths / n_episodes
        tr   = timeouts / n_episodes
        mr   = total_r / n_episodes
        ms   = total_steps / n_episodes
        cpct = total_carrot_pct / n_episodes

        all_results[level] = dict(win_rate=wr, mean_r=mr, mean_steps=ms,
                                  death_rate=dr, timeout_rate=tr, carrot_pct=cpct)
        heatmaps[level] = heatmap / max(heatmap.sum(), 1)

        print(f"L{level:<7} {wr:<8.0%} {mr:>9.2f} {ms:>8.1f} "
              f"{dr:>8.0%} {tr:>10.0%} {cpct:>9.0%}")

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
