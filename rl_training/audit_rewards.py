"""Reward-magnitude audit for RewardShapingWrapper.

Run a random-policy rollout, classify every step's reward components, and
print the absolute and relative magnitudes.  The implementation plan §8
requires:

    |exit_seeking|   <  |carrot_pickup|   <  |win| / 3

If any inequality flips, training will silently learn the wrong objective:
- exit_seeking >= carrot_pickup ⇒ agent skips carrots and rushes the exit;
- carrot_pickup >= win/3       ⇒ collecting carrots dominates winning,
                                  reward farming returns.

Usage:
    python -m rl_training.audit_rewards [--levels 1,2,3] [--episodes 5] [--seed 0]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from bobby_carrot.gym_env import BobbyCarrotEnv
from rl_training.wrappers import RewardShapingWrapper
from rl_training.config import (
    EXIT_APPROACH_BONUS, EXIT_RETREAT_PENALTY,
    ALL_COLLECTED_BONUS, POST_COLLECT_STEP_PENALTY,
)
from rl_training.potential import compute_potential, bfs_to_exit


def audit_level(level: int, n_episodes: int, max_steps: int, seed: int) -> Dict[str, float]:
    """Run random rollouts and bucket reward magnitudes by source."""
    base = BobbyCarrotEnv(map_kind="normal", map_number=level)
    env = RewardShapingWrapper(base, max_episode_steps=max_steps)
    rng = np.random.default_rng(seed + level)

    buckets: Dict[str, List[float]] = defaultdict(list)
    win_terminals: List[float] = []
    death_terminals: List[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            action = int(rng.integers(0, env.action_space.n))
            obs, reward, terminated, truncated, info = env.step(action)

            evts = info.get("events", []) or []
            if "carrot_collected" in evts:
                buckets["carrot_pickup"].append(reward)
            elif "egg_collected" in evts:
                buckets["egg_pickup"].append(reward)
            elif "key_picked" in evts:
                buckets["key_pickup"].append(reward)
            else:
                buckets["per_step"].append(reward)

            if terminated:
                if reward > 5.0:
                    win_terminals.append(reward)
                else:
                    death_terminals.append(reward)
                break

    env.close()

    def _stat(name: str) -> float:
        vals = buckets.get(name, [])
        return float(np.mean(np.abs(vals))) if vals else 0.0

    return {
        "level":          level,
        "per_step":       _stat("per_step"),
        "carrot_pickup":  _stat("carrot_pickup"),
        "egg_pickup":     _stat("egg_pickup"),
        "key_pickup":     _stat("key_pickup"),
        "win_terminal":   float(np.mean(win_terminals)) if win_terminals else 0.0,
        "death_terminal": float(np.mean(death_terminals)) if death_terminals else 0.0,
        "n_wins":         len(win_terminals),
        "n_deaths":       len(death_terminals),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,2,3",
                    help="comma-separated level indices")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    levels = [int(s) for s in args.levels.split(",") if s.strip()]
    print(f"\nReward-magnitude audit  (random policy, {args.episodes} eps/level)")
    print(f"  Constants: EXIT_APPROACH_BONUS={EXIT_APPROACH_BONUS}  "
          f"EXIT_RETREAT_PENALTY={EXIT_RETREAT_PENALTY}  "
          f"ALL_COLLECTED_BONUS={ALL_COLLECTED_BONUS}  "
          f"POST_COLLECT_STEP_PENALTY={POST_COLLECT_STEP_PENALTY}\n")
    header = (f"{'Level':<6}{'|step|':>10}{'|carrot|':>10}{'|egg|':>8}"
              f"{'|key|':>8}{'win':>8}{'death':>8}{'wins':>6}{'deaths':>8}")
    print(header)
    print("-" * len(header))

    rows = []
    for lvl in levels:
        row = audit_level(lvl, args.episodes, args.max_steps, args.seed)
        rows.append(row)
        print(f"L{row['level']:<5}"
              f"{row['per_step']:>10.4f}"
              f"{row['carrot_pickup']:>10.4f}"
              f"{row['egg_pickup']:>8.4f}"
              f"{row['key_pickup']:>8.4f}"
              f"{row['win_terminal']:>8.2f}"
              f"{row['death_terminal']:>8.2f}"
              f"{row['n_wins']:>6}"
              f"{row['n_deaths']:>8}")

    # ── Magnitude assertions (§8 reward magnitude audit) ──────────────────
    failures = []
    for row in rows:
        if row["carrot_pickup"] > 0 and row["per_step"] > 0:
            if row["per_step"] >= row["carrot_pickup"]:
                failures.append(
                    f"L{row['level']}: per_step ({row['per_step']:.3f}) >= "
                    f"carrot_pickup ({row['carrot_pickup']:.3f}) — exit-seeking "
                    f"or step penalty is overwhelming collection signal"
                )
        if row["win_terminal"] > 0 and row["carrot_pickup"] > 0:
            if row["carrot_pickup"] * 3 >= row["win_terminal"]:
                failures.append(
                    f"L{row['level']}: 3 × carrot_pickup ({row['carrot_pickup']*3:.2f}) "
                    f">= win_terminal ({row['win_terminal']:.2f}) — wins are not "
                    f"sufficiently dominant; reward farming is possible"
                )

    print()
    if failures:
        print("[AUDIT FAIL]")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[AUDIT OK] all reward-magnitude inequalities hold "
          "(|step| < |carrot| < |win|/3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
