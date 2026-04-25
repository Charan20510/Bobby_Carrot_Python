"""Headless benchmark: measures logical steps/second for the RL env.

Run with:
    python -m bobby_carrot.bench [N_STEPS] [MAP]

Examples:
    python -m bobby_carrot.bench
    python -m bobby_carrot.bench 500000
    python -m bobby_carrot.bench 100000 egg-5
"""
from __future__ import annotations

import sys
import time

from .core.loader import parse_map_arg
from .core.state import Action
from .env import GameEnv

_ACTIONS = list(Action)


def run(n_steps: int = 200_000, map_kind: str = "normal", map_number: int = 1) -> None:
    env = GameEnv()
    env.reset(map_kind, map_number)

    t0 = time.perf_counter()
    resets = 0
    for i in range(n_steps):
        action = _ACTIONS[i % len(_ACTIONS)]
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset(map_kind, map_number)
            resets += 1
    elapsed = time.perf_counter() - t0

    rate = n_steps / elapsed
    print(
        f"{n_steps:,} steps in {elapsed:.3f}s  →  {rate:,.0f} steps/sec"
        f"  ({resets} resets)"
    )


def main() -> None:
    n_steps = 200_000
    map_kind = "normal"
    map_number = 1

    args = sys.argv[1:]
    if args:
        try:
            n_steps = int(args[0])
        except ValueError:
            print(f"Invalid step count: {args[0]}", file=sys.stderr)
            sys.exit(1)
    if len(args) >= 2:
        try:
            m = parse_map_arg(args[1])
            map_kind, map_number = m.kind, m.number
        except ValueError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)

    print(f"Running {n_steps:,} steps on {map_kind}-{map_number:02} …")
    run(n_steps, map_kind, map_number)


if __name__ == "__main__":
    main()
