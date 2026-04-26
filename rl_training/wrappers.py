"""Gymnasium wrappers for curriculum training.

RewardShapingWrapper adds potential-based shaping + blocked-move penalty.
CurriculumEnv handles stage-based level sampling with inverse-win-rate weighting.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import gymnasium as gym

from bobby_carrot.gym_env import BobbyCarrotEnv

from .potential import compute_potential
from .config import (
    STAGE_ALL_LEVELS,
    STAGE_NEW_START,
)


class RewardShapingWrapper(gym.Wrapper):
    """Potential-based reward shaping + key-pickup bonus + win efficiency bonus.

    Shaped reward on each step:
        r'(s,a,s') = r(s,a,s') + γ·Φ(s') - Φ(s)   (Ng et al. 1999)

    On terminal steps the underlying env reloads/advances before we can
    read the next state.  For wins Φ(s') = 0 (correct terminal value).
    For deaths Φ(s') = Φ(s) so shaping ≈ (γ-1)·Φ ≈ 0, keeping the
    -1.0 death penalty as the sole death signal.

    Bonuses:
        +key_pickup:   +0.3   (matters from stage 5 onward; harmless earlier)
        +efficiency:   +5.0 × (1 - steps/max_steps)  on win

    Removed (as of stage-1 reward redesign — they caused reward farming):
        +0.1 × (collected/total)  per carrot_collected
        +0.2  per crumble_survived
    These dense non-potential bonuses created a local optimum where the agent
    farmed partial-progress reward without ever reaching the exit. Empirically
    that produced ep_rew_mean ≈ 6 with 0% L01–L03 win rate at 500K steps.
    Removing them restores the property that the only large reward path is
    "collect all → reach exit", which the potential function already shapes.

    The efficiency bonus was also raised from +2.0 → +5.0 to widen the gap
    between "win" (~+15 shaped return on L01) and "wander" (~0), giving the
    value head a sharper signal to latch onto early.
    """
    GAMMA = 0.995
    EFFICIENCY_BONUS = 5.0

    def __init__(self, env, max_episode_steps: int = 500):
        super().__init__(env)
        self._max_steps = max_episode_steps
        self._prev_potential = 0.0
        self._step_count = 0

    DEBUG_RESET = False   # flip to True to log per-reset potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        gs = self._get_gs()
        self._prev_potential = compute_potential(gs) if gs else 0.0
        if self.DEBUG_RESET:
            import sys
            sys.__stdout__.write(
                f"[RewardShapingWrapper.reset] prev_potential="
                f"{self._prev_potential:.4f}\n"
            )
        return obs, info

    def step(self, action):
        # Capture pre-step position for blocked-move detection
        gs_before = self._get_gs()
        prev_pos = (gs_before.coord_src if gs_before else None)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # Penalise blocked moves (wall bumps / IDLE): the agent moved but
        # position didn't change.  -0.05 is 5× the base step penalty,
        # enough to deter wall-bumping without overwhelming carrot signals.
        if not terminated and prev_pos is not None:
            gs_after = self._get_gs()
            if gs_after and gs_after.coord_src == prev_pos:
                reward -= 0.05

        if not terminated:
            gs = self._get_gs()
            curr_potential = compute_potential(gs) if gs else 0.0
        elif reward > 5.0:
            # Win: all goals reached; Φ(terminal) = 0 by definition.
            curr_potential = 0.0
        else:
            # Death: env already reloaded; cancel spurious positive shaping.
            # The -1.0 death penalty is the sole death signal.
            curr_potential = self._prev_potential

        # Potential-based shaping (policy-invariant, Ng et al. 1999)
        reward += self.GAMMA * curr_potential - self._prev_potential
        self._prev_potential = curr_potential

        # Only keep the key-pickup bonus — it matters for stage-5 sequencing
        # and is harmless on stages without keys.
        events = info.get("events", [])
        for e in events:
            if e == "key_picked":
                reward += 0.3

        # Efficiency bonus on win (terminal_r > 5.0 only when native +10.0 fires)
        if terminated and reward > 5.0:
            reward += self.EFFICIENCY_BONUS * (1.0 - self._step_count / self._max_steps)

        return obs, reward, terminated, truncated, info

    def _get_gs(self):
        """Unwrap to the innermost GameEnv and return its GameState."""
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return getattr(getattr(env, "_env", None), "gs", None)


class CurriculumEnv(gym.Env):
    """Curriculum-based level sampling with inverse-win-rate weighting.

    Level sampling:
      stage 1 : uniform over stage levels
      stage k>1: weighted by (1 - rolling_win_rate) per level;
                 70% new levels, 30% replay from prior stages
    """
    metadata = {"render_modes": [None]}

    def __init__(self, stage: int = 1, max_episode_steps: int = 500):
        super().__init__()
        assert stage in STAGE_ALL_LEVELS, f"Invalid stage {stage}"
        self.stage = stage
        self.max_episode_steps = max_episode_steps

        new_start = STAGE_NEW_START[stage]
        all_end   = STAGE_ALL_LEVELS[stage][-1]
        self._new_levels   = list(range(new_start, all_end + 1))
        self._prior_levels = list(range(1, new_start)) if new_start > 1 else []
        self._all_levels   = self._prior_levels + self._new_levels

        # Rolling win-rate tracking: deque of last 100 outcomes per level
        self._outcomes: dict = {l: deque(maxlen=100) for l in self._all_levels}

        _dummy = BobbyCarrotEnv(map_kind="normal", map_number=1)
        self.observation_space = _dummy.observation_space
        self.action_space      = _dummy.action_space
        _dummy.close()

        self._env_cache: dict = {}   # level -> RewardShapingWrapper, created lazily
        self._env        = None
        self._step_count = 0
        self._cur_level  = None

    def _win_rate(self, level: int) -> float:
        outcomes = self._outcomes[level]
        return float(np.mean(outcomes)) if outcomes else 0.5   # optimistic prior

    def _pick_level(self) -> int:
        # 70/30 split between new and prior levels (same as before)
        if self._prior_levels and np.random.random() < 0.30:
            pool = self._prior_levels
        else:
            pool = self._new_levels

        # Inverse win-rate weights within the chosen pool
        weights = np.array([1.0 - self._win_rate(l) for l in pool])
        weights = np.clip(weights, 0.05, 1.0)   # floor prevents starvation
        weights /= weights.sum()
        return int(np.random.choice(pool, p=weights))

    def record_outcome(self, level: int, won: bool) -> None:
        """Called externally (e.g. from WinRateCallback) to log episode results."""
        if level in self._outcomes:
            self._outcomes[level].append(float(won))

    def reset(self, **kwargs):
        self._cur_level = self._pick_level()
        if self._cur_level not in self._env_cache:
            base = BobbyCarrotEnv(map_kind="normal", map_number=self._cur_level)
            self._env_cache[self._cur_level] = RewardShapingWrapper(
                base, max_episode_steps=self.max_episode_steps)
        self._env = self._env_cache[self._cur_level]
        self._step_count = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        self._step_count += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        if self._step_count >= self.max_episode_steps:
            truncated = True
        if terminated or truncated:
            # Native win reward is +10.0 on the terminal step; shaped win step
            # reward is ~14–15 after the +5.0·(1−steps/max) efficiency bonus.
            # Death/timeout return < 0.
            won = reward > 5.0
            self.record_outcome(self._cur_level, won)
        return obs, reward, terminated, truncated, info

    def close(self):
        for env in self._env_cache.values():
            env.close()
        self._env_cache.clear()
        self._env = None
