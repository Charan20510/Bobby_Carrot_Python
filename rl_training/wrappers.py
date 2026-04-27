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

from .potential import compute_potential, bfs_to_exit
from .config import (
    STAGE_ALL_LEVELS,
    STAGE_NEW_START,
    EXIT_APPROACH_BONUS,
    EXIT_RETREAT_PENALTY,
    POST_COLLECT_STEP_PENALTY,
    ALL_COLLECTED_BONUS,
)


class RewardShapingWrapper(gym.Wrapper):
    """Potential-based reward shaping + exit-seeking bonus + key-pickup bonus.

    Shaped reward on each step:
        r'(s,a,s') = r(s,a,s') + γ·Φ(s') - Φ(s)   (Ng et al. 1999)

    On terminal steps the underlying env reloads/advances before we can
    read the next state.  For wins Φ(s') = 0 (correct terminal value).
    For deaths Φ(s') = Φ(s) so shaping ≈ (γ-1)·Φ ≈ 0, keeping the
    -1.0 death penalty as the sole death signal.

    Exit-seeking (the critical fix for 0% win rate):
        After all collectibles are gathered, gives +0.3 per step that
        moves closer to the exit, -0.1 per step that moves away.  This
        is 50× stronger than the bare potential shaping gradient (~0.006).
        Also increases the step penalty from -0.01 to -0.05 to create
        urgency in reaching the exit.

    Bonuses:
        +key_pickup:   +0.3   (matters from stage 5 onward; harmless earlier)
        +efficiency:   +5.0 × (1 - steps/max_steps)  on win
    """
    GAMMA = 0.995
    EFFICIENCY_BONUS = 5.0

    # ── Edge-case termination tunables (Phase 2) ───────────────────────────
    # Conveyor-trap: 3 consecutive same-position steps ON a conveyor tile.
    # Two conveyors pointing at each other trap the agent forever; without
    # this, episodes burn the full 500-step horizon in a stuck state.
    CONVEYOR_TILE_IDS = frozenset({24, 25, 26, 27, 28, 29, 40, 41, 42, 43})
    CONVEYOR_TRAP_THRESHOLD = 3      # steps stuck before forced termination
    CONVEYOR_TRAP_PENALTY   = 0.5

    # Loop-detection: penalise revisiting (x,y) > LOOP_VISIT_LIMIT times since
    # the last collection event.  Counter resets on carrot/egg/key pickup.
    LOOP_VISIT_LIMIT = 3
    LOOP_PENALTY     = 0.1

    # Unwinnable: BFS potential collapses to -1.0 when no target is reachable
    # (typically because crumble-critical tiles all collapsed).  Terminate
    # immediately rather than wandering 500 steps in a lost state.
    UNWINNABLE_PENALTY = 1.0

    def __init__(self, env, max_episode_steps: int = 500):
        super().__init__(env)
        self._max_steps = max_episode_steps
        self._prev_potential = 0.0
        self._step_count = 0
        self._prev_exit_dist = None
        self._all_collected = False

        # Phase 2 state
        self._visit_count: dict = {}        # (x,y) → visit count since last event
        self._stuck_steps = 0                # consecutive same-position steps

    DEBUG_RESET = False   # flip to True to log per-reset potential

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        self._prev_exit_dist = None
        self._all_collected = False
        self._visit_count = {}
        self._stuck_steps = 0
        gs = self._get_gs()
        self._prev_potential = compute_potential(gs) if gs else 0.0
        # Zero-collectibles edge case: if a map starts already collected,
        # initialise exit-seeking state immediately so the +0.5/-0.3 signal
        # is active from step 1 instead of waiting for a phase transition
        # that has already happened.
        if gs and gs.carrot_count >= gs.carrot_total and gs.egg_count >= gs.egg_total:
            self._all_collected = True
            self._prev_exit_dist = bfs_to_exit(gs)
        if self.DEBUG_RESET:
            import sys
            sys.__stdout__.write(
                f"[RewardShapingWrapper.reset] prev_potential="
                f"{self._prev_potential:.4f}  all_collected={self._all_collected}\n"
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

        # ── Phase 2 edge cases (loop / conveyor-trap / unwinnable) ──────
        # Run BEFORE the exit-seeking block so an unwinnable termination
        # cancels potential shaping cleanly.
        if not terminated:
            gs = self._get_gs()
            if gs is not None:
                pos = gs.coord_src

                # Loop detection: visit-count, reset on collection events.
                # Frees the agent from grinding over a single cell when the
                # potential gradient is flat (e.g. before any phase change).
                evts = info.get("events", []) or []
                if any(e in ("carrot_collected", "egg_collected", "key_picked")
                       for e in evts):
                    self._visit_count = {}
                visits = self._visit_count.get(pos, 0) + 1
                self._visit_count[pos] = visits
                if visits > self.LOOP_VISIT_LIMIT:
                    reward -= self.LOOP_PENALTY

                # Conveyor-trap: same position 3 consecutive steps AND
                # standing on a conveyor tile (forced movement). Two
                # conveyors pointing at each other otherwise burn the full
                # 500-step horizon.
                if prev_pos is not None and pos == prev_pos:
                    self._stuck_steps += 1
                else:
                    self._stuck_steps = 0
                if self._stuck_steps >= self.CONVEYOR_TRAP_THRESHOLD:
                    tile_id = gs.tiles[pos[0] + pos[1] * 16]
                    if tile_id in self.CONVEYOR_TILE_IDS:
                        reward -= self.CONVEYOR_TRAP_PENALTY
                        terminated = True

                # Unwinnable: BFS to nearest target unreachable.  Triggers
                # for crumble-critical levels where the wrong crumble
                # collapsed (e.g. L03 if all 3 crumbles are exhausted).
                # compute_potential returns -1.0 when _bfs_distance returns
                # -1; we read it directly to avoid a redundant BFS call.
                if not terminated:
                    pot = compute_potential(gs)
                    if pot <= -0.999:   # _bfs_distance unreachable sentinel
                        reward -= self.UNWINNABLE_PENALTY
                        terminated = True

        if not terminated:
            gs = self._get_gs()
            curr_potential = compute_potential(gs) if gs else 0.0

            # ── Exit-seeking reward (the critical fix) ──────────────────
            # After all collectibles are gathered, give explicit per-step
            # reward for approaching the exit.  The bare potential shaping
            # gradient (~0.006/step) is 150× weaker than carrot pickup
            # (+1.0) and gets lost in noise.  This 0.3/step bonus makes
            # exit-seeking a clear, learnable signal.
            if gs:
                all_collected = (
                    gs.carrot_count >= gs.carrot_total
                    and gs.egg_count >= gs.egg_total
                )
                if all_collected:
                    if not self._all_collected:
                        # First step after collecting all: initialise and
                        # give a one-time bonus to mark the phase transition
                        self._all_collected = True
                        self._prev_exit_dist = bfs_to_exit(gs)
                        reward += ALL_COLLECTED_BONUS

                    curr_exit_dist = bfs_to_exit(gs)
                    if (self._prev_exit_dist is not None
                            and self._prev_exit_dist >= 0
                            and curr_exit_dist >= 0):
                        delta = self._prev_exit_dist - curr_exit_dist
                        if delta > 0:
                            reward += EXIT_APPROACH_BONUS
                        elif delta < 0:
                            reward -= EXIT_RETREAT_PENALTY
                    self._prev_exit_dist = curr_exit_dist

                    # Increased step penalty after collection to create
                    # urgency: -0.05 instead of the base -0.01
                    reward -= POST_COLLECT_STEP_PENALTY

        elif reward > 5.0:
            # Win: all goals reached; Φ(terminal) = 0 by definition.
            curr_potential = 0.0
        else:
            # Death OR Phase-2 forced termination (unwinnable / conveyor
            # trap): cancel spurious potential shaping; the per-edge-case
            # negative reward is the sole signal.
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
