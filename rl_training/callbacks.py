"""Training callbacks: TabularLogCallback, WinRateCallback, safe_print.

All stdout output goes through ``safe_print`` and tqdm bars are pinned to
``sys.__stdout__`` to avoid recursion crashes when tqdm.rich or IPython
have hijacked ``sys.stdout``.
"""
from __future__ import annotations

import gc
import sys
import time
import traceback
from typing import List, Optional

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

from bobby_carrot.gym_env import BobbyCarrotEnv
from .config import (
    STAGE_WIN_THRESHOLD,
    STAGE_MIN_STEPS,
    STAGE_MAX_STEPS,
)


# ── Safe stdout helper ───────────────────────────────────────────────────────
def safe_print(msg: str) -> None:
    """Print that won't recurse-crash in Colab.

    tqdm.rich installs a rich.console.Console as sys.stdout.  Bare print()
    from a callback flows through that proxy and IPython's flusher, each
    flushing the other → RecursionError after ~1000 frames.  Routing
    through tqdm.write() pinned to the *real* stdout avoids the loop.

    Call hierarchy:
      safe_print  →  tqdm.write(file=sys.__stdout__)
                        ↓  (writes directly to fd 1)
                  no proxy, no IPython flush, no recursion.
    """
    try:
        tqdm.write(msg, file=sys.__stdout__)
    except Exception:
        try:
            sys.__stdout__.write(msg + "\n")
            sys.__stdout__.flush()
        except Exception:
            print(msg, flush=True)


# ── Tabular Log Callback ─────────────────────────────────────────────────────
class TabularLogCallback(BaseCallback):
    """Replaces SB3's per-rollout box output with a single header + compact rows."""

    COLS = [
        ("time/total_timesteps",     "Timesteps", 11),
        ("rollout/ep_rew_mean",      "ep_rew",     8),
        ("rollout/ep_len_mean",      "ep_len",     7),
        ("time/fps",                 "FPS",        5),
        ("train/approx_kl",          "kl",        10),
        ("train/clip_fraction",      "clip",       7),
        ("train/entropy_loss",       "entropy",    9),
        ("train/explained_variance", "expl_var",   9),
        ("train/value_loss",         "val_loss",   9),
        ("train/loss",               "loss",       9),
    ]

    def __init__(self):
        super().__init__(verbose=0)
        self._header_printed = False

    def _on_training_start(self) -> None:
        logger = self.model.logger
        original_dump = logger.dump
        cb = self

        def _row_dump(step: int = 0) -> None:
            kvs = dict(logger.name_to_value)
            cb._print_row(kvs)
            original_dump(step)

        logger.dump = _row_dump

    def _on_step(self) -> bool:
        return True

    def _print_row(self, kvs: dict) -> None:
        if not self._header_printed:
            header = " | ".join(f"{col:^{w}}" for _, col, w in self.COLS)
            sep    = "-+-".join("-" * w for _, _, w in self.COLS)
            safe_print(header)
            safe_print(sep)
            self._header_printed = True

        parts = []
        for key, _, w in self.COLS:
            v = kvs.get(key, "")
            if isinstance(v, float):
                s = f"{v:.5g}"
            elif v == "":
                s = "-"
            else:
                s = str(v)
            parts.append(f"{s:>{w}}")
        safe_print(" | ".join(parts))


# ── Failure-mode classifier ──────────────────────────────────────────────────
_FAILURE_CATEGORIES = (
    "win",
    "death_crumble",
    "death_other",
    "timeout",
    "stuck_near_exit",
    "unwinnable",
    "unknown",
)


def _classify_episode(
    won: bool,
    terminal_reward: float,
    ep_length: int,
    max_eval_steps: int,
    crumble_death: bool,
    carrot_pct: float,
) -> str:
    """Classify a single episode outcome.

    Categories:
      win              — terminal_reward > 5.0 (native +10 win burst fired)
      death_crumble    — env emitted "died_on_crumble" event
      death_other      — env emitted "died_other" event (no death tiles besides
                          31 in the current tile spec, but kept for forward-
                          compat with future hazards)
      timeout          — episode hit max_eval_steps without terminating
      stuck_near_exit  — timed out with carrot_pct >= 0.9 (collected most
                          items but couldn't reach the exit) — mirrors
                          evaluate.py::_classify
      unwinnable       — Phase-2 forced termination (BFS=-1 / conveyor trap)
                          with terminal reward in (-1.0, -0.5)
      unknown          — fallback
    """
    if won:
        return "win"
    if crumble_death:
        return "death_crumble"
    if ep_length >= max_eval_steps:
        return "stuck_near_exit" if carrot_pct >= 0.9 else "timeout"
    if terminal_reward <= -0.999:
        return "death_other"
    if terminal_reward < -0.4:
        return "unwinnable"
    if terminal_reward < 0.0:
        return "stuck_near_exit"
    return "unknown"


def _classify_failure(
    wins: np.ndarray,
    done_mask: np.ndarray,
    terminal_rewards: np.ndarray,
    ep_lengths: np.ndarray,
    max_eval_steps: int,
    crumble_deaths: np.ndarray | None = None,
    carrot_pcts: np.ndarray | None = None,
) -> tuple:
    """Return (dominant_mode, breakdown_dict) across all episodes.

    ``crumble_deaths[i]``: bool, True if env emitted "died_on_crumble".
    ``carrot_pcts[i]``:    float in [0, 1], collection ratio at episode end.
    Both may be None for callers that don't track them; classification then
    falls back to the reward-only heuristic.
    """
    n = len(wins)
    counts = {k: 0 for k in _FAILURE_CATEGORIES}
    for i in range(n):
        crumble = bool(crumble_deaths[i]) if crumble_deaths is not None else False
        cpct = float(carrot_pcts[i]) if carrot_pcts is not None else 0.0
        mode = _classify_episode(
            won=bool(wins[i]),
            terminal_reward=float(terminal_rewards[i]),
            ep_length=int(ep_lengths[i]),
            max_eval_steps=max_eval_steps,
            crumble_death=crumble,
            carrot_pct=cpct,
        )
        counts[mode] += 1
    non_win = {k: v for k, v in counts.items() if k != "win" and v > 0}
    if not non_win:
        return "win", counts
    # Priority order for tie-breaks: crumble > other > unwinnable > stuck > timeout
    priority = ["death_crumble", "death_other", "unwinnable",
                "stuck_near_exit", "timeout", "unknown"]
    dominant = max(non_win, key=lambda k: (non_win[k], -priority.index(k)))
    return dominant, counts


# ── Win Rate Callback ─────────────────────────────────────────────────────────
class WinRateCallback(BaseCallback):
    """Periodic per-level evaluation with promotion detection.

    Runs vectorized deterministic rollouts per level, logs win rates and
    mean rewards, and triggers stage promotion when the hardest level
    crosses its threshold AND enough steps have been taken.

    Design notes:
      - all status output goes through ``safe_print`` and tqdm bars are
        pinned to ``sys.__stdout__`` to avoid the rich/IPython recursion
        crash (see the safe_print docstring above);
      - eval is wrapped in try/except so a flaky eval cannot kill a
        multi-hour training run;
      - after each eval, ``gc.collect`` + ``torch.cuda.empty_cache``
        returns VRAM to the training loop.
    """

    def __init__(
        self,
        stage: int,
        eval_levels: List[int],
        n_eval_episodes: int = 10,
        check_freq: int = 100_000,
        max_eval_steps: int = 500,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.stage = stage
        self.eval_levels = eval_levels
        self.hardest_level = max(eval_levels)
        self.n_eval_episodes = n_eval_episodes
        self.check_freq = check_freq
        self.max_eval_steps = max_eval_steps
        self.promote = False

        self._stage_start = 0
        self._steps_at_last_check = 0
        self._eval_count = 0
        self._prev_win_rates: dict = {}
        self._prev_mean_rewards: dict = {}
        self._is_recurrent = False

    def _stage_steps(self) -> int:
        return self.num_timesteps - self._stage_start

    def _on_training_start(self) -> None:
        self._stage_start = self.num_timesteps
        self._steps_at_last_check = self.num_timesteps
        try:
            from sb3_contrib import RecurrentPPO
            self._is_recurrent = isinstance(self.model, RecurrentPPO)
        except ImportError:
            self._is_recurrent = False

    # ------------------------------------------------------------------
    # Vectorized eval for one level
    # Returns dict with: win_rate, mean_reward, mean_steps,
    #                    failure_mode, failure_breakdown, carrot_pct
    # ------------------------------------------------------------------
    def _eval_one_level(self, level: int, deterministic: bool = True) -> dict:
        n = self.n_eval_episodes

        def _make(level=level):
            def _init():
                return BobbyCarrotEnv(map_kind="normal", map_number=level)
            return _init

        vec = DummyVecEnv([_make() for _ in range(n)])
        try:
            obs = vec.reset()
            wins = np.zeros(n, dtype=bool)
            done_mask = np.zeros(n, dtype=bool)
            ep_rewards = np.zeros(n, dtype=np.float64)
            ep_lengths = np.zeros(n, dtype=np.int32)
            terminal_rewards = np.zeros(n, dtype=np.float64)
            crumble_deaths = np.zeros(n, dtype=bool)
            carrot_pcts = np.zeros(n, dtype=np.float64)

            lstm_states = None
            episode_starts = np.ones(n, dtype=bool)

            self.model.policy.set_training_mode(False)

            mode = "det" if deterministic else "sto"
            pbar = tqdm(
                total=self.max_eval_steps,
                desc=f"   eval L{level:02d} ({mode}, n={n})",
                leave=False,
                dynamic_ncols=True,
                file=sys.__stdout__,
            )
            with torch.no_grad():
                for _ in range(self.max_eval_steps):
                    if done_mask.all():
                        break

                    if self._is_recurrent:
                        actions, lstm_states = self.model.predict(
                            obs,
                            state=lstm_states,
                            episode_start=episode_starts,
                            deterministic=deterministic,
                        )
                    else:
                        actions, _ = self.model.predict(
                            obs, deterministic=deterministic
                        )

                    # Snapshot pre-step carrot completion for the soon-to-
                    # terminate envs.  obs is a vec dict with leading axis
                    # of size n; auto-reset overwrites it on done=True.
                    pre_carrot = obs["carrot_count"].astype(np.float64)
                    pre_carrot_total = obs["carrot_total"].astype(np.float64)
                    pre_egg = obs["egg_count"].astype(np.float64)
                    pre_egg_total = obs["egg_total"].astype(np.float64)

                    obs, rewards, dones, infos = vec.step(actions)
                    episode_starts = dones

                    for i in range(n):
                        if done_mask[i]:
                            continue
                        ep_rewards[i] += float(rewards[i])
                        ep_lengths[i] += 1
                        if dones[i]:
                            done_mask[i] = True
                            terminal_rewards[i] = float(rewards[i])
                            wins[i] = float(rewards[i]) > 5.0
                            evts = infos[i].get("events", []) or []
                            crumble_deaths[i] = "died_on_crumble" in evts
                            # Approximate carrot completion at termination:
                            # use pre-step counts since the env auto-resets
                            # post-done (info also carries a terminal_obs
                            # in some SB3 versions but not reliably).
                            total = pre_carrot_total[i] + pre_egg_total[i]
                            collected = pre_carrot[i] + pre_egg[i]
                            carrot_pcts[i] = (
                                float(collected) / float(total) if total > 0 else 1.0
                            )

                    pbar.update(1)
                    pbar.set_postfix(done=int(done_mask.sum()), wins=int(wins.sum()))
            pbar.close()

            # For envs that never terminated within max_eval_steps, capture
            # the live carrot completion from the last observation.
            for i in range(n):
                if not done_mask[i]:
                    total = float(obs["carrot_total"][i] + obs["egg_total"][i])
                    collected = float(obs["carrot_count"][i] + obs["egg_count"][i])
                    carrot_pcts[i] = collected / total if total > 0 else 1.0
                    ep_lengths[i] = self.max_eval_steps

            # Classify the dominant failure mode + full breakdown counts.
            failure_mode, failure_breakdown = _classify_failure(
                wins, done_mask, terminal_rewards, ep_lengths,
                self.max_eval_steps, crumble_deaths, carrot_pcts,
            )
            return {
                "win_rate":          float(wins.mean()),
                "mean_reward":       float(ep_rewards.mean()),
                "mean_steps":        float(ep_lengths.mean()),
                "failure_mode":      failure_mode,
                "failure_breakdown": failure_breakdown,
                "carrot_pct":        float(carrot_pcts.mean()),
            }
        finally:
            vec.close()

    def _run_eval(self) -> None:
        t0 = time.time()
        try:
            results: dict = {}            # level -> deterministic result dict
            sto_win_rates: dict = {}      # level -> stochastic win rate
            outer_pbar = tqdm(
                total=len(self.eval_levels),
                desc=f"  [eval stage {self.stage}] levels",
                leave=False,
                dynamic_ncols=True,
                file=sys.__stdout__,
            )
            for level in self.eval_levels:
                outer_pbar.set_postfix_str(f"L{level:02d}")
                # Primary deterministic eval — the official metric.
                det = self._eval_one_level(level, deterministic=True)
                results[level] = det
                # Diagnostic stochastic eval — detects policy-flatness when
                # det wins ≪ sto wins (the failure mode at Stage 1 ent_coef=0.01).
                sto = self._eval_one_level(level, deterministic=False)
                sto_win_rates[level] = sto["win_rate"]
                outer_pbar.update(1)
            outer_pbar.close()

            for level, det in results.items():
                self.logger.record(f"eval/win_rate_L{level:02d}", det["win_rate"])
                self.logger.record(f"eval/mean_reward_L{level:02d}", det["mean_reward"])
                self.logger.record(f"eval/mean_steps_L{level:02d}",  det["mean_steps"])
                self.logger.record(f"eval/carrot_pct_L{level:02d}",  det["carrot_pct"])
                self.logger.record(f"eval/sto_win_rate_L{level:02d}", sto_win_rates[level])
                # Failure-mode counts as separate scalars for TensorBoard
                for cat, count in det["failure_breakdown"].items():
                    if count > 0:
                        self.logger.record(
                            f"eval/fail_{cat}_L{level:02d}", int(count)
                        )

            hardest_wr = results[self.hardest_level]["win_rate"]
            threshold  = STAGE_WIN_THRESHOLD.get(self.stage)
            if threshold is None:
                from .config import LEVEL_WIN_THRESHOLD, LEVEL_MIN_STEPS
                threshold = LEVEL_WIN_THRESHOLD[self.stage]
                min_steps = LEVEL_MIN_STEPS[self.stage]
            else:
                min_steps = STAGE_MIN_STEPS[self.stage]
            stage_steps = self._stage_steps()
            promote_now = hardest_wr >= threshold and stage_steps >= min_steps

            if self.verbose:
                # ── Per-level [EVAL S{stage} @{steps}] lines (§4 format) ───
                for level in self.eval_levels:
                    det = results[level]
                    bd = det["failure_breakdown"]
                    # Compact non-zero breakdown for the failure-mode column
                    nonzero = " ".join(
                        f"{k}={v}" for k, v in bd.items()
                        if v > 0 and k != "win"
                    )
                    fm = (
                        f"{det['failure_mode']}"
                        f"({nonzero})" if nonzero else det['failure_mode']
                    )
                    safe_print(
                        f"  [EVAL S{self.stage} @{self.num_timesteps:,}] "
                        f"L{level:02d}: {det['win_rate']:.0%} wins | "
                        f"{det['mean_steps']:.0f} steps | "
                        f"{det['mean_reward']:+.2f} rew | "
                        f"carrot={det['carrot_pct']:.0%} | {fm}"
                    )

                # ── Stage summary ────────────────────────────────────────
                stage_wr = float(np.mean([r["win_rate"] for r in results.values()]))
                stage_carrot = float(
                    np.mean([r["carrot_pct"] for r in results.values()])
                )
                tag = " → PROMOTE" if promote_now else ""
                safe_print(
                    f"  [EVAL S{self.stage} @{self.num_timesteps:,}] STAGE: "
                    f"{stage_wr:.0%} | carrot_completion: {stage_carrot:.0%} | "
                    f"hardest L{self.hardest_level:02d}="
                    f"{hardest_wr:.0%}/{threshold:.0%}{tag}"
                )

                # ── Policy-flatness diagnostic ──────────────────────────
                # Large det/sto gap ⇒ policy is near-uniform; argmax picks the
                # wrong action even though sampling finds wins.  This is the
                # exact failure pattern at Stage 1 ent_coef=0.01.
                gaps = {
                    l: sto_win_rates[l] - results[l]["win_rate"]
                    for l in self.eval_levels
                }
                flat_levels = [l for l, g in gaps.items() if g >= 0.30]
                if flat_levels:
                    flat_str = " ".join(
                        f"L{l:02d}(det={results[l]['win_rate']:.0%},"
                        f"sto={sto_win_rates[l]:.0%})"
                        for l in flat_levels
                    )
                    safe_print(
                        f"  [POLICY-FLAT WARNING] {flat_str}  "
                        f"→ stochastic ≫ deterministic; consider lowering ent_coef"
                    )

                # ── Delta vs previous eval ───────────────────────────────
                if self._eval_count == 0:
                    safe_print(
                        f"  [EVAL S{self.stage}] first eval — subsequent evals "
                        f"print deltas (took {time.time()-t0:.1f}s)"
                    )
                else:
                    improved = regressed = flat = 0
                    for l in self.eval_levels:
                        d_wr = results[l]["win_rate"] - self._prev_win_rates.get(l, 0.0)
                        if d_wr >= 0.05:
                            improved += 1
                        elif d_wr <= -0.05:
                            regressed += 1
                        else:
                            flat += 1
                    verdict = "LEARNING" if improved > 0 else "STALLED"
                    safe_print(
                        f"  [EVAL S{self.stage}] {verdict}  "
                        f"up={improved} down={regressed} flat={flat}  "
                        f"(took {time.time()-t0:.1f}s)"
                    )

                self._prev_win_rates = {l: results[l]["win_rate"] for l in self.eval_levels}
                self._prev_mean_rewards = {l: results[l]["mean_reward"] for l in self.eval_levels}
                self._eval_count += 1

            if promote_now:
                if self.verbose:
                    safe_print(f"  → Promotion threshold met! Stopping stage {self.stage}.")
                self.promote = True

        except Exception as exc:
            safe_print(f"\n  [WinRateCallback] EVAL FAILED ({type(exc).__name__}): {exc}")
            try:
                traceback.print_exc(file=sys.__stdout__)
            except Exception:
                pass
            safe_print("  [WinRateCallback] continuing training despite eval failure")
        finally:
            self.model.policy.set_training_mode(True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self._steps_at_last_check < self.check_freq:
            return
        self._steps_at_last_check = self.num_timesteps
        self._run_eval()

    def _on_step(self) -> bool:
        return not self.promote


# ── Stage Progress Callback ──────────────────────────────────────────────────
class StageProgressCallback(BaseCallback):
    """Displays a tqdm progress bar tracking timesteps within the current stage.

    Updates every rollout to avoid overhead.  The bar shows
    ``num_timesteps / max_steps`` for the stage and includes the current
    episode reward and length in the postfix.
    """

    def __init__(self, stage: int, resume_steps: int = 0):
        super().__init__(verbose=0)
        self.stage = stage
        self._resume_steps = resume_steps
        self._pbar = None
        self._stage_start = 0

    def _on_training_start(self) -> None:
        self._stage_start = self.num_timesteps
        max_steps = STAGE_MAX_STEPS.get(self.stage)
        if max_steps is None:
            from .config import LEVEL_MAX_STEPS
            max_steps = LEVEL_MAX_STEPS[self.stage]
        initial = self._resume_steps
        self._pbar = tqdm(
            total=max_steps,
            initial=initial,
            desc=f"  Stage {self.stage}",
            unit="step",
            unit_scale=True,
            dynamic_ncols=True,
            file=sys.__stdout__,
            miniters=1,
        )

    def _on_rollout_end(self) -> None:
        if self._pbar is None:
            return
        stage_steps = self.num_timesteps - self._stage_start + self._resume_steps
        self._pbar.n = min(stage_steps, self._pbar.total)
        self._pbar.refresh()

        # Show latest episode stats in the postfix
        try:
            info = self.model.logger.name_to_value
            ep_rew = info.get("rollout/ep_rew_mean", None)
            ep_len = info.get("rollout/ep_len_mean", None)
            postfix = {}
            if ep_rew is not None:
                postfix["rew"] = f"{ep_rew:.1f}"
            if ep_len is not None:
                postfix["len"] = f"{ep_len:.0f}"
            if postfix:
                self._pbar.set_postfix(postfix)
        except Exception:
            pass

    def _on_training_end(self) -> None:
        if self._pbar:
            self._pbar.close()
            self._pbar = None

    def _on_step(self) -> bool:
        return True
