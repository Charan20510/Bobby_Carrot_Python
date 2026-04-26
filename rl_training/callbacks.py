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
            print(header)
            print(sep)
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
        print(" | ".join(parts))
        sys.stdout.flush()


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
        max_eval_steps: int = 250,
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
    # Vectorized eval for one level — returns (win_rate, mean_reward)
    # ------------------------------------------------------------------
    def _eval_one_level(self, level: int) -> tuple:
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

            lstm_states = None
            episode_starts = np.ones(n, dtype=bool)

            self.model.policy.set_training_mode(False)

            pbar = tqdm(
                total=self.max_eval_steps,
                desc=f"   eval L{level:02d} (n={n})",
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
                            deterministic=True,
                        )
                    else:
                        actions, _ = self.model.predict(obs, deterministic=True)

                    obs, rewards, dones, infos = vec.step(actions)
                    episode_starts = dones

                    for i in range(n):
                        if done_mask[i]:
                            continue
                        ep_rewards[i] += float(rewards[i])
                        if dones[i]:
                            done_mask[i] = True
                            wins[i] = float(rewards[i]) > 5.0

                    pbar.update(1)
                    pbar.set_postfix(done=int(done_mask.sum()), wins=int(wins.sum()))
            pbar.close()
            return float(wins.mean()), float(ep_rewards.mean())
        finally:
            vec.close()

    def _run_eval(self) -> None:
        t0 = time.time()
        try:
            win_rates = {}
            mean_rewards = {}
            outer_pbar = tqdm(
                total=len(self.eval_levels),
                desc=f"  [eval stage {self.stage}] levels",
                leave=False,
                dynamic_ncols=True,
                file=sys.__stdout__,
            )
            for level in self.eval_levels:
                outer_pbar.set_postfix_str(f"L{level:02d}")
                wr, mr = self._eval_one_level(level)
                win_rates[level] = wr
                mean_rewards[level] = mr
                outer_pbar.update(1)
            outer_pbar.close()

            for level, wr in win_rates.items():
                self.logger.record(f"eval/win_rate_L{level:02d}", wr)
                self.logger.record(f"eval/mean_reward_L{level:02d}", mean_rewards[level])

            hardest_wr = win_rates.get(self.hardest_level, 0.0)
            threshold  = STAGE_WIN_THRESHOLD[self.stage]
            min_steps  = STAGE_MIN_STEPS[self.stage]
            stage_steps = self._stage_steps()
            took = time.time() - t0

            if self.verbose:
                # Format as: L01=80%(r=12.4) L02=70%(r=11.8) ...
                wr_str = " ".join(
                    f"L{l:02d}={win_rates[l]:.0%}(r={mean_rewards[l]:+.1f})"
                    for l in self.eval_levels
                )
                safe_print(
                    f"\n  [WinRateCallback] stage={self.stage}  {wr_str}"
                    f"  threshold={threshold:.0%}  stage_steps={stage_steps:,}"
                    f"  min={min_steps:,}  took={took:.1f}s"
                )

                # Compact per-level win-rate summary in the requested format
                level_summary = ", ".join(
                    f"L{l}={win_rates[l]:.0%}" for l in self.eval_levels
                )
                safe_print(f"  [LEVEL RESULTS] {level_summary}")

                if self._eval_count == 0:
                    safe_print(
                        "  [WinRateCallback] PRIORS (first eval) — "
                        "subsequent evals will print per-level deltas"
                    )
                else:
                    deltas = []
                    improved = regressed = flat = 0
                    for l in self.eval_levels:
                        d_wr = win_rates[l] - self._prev_win_rates.get(l, 0.0)
                        d_mr = mean_rewards[l] - self._prev_mean_rewards.get(l, 0.0)
                        if d_wr >= 0.05 or d_mr >= 0.5:
                            tag = "UP"
                            improved += 1
                        elif d_wr <= -0.05 or d_mr <= -0.5:
                            tag = "DOWN"
                            regressed += 1
                        else:
                            tag = "FLAT"
                            flat += 1
                        deltas.append(
                            f"L{l:02d}:{tag}(dwr={d_wr:+.0%},dr={d_mr:+.1f})"
                        )
                    verdict = "LEARNING" if improved > 0 else "STALLED"
                    safe_print(
                        f"  [WinRateCallback] {verdict}  "
                        f"up={improved} down={regressed} flat={flat}  "
                        + " ".join(deltas)
                    )

                self._prev_win_rates = dict(win_rates)
                self._prev_mean_rewards = dict(mean_rewards)
                self._eval_count += 1

            if hardest_wr >= threshold and stage_steps >= min_steps:
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
