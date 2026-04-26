"""Training configuration constants and hyperparameters.

All STAGE_* dicts, PPO kwargs, and environment sizing live here so they
are defined in exactly one place.  The notebook and the training loop both
import from this module.
"""
from __future__ import annotations

# ── Level pools per curriculum stage ─────────────────────────────────────────
STAGE_ALL_LEVELS = {
    1: list(range(1,  4)),   # L01–L03: walls + floor + carrots
    2: list(range(1,  8)),   # L01–L07: + crumble tiles
    3: list(range(1, 13)),   # L01–L12: + directional conveyors
    4: list(range(1, 18)),   # L01–L17: + global switches
    5: list(range(1, 26)),   # L01–L25: + keys/locks, full mix
}
STAGE_NEW_START = {1: 1, 2: 4, 3: 8, 4: 13, 5: 18}

# Win-rate thresholds for adaptive stage promotion (on hardest level of stage)
STAGE_WIN_THRESHOLD = {1: 0.70, 2: 0.65, 3: 0.55, 4: 0.50, 5: 0.45}
# Minimum and maximum steps per stage regardless of win rate.
STAGE_MIN_STEPS = {
    1: 100_000, 2: 750_000, 3: 2_000_000, 4: 3_000_000, 5: 4_000_000,
}
STAGE_MAX_STEPS = {
    1: 5_000_000, 2: 8_000_000, 3: 12_000_000, 4: 15_000_000, 5: 20_000_000,
}

# Per-stage episode horizons.
STAGE_MAX_EPISODE_STEPS = {1: 500, 2: 500, 3: 500, 4: 600, 5: 600}

# Per-stage entropy coefficient.
# Stage 1 was originally 0.02 — far too high; 0.005 is the SB3 default and
# lets the policy commit without premature convergence.
STAGE_ENT_COEF = {1: 0.01, 2: 0.01, 3: 0.01, 4: 0.005, 5: 0.005}

# Per-stage eval cadence and episodes.
STAGE_CHECK_FREQ = {
    1: 100_000, 2: 250_000, 3: 500_000, 4: 1_000_000, 5: 1_000_000,
}
STAGE_EVAL_EPS = {1: 20, 2: 15, 3: 10, 4: 10, 5: 10}

# ── PPO hyperparameters ──────────────────────────────────────────────────────
# Stages 1-2: standard PPO (fast, no LSTM overhead)
# Stages 3-5: RecurrentPPO with LSTM (better multi-step planning)
RECURRENT_FROM_STAGE = 3   # set to 6 to disable RecurrentPPO entirely

# ── Exit-seeking reward constants ────────────────────────────────────────────
# After all collectibles are gathered, reward/penalise movement toward/away
# from the exit tile.  Without this, the BFS potential shaping alone gives
# only ~0.006 per step — 150× weaker than a carrot (+1.0) — so the agent
# collects all carrots and then wanders.
EXIT_APPROACH_BONUS  = 0.5    # per-step bonus for moving closer to exit
EXIT_RETREAT_PENALTY = 0.3    # per-step penalty for moving away from exit
POST_COLLECT_STEP_PENALTY = 0.05  # increased step penalty after all collected
ALL_COLLECTED_BONUS  = 2.0    # one-time bonus when all items are collected

N_STEPS    = 256   # shorter rollouts → more frequent updates

# Batch size: keep GPU saturated but don't bloat CPU RAM with rollout buffers.
BATCH_SIZE = 512

# Shared PPO kwargs (used for both PPO and RecurrentPPO)
PPO_BASE_KWARGS = dict(
    n_steps       = N_STEPS,
    batch_size    = BATCH_SIZE,
    n_epochs      = 10,
    gamma         = 0.995,    # higher discount for long-horizon planning
    gae_lambda    = 0.97,
    clip_range    = 0.2,
    max_grad_norm = 0.5,
)

# RecurrentPPO-specific: LSTM hidden size
RECURRENT_KWARGS = dict(
    **PPO_BASE_KWARGS,
    lstm_hidden_size = 256,
    n_lstm_layers    = 1,
)


def get_n_envs(cpu_ram_gb: float | None = None) -> int:
    """Return the number of DummyVecEnv workers based on CPU RAM.

    DummyVecEnv runs all envs sequentially in the main process.  The game
    runs at ~2 k headless steps/s per env, so 8 envs gives ~16 k steps/s —
    well above GPU-bound throughput.
    """
    if cpu_ram_gb is None:
        try:
            import psutil
            cpu_ram_gb = psutil.virtual_memory().total / 1e9
        except ImportError:
            cpu_ram_gb = 12.0
    if cpu_ram_gb >= 50:
        return 32
    elif cpu_ram_gb >= 25:
        return 16
    else:
        return 8


def get_policy_kwargs(features_dim: int = 256) -> dict:
    """Return policy_kwargs dict referencing BobbyExtractor."""
    from .extractor import BobbyExtractor
    return dict(
        features_extractor_class  = BobbyExtractor,
        features_extractor_kwargs = {"features_dim": features_dim},
        net_arch                  = [],   # extractor outputs 256-dim; no extra MLP
    )
