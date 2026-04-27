"""rl_training — Bobby Carrot PPO training package.

Extracted from bobby_carrot_rl.ipynb for maintainability and testability.
The notebook imports from this package instead of inlining 1000+ lines of
training infrastructure.

Note: Submodules that depend on stable_baselines3 / sb3_contrib (extractor,
callbacks, train) are NOT eagerly imported here.  Import them directly when
needed so the package is importable in environments without SB3 installed.
"""

from .config import (
    STAGE_ALL_LEVELS,
    STAGE_NEW_START,
    STAGE_WIN_THRESHOLD,
    STAGE_MIN_STEPS,
    STAGE_MAX_STEPS,
    STAGE_MAX_EPISODE_STEPS,
    STAGE_ENT_COEF,
    STAGE_CHECK_FREQ,
    STAGE_EVAL_EPS,
    RECURRENT_FROM_STAGE,
    EXIT_APPROACH_BONUS,
    EXIT_RETREAT_PENALTY,
    POST_COLLECT_STEP_PENALTY,
    N_STEPS,
    BATCH_SIZE,
    PPO_BASE_KWARGS,
    RECURRENT_KWARGS,
    get_n_envs,
    get_policy_kwargs,
    INDIVIDUAL_LEVELS,
    LEVEL_MIN_STEPS,
    LEVEL_MAX_STEPS,
    LEVEL_MAX_EPISODE_STEPS,
    LEVEL_ENT_COEF,
    LEVEL_WIN_THRESHOLD,
    LEVEL_CHECK_FREQ,
    LEVEL_EVAL_EPS,
    LEVEL_USE_RECURRENT,
)
from .potential import compute_potential, bfs_to_exit

# Gymnasium wrappers — require gymnasium + bobby_carrot but NOT stable_baselines3
try:
    from .wrappers import RewardShapingWrapper, CurriculumEnv, SingleLevelEnv
except ImportError:
    pass

# SB3-dependent modules — lazy import; these fail without stable_baselines3
# Import them directly: from rl_training.extractor import BobbyExtractor
# from rl_training.callbacks import TabularLogCallback, WinRateCallback
# from rl_training.train import run_training, run_level_training
# from rl_training.evaluate import run_evaluation

