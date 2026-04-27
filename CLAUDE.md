# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Pure-Python port of the Bobby Carrot tile-puzzle game, structured so it can be played interactively *and* used as a headless RL training environment. All game assets (tiles, audio, `.blm` level files) live under `assets/`.

The README still mentions a Rust launcher (`run.py`); it is not in the tree. The active code is the `bobby_carrot/` Python package.

## Commands

Setup (editable install + pygame for interactive play):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pygame        # only needed for interactive play / rendering
pip install gymnasium numpy   # only needed for the gym wrapper / RL training
```

Running:

```bash
python -m bobby_carrot [map]            # interactive play, e.g. `5`, `normal-3`, `egg-10`
python -m bobby_carrot.bench [N] [MAP]  # headless steps/sec benchmark
```

There is no test suite, lint config, or build step. `bench.py` is the closest thing to a smoke test (`python -m bobby_carrot.bench 10000`).

RL training & evaluation:

```bash
# Train all 5 stages (resumes from latest checkpoint per stage automatically)
python -c "from rl_training.train import run_training; run_training('.', 'cuda', 8)"

# Headless held-out evaluation on L26–L30
python -c "from rl_training.evaluate import run_evaluation; print(run_evaluation('.'))"

# GUI / visual evaluation (Pygame rendering of a trained agent)
python rl_training/evaluate_gui.py --model models/stage_5/best_model.zip --level 26 --fps 5
python rl_training/evaluate_gui.py --model models/stage_5/best_model.zip --levels 26-30 --episodes 20 --headless --report
```

The Colab notebook `bobby_carrot_rl.ipynb` is now a thin wrapper that imports `rl_training.*` — long-form training/eval logic lives in the package, not the notebook.

## Architecture

The codebase is built around a strict **logic / renderer separation** so RL training does not pay any rendering cost (~175k logical steps/sec headless vs. 60 FPS interactive).

```
bobby_carrot/                  Game core (no RL, no training)
├── core/
│   ├── state.py    GameState dataclass, State + Action enums, tile/grid constants. NO pygame.
│   ├── loader.py   Map, MapInfo, .blm parsing, CLI map-arg parsing. NO pygame.
│   └── logic.py    Pure logic: start_move, apply_landing, advance_frame, logical_step. NO pygame.
├── env.py          GameEnv: headless reset/step/observe Gym-style API. NO pygame.
├── renderer.py     Renderer + Assets + compute_sprite_rects. ALL pygame code lives here.
├── game.py         Slim main(): wires GameEnv + Renderer + pygame event loop for interactive play.
├── gym_env.py      BobbyCarrotEnv(gymnasium.Env) wrapper around GameEnv.
└── bench.py        Steps/sec benchmark for the headless env.

rl_training/                   PPO training stack (extracted from the notebook)
├── config.py       STAGE_* dicts, PPO/RecurrentPPO kwargs, get_n_envs, get_policy_kwargs.
├── potential.py    BFS shortest-path potential Φ(s) and bfs_to_exit (replaced Manhattan).
├── wrappers.py     RewardShapingWrapper (potential + exit-seeking + bonuses) and CurriculumEnv.
├── extractor.py    BobbyExtractor — tile-embed → CNN+ResBlock → 4-head self-attn → MLP fuse.
├── callbacks.py    TabularLogCallback, WinRateCallback, StageProgressCallback, safe_print.
├── train.py        run_training: 5-stage loop with checkpoint resume, heartbeat, warm-start.
├── evaluate.py     run_evaluation: held-out per-level metrics with failure-mode classification.
└── evaluate_gui.py CLI tool: Pygame rendering, frame-saving, batch report (importable too).
```

`rl_training/__init__.py` re-exports config + potential eagerly and `wrappers` if `gymnasium` is available; `extractor`, `callbacks`, `train`, `evaluate` require `stable_baselines3` / `sb3_contrib` and are imported lazily so the package can be loaded in environments without SB3.

**Invariant:** `core/` and `env.py` must never import pygame. `renderer.py` is the *only* pygame entry point and is imported lazily from `game.py`/`gym_env.py` after `pygame.init()`. Breaking this invariant kills RL throughput.

**Two execution modes share the same logic module:**

- **Interactive (60 FPS):** `game.py` calls `advance_frame(gs)` once per render frame. Tile effects fire at animation step 8 (`step = (gs.frame - gs.start_frame) // FRAMES_PER_STEP`). The renderer reads `gs` and computes sprite rects via `compute_sprite_rects(gs)` — pure, no mutation.
- **Headless (RL):** `env.py` calls `logical_step(gs, action)` once per agent action. It collapses one full tile move + chained conveyor moves immediately, with no frame counting. `GameEnv.reset()` starts the player in `State.Down` (skipping FadeIn) so an agent can act on the first step; `game.py._reset_interactive()` overrides this to `State.FadeIn` for the visible intro animation.

**Key split that enables this:** the original `Bobby.update_texture_position()` was decomposed into `apply_landing(gs)` (logic-only tile effects: carrot/egg pickup, switches, conveyor entry) in `logic.py`, and `compute_sprite_rects(gs)` (rendering math) in `renderer.py`.

**Tile IDs are hardcoded throughout `logic.py`** (e.g. 19 = carrot, 22/38 = red/yellow switches, 24–29 = red-controlled conveyors, 30→31 = crumble, 32–37 = key/door pairs by colour, 40–43 = forced-direction conveyors, 44 = exit, 45/46 = egg/empty). When changing tile semantics, both the entry restrictions in `_check_dest` and the effect tables in `apply_landing` (and the toggle helpers `_toggle_red_switch`/`_toggle_yellow_switch`) must stay in sync.

**Native reward** (`GameEnv._compute_reward`): −0.01 per step, +1 per carrot/egg, +10 on level complete, −1 on death. Death and win both set `terminated=True`; on win, the env auto-advances to `Map.next()` before returning. The `+10.0` win burst is the only signal that crosses the `+5.0` win-detection threshold used by callbacks and eval.

**Shaped reward** (`rl_training/wrappers.py::RewardShapingWrapper`) layers on top of the native reward and is what training actually sees:
- Potential-based shaping `r' += γΦ(s') − Φ(s)` with `γ=0.995` (Ng et al. 1999). `Φ(s) = −BFS_dist/32` toward nearest carrot/egg, switching to `−BFS_dist/16` toward the exit once all collectibles are gathered (the stronger normaliser makes the exit gradient compete with the carrot signal). On wins `Φ(terminal)=0`; on deaths `Φ(s')=Φ(s)` so `(γ−1)·Φ ≈ 0` and the native `−1.0` death penalty stays the sole death signal.
- **Exit-seeking block** (added in commit `Target Navigation`, the critical fix for 0% L01–L03 win rate at 500K steps). After all items are collected: `+ALL_COLLECTED_BONUS=2.0` one-time, then per-step `+EXIT_APPROACH_BONUS=0.5` / `−EXIT_RETREAT_PENALTY=0.3` based on BFS-distance delta to the exit, plus an extra `−POST_COLLECT_STEP_PENALTY=0.05` urgency penalty. Without this, bare potential shaping (~0.006/step) was 150× weaker than a carrot pickup and the agent would collect everything then wander.
- Blocked-move penalty: `−0.05` when the player tried to move but `coord_src` didn't change (wall bumps / IDLE).
- Key-pickup bonus: `+0.3` on `events == "key_picked"` (matters from stage 5; harmless earlier).
- Efficiency bonus on win: `+5.0 · (1 − steps / max_steps)`.

**Frame counter convention:** `gs.frame` is a unitless tick that increments once per interactive render frame *and* once per `logical_step`. `last_action_time` and `start_time` are stored as `gs.frame` snapshots, not milliseconds — do not mix units.

## Project goal

Train an RL agent on **normal levels 1–25** via a 5-stage curriculum and evaluate it **zero-shot on held-out normal levels 26–30**. Levels 26–30 are never seen during training; they are the sole test set. Success is measured per-level by:

- **Win rate** over 50 deterministic episodes (`model.predict(deterministic=True)`),
- **Mean episode reward** (shaped),
- **Mean steps** to win,
- **Failure-mode breakdown**: death / timeout / stuck-near-exit / unknown,
- **Mean carrot completion %**.

A win is detected by the *terminal* shaped reward exceeding `+5.0`. The native `+10.0` win burst is the only thing that pushes a terminal step that high; the per-step shaped values (carrot=+1, exit-approach=+0.5, etc.) stay well below it. Training/promotion lives in `rl_training/train.py` + `callbacks.py`; held-out eval in `rl_training/evaluate.py` (and `evaluate_gui.py` for the visual variant).

Curriculum stages (`rl_training/config.py::STAGE_ALL_LEVELS`):

| Stage | Levels  | New mechanic                          |
|-------|---------|---------------------------------------|
| 1     | L01–L03 | Walls, floor, carrots                 |
| 2     | L01–L07 | + Crumble tiles                       |
| 3     | L01–L12 | + Directional conveyors               |
| 4     | L01–L17 | + Red/Yellow switches                 |
| 5     | L01–L25 | + Keys/Locks, full mix                |

## Coding standards

- **Logic / renderer separation is non-negotiable.** Anything under `bobby_carrot/core/` and `bobby_carrot/env.py` must stay pygame-free (see invariant above). RL throughput depends on it.
- **Pure functions for game logic.** `apply_landing`, `start_move`, `advance_frame`, `logical_step`, and `compute_sprite_rects` take `GameState` and return new state / sprite rects — they don't reach into globals or pygame.
- **Type hints on public surface.** Module-level functions, class methods, and dataclass fields are annotated (see [bobby_carrot/gym_env.py](bobby_carrot/gym_env.py), `core/state.py`).
- **Comments explain *why*, not *what*.** Modules in `rl_training/` and notebook cells routinely cite the incident or paper that justifies a value (e.g. "lowered from 500K → 100K because…", "Ng et al. 1999", "BFS replaces Manhattan: L01 was 7 by Manhattan but 13 by BFS"). Avoid restating what the code already says; do record the failure mode that motivated a tuning change.
- **Per-stage configuration is dict-keyed by stage int** in `rl_training/config.py`: `STAGE_ALL_LEVELS`, `STAGE_NEW_START`, `STAGE_WIN_THRESHOLD`, `STAGE_MIN_STEPS`, `STAGE_MAX_STEPS`, `STAGE_MAX_EPISODE_STEPS`, `STAGE_ENT_COEF`, `STAGE_CHECK_FREQ`, `STAGE_EVAL_EPS`. Adding a stage means adding entries to all of them; changing a curve means editing one dict, not branching on `if stage == k` throughout.
- **Defensive callbacks.** `WinRateCallback._run_eval` wraps the entire eval in `try/except` so a flaky eval cannot kill a multi-hour training run. Heartbeat / stall-warning threads are daemons and never raise.
- **Stdout safety in Colab.** Never call bare `print` from a callback while `tqdm.rich` / IPython has hijacked `sys.stdout` — it recurses and crashes. Use `safe_print` from `rl_training/callbacks.py` (routes through `tqdm.write(file=sys.__stdout__)`) and pin tqdm bars to `sys.__stdout__`. Keep `progress_bar=False` on `model.learn(...)`.
- **RecurrentPPO state threading.** When the loaded model is `RecurrentPPO`, every `model.predict` call must thread `state=lstm_states` and `episode_start=episode_starts`, and reset the start mask whenever an env terminates. `WinRateCallback._eval_one_level`, `evaluate.run_evaluation`, and `evaluate_gui.play_level_gui` all branch on `is_recurrent` for this reason.
- **Tile IDs are the cross-cutting contract.** Magic numbers (19 carrot, 22/38 switches, 24–29 red conveyors, 30→31 crumble, 32–37 keys/locks, 40–43 forced conveyors, 44 exit, 45/46 egg/empty) appear in `core/logic.py`, `_check_dest`, `apply_landing`, `_toggle_*_switch`, *and* in `rl_training/potential.py` (BFS walkability uses `tile >= 18 and tile != 31`; targets scan for 19, 45, 44). When changing tile semantics, update all of them in one PR.
- **Frame counter is a tick, not milliseconds.** See the frame-counter convention above. Do not mix units.
- **No new top-level files casually.** The layout is intentional: game logic in `bobby_carrot/`, RL wiring in `rl_training/`, benchmarks in `bench.py`, gym wrapping in `gym_env.py`. The notebook is a thin orchestrator over `rl_training`, not the source of truth — add training code to `rl_training/`, not to new notebook cells.

## Architecture decisions

1. **Headless `GameEnv` separate from `BobbyCarrotEnv` (gym wrapper).** `core/` + `env.py` know nothing about gymnasium *or* pygame. `gym_env.py` adapts `GameEnv` to the gymnasium API and converts the dict observation. This lets `bench.py` skip both layers and lets the renderer be imported lazily after `pygame.init()`.
2. **`DummyVecEnv`, not `SubprocVecEnv`.** Each env runs ~2k headless steps/s, so 8 in-process envs give ~16k steps/s — already past GPU-bound. Subprocesses would cost CPU RAM (rollout buffer × N_ENVS) for no throughput gain. `N_ENVS` is auto-picked from CPU RAM (≥50 GB → 32, ≥25 GB → 16, else 8) by `rl_training.config.get_n_envs`.
3. **5-stage curriculum keyed by mechanic introduction.** Each stage adds the next class of tile (crumble → conveyors → switches → keys). Stage k+1 warm-starts from stage k's `best_model.zip` / `_final.zip` when (and only when) the algorithm class matches; otherwise it starts fresh (`rl_training/train.py::run_training`, `_use_recurrent` / warm-start branch).
4. **Adaptive stage promotion.** Promotion fires when the *hardest* level in the stage hits `STAGE_WIN_THRESHOLD[stage]` *and* total stage steps cleared `STAGE_MIN_STEPS[stage]`. Hard cap at `STAGE_MAX_STEPS[stage]`. Stage-local step counters mean `reset_num_timesteps=False` resumes do the right thing. Promotion sets `WinRateCallback.promote=True` and is read in `_on_step` to early-stop `model.learn`.
5. **Inverse-win-rate level sampling.** `CurriculumEnv` samples levels with weight `1 − rolling_win_rate` (clamped to ≥ 0.05 to prevent starvation). Stages > 1 use a 70/30 new-vs-prior split so old levels don't catastrophically forget. Win rate is a deque of the last 100 outcomes per level.
6. **PPO for stages 1–2, RecurrentPPO (LSTM) from stage 3.** Conveyors / switches / keys all require multi-step planning where the optimal action depends on history (e.g. "I just flipped the red switch"). LSTM provides that memory. Earlier stages are stateless puzzles where the LSTM overhead is pure cost. `RECURRENT_FROM_STAGE = 3` in `config.py` (set to 6 to disable LSTM entirely).
7. **BFS, not Manhattan, potential.** `compute_potential(gs)` walks the maze with BFS over walkable tiles (`tile >= 18 and tile != 31`). Manhattan ignored walls and actively misled the agent — on L01 the exit was 7 tiles by Manhattan but 13 by BFS, a 1.86× error. `Φ` is normalised by `/32` toward carrots/eggs and `/16` toward the exit (the stronger normaliser is deliberate so exit-seeking competes with collection).
8. **Exit-seeking reward (commit `Target Navigation`).** Bare potential shaping yields ~0.006/step gradient, 150× weaker than a carrot pickup. Once all collectibles are gathered, `RewardShapingWrapper` adds an explicit per-step `+0.5 / −0.3` for BFS-distance delta to the exit, plus a one-time `+2.0` phase-transition bonus and a `−0.05` urgency penalty. This was the fix for the long-running "0% L01–L03 wins despite `ep_rew_mean ≈ 6`" failure mode.
9. **Dense non-potential bonuses removed.** Earlier per-carrot and per-crumble bonuses caused reward farming (the same `ep_rew_mean ≈ 6` / 0% wins failure). Only the key-pickup bonus (+0.3, harmless pre-stage-5), the +5.0 efficiency bonus on win, the −0.05 blocked-move penalty, and the exit-seeking block survived.
10. **`BobbyExtractor` features (`rl_training/extractor.py`).** Tile IDs → `Embedding(64, 32)` → reshape to `(B, 32, 16, 16)` → CNN+ResBlock → CNN+ResBlock → 4-head self-attention over the token grid → adaptive pool → concat with an MLP over **11** normalised scalars (`player_x/y`, `carrot/egg_count/total`, 3 keys, `completion_ratio`, `all_collected`) → 256-dim features. The two completion features were added to give the network an explicit "collect vs. seek-exit" phase signal. `net_arch=[]` (no extra MLP head — the extractor is the head).
11. **Cosine LR schedule** `3e-4 → 1e-5` per stage; `gamma=0.995`, `gae_lambda=0.97`, `n_steps=256`, `batch_size=512`, `n_epochs=10`, `clip_range=0.2`. Per-stage `ent_coef` is `{1: 0.01, 2: 0.01, 3: 0.01, 4: 0.005, 5: 0.005}` — stage 1 was originally 0.02 and that drowned the policy gradient (entropy ≈ −1.5, near-uniform, 0% wins). When raising `ent_coef` here, also bump `STAGE_ENT_COEF` in `config.py` and rerun stage 1 from scratch.
12. **Vectorised eval.** `WinRateCallback._eval_one_level` runs `n_eval_episodes` rollouts in lockstep through one `DummyVecEnv` so each `model.predict` is a single batched GPU call. Per-level eval is capped at `max_eval_steps=500` so a stuck policy can't stall the training loop.
13. **Heartbeat thread.** `_Heartbeat` in `rl_training/train.py` prints `[heartbeat]` every 30 s and `[FREEZE WARNING +Ns]` if `num_timesteps` hasn't advanced in 120 s. It is the single best signal that training is stuck vs. just slow.
14. **Resume strategy.** `_find_latest_ckpt` parses both `ckpt_{N}_steps.zip` and `ckpt_interrupt_{N}_steps.zip` and resumes from the highest step. `KeyboardInterrupt` saves a rescue checkpoint (`ckpt_interrupt_{N}_steps.zip`) before re-raising. `stage_{N}_final.zip` is the "stage complete" sentinel — re-runs skip completed stages automatically. On resume, `WinRateCallback._run_eval` is fired once before training continues so the next eval delta is meaningful.

## Project-specific context

- **Training environment is Google Colab** (T4 free / L4 / Pro+ 52 GB). The notebook mounts Google Drive at `/content/drive/MyDrive/bobby_carrot_rl`, clones `https://github.com/Charan20510/Bobby_Carrot_Python.git` into `/content/bobby_carrot`, `pip install -e`s it, then calls `rl_training.train.run_training(...)`. Pygame display is suppressed via `SDL_VIDEODRIVER=dummy`.
- **Checkpoints + TB logs persist on Drive.** Path layout: `models/stage_{N}/ckpt_*_steps.zip`, `models/stage_{N}/best_model.zip`, `models/stage_{N}_final.zip`, and `tb_logs/`. A disconnected session re-runs the training cell and resumes from the latest ckpt; finished stages skip via the `stage_{N}_final.zip` sentinel. Local development: training writes the same layout under `./models/` and `./tb_logs/` (note `models/` is gitignored).
- **Episode horizons.** Per-stage curriculum horizon `STAGE_MAX_EPISODE_STEPS = {1–3: 500, 4–5: 600}` (longer for keys/locks). `WinRateCallback.max_eval_steps=500`. Held-out eval (`evaluate.run_evaluation`) hard-stops at `max_steps=500`. `evaluate_gui` defaults to `max_steps=500` per episode.
- **Stage step budgets** (`config.STAGE_MIN_STEPS` / `STAGE_MAX_STEPS`):
  - S1: 100K / 5M, S2: 750K / 8M, S3: 2M / 12M, S4: 3M / 15M, S5: 4M / 20M.
  Eval cadence (`STAGE_CHECK_FREQ`): S1: 100K, S2: 250K, S3: 500K, S4–S5: 1M. Eval episodes per level (`STAGE_EVAL_EPS`): S1: 20, S2: 15, S3–S5: 10. Stage 1 was retuned (500K → 100K min, 3 → 20 episodes) so the 70% promotion threshold is statistically resolvable.
- **Held-out evaluation pipeline (`rl_training/evaluate.py`).** `_load_best_model` walks stages 5→1 looking for `best_model.zip` then `_final.zip`, prefers `RecurrentPPO` and falls back to `PPO`. `run_evaluation` runs 50 deterministic episodes per level for L26–L30 (overridable) and returns per-level dicts (win rate, mean reward, mean steps, failure-mode breakdown via `_classify`: win/death/timeout/stuck_near_exit/unknown, mean carrot %).
- **GUI evaluator (`rl_training/evaluate_gui.py`).** A standalone CLI that wraps the headless eval with Pygame rendering. Supports single-level live play (`--level N --fps 5`), batch headless reports (`--levels 1-3 --episodes 20 --headless --report`), and per-step frame dumps for video creation (`--save-frames ./frames/`). Falls back to the newest `ckpt_*_steps.zip` in the same directory when the requested `best_model.zip` is missing.
- **Sanity check.** If you change the observation space in [bobby_carrot/gym_env.py](bobby_carrot/gym_env.py), update `BobbyExtractor.forward` in `rl_training/extractor.py` in lockstep — `tiles=(256,) uint8`, scalars `Discrete`, `keys=MultiBinary(3)` is the documented shape, and the extractor's scalar-net input dimension (currently 11) must match.
- **Colab stdout caveats.** `progress_bar=True` activates `tqdm.rich` which installs a `rich.console.Console` as `sys.stdout`. Bare `print` from a callback then recurses with IPython's flusher → `RecursionError` kills training. Hence `safe_print` everywhere in `rl_training/callbacks.py` and `progress_bar=False` on `model.learn`. Don't "fix" this by adding a regular print — it will silently work locally and crash in Colab after hours of training.
- **`Project_Summary/`** contains running project reports (e.g. `summary_1.txt`) — narrative timeline of failures, fixes, and ablations. Treat it as append-only documentation; do not rely on it as the source of truth for current hyperparameters (those live in `config.py`).
