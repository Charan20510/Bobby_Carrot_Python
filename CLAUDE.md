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

The RL training notebook is `bobby_carrot_rl.ipynb` at the repo root.

## Architecture

The codebase is built around a strict **logic / renderer separation** so RL training does not pay any rendering cost (~175k logical steps/sec headless vs. 60 FPS interactive).

```
bobby_carrot/
├── core/
│   ├── state.py    GameState dataclass, State + Action enums, tile/grid constants. NO pygame.
│   ├── loader.py   Map, MapInfo, .blm parsing, CLI map-arg parsing. NO pygame.
│   └── logic.py    Pure logic: start_move, apply_landing, advance_frame, logical_step. NO pygame.
├── env.py          GameEnv: headless reset/step/observe Gym-style API. NO pygame.
├── renderer.py     Renderer + Assets + compute_sprite_rects. ALL pygame code lives here.
├── game.py         Slim main(): wires GameEnv + Renderer + pygame event loop for interactive play.
├── gym_env.py      BobbyCarrotEnv(gymnasium.Env) wrapper around GameEnv.
└── bench.py        Steps/sec benchmark for the headless env.
```

**Invariant:** `core/` and `env.py` must never import pygame. `renderer.py` is the *only* pygame entry point and is imported lazily from `game.py`/`gym_env.py` after `pygame.init()`. Breaking this invariant kills RL throughput.

**Two execution modes share the same logic module:**

- **Interactive (60 FPS):** `game.py` calls `advance_frame(gs)` once per render frame. Tile effects fire at animation step 8 (`step = (gs.frame - gs.start_frame) // FRAMES_PER_STEP`). The renderer reads `gs` and computes sprite rects via `compute_sprite_rects(gs)` — pure, no mutation.
- **Headless (RL):** `env.py` calls `logical_step(gs, action)` once per agent action. It collapses one full tile move + chained conveyor moves immediately, with no frame counting. `GameEnv.reset()` starts the player in `State.Down` (skipping FadeIn) so an agent can act on the first step; `game.py._reset_interactive()` overrides this to `State.FadeIn` for the visible intro animation.

**Key split that enables this:** the original `Bobby.update_texture_position()` was decomposed into `apply_landing(gs)` (logic-only tile effects: carrot/egg pickup, switches, conveyor entry) in `logic.py`, and `compute_sprite_rects(gs)` (rendering math) in `renderer.py`.

**Tile IDs are hardcoded throughout `logic.py`** (e.g. 19 = carrot, 22/38 = red/yellow switches, 24–29 = red-controlled conveyors, 30→31 = crumble, 32–37 = key/door pairs by colour, 40–43 = forced-direction conveyors, 44 = exit, 45/46 = egg/empty). When changing tile semantics, both the entry restrictions in `_check_dest` and the effect tables in `apply_landing` (and the toggle helpers `_toggle_red_switch`/`_toggle_yellow_switch`) must stay in sync.

**Reward shaping** (`GameEnv._compute_reward`): −0.01 per step, +1 per carrot/egg, +10 on level complete, −1 on death. Death and win both set `terminated=True`; on win, the env auto-advances to `Map.next()` before returning.

**Frame counter convention:** `gs.frame` is a unitless tick that increments once per interactive render frame *and* once per `logical_step`. `last_action_time` and `start_time` are stored as `gs.frame` snapshots, not milliseconds — do not mix units.

## Project goal

Train an RL agent on **normal levels 1–25** via a 5-stage curriculum and evaluate it **zero-shot on held-out normal levels 26–30**. Levels 26–30 are never seen during training; they are the sole test set. Success is measured per-level by:

- **Win rate** over 50 deterministic episodes (`model.predict(deterministic=True)`),
- **Mean episode reward** (shaped),
- **Mean steps** to win,
- **Failure-mode breakdown**: death / timeout / stuck-near-exit / unknown,
- **Mean carrot completion %**.

A win is detected by the native terminal reward exceeding `+5.0` — the native `+10.0` win burst in [bobby_carrot/env.py](bobby_carrot/env.py) is the only signal that crosses that threshold. Per-step shaped reward stays well below it. Eval logic lives in cells 15–16 of [bobby_carrot_rl.ipynb](bobby_carrot_rl.ipynb); training in cells 6–12.

Curriculum stages (cell 6, `STAGE_ALL_LEVELS`):

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
- **Comments explain *why*, not *what*.** Notebook cells routinely cite the incident or paper that justifies a value (e.g. "lowered from 500K → 100K because…", "Ng et al. 1999"). Avoid restating what the code already says; do record the failure mode that motivated a tuning change.
- **Per-stage configuration is dict-keyed by stage int**: `STAGE_ALL_LEVELS`, `STAGE_NEW_START`, `STAGE_WIN_THRESHOLD`, `STAGE_MIN_STEPS`, `STAGE_MAX_STEPS`, `STAGE_ENT_COEF`, `STAGE_CHECK_FREQ`, `STAGE_EVAL_EPS`. Adding a stage means adding entries to all of them; changing a curve means editing one dict, not branching on `if stage == k` throughout.
- **Defensive callbacks.** `WinRateCallback._run_eval` wraps the entire eval in `try/except` so a flaky eval cannot kill a multi-hour training run. Heartbeat / stall-warning threads are daemons and never raise.
- **Stdout safety in Colab.** Never call bare `print` from a callback while `tqdm.rich` / IPython has hijacked `sys.stdout` — it recurses and crashes. Use `_safe_print` (routes through `tqdm.write(file=sys.__stdout__)`) and pin tqdm bars to `sys.__stdout__`. Keep `progress_bar=False` on `model.learn(...)`.
- **RecurrentPPO state threading.** When the loaded model is `RecurrentPPO`, every `model.predict` call must thread `state=lstm_states` and `episode_start=episode_starts`, and reset the start mask whenever an env terminates. The eval, replay, and held-out test cells all branch on `is_recurrent` for this reason.
- **Tile IDs are the cross-cutting contract.** Magic numbers (19 carrot, 22/38 switches, 24–29 red conveyors, 30→31 crumble, 32–37 keys/locks, 40–43 forced conveyors, 44 exit, 45/46 egg/empty) appear in `core/logic.py`, `_check_dest`, `apply_landing`, `_toggle_*_switch`, *and* in the notebook's `_compute_potential` (which scans for 19, 45, 44). When changing tile semantics, update all of them in one PR.
- **Frame counter is a tick, not milliseconds.** See the frame-counter convention above. Do not mix units.
- **No new top-level files casually.** The package layout is small and intentional; benchmarks go in `bench.py`, RL wiring in the notebook, gym wrapping in `gym_env.py`.

## Architecture decisions

1. **Headless `GameEnv` separate from `BobbyCarrotEnv` (gym wrapper).** `core/` + `env.py` know nothing about gymnasium *or* pygame. `gym_env.py` adapts `GameEnv` to the gymnasium API and converts the dict observation. This lets `bench.py` skip both layers and lets the renderer be imported lazily after `pygame.init()`.
2. **`DummyVecEnv`, not `SubprocVecEnv`.** Each env runs ~2k headless steps/s, so 8 in-process envs give ~16k steps/s — already past GPU-bound. Subprocesses would cost CPU RAM (rollout buffer × N_ENVS) for no throughput gain. `N_ENVS` is auto-picked from CPU RAM (≥50 GB → 32, ≥25 GB → 16, else 8) in cell 2.
3. **5-stage curriculum keyed by mechanic introduction.** Each stage adds the next class of tile (crumble → conveyors → switches → keys). Stage k+1 warm-starts from stage k's checkpoint when (and only when) the algorithm class matches; otherwise it starts fresh (cell 12, `_use_recurrent` / warm-start branch).
4. **Adaptive stage promotion.** Promotion fires when the *hardest* level in the stage hits `STAGE_WIN_THRESHOLD[stage]` *and* total stage steps cleared `STAGE_MIN_STEPS[stage]`. Hard cap at `STAGE_MAX_STEPS[stage]`. Stage-local step counters mean `reset_num_timesteps=False` resumes do the right thing.
5. **Inverse-win-rate level sampling.** Within a stage, levels are sampled with weight `1 − rolling_win_rate` (clamped to ≥ 0.05 to prevent starvation). Stages > 1 use a 70/30 new-vs-prior split so old levels don't catastrophically forget. Win rate is a deque of the last 100 outcomes per level.
6. **PPO for stages 1–2, RecurrentPPO (LSTM) from stage 3.** Conveyors / switches / keys all require multi-step planning where the optimal action depends on history (e.g. "I just flipped the red switch"). LSTM provides that memory. Earlier stages are stateless puzzles where the LSTM overhead is pure cost. `RECURRENT_FROM_STAGE = 3` (cell 11).
7. **Potential-based reward shaping (Ng et al. 1999).** `Φ(s) = −Manhattan(player, nearest carrot/egg)/32`, switching to the exit tile when all collectibles are gathered. Shaping `r' = r + γΦ(s') − Φ(s)` is policy-invariant in theory; we set `Φ(terminal) = 0` on win and `Φ(s')=Φ(s)` on death (so the −1.0 native death penalty is the sole death signal — the env reloads before we can read the next state).
8. **Dense non-potential bonuses removed.** Per-carrot and per-crumble bonuses caused reward farming (a local optimum where `ep_rew_mean ≈ 6` with **0% L01–L03 win rate** at 500K steps). Only the key-pickup bonus (+0.3, harmless pre-stage-5) and a +5.0 efficiency bonus on win survived.
9. **`BobbyExtractor` features (cell 9).** Tile IDs → `Embedding(64, 32)` → reshape to `(B, 32, 16, 16)` → CNN+ResBlock → CNN+ResBlock → 4-head self-attention over the 16×16 token grid → adaptive pool → concat with an MLP over 9 normalised scalars (`player_x/y`, `carrot/egg_count/total`, 3 keys) → 256-dim features. `net_arch=[]` (no extra MLP head — the extractor is the head).
10. **Cosine LR schedule** `3e-4 → 1e-5` per stage; `gamma=0.995`, `gae_lambda=0.97`, `n_steps=256`, `batch_size=512`. Per-stage `ent_coef` is `{1: 0.005, 2: 0.01, 3: 0.01, 4: 0.005, 5: 0.005}` — stage 1 was originally 0.02 and that drowned the policy gradient (entropy ≈ −1.5, near-uniform, 0% wins).
11. **Vectorised eval.** `WinRateCallback._eval_one_level` runs `n_eval_episodes` rollouts in lockstep through one `DummyVecEnv` so each `model.predict` is a single batched GPU call. Per-level eval is capped at `max_eval_steps=250` so a stuck policy can't stall the training loop.
12. **Heartbeat thread.** `_Heartbeat` (cell 12) prints `[heartbeat]` every 30 s and `[FREEZE WARNING]` if `num_timesteps` hasn't advanced in 120 s. It is the single best signal that training is stuck vs. just slow.
13. **Resume strategy.** `_find_latest_ckpt` parses both `ckpt_{N}_steps.zip` and `ckpt_interrupt_{N}_steps.zip` and resumes from the highest step. `KeyboardInterrupt` saves a rescue checkpoint before re-raising. `stage_{N}_final.zip` is the "stage complete" sentinel — re-runs skip completed stages automatically.

## Project-specific context

- **Training environment is Google Colab** (T4 free / L4 / Pro+ 52 GB). Cell 1 mounts Google Drive at `/content/drive/MyDrive/bobby_carrot_rl`; cell 3 clones `https://github.com/Charan20510/Bobby_Carrot_Python.git` into `/content/bobby_carrot` and `pip install -e`s it. Pygame display is suppressed via `SDL_VIDEODRIVER=dummy` (cell 4).
- **Checkpoints + TB logs persist on Drive.** Path layout: `models/stage_{N}/ckpt_*_steps.zip`, `models/stage_{N}/best_model.zip`, `models/stage_{N}_final.zip`, and `tb_logs/`. A disconnected session re-runs the training cell and resumes from the latest ckpt; finished stages skip via the `stage_{N}_final.zip` sentinel.
- **Episode horizons.** Curriculum wrapper `max_episode_steps=500`. Eval cap `max_eval_steps=250`. Replay cell `MAX_STEPS=400`. Held-out eval (cell 15) hard-stops episodes at 500.
- **Stage step budgets** (cell 6): min/max steps per stage are
  - S1: 100K / 5M, S2: 750K / 8M, S3: 2M / 12M, S4: 3M / 15M, S5: 4M / 20M.
  Eval cadence (cell 12): S1: 100K, S2: 250K, S3: 500K, S4–S5: 1M. Eval episodes per level: S1: 20, S2: 15, S3–S5: 10. Stage 1 was retuned (500K → 100K min, 3 → 20 episodes) so the 70% promotion threshold is statistically resolvable.
- **Held-out evaluation pipeline (cells 15–16).** Loads the highest-stage `best_model.zip` / `_final.zip` (preferring RecurrentPPO and falling back to PPO), runs 50 deterministic episodes per level for L26–30, and produces a 6-panel matplotlib chart: win rate, mean reward, failure-mode stack, carrot %, plus visitation heatmaps for L26 and L27. Chart is saved to `${DRIVE_DIR}/eval_results_v2.png`.
- **Replay cell (18)** records per-step tile grids for one chosen level and animates them with `matplotlib.animation` — no pygame display required.
- **Sanity check (cell 4)** confirms the gym wrapper observation matches the documented dict shape (`tiles=(256,) uint8`, scalars Discrete, `keys=MultiBinary(3)`) before any training begins. If you change the observation space in [bobby_carrot/gym_env.py](bobby_carrot/gym_env.py), update `BobbyExtractor.forward` (cell 9) in lockstep.
- **Colab stdout caveats** (cell 7). `progress_bar=True` activates `tqdm.rich` which installs a `rich.console.Console` as `sys.stdout`. Bare `print` from a callback then recurses with IPython's flusher → `RecursionError` kills training. Hence `_safe_print` everywhere and `progress_bar=False` on `model.learn`. Don't "fix" this by adding a regular print — it will silently work locally and crash in Colab after hours of training.
