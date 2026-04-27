#!/usr/bin/env python3
"""GUI Gameplay Evaluator — visualise a trained agent playing Bobby Carrot.

This tool renders the agent's actual gameplay using Pygame, captures frames
for video creation, and produces per-level evaluation reports.

Usage
-----
# Watch a single level with live Pygame rendering:
python evaluate_gui.py --model models/stage_1/best_model.zip --level 1 --fps 5

# Batch evaluate levels 1-3, generate per-level report:
python evaluate_gui.py --model models/stage_1/best_model.zip --levels 1-3 --episodes 10 --report

# Save frames for video creation (no live display needed):
python evaluate_gui.py --model models/stage_1/best_model.zip --level 2 --save-frames ./frames/

# Headless batch report (no Pygame window):
python evaluate_gui.py --model models/stage_1/best_model.zip --levels 1-3 --episodes 20 --headless
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT = True
except ImportError:
    RecurrentPPO = None
    _HAS_RECURRENT = False


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Model loading ────────────────────────────────────────────────────────────

def _resolve_model_path(path: str) -> Path:
    """Resolve a requested model path to an existing checkpoint file.

    The GUI accepts a best_model.zip-style path, but training in this repo may
    only leave behind checkpoints or final stage exports. When the exact file
    is missing, fall back to the newest model artifact in the same directory.
    """
    requested = Path(path).expanduser()
    repo_root = Path(__file__).resolve().parents[1]

    candidates = [requested]
    if requested.suffix != ".zip":
        candidates.append(requested.with_suffix(".zip"))

    if not requested.is_absolute():
        candidates.extend([repo_root / candidate for candidate in list(candidates)])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    search_dir = requested if requested.is_dir() else requested.parent
    if not search_dir.is_absolute():
        search_dir = repo_root / search_dir
    if search_dir.exists():
        for pattern in ("best_model.zip", "*_final.zip", "ckpt_*_steps.zip"):
            matches = sorted(search_dir.glob(pattern))
            if matches:
                return matches[-1]

    raise FileNotFoundError(
        f"Could not find model file for '{path}'. Looked for the requested file, "
        f"a .zip variant, and stage artifacts in '{search_dir}'."
    )

def load_model(path: str) -> Tuple[object, bool]:
    """Load a trained model, trying RecurrentPPO first then PPO."""
    resolved_path = _resolve_model_path(path)
    if resolved_path != Path(path):
        print(f"Resolved model path: {resolved_path}")
    custom_objects = {
        "learning_rate": 3e-4,
        "clip_range": 0.2,
    }
    if _HAS_RECURRENT:
        try:
            model = RecurrentPPO.load(resolved_path, custom_objects=custom_objects)
            return model, True
        except Exception:
            pass
    model = PPO.load(resolved_path, custom_objects=custom_objects)
    return model, False


# ── Level range parsing ──────────────────────────────────────────────────────

def parse_levels(spec: str) -> List[int]:
    """Parse a level spec like '1-3' or '1,2,5' into a list of ints."""
    levels = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            levels.extend(range(int(a), int(b) + 1))
        else:
            levels.append(int(part))
    return sorted(set(levels))


# ── Single-level GUI playback ────────────────────────────────────────────────

def play_level_gui(
    model,
    is_recurrent: bool,
    level: int,
    fps: float = 5.0,
    max_steps: int = 500,
    save_frames_dir: Optional[str] = None,
    headless: bool = False,
) -> dict:
    """Play one level with Pygame rendering, return episode stats.

    Parameters
    ----------
    model : Trained PPO/RecurrentPPO model.
    is_recurrent : Whether the model uses LSTM.
    level : Level number (1-30).
    fps : Rendering FPS (lower = easier to watch).
    max_steps : Hard step cap.
    save_frames_dir : If set, save each frame as PNG here.
    headless : If True, skip Pygame rendering (for batch eval).

    Returns
    -------
    dict with keys: won, reward, steps, carrot_pct, frames_saved
    """
    import pygame
    from bobby_carrot.gym_env import BobbyCarrotEnv
    from bobby_carrot.renderer import Renderer, Assets
    from bobby_carrot.core.state import VIEW_WIDTH, VIEW_HEIGHT

    render_mode = None if headless else "rgb_array"
    env = BobbyCarrotEnv(map_kind="normal", map_number=level, render_mode=render_mode)

    # Init Pygame display
    if not headless:
        if not pygame.get_init():
            pygame.init()
        screen = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
        pygame.display.set_caption(f"Bobby Carrot — Level {level} (Agent Playback)")
        clock = pygame.time.Clock()
        renderer = Renderer()
        assets = Assets()
    else:
        renderer = assets = screen = clock = None

    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)

    obs, _ = env.reset()
    total_reward = 0.0
    step_count = 0
    frames_saved = 0
    won = False

    lstm_states = None
    episode_start = np.array([True])
    _hl_renderer = None
    _hl_assets = None
    _hl_surface = None

    for step_i in range(max_steps):
        # Handle Pygame events (quit, etc)
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return {
                        "won": False, "reward": total_reward,
                        "steps": step_count, "carrot_pct": 0.0,
                        "frames_saved": frames_saved,
                    }
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return {
                        "won": False, "reward": total_reward,
                        "steps": step_count, "carrot_pct": 0.0,
                        "frames_saved": frames_saved,
                    }

        # Predict action
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.array([False])
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, r, terminated, truncated, info = env.step(action)
        total_reward += r
        step_count += 1

        # Render
        if not headless:
            gs = env._env.gs
            renderer.draw(gs, assets)

            # Draw HUD overlay with agent stats
            font = pygame.font.SysFont("monospace", 14, bold=True)
            overlay_lines = [
                f"Step: {step_count}/{max_steps}",
                f"Reward: {total_reward:+.2f}",
                f"Carrots: {gs.carrot_count}/{gs.carrot_total}",
            ]
            if gs.egg_total > 0:
                overlay_lines.append(f"Eggs: {gs.egg_count}/{gs.egg_total}")
            if gs.key_gray or gs.key_yellow or gs.key_red:
                overlay_lines.append(
                    f"Keys: G={gs.key_gray} Y={gs.key_yellow} R={gs.key_red}"
                )

            # Semi-transparent background for readability
            overlay_h = len(overlay_lines) * 18 + 8
            overlay_surf = pygame.Surface((180, overlay_h), pygame.SRCALPHA)
            overlay_surf.fill((0, 0, 0, 160))
            screen.blit(overlay_surf, (4, VIEW_HEIGHT - overlay_h - 4))

            for i, line in enumerate(overlay_lines):
                text = font.render(line, True, (255, 255, 255))
                screen.blit(text, (8, VIEW_HEIGHT - overlay_h + i * 18))

            pygame.display.flip()
            clock.tick(fps)

        # Save frame (works in both headed and headless modes)
        if save_frames_dir:
            if not headless:
                frame_path = os.path.join(
                    save_frames_dir, f"frame_{frames_saved:05d}.png"
                )
                pygame.image.save(pygame.display.get_surface(), frame_path)
                frames_saved += 1
            else:
                # Headless frame capture: write to an off-screen Surface.
                # The previous implementation called pygame.display.get_surface()
                # without ever calling pygame.display.set_mode(), so it returned
                # None and frames were silently dropped.  Now we set up a dummy
                # display surface once on first use (works under SDL_VIDEODRIVER=
                # dummy as well) so renderer.draw has a canvas to blit to.
                gs = env._env.gs
                if _hl_renderer is None:
                    from bobby_carrot.renderer import Renderer, Assets
                    if not pygame.get_init():
                        pygame.init()
                    if pygame.display.get_surface() is None:
                        # SDL_VIDEODRIVER=dummy makes this a no-op visually
                        pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))
                    _hl_renderer = Renderer()
                    _hl_assets = Assets()
                _hl_renderer.draw(gs, _hl_assets)
                _hl_surface = pygame.display.get_surface()
                if _hl_surface is not None:
                    frame_path = os.path.join(
                        save_frames_dir, f"frame_{frames_saved:05d}.png"
                    )
                    pygame.image.save(_hl_surface, frame_path)
                    frames_saved += 1

        if terminated or truncated:
            won = total_reward > 5.0
            break

    # Get carrot completion
    env_inner = env._env
    gs = env_inner.gs if env_inner else None
    if gs:
        total_ct = max(gs.carrot_total + gs.egg_total, 1)
        carrot_pct = (gs.carrot_count + gs.egg_count) / total_ct
    else:
        carrot_pct = 0.0

    # Show result on screen briefly
    if not headless and screen:
        font = pygame.font.SysFont("monospace", 24, bold=True)
        result_text = "✓ WON!" if won else "✗ LOST"
        color = (0, 255, 0) if won else (255, 0, 0)
        text = font.render(
            f"Level {level}: {result_text}  R={total_reward:.1f}  "
            f"Steps={step_count}  Carrots={carrot_pct:.0%}",
            True, color
        )
        rect = text.get_rect(center=(VIEW_WIDTH // 2, VIEW_HEIGHT // 2))
        bg = pygame.Surface((rect.width + 20, rect.height + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 200))
        screen.blit(bg, (rect.x - 10, rect.y - 5))
        screen.blit(text, rect)
        pygame.display.flip()
        time.sleep(2.0)

    env.close()
    return {
        "won": won,
        "reward": total_reward,
        "steps": step_count,
        "carrot_pct": carrot_pct,
        "frames_saved": frames_saved,
    }


# ── Batch evaluation with per-level report ───────────────────────────────────

def batch_evaluate(
    model,
    is_recurrent: bool,
    levels: List[int],
    n_episodes: int = 10,
    max_steps: int = 500,
    headless: bool = True,
    save_frames_dir: Optional[str] = None,
) -> dict:
    """Run batch evaluation on multiple levels, return per-level results."""
    all_results = {}

    print(f"\n{'='*72}")
    print(f"  EVALUATION REPORT — {len(levels)} levels × {n_episodes} episodes")
    print(f"{'='*72}")
    print(f"\n{'Level':<8} {'Win%':>6} {'Reward':>9} {'Steps':>7} "
          f"{'Carrot%':>9} {'Deaths':>7} {'Timeouts':>9}")
    print("─" * 64)

    outer_pbar = tqdm(
        levels, desc="Evaluating levels",
        file=sys.__stdout__, leave=True,
    )

    for level in outer_pbar:
        outer_pbar.set_postfix_str(f"L{level:02d}")
        wins = 0
        total_r = total_steps = total_carrot = 0.0
        deaths = timeouts = 0

        ep_pbar = tqdm(
            range(n_episodes),
            desc=f"  L{level:02d}",
            leave=False,
            file=sys.__stdout__,
        )
        for ep in ep_pbar:
            # Per-level frame directory: e.g. frames/L01/
            ep_frame_dir = None
            if save_frames_dir:
                ep_frame_dir = os.path.join(
                    save_frames_dir, f"L{level:02d}"
                )
                if n_episodes > 1:
                    ep_frame_dir = os.path.join(ep_frame_dir, f"ep_{ep:03d}")

            result = play_level_gui(
                model, is_recurrent, level,
                fps=60, max_steps=max_steps,
                save_frames_dir=ep_frame_dir,
                headless=headless,
            )
            if result["won"]:
                wins += 1
            elif result["reward"] < -0.5:
                deaths += 1
            elif result["steps"] >= max_steps:
                timeouts += 1

            total_r += result["reward"]
            total_steps += result["steps"]
            total_carrot += result["carrot_pct"]

            ep_pbar.set_postfix(
                wins=wins,
                rew=f"{total_r/(ep+1):.1f}",
            )
        ep_pbar.close()

        wr = wins / n_episodes
        mr = total_r / n_episodes
        ms = total_steps / n_episodes
        mc = total_carrot / n_episodes
        dr = deaths / n_episodes
        tr = timeouts / n_episodes

        all_results[level] = {
            "win_rate": wr, "mean_reward": mr, "mean_steps": ms,
            "carrot_pct": mc, "death_rate": dr, "timeout_rate": tr,
        }

        print(f"L{level:<7} {wr:>5.0%} {mr:>+9.2f} {ms:>7.1f} "
              f"{mc:>8.0%} {dr:>6.0%} {tr:>8.0%}")

    outer_pbar.close()

    # Summary
    avg_wr = np.mean([v["win_rate"] for v in all_results.values()])
    avg_r = np.mean([v["mean_reward"] for v in all_results.values()])
    avg_c = np.mean([v["carrot_pct"] for v in all_results.values()])
    print("─" * 64)
    print(f"{'AVG':<8} {avg_wr:>5.0%} {avg_r:>+9.2f} {'':>7} "
          f"{avg_c:>8.0%}")
    print(f"\n{'='*72}\n")

    return all_results


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bobby Carrot — GUI Gameplay Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained model checkpoint (.zip)",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="Single level to play (1-30)",
    )
    parser.add_argument(
        "--levels", type=str, default=None,
        help="Level range for batch eval, e.g. '1-3' or '1,2,5'",
    )
    parser.add_argument(
        "--episodes", type=int, default=1,
        help="Episodes per level (default: 1 for GUI, 10 for report)",
    )
    parser.add_argument(
        "--fps", type=float, default=5.0,
        help="Rendering FPS for GUI playback (default: 5)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--save-frames", type=str, default=None,
        help="Directory to save per-frame PNGs",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate per-level evaluation report",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without Pygame display (for batch eval)",
    )

    args = parser.parse_args()

    if args.level is None and args.levels is None:
        parser.error("Specify --level N or --levels 1-3")

    # Suppress Pygame display for headless mode
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    print(f"Loading model: {args.model}")
    model, is_recurrent = load_model(args.model)
    algo = "RecurrentPPO" if is_recurrent else "PPO"
    print(f"Algorithm: {algo}")

    if args.level is not None and not args.report:
        # Single level GUI playback
        print(f"\nPlaying Level {args.level} (FPS={args.fps}, max_steps={args.max_steps})")
        for ep in range(args.episodes):
            if args.episodes > 1:
                print(f"\n--- Episode {ep+1}/{args.episodes} ---")

            save_dir = None
            if args.save_frames:
                save_dir = os.path.join(args.save_frames, f"ep_{ep:03d}")

            result = play_level_gui(
                model, is_recurrent, args.level,
                fps=args.fps,
                max_steps=args.max_steps,
                save_frames_dir=save_dir,
                headless=args.headless,
            )
            status = "WON ✓" if result["won"] else "LOST ✗"
            print(f"  {status}  reward={result['reward']:+.2f}  "
                  f"steps={result['steps']}  carrots={result['carrot_pct']:.0%}")
            if save_dir and result["frames_saved"] > 0:
                print(f"  Frames saved: {result['frames_saved']} → {save_dir}/")
    else:
        # Batch evaluation with report
        levels = parse_levels(args.levels) if args.levels else [args.level]
        # When saving frames, default to 1 episode per level;
        # otherwise default to 10 for statistical reports.
        if args.episodes > 1:
            n_eps = args.episodes
        elif args.save_frames:
            n_eps = 1
        else:
            n_eps = 10
        batch_evaluate(
            model, is_recurrent, levels,
            n_episodes=n_eps,
            max_steps=args.max_steps,
            headless=args.headless or args.report,
            save_frames_dir=args.save_frames,
        )


if __name__ == "__main__":
    main()
