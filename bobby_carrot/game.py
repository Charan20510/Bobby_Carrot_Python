"""Interactive entry point: wires GameEnv + Renderer + pygame event loop.

Run with:
    python -m bobby_carrot [map]
"""
from __future__ import annotations

import sys
from typing import Optional

try:
    import pygame
except ImportError:
    pygame = None

from .core.loader import choose_map_interactive, parse_map_arg
from .core.logic import advance_frame, queue_next_move, start_move
from .core.state import Action, ACTION_TO_STATE, FRAMES, State
from .env import GameEnv

# 4 seconds at 60 FPS before the idle animation triggers
_IDLE_TIMEOUT_FRAMES = 4 * FRAMES


def _reset_interactive(env: GameEnv, map_kind: str, map_number: int) -> None:
    """Reset env and apply interactive-mode overrides (FadeIn animation)."""
    env.reset(map_kind, map_number)
    gs = env.gs.copy()
    gs.player_state = State.FadeIn
    gs.start_frame = gs.frame
    gs.last_action_time = gs.frame   # frame count, matched by _IDLE_TIMEOUT_FRAMES check
    env.gs = gs


def main() -> None:
    if pygame is None:
        print(
            "pygame is not installed or failed to import.\n"
            "Install it with:  pip install pygame"
        )
        sys.exit(1)

    pygame.init()

    # Determine starting map
    try:
        if len(sys.argv) > 1:
            map_obj_arg = parse_map_arg(sys.argv[1])
            map_kind, map_number = map_obj_arg.kind, map_obj_arg.number
        else:
            map_obj_arg = choose_map_interactive()
            map_kind, map_number = map_obj_arg.kind, map_obj_arg.number
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to load map: {exc}")
        sys.exit(2)

    # Late import so that renderer (pygame) is only touched after pygame.init()
    from .renderer import Assets, Renderer

    env = GameEnv()
    _reset_interactive(env, map_kind, map_number)

    renderer = Renderer()
    renderer.set_caption(f"Bobby Carrot ({env._map_obj})")
    assets = Assets()

    show_help = False
    running = True

    while running:
        gs = env.gs
        now_ms = pygame.time.get_ticks()

        # --- event handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                code = event.key
                if code == pygame.K_q:
                    running = False
                elif code == pygame.K_r:
                    _reset_interactive(env, env._map_obj.kind, env._map_obj.number)
                elif code == pygame.K_n:
                    nxt = env._map_obj.next()
                    _reset_interactive(env, nxt.kind, nxt.number)
                    renderer.set_caption(f"Bobby Carrot ({env._map_obj})")
                elif code == pygame.K_p:
                    prv = env._map_obj.previous()
                    _reset_interactive(env, prv.kind, prv.number)
                    renderer.set_caption(f"Bobby Carrot ({env._map_obj})")
                elif code == pygame.K_f:
                    renderer.toggle_fullscreen()
                elif code in (pygame.K_h, pygame.K_F1):
                    show_help = not show_help
                else:
                    show_help = False
            # Re-read gs in case reset changed it
            gs = env.gs

        # --- keyboard input → action ---
        keys = pygame.key.get_pressed()
        action: Optional[Action] = None
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action = Action.LEFT
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action = Action.RIGHT
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            action = Action.UP
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = Action.DOWN

        if action is not None:
            gs = env.gs.copy()
            gs.last_action_time = gs.frame   # frame count, compared against gs.frame later
            if not gs.is_walking():
                direction = ACTION_TO_STATE[action]
                gs = start_move(gs, direction)
            else:
                gs = queue_next_move(gs, action)
            env.gs = gs

        # --- advance animation logic (fires tile effects at step 8) ---
        prev_carrots = env.gs.carrot_count
        prev_eggs = env.gs.egg_count
        prev_dead = env.gs.dead

        env.gs, events = advance_frame(env.gs)
        gs = env.gs

        # --- sound triggers ---
        if gs.carrot_count != prev_carrots or gs.egg_count != prev_eggs:
            assets.play_event("carrot_collected")
        if gs.dead and not prev_dead:
            assets.play_death()

        # --- win / death / idle state transitions ---
        gs = env.gs
        if gs.dead:
            _reset_interactive(env, env._map_obj.kind, env._map_obj.number)
        elif gs.is_finished() and gs.tiles[gs.coord_src[0] + gs.coord_src[1] * 16] == 44:
            if gs.faded_out:
                nxt = env._map_obj.next()
                _reset_interactive(env, nxt.kind, nxt.number)
                renderer.set_caption(f"Bobby Carrot ({env._map_obj})")
            elif gs.player_state != State.FadeOut:
                gs = gs.copy()
                gs.start_frame = gs.frame
                gs.player_state = State.FadeOut
                env.gs = gs
        elif (
            env.gs.frame - env.gs.last_action_time >= _IDLE_TIMEOUT_FRAMES
            and not env.gs.is_walking()
            and env.gs.player_state not in {
                State.Idle, State.Death, State.FadeIn, State.FadeOut
            }
            and env.gs.next_state is None
        ):
            gs = env.gs.copy()
            gs.start_frame = gs.frame
            gs.player_state = State.Idle
            env.gs = gs

        # --- render ---
        renderer.draw(env.gs, assets, show_help)

        # advance frame counter and cap FPS
        env.gs = env.gs.copy()
        env.gs.frame += 1
        renderer.tick()

    pygame.quit()


if __name__ == "__main__":
    main()
