"""Pygame renderer: Assets loading and all screen-drawing code.

Import this module only when a display is needed.  The core logic and env
modules must never import from here.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import pygame
from pygame import Rect, Surface

from .core.loader import asset_path
from .core.state import (
    FRAMES, FRAMES_PER_STEP,
    WIDTH, HEIGHT, VIEW_WIDTH, VIEW_HEIGHT,
    WIDTH_POINTS, HEIGHT_POINTS, VIEW_WIDTH_POINTS, VIEW_HEIGHT_POINTS,
    WIDTH_POINTS_DELTA, HEIGHT_POINTS_DELTA,
    GameState, State,
)


def _load_image(sub: str) -> Surface:
    return pygame.image.load(str(asset_path(sub))).convert_alpha()


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


class Assets:
    """All loaded pygame Surfaces and Sounds."""

    def __init__(self) -> None:
        self.bobby_idle = _load_image("image/bobby_idle.png")
        self.bobby_death = _load_image("image/bobby_death.png")
        self.bobby_fade = _load_image("image/bobby_fade.png")
        self.bobby_left = _load_image("image/bobby_left.png")
        self.bobby_right = _load_image("image/bobby_right.png")
        self.bobby_up = _load_image("image/bobby_up.png")
        self.bobby_down = _load_image("image/bobby_down.png")
        self.tile_conveyor_left = _load_image("image/tile_conveyor_left.png")
        self.tile_conveyor_right = _load_image("image/tile_conveyor_right.png")
        self.tile_conveyor_up = _load_image("image/tile_conveyor_up.png")
        self.tile_conveyor_down = _load_image("image/tile_conveyor_down.png")
        self.tileset = _load_image("image/tileset.png")
        self.tile_finish = _load_image("image/tile_finish.png")
        self.hud = _load_image("image/hud.png")
        self.numbers = _load_image("image/numbers.png")
        self.help = _load_image("image/help.png")

        self.snd_carrot: Optional[object] = None
        self.audio_enabled = False
        self._beep_fn = _make_beep()

        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except Exception:
            pass

        if self.audio_enabled:
            self.snd_carrot = _load_sound(
                ["carrot.mid", "carrot.mmf", "carrot.mfm"]
            )
            try:
                pygame.mixer.music.load(
                    str(asset_path("audio/title.mid"))
                )
                pygame.mixer.music.play(-1)
            except Exception:
                self._beep_fn()

    def play_event(self, event: str) -> None:
        if event in ("carrot_collected", "egg_collected", "key_picked"):
            if self.snd_carrot:
                self.snd_carrot.play()
            else:
                self._beep_fn()

    def play_death(self) -> None:
        self._beep_fn()


def _make_beep():
    def _beep():
        try:
            import winsound
            winsound.Beep(1000, 100)
        except Exception:
            print("\a", end="", flush=True)
    return _beep


def _load_sound(candidates: List[str]) -> Optional[object]:
    for name in candidates:
        p = asset_path(f"audio/{name}")
        if not p.exists():
            continue
        try:
            return pygame.mixer.Sound(str(p))
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Sprite rect computation (extracted from Bobby.update_texture_position)
# ---------------------------------------------------------------------------


def compute_sprite_rects(gs: GameState) -> Tuple[Rect, Rect]:
    """Return (src_rect, dest_rect) for the player sprite at the current frame.

    Pure rendering calculation — does *not* mutate gs.
    """
    delta_frame = gs.frame - gs.start_frame
    is_walking = gs.coord_src != gs.coord_dest
    step = delta_frame // FRAMES_PER_STEP

    cx, cy = gs.coord_src
    base_x = cx * 32 + 16 - (36 // 2)
    base_y = cy * 32 + 16 - (50 - 32 // 2)

    if gs.player_state == State.Idle:
        step_idle = (step // 2) % 3
        src = Rect(36 * step_idle, 0, 36, 50)
        dest = Rect(base_x, base_y, 36, 50)
        return src, dest

    if gs.player_state == State.Death:
        step_death = min(step // 3, 7)
        src = Rect((step_death % 8) * 44, 0, 44, 54)
        x0 = gs.coord_src[0] * 32
        y0 = gs.coord_src[1] * 32
        x1 = gs.coord_dest[0] * 32
        y1 = gs.coord_dest[1] * 32
        x = (x1 - x0) // 2 + x0
        y = (y1 - y0) // 2 + y0
        dest = Rect(x + 16 - (44 // 2), y + 16 - (54 - 32 // 2), 44, 54)
        return src, dest

    if gs.player_state == State.FadeIn:
        src = Rect((7 - step) * 36, 0, 36, 50)
        return src, Rect(base_x, base_y, 36, 50)

    if gs.player_state == State.FadeOut:
        src = Rect(step * 36, 0, 36, 50)
        return src, Rect(base_x, base_y, 36, 50)

    # Movement states
    src_x = 36 * ((step + 7) % 8) if is_walking else 36 * 7

    if gs.player_state == State.Left:
        dest_x = (gs.coord_src[0] * 8 - step) * 32 // 8 + 16 - (36 // 2) if is_walking else base_x
        dest_y = base_y
    elif gs.player_state == State.Right:
        dest_x = (gs.coord_src[0] * 8 + step) * 32 // 8 + 16 - (36 // 2) if is_walking else base_x
        dest_y = base_y
    elif gs.player_state == State.Up:
        dest_x = base_x
        dest_y = (gs.coord_src[1] * 8 - step) * 32 // 8 + 16 - (50 - 32 // 2) if is_walking else base_y
    elif gs.player_state == State.Down:
        dest_x = base_x
        dest_y = (gs.coord_src[1] * 8 + step) * 32 // 8 + 16 - (50 - 32 // 2) if is_walking else base_y
    else:
        dest_x, dest_y = base_x, base_y

    return Rect(src_x, 0, 36, 50), Rect(dest_x, dest_y, 36, 50)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


class Renderer:
    """Manages the pygame window and draws a GameState each frame."""

    def __init__(self, fullscreen: bool = False) -> None:
        self._fullscreen = fullscreen
        self._clock = pygame.time.Clock()
        if fullscreen:
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))

    def set_caption(self, caption: str) -> None:
        pygame.display.set_caption(caption)

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT))

    def tick(self) -> int:
        """Cap to 60 FPS and return elapsed ms."""
        return self._clock.tick(FRAMES)

    def draw(
        self,
        gs: GameState,
        assets: Assets,
        show_help: bool = False,
    ) -> None:
        """Render the full frame for the given GameState."""
        screen = pygame.display.get_surface()
        # HUD is anchored to the right edge of the *game content*, not the OS window.
        hud_right = WIDTH if self._fullscreen else VIEW_WIDTH
        screen.fill((0, 0, 0))

        cam_x, cam_y = self._camera(gs, self._fullscreen)

        # --- map tiles ---
        anim_frame_idx = (gs.frame // (FRAMES // 10)) % 4
        for x in range(WIDTH_POINTS):
            for y in range(HEIGHT_POINTS):
                item = gs.tiles[x + y * 16]
                texture = assets.tileset
                animated = False
                if item == 44 and gs.is_finished():
                    texture = assets.tile_finish
                    animated = True
                elif item == 40:
                    texture = assets.tile_conveyor_left
                    animated = True
                elif item == 41:
                    texture = assets.tile_conveyor_right
                    animated = True
                elif item == 42:
                    texture = assets.tile_conveyor_up
                    animated = True
                elif item == 43:
                    texture = assets.tile_conveyor_down
                    animated = True

                if animated:
                    src = Rect(32 * anim_frame_idx, 0, 32, 32)
                else:
                    src = Rect(32 * (item % 8), 32 * (item // 8), 32, 32)
                dest = Rect(x * 32 - cam_x, y * 32 - cam_y, 32, 32)
                screen.blit(texture, dest, src)

        # --- player sprite ---
        bobby_src, bobby_dest = compute_sprite_rects(gs)
        bobby_tex = {
            State.Idle:    assets.bobby_idle,
            State.Death:   assets.bobby_death,
            State.FadeIn:  assets.bobby_fade,
            State.FadeOut: assets.bobby_fade,
            State.Left:    assets.bobby_left,
            State.Right:   assets.bobby_right,
            State.Up:      assets.bobby_up,
            State.Down:    assets.bobby_down,
        }[gs.player_state]
        screen.blit(bobby_tex, bobby_dest.move(-cam_x, -cam_y), bobby_src)

        # --- HUD ---
        self._draw_hud(screen, assets, gs, hud_right)

        # --- help overlay ---
        if show_help:
            s = pygame.Surface((158, 160), pygame.SRCALPHA)
            s.fill((0, 0, 0, 200))
            screen.blit(s, ((hud_right - 158) // 2, 32 * 3 - (160 - 142) // 2))
            screen.blit(
                assets.help,
                ((hud_right - 133) // 2, 32 * 3),
                Rect(0, 0, 133, 142),
            )

        pygame.display.flip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _camera(gs: GameState, fullscreen: bool) -> Tuple[int, int]:
        if fullscreen:
            return 0, 0
        step = gs.frame - gs.start_frame
        x0 = gs.coord_src[0] * 32
        y0 = gs.coord_src[1] * 32
        x1 = gs.coord_dest[0] * 32
        y1 = gs.coord_dest[1] * 32
        if gs.player_state == State.Death:
            cam_x = (x1 - x0) * 6 // 8 + x0 - (VIEW_WIDTH_POINTS // 2) * 32
            cam_y = (y1 - y0) * 6 // 8 + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
        else:
            cam_x = (x1 - x0) * step // (8 * FRAMES_PER_STEP) + x0 - (VIEW_WIDTH_POINTS // 2) * 32
            cam_y = (y1 - y0) * step // (8 * FRAMES_PER_STEP) + y0 - (VIEW_HEIGHT_POINTS // 2) * 32
        cam_x += 16
        cam_y += 16
        cam_x = max(0, min(cam_x, WIDTH_POINTS_DELTA * 32))
        cam_y = max(0, min(cam_y, HEIGHT_POINTS_DELTA * 32))
        return cam_x, cam_y

    @staticmethod
    def _draw_hud(
        screen: Surface, assets: Assets, gs: GameState, hud_right: int
    ) -> None:
        if gs.carrot_total > 0:
            icon_rect = Rect(0, 0, 46, 44)
            num_left = gs.carrot_total - gs.carrot_count
            icon_width = 46
        else:
            icon_rect = Rect(46, 0, 34, 44)
            num_left = gs.egg_total - gs.egg_count
            icon_width = 34

        screen.blit(assets.hud, (hud_right - (icon_width + 4), 4), icon_rect)
        num_10 = num_left // 10
        num_01 = num_left % 10
        screen.blit(
            assets.numbers,
            (hud_right - (icon_width + 4) - 2 - 12, 4 + 14),
            Rect(num_01 * 12, 0, 12, 18),
        )
        screen.blit(
            assets.numbers,
            (hud_right - (icon_width + 4) - 2 - 12 * 2 - 1, 4 + 14),
            Rect(num_10 * 12, 0, 12, 18),
        )

        key_slots = []
        for _ in range(gs.key_gray):
            key_slots.append((122, len(key_slots)))
        for _ in range(gs.key_yellow):
            key_slots.append((122 + 22, len(key_slots)))
        for _ in range(gs.key_red):
            key_slots.append((122 + 22 + 22, len(key_slots)))
        for offset, count in key_slots:
            screen.blit(
                assets.hud,
                (hud_right - (22 + 4) - count * 22, 4 + 44 + 2),
                Rect(offset, 0, 22, 44),
            )

        passed_secs = (gs.frame - gs.start_time) // FRAMES
        minutes = min(passed_secs // 60, 99)
        seconds = min(passed_secs % 60, 99) if minutes < 99 else 99
        for idx, offset in enumerate(
            [minutes // 10, minutes % 10, 10, seconds // 10, seconds % 10]
        ):
            screen.blit(
                assets.numbers,
                (4 + 12 * idx, 4),
                Rect(offset * 12, 0, 12, 18),
            )
