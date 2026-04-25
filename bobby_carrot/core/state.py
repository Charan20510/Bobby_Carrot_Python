"""Pure data types: constants, enums, and the GameState dataclass."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAMES = 60
FRAMES_PER_STEP = 2
WIDTH_POINTS = 16
HEIGHT_POINTS = 16
VIEW_WIDTH_POINTS = 10
VIEW_HEIGHT_POINTS = 12
MS_PER_FRAME = 1000 // FRAMES
WIDTH = 32 * WIDTH_POINTS
HEIGHT = 32 * HEIGHT_POINTS
VIEW_WIDTH = 32 * VIEW_WIDTH_POINTS
VIEW_HEIGHT = 32 * VIEW_HEIGHT_POINTS
WIDTH_POINTS_DELTA = WIDTH_POINTS - VIEW_WIDTH_POINTS
HEIGHT_POINTS_DELTA = HEIGHT_POINTS - VIEW_HEIGHT_POINTS
MAX_NORMAL_MAP = 30
MAX_EGG_MAP = 20

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class State(Enum):
    """Player animation / movement state."""
    Idle = 0
    Death = 1
    FadeIn = 2
    FadeOut = 3
    Left = 4
    Right = 5
    Up = 6
    Down = 7


class Action(Enum):
    """RL action space."""
    IDLE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


ACTION_TO_STATE: dict = {
    Action.UP: State.Up,
    Action.DOWN: State.Down,
    Action.LEFT: State.Left,
    Action.RIGHT: State.Right,
}

# ---------------------------------------------------------------------------
# GameState
# ---------------------------------------------------------------------------


class GameState:
    """All mutable game state, with no pygame dependency."""

    __slots__ = (
        "tiles", "carrot_total", "egg_total",
        "coord_src", "coord_dest",
        "player_state", "next_state", "start_frame",
        "carrot_count", "egg_count",
        "key_gray", "key_yellow", "key_red",
        "dead", "faded_out",
        "last_action_time", "start_time",
        "frame",
    )

    def __init__(
        self,
        tiles: List[int],
        carrot_total: int,
        egg_total: int,
        coord_src: Tuple[int, int],
        coord_dest: Tuple[int, int],
        player_state: State,
        next_state: Optional[State],
        start_frame: int,
        carrot_count: int,
        egg_count: int,
        key_gray: int,
        key_yellow: int,
        key_red: int,
        dead: bool,
        faded_out: bool,
        last_action_time: int,
        start_time: int,
        frame: int,
    ) -> None:
        self.tiles = tiles
        self.carrot_total = carrot_total
        self.egg_total = egg_total
        self.coord_src = coord_src
        self.coord_dest = coord_dest
        self.player_state = player_state
        self.next_state = next_state
        self.start_frame = start_frame
        self.carrot_count = carrot_count
        self.egg_count = egg_count
        self.key_gray = key_gray
        self.key_yellow = key_yellow
        self.key_red = key_red
        self.dead = dead
        self.faded_out = faded_out
        self.last_action_time = last_action_time
        self.start_time = start_time
        self.frame = frame

    def is_walking(self) -> bool:
        return self.coord_src != self.coord_dest

    def is_finished(self) -> bool:
        if self.carrot_total > 0:
            return self.carrot_count == self.carrot_total
        return self.egg_count == self.egg_total

    def copy(self) -> GameState:
        return GameState(
            tiles=self.tiles[:],
            carrot_total=self.carrot_total,
            egg_total=self.egg_total,
            coord_src=self.coord_src,
            coord_dest=self.coord_dest,
            player_state=self.player_state,
            next_state=self.next_state,
            start_frame=self.start_frame,
            carrot_count=self.carrot_count,
            egg_count=self.egg_count,
            key_gray=self.key_gray,
            key_yellow=self.key_yellow,
            key_red=self.key_red,
            dead=self.dead,
            faded_out=self.faded_out,
            last_action_time=self.last_action_time,
            start_time=self.start_time,
            frame=self.frame,
        )
