"""Headless game environment for RL training.

No pygame import.  Implements a Gym-style interface over the pure-logic layer.

Usage
-----
    from bobby_carrot.env import GameEnv
    from bobby_carrot.core.state import Action

    env = GameEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .core.loader import Map, MapInfo
from .core.logic import logical_step
from .core.state import Action, GameState, State


class GameEnv:
    """Headless Bobby Carrot environment.

    One ``step()`` call == one complete tile move (plus any conveyor chains).
    No frame-rate cap; runs as fast as the CPU allows.
    """

    def __init__(self) -> None:
        self._map_obj: Optional[Map] = None
        self._map_info_fresh: Optional[MapInfo] = None
        self.gs: Optional[GameState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        map_kind: str = "normal",
        map_number: int = 1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load *map_kind*/*map_number* and return (observation, info)."""
        self._map_obj = Map(map_kind, map_number)
        self._map_info_fresh = self._map_obj.load_map_info()
        self.gs = self._build_state(self._map_info_fresh, frame=0)
        return self._observe(), {}

    def step(
        self, action: Action
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute *action* and return (obs, reward, terminated, truncated, info).

        ``terminated`` is True when the level is won (faded_out) or the player
        died and was reset.  ``truncated`` is always False.
        """
        assert self.gs is not None, "call reset() before step()"

        self.gs, events = logical_step(self.gs, action)

        reward = self._compute_reward(events)
        terminated = False

        if self.gs.dead:
            reward -= 1.0
            terminated = True
            self._reload_level()

        elif self.gs.is_finished() and self._at_exit():
            # In RL mode there is no fade-out animation; terminate immediately.
            reward += 10.0
            terminated = True
            self._advance_level()

        return self._observe(), reward, terminated, False, {"events": events}

    def get_observation(self) -> Dict[str, Any]:
        return self._observe()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _observe(self) -> Dict[str, Any]:
        gs = self.gs
        return {
            "tiles": gs.tiles[:],
            "player_x": gs.coord_src[0],
            "player_y": gs.coord_src[1],
            "carrot_count": gs.carrot_count,
            "carrot_total": gs.carrot_total,
            "egg_count": gs.egg_count,
            "egg_total": gs.egg_total,
            "key_gray": gs.key_gray,
            "key_yellow": gs.key_yellow,
            "key_red": gs.key_red,
            "dead": gs.dead,
            "won": gs.faded_out,
        }

    @staticmethod
    def _compute_reward(events: List[str]) -> float:
        reward = -0.01   # small step penalty
        for e in events:
            if e == "carrot_collected":
                reward += 1.0
            elif e == "egg_collected":
                reward += 1.0
        return reward

    def _at_exit(self) -> bool:
        gs = self.gs
        pos = gs.coord_src[0] + gs.coord_src[1] * 16
        return gs.tiles[pos] == 44

    def _build_state(self, map_info: MapInfo, frame: int) -> GameState:
        """Build a fresh GameState for *map_info* starting at *frame*.

        ``last_action_time`` and ``start_time`` are stored as frame counts so
        that game.py can compare them against ``gs.frame`` without mixing units.
        """
        return GameState(
            tiles=map_info.data[:],
            carrot_total=map_info.carrot_total,
            egg_total=map_info.egg_total,
            coord_src=map_info.coord_start,
            coord_dest=map_info.coord_start,
            player_state=State.Down,   # skip FadeIn for RL; game.py overrides to FadeIn
            next_state=None,
            start_frame=frame,
            carrot_count=0,
            egg_count=0,
            key_gray=0,
            key_yellow=0,
            key_red=0,
            dead=False,
            faded_out=False,
            last_action_time=frame,   # frame count (not ms)
            start_time=frame,         # frame count (not ms)
            frame=frame,
        )

    def _reload_level(self) -> None:
        self.gs = self._build_state(self._map_info_fresh, self.gs.frame)

    def _advance_level(self) -> None:
        self._map_obj = self._map_obj.next()
        self._map_info_fresh = self._map_obj.load_map_info()
        self.gs = self._build_state(self._map_info_fresh, self.gs.frame)
