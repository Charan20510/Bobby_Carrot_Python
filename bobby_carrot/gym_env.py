"""Gymnasium-compatible wrapper around GameEnv.

Install gymnasium first:
    pip install gymnasium

Usage
-----
    import gymnasium as gym
    from bobby_carrot.gym_env import BobbyCarrotEnv

    env = BobbyCarrotEnv(map_kind="normal", map_number=1)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

from .env import GameEnv
from .core.state import Action, MAX_NORMAL_MAP, MAX_EGG_MAP


def _require_gym() -> None:
    if not _GYM_AVAILABLE:
        raise ImportError(
            "gymnasium is required for BobbyCarrotEnv.\n"
            "Install it with:  pip install gymnasium"
        )


if _GYM_AVAILABLE:
    class BobbyCarrotEnv(gym.Env):
        """Bobby Carrot as a Gymnasium environment.

        Observation space
        -----------------
        Dict with:
            tiles       : Box(uint8, shape=(256,)) — raw tile IDs
            player_x    : Discrete(16)
            player_y    : Discrete(16)
            carrot_count: Discrete(64)
            carrot_total: Discrete(64)
            egg_count   : Discrete(64)
            egg_total   : Discrete(64)
            keys        : MultiBinary(3) — [gray, yellow, red]

        Action space
        ------------
        Discrete(5) — 0=IDLE, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT
        """

        metadata = {"render_modes": ["human", "rgb_array", None]}

        def __init__(
            self,
            map_kind: str = "normal",
            map_number: int = 1,
            render_mode: Optional[str] = None,
        ) -> None:
            super().__init__()
            self._map_kind = map_kind
            self._map_number = map_number
            self.render_mode = render_mode
            self._env = GameEnv()
            self._renderer = None
            self._assets = None

            self.observation_space = spaces.Dict({
                "tiles":        spaces.Box(0, 255, shape=(256,), dtype=np.uint8),
                "player_x":     spaces.Discrete(16),
                "player_y":     spaces.Discrete(16),
                "carrot_count": spaces.Discrete(64),
                "carrot_total": spaces.Discrete(64),
                "egg_count":    spaces.Discrete(64),
                "egg_total":    spaces.Discrete(64),
                "keys":         spaces.MultiBinary(3),
            })
            self.action_space = spaces.Discrete(len(Action))

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[Dict[str, Any]] = None,
        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            super().reset(seed=seed)
            obs, info = self._env.reset(self._map_kind, self._map_number)
            return self._convert_obs(obs), info

        def step(
            self, action: int
        ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
            obs, reward, terminated, truncated, info = self._env.step(
                Action(int(action))
            )
            return self._convert_obs(obs), float(reward), terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            if self.render_mode is None:
                return None
            try:
                import pygame
            except ImportError:
                return None

            if self._renderer is None:
                pygame.init()
                from .renderer import Assets, Renderer
                self._renderer = Renderer()
                self._assets = Assets()

            if self.render_mode == "human":
                self._renderer.draw(self._env.gs, self._assets)
                self._renderer.tick()
                return None

            if self.render_mode == "rgb_array":
                self._renderer.draw(self._env.gs, self._assets)
                surface = pygame.display.get_surface()
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2)
                )

        def close(self) -> None:
            if self._renderer is not None:
                try:
                    import pygame
                    pygame.quit()
                except Exception:
                    pass
                self._renderer = None
                self._assets = None

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        @staticmethod
        def _convert_obs(obs: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "tiles":        np.array(obs["tiles"], dtype=np.uint8),
                "player_x":     int(obs["player_x"]),
                "player_y":     int(obs["player_y"]),
                "carrot_count": int(obs["carrot_count"]),
                "carrot_total": int(obs["carrot_total"]),
                "egg_count":    int(obs["egg_count"]),
                "egg_total":    int(obs["egg_total"]),
                "keys":         np.array(
                    [obs["key_gray"] > 0, obs["key_yellow"] > 0, obs["key_red"] > 0],
                    dtype=np.int8,
                ),
            }

else:
    # Stub so the module is importable even without gymnasium
    class BobbyCarrotEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_gym()
