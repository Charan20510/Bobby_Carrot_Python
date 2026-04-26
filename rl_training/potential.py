"""BFS-based potential function for reward shaping.

Uses BFS shortest-path distance through the maze (respecting walls) instead
of Manhattan distance, which ignores walls and misleads the agent toward
wall-blocked shortcuts.  On L01, the exit is 7 tiles away by Manhattan but
13 by BFS — a 1.86× error that actively misled the agent.

Performance: BFS on a 16×16 grid (256 nodes) takes ~50 µs per call.
"""
from __future__ import annotations

from collections import deque


def _bfs_distance(tiles, px, py, targets):
    """BFS shortest-path distance from (px, py) to the nearest target.

    Returns the distance in tiles, or -1 if unreachable.
    """
    if not targets:
        return -1
    if (px, py) in targets:
        return 0

    visited = {(px, py)}
    queue = deque([(px, py, 0)])
    while queue:
        x, y, dist = queue.popleft()
        if (x, y) in targets:
            return dist
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < 16 and 0 <= ny < 16 and (nx, ny) not in visited:
                pos = nx + ny * 16
                # Walkable: tile >= 18 and not a collapsed crumble (31)
                if tiles[pos] >= 18 and tiles[pos] != 31:
                    visited.add((nx, ny))
                    queue.append((nx, ny, dist + 1))
    return -1


def compute_potential(gs) -> float:
    """BFS shortest-path distance to nearest uncollected carrot/egg.

    When all collectibles are gathered, targets the exit tile.
    BFS respects walls (tile < 18 → impassable), giving the true
    shortest path through the maze — unlike Manhattan which ignores walls.
    Potential is in [-1, 0]; closer to goal → closer to 0.

    After all collectibles are gathered, exit potential is normalised by /16
    (twice as strong as carrot potential at /32) so the exit gradient is
    competitive with the carrot collection signal.
    """
    tiles = gs.tiles
    px, py = gs.coord_src

    # Identify targets: uncollected carrots (19) or eggs (45)
    targets = set()
    for i, t in enumerate(tiles):
        if t in (19, 45):
            targets.add((i % 16, i // 16))

    all_collected = len(targets) == 0
    if all_collected:
        exit_idx = next((i for i, t in enumerate(tiles) if t == 44), None)
        if exit_idx is not None:
            targets = {(exit_idx % 16, exit_idx // 16)}

    if not targets:
        return 0.0

    dist = _bfs_distance(tiles, px, py, targets)
    if dist < 0:
        return -1.0

    # Exit potential uses /16 for stronger gradient; carrot uses /32
    norm = 16.0 if all_collected else 32.0
    return -dist / norm


def bfs_to_exit(gs) -> int:
    """BFS shortest-path distance from the player to the exit tile (44).

    Returns the distance in tiles, or -1 if unreachable.
    Used by RewardShapingWrapper for explicit exit-seeking reward.
    """
    tiles = gs.tiles
    px, py = gs.coord_src
    exit_idx = next((i for i, t in enumerate(tiles) if t == 44), None)
    if exit_idx is None:
        return -1
    target = {(exit_idx % 16, exit_idx // 16)}
    return _bfs_distance(tiles, px, py, target)
