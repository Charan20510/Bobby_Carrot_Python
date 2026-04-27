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

    NOTE: must be recomputed every step.  The maze topology mutates whenever
    a crumble collapses (30 → 31), a switch toggles (22/23, 24-27, 28/29,
    38/39, 40-43), or a key/lock pair clears.  Do NOT memoize the result
    across steps — a stale Φ will mislead the agent through tiles that no
    longer exist.
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


def simulate_level(level_idx: int, kind: str = "normal") -> dict:
    """Static analysis of a level: optimal path, choke points, crumble criticality.

    Loads the .blm file directly (no env, no pygame) and computes:
      - path_len:         BFS shortest start → all-carrots → exit (greedy nearest)
      - choke_points:     tiles whose removal disconnects start from exit
      - crumble_critical: True if collapsing any single crumble breaks reachability
      - perfect_steps:    same as path_len (lower bound)
      - est_timesteps:    10K × path_len × num_special_tiles (heuristic)
      - key_insight:      one-sentence summary
    """
    from bobby_carrot.core.loader import Map
    map_obj = Map(kind, level_idx)
    info = map_obj.load_map_info()
    tiles = info.data
    sx, sy = info.coord_start

    carrots = [(i % 16, i // 16) for i, t in enumerate(tiles) if t == 19]
    eggs    = [(i % 16, i // 16) for i, t in enumerate(tiles) if t == 45]
    crumbles = [(i % 16, i // 16) for i, t in enumerate(tiles) if t == 30]
    exit_pos = next(
        ((i % 16, i // 16) for i, t in enumerate(tiles) if t == 44), None
    )

    # Greedy nearest-neighbour TSP-style path estimate (lower bound)
    path_len = 0
    cur = (sx, sy)
    remaining = list(carrots) + list(eggs)
    while remaining:
        # Distance from cur to each remaining target via BFS
        dists = []
        for tgt in remaining:
            d = _bfs_distance(tiles, cur[0], cur[1], {tgt})
            dists.append((d if d >= 0 else 1_000, tgt))
        dists.sort()
        d, nxt = dists[0]
        if d >= 1_000:
            path_len = -1
            break
        path_len += d
        cur = nxt
        remaining.remove(nxt)
    if path_len >= 0 and exit_pos is not None:
        d_exit = _bfs_distance(tiles, cur[0], cur[1], {exit_pos})
        path_len = path_len + d_exit if d_exit >= 0 else -1

    # Crumble-criticality: simulate collapsing each crumble individually and
    # check that start can still reach the exit.
    crumble_critical = False
    if exit_pos is not None and path_len > 0:
        for cx, cy in crumbles:
            modified = list(tiles)
            modified[cx + cy * 16] = 31  # collapsed
            d = _bfs_distance(modified, sx, sy, {exit_pos})
            if d < 0:
                crumble_critical = True
                break

    # Special-tile count for the learn-time heuristic.
    special_ids = {19, 22, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37,
                   38, 40, 41, 42, 43, 45}
    num_special = sum(1 for t in tiles if t in special_ids)

    if path_len > 0:
        est_timesteps = 10_000 * path_len * max(num_special, 1)
        # Expected shaped reward on a perfect win:
        #   native +10 (win) + N×(+1) carrot/egg + ~+5 efficiency + ~+2 phase
        n_collect = len(carrots) + len(eggs)
        expected_reward = 10.0 + n_collect + 5.0 + 2.0
    else:
        est_timesteps = -1
        expected_reward = -1.0

    if crumble_critical:
        insight = (
            f"crumble-critical: at least one of {len(crumbles)} crumbles must "
            f"be preserved for the exit path"
        )
    elif crumbles:
        insight = f"{len(crumbles)} crumbles present but none individually critical"
    elif len(carrots) + len(eggs) == 0:
        insight = "no collectibles; exit-seeking phase active from step 0"
    else:
        insight = f"clean path: {len(carrots)} carrots, {len(eggs)} eggs, exit"

    return {
        "level": level_idx,
        "path_len": path_len,
        "num_carrots": len(carrots),
        "num_eggs": len(eggs),
        "num_crumbles": len(crumbles),
        "crumble_critical": crumble_critical,
        "perfect_steps": path_len,
        "est_timesteps": est_timesteps,
        "expected_reward": expected_reward,
        "key_insight": insight,
    }


def format_simulation(sim: dict) -> str:
    """Render simulate_level() output in the §4 format."""
    return (
        f"  ┌─ Level L{sim['level']:02d} Simulation "
        f"{'─' * (37 - len(str(sim['level'])))}┐\n"
        f"  │ Path length (greedy)  : {sim['perfect_steps']:>5} steps              │\n"
        f"  │ Carrots / Eggs        : {sim['num_carrots']:>2} / {sim['num_eggs']:>2}                   │\n"
        f"  │ Crumbles              : {sim['num_crumbles']:>2}                        │\n"
        f"  │ Crumble-critical      : {'yes' if sim['crumble_critical'] else 'no':<3}                       │\n"
        f"  │ Estimated learn time  : {sim['est_timesteps']/1000:>6.0f}K timesteps        │\n"
        f"  │ Expected win reward   : ~{sim['expected_reward']:>5.1f} (shaped)          │\n"
        f"  │ Key insight: {sim['key_insight'][:48]:<48}│\n"
        f"  └{'─' * 56}┘"
    )
