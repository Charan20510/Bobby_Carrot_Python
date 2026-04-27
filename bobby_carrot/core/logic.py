"""Pure game logic: movement, collision, tile effects, and frame advance.

No pygame import anywhere in this module.

Two usage modes
---------------
Interactive (renderer present):
    Each render frame call ``advance_frame(gs)`` which fires tile effects at
    animation step 8 and handles special-state transitions.  The renderer then
    calls ``compute_sprite_rects`` (in renderer.py) for the visual positions.

Headless / RL:
    Call ``logical_step(gs, action)`` once per agent action.  It immediately
    executes the full tile move and all chained forced movements (conveyors)
    without any frame-counter dependency.
"""
from __future__ import annotations

from typing import List, Tuple

from .state import (
    Action, ACTION_TO_STATE, GameState, State,
    FRAMES_PER_STEP, WIDTH_POINTS, HEIGHT_POINTS,
)

# ---------------------------------------------------------------------------
# Switch helpers
# ---------------------------------------------------------------------------


def _toggle_red_switch(tiles: List[int]) -> None:
    """Toggle all red-switch-controlled tiles (tile groups 22/23 and 24-29)."""
    for i in range(len(tiles)):
        v = tiles[i]
        if v == 22:   tiles[i] = 23
        elif v == 23: tiles[i] = 22
        elif v == 24: tiles[i] = 25
        elif v == 25: tiles[i] = 26
        elif v == 26: tiles[i] = 27
        elif v == 27: tiles[i] = 24
        elif v == 28: tiles[i] = 29
        elif v == 29: tiles[i] = 28


def _toggle_yellow_switch(tiles: List[int]) -> None:
    """Toggle all yellow-switch-controlled tiles (tile groups 38/39 and 40-43)."""
    for i in range(len(tiles)):
        v = tiles[i]
        if v == 38:   tiles[i] = 39
        elif v == 39: tiles[i] = 38
        elif v == 40: tiles[i] = 41
        elif v == 41: tiles[i] = 40
        elif v == 42: tiles[i] = 43
        elif v == 43: tiles[i] = 42

# ---------------------------------------------------------------------------
# Movement validation
# ---------------------------------------------------------------------------


def _compute_tentative_dest(
    coord: Tuple[int, int], direction: State
) -> Tuple[int, int]:
    x, y = coord
    if direction == State.Left and x > 0:
        return (x - 1, y)
    if direction == State.Right and x < WIDTH_POINTS - 1:
        return (x + 1, y)
    if direction == State.Up and y > 0:
        return (x, y - 1)
    if direction == State.Down and y < HEIGHT_POINTS - 1:
        return (x, y + 1)
    return coord   # wall boundary → stays put


def _check_dest(
    tiles: List[int],
    coord_src: Tuple[int, int],
    new_dest: Tuple[int, int],
    old_dest: Tuple[int, int],
    direction: State,
    key_gray: int,
    key_yellow: int,
    key_red: int,
) -> Tuple[bool, bool]:
    """Return (forbidden, death_queued) for a proposed move.

    ``old_dest`` is the coord_dest value *before* this move attempt — used as
    the fallback if the move is forbidden (matching the original Bobby logic).
    """
    old_pos = coord_src[0] + coord_src[1] * 16
    new_pos = new_dest[0] + new_dest[1] * 16
    old_item = tiles[old_pos]
    new_item = tiles[new_pos]

    forbidden = False

    if new_item < 18:
        forbidden = True
    if new_item == 33 and key_gray == 0:
        forbidden = True
    if new_item == 35 and key_yellow == 0:
        forbidden = True
    if new_item == 37 and key_red == 0:
        forbidden = True

    # new-tile conveyor entry restrictions
    if new_item == 24 and direction in {State.Right, State.Down}:
        forbidden = True
    if new_item == 25 and direction in {State.Left, State.Down}:
        forbidden = True
    if new_item == 26 and direction in {State.Left, State.Up}:
        forbidden = True
    if new_item == 27 and direction in {State.Right, State.Up}:
        forbidden = True
    if new_item in {28, 40, 41} and direction in {State.Up, State.Down}:
        forbidden = True
    if new_item in {29, 42, 43} and direction in {State.Left, State.Right}:
        forbidden = True
    if new_item == 40 and direction == State.Right:
        forbidden = True
    if new_item == 41 and direction == State.Left:
        forbidden = True
    if new_item == 42 and direction == State.Down:
        forbidden = True
    if new_item == 43 and direction == State.Up:
        forbidden = True
    if new_item == 46:
        forbidden = True

    # old-tile conveyor exit restrictions
    if old_item == 24 and direction in {State.Left, State.Up}:
        forbidden = True
    if old_item == 25 and direction in {State.Right, State.Up}:
        forbidden = True
    if old_item == 26 and direction in {State.Right, State.Down}:
        forbidden = True
    if old_item == 27 and direction in {State.Left, State.Down}:
        forbidden = True
    if old_item in {28, 40, 41} and direction in {State.Up, State.Down}:
        forbidden = True
    if old_item in {29, 42, 43} and direction in {State.Left, State.Right}:
        forbidden = True
    if old_item == 40 and direction == State.Right:
        forbidden = True
    if old_item == 41 and direction == State.Left:
        forbidden = True
    if old_item == 42 and direction == State.Down:
        forbidden = True
    if old_item == 43 and direction == State.Up:
        forbidden = True

    death_queued = new_item == 31

    return forbidden, death_queued


def _update_dest(gs: GameState) -> GameState:
    """Compute and validate coord_dest for the current player_state direction."""
    old_dest = gs.coord_dest
    tentative = _compute_tentative_dest(gs.coord_dest, gs.player_state)
    forbidden, death_queued = _check_dest(
        gs.tiles, gs.coord_src, tentative, old_dest, gs.player_state,
        gs.key_gray, gs.key_yellow, gs.key_red,
    )
    if death_queued:
        gs.next_state = State.Death
    if forbidden:
        gs.coord_dest = old_dest
    else:
        gs.coord_dest = tentative
    return gs

# ---------------------------------------------------------------------------
# Core state-transition functions
# ---------------------------------------------------------------------------


def start_move(gs: GameState, direction: State) -> GameState:
    """Begin moving in *direction*: sets player_state, start_frame, coord_dest."""
    gs = gs.copy()
    gs.start_frame = gs.frame
    gs.player_state = direction
    gs = _update_dest(gs)
    return gs


def apply_landing(gs: GameState) -> Tuple[GameState, List[str]]:
    """Process tile effects when a tile move completes.

    Handles departure effects on coord_src and arrival effects on coord_dest,
    then advances coord_src to coord_dest.  If a forced move (conveyor) or
    death is queued via next_state it is processed before returning.

    Returns the updated GameState and a list of event strings:
        "carrot_collected", "egg_collected", "key_picked"
    """
    gs = gs.copy()
    events: List[str] = []

    old_pos = gs.coord_src[0] + gs.coord_src[1] * 16
    new_pos = gs.coord_dest[0] + gs.coord_dest[1] * 16

    # --- departure tile effects (tile the player is leaving) ---
    item = gs.tiles[old_pos]
    if item == 24:   gs.tiles[old_pos] = 25
    elif item == 25: gs.tiles[old_pos] = 26
    elif item == 26: gs.tiles[old_pos] = 27
    elif item == 27: gs.tiles[old_pos] = 24
    elif item == 28: gs.tiles[old_pos] = 29
    elif item == 29: gs.tiles[old_pos] = 28
    elif item == 30:
        gs.tiles[old_pos] = 31   # crumble tile becomes deadly
        events.append("crumble_survived")
    elif item == 45:
        gs.tiles[old_pos] = 46
        gs.egg_count += 1
        events.append("egg_collected")

    # --- arrival tile effects (tile the player just landed on) ---
    new_item = gs.tiles[new_pos]
    if new_item == 19:
        gs.tiles[new_pos] = 20
        gs.carrot_count += 1
        events.append("carrot_collected")
    elif new_item == 22:
        _toggle_red_switch(gs.tiles)
    elif new_item == 31:
        pass   # crumble tile already deadly; death handled via next_state
    elif new_item == 32:
        gs.tiles[new_pos] = 18
        gs.key_gray += 1
        events.append("key_picked")
    elif new_item == 33 and gs.key_gray > 0:
        gs.tiles[new_pos] = 18
        gs.key_gray -= 1
    elif new_item == 34:
        gs.tiles[new_pos] = 18
        gs.key_yellow += 1
        events.append("key_picked")
    elif new_item == 35 and gs.key_yellow > 0:
        gs.tiles[new_pos] = 18
        gs.key_yellow -= 1
    elif new_item == 36:
        gs.tiles[new_pos] = 18
        gs.key_red += 1
        events.append("key_picked")
    elif new_item == 37 and gs.key_red > 0:
        gs.tiles[new_pos] = 18
        gs.key_red -= 1
    elif new_item == 38:
        _toggle_yellow_switch(gs.tiles)
    elif new_item == 40:
        gs.next_state = State.Left
    elif new_item == 41:
        gs.next_state = State.Right
    elif new_item == 42:
        gs.next_state = State.Up
    elif new_item == 43:
        gs.next_state = State.Down

    # finalise position
    gs.coord_src = gs.coord_dest
    gs.start_frame = gs.frame

    # process any queued move (forced conveyor movement or death)
    if gs.next_state is not None:
        queued = gs.next_state
        gs.next_state = None
        if queued == State.Death:
            gs.dead = True
            gs.player_state = State.Death
            # Emit a typed death cause so RL eval / failure-mode breakdown
            # can distinguish crumble deaths from other deaths.  Tile 31 is
            # the only deadly tile in the game; this branch fires whenever
            # the agent stepped onto a collapsed crumble.
            death_pos = gs.coord_src[0] + gs.coord_src[1] * 16
            if gs.tiles[death_pos] == 31:
                events.append("died_on_crumble")
            else:
                events.append("died_other")
        else:
            gs = start_move(gs, queued)

    return gs, events

# ---------------------------------------------------------------------------
# Interactive-mode frame advance
# ---------------------------------------------------------------------------


def advance_frame(gs: GameState) -> Tuple[GameState, List[str]]:
    """Process one animation frame's worth of game logic (interactive mode).

    Called once per rendered frame *before* the renderer draws.  Fires tile
    effects at the right animation step boundary and handles special-state
    transitions (FadeIn→Down, FadeOut completion, Death completion).
    """
    events: List[str] = []
    delta_frame = gs.frame - gs.start_frame
    step = delta_frame // FRAMES_PER_STEP

    if gs.player_state == State.FadeIn:
        if step >= 8:
            gs = gs.copy()
            gs.start_frame = gs.frame
            gs.player_state = State.Down

    elif gs.player_state == State.FadeOut:
        if step >= 8:
            gs = gs.copy()
            gs.faded_out = True

    elif gs.player_state == State.Death:
        if step // 3 >= 12:   # 36 animation steps → death confirmed
            gs = gs.copy()
            gs.dead = True

    elif gs.is_walking():
        # Pre-land death: start death animation 2 steps before landing
        if step == 6 and gs.next_state == State.Death:
            gs = gs.copy()
            gs.start_frame = gs.frame
            gs.player_state = State.Death
        elif step >= 8:
            gs, events = apply_landing(gs)

    return gs, events

# ---------------------------------------------------------------------------
# RL logical step (headless, no frame counting)
# ---------------------------------------------------------------------------


def queue_next_move(gs: GameState, action: Action) -> GameState:
    """Queue a move to execute after the current walking animation ends.

    Mirrors Bobby.update_next_state: only queues after step > 3 so that
    very-early input during a walk is ignored (matching original behaviour).
    """
    direction = ACTION_TO_STATE.get(action)
    if direction is None:
        return gs
    step = (gs.frame - gs.start_frame) // FRAMES_PER_STEP
    if step > 3 and gs.next_state not in {
        State.Idle, State.Death, State.FadeIn, State.FadeOut
    }:
        gs = gs.copy()
        gs.next_state = direction
    return gs


def logical_step(gs: GameState, action: Action) -> Tuple[GameState, List[str]]:
    """Execute one complete logical tile move for RL (no animation).

    Applies the player action, resolves the move immediately, and chains
    any forced movements from conveyors before returning.  The caller should
    check ``gs.dead`` and ``gs.faded_out`` (level won) after each call.
    """
    all_events: List[str] = []

    # Cannot act while in a special state or already walking
    if gs.dead or gs.faded_out:
        return gs, all_events
    if gs.player_state in (State.Death, State.FadeIn, State.FadeOut):
        return gs, all_events
    if gs.is_walking():
        return gs, all_events
    if action == Action.IDLE:
        return gs, all_events

    direction = ACTION_TO_STATE[action]
    gs = start_move(gs, direction)
    gs.frame += 1   # advance logical frame counter

    # Move was blocked — coord_src == coord_dest, nothing to land on
    if not gs.is_walking() and gs.next_state != State.Death:
        return gs, all_events

    # Execute the move (handles death-queued moves too via apply_landing)
    gs, events = apply_landing(gs)
    all_events.extend(events)

    # Auto-chain forced movements (conveyor tiles).  Death stops the chain.
    for _ in range(64):   # cap prevents infinite loops on pathological maps
        if not gs.is_walking() or gs.dead:
            break
        gs, events = apply_landing(gs)
        all_events.extend(events)

    return gs, all_events
