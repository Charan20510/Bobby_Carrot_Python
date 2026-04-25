"""Level loading: Map, MapInfo, and CLI map-selection helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

from .state import MAX_NORMAL_MAP, MAX_EGG_MAP


def asset_path(sub: str) -> Path:
    # __file__ = bobby_carrot/core/loader.py  →  .parent.parent.parent = project root
    root = Path(__file__).parent.parent.parent
    return root / "assets" / sub


def _is_valid_map_number(kind: str, number: int) -> bool:
    if number < 1:
        return False
    if kind == "normal":
        return number <= MAX_NORMAL_MAP
    if kind == "egg":
        return number <= MAX_EGG_MAP
    return False


class MapInfo:
    """Parsed level data."""

    def __init__(
        self,
        data: List[int],
        coord_start: Tuple[int, int],
        carrot_total: int,
        egg_total: int,
    ) -> None:
        self.data = data
        self.coord_start = coord_start
        self.carrot_total = carrot_total
        self.egg_total = egg_total


class Map:
    """Identifies a level and knows how to load it."""

    def __init__(self, kind: str, number: int) -> None:
        self.kind = kind   # "normal" or "egg"
        self.number = number

    def __str__(self) -> str:
        if self.kind == "normal":
            return f"Normal-{self.number:02}"
        return f"Egg-{self.number:02}"

    def load_map_info(self) -> MapInfo:
        fname = f"{self.kind}{self.number:02}.blm"
        path = asset_path(f"level/{fname}")
        data = path.read_bytes()[4:]   # skip 4-byte header
        start_idx = 0
        carrot_total = 0
        egg_total = 0
        for idx, byte in enumerate(data):
            if byte == 19:
                carrot_total += 1
            elif byte == 45:
                egg_total += 1
            elif byte == 21:
                start_idx = idx
        coord_start = (start_idx % 16, start_idx // 16)
        return MapInfo(
            data=list(data),
            coord_start=coord_start,
            carrot_total=carrot_total,
            egg_total=egg_total,
        )

    def next(self) -> Map:
        if self.kind == "normal":
            if self.number < 30:
                return Map("normal", self.number + 1)
            return Map("egg", 1)
        if self.number < 20:
            return Map("egg", self.number + 1)
        return Map("normal", 1)

    def previous(self) -> Map:
        if self.kind == "normal":
            if self.number <= 1:
                return Map("egg", 20)
            return Map("normal", self.number - 1)
        if self.number <= 1:
            return Map("normal", 30)
        return Map("egg", self.number - 1)


def parse_map_arg(arg: str) -> Map:
    try:
        num = int(arg)
    except ValueError:
        num = None
    if num is not None:
        if not _is_valid_map_number("normal", num):
            raise ValueError(
                f"Normal map out of range: {num} (expected 1-{MAX_NORMAL_MAP})"
            )
        return Map("normal", num)
    if "-" in arg:
        type_str, num_str = arg.split("-", 1)
        try:
            num = int(num_str)
        except ValueError as exc:
            raise ValueError(f"Invalid map: {arg}") from exc
        if type_str.lower() == "normal":
            if not _is_valid_map_number("normal", num):
                raise ValueError(
                    f"Normal map out of range: {num} (expected 1-{MAX_NORMAL_MAP})"
                )
            return Map("normal", num)
        if type_str.lower() == "egg":
            if not _is_valid_map_number("egg", num):
                raise ValueError(
                    f"Egg map out of range: {num} (expected 1-{MAX_EGG_MAP})"
                )
            return Map("egg", num)
    raise ValueError(f"Invalid map: {arg}")


def choose_map_interactive() -> Map:
    print(
        "Select a level to play (examples: 5, normal-3, egg-10).\n"
        "Press Enter for normal level 1:"
    )
    if not sys.stdin.isatty():
        return Map("normal", 1)
    try:
        choice = input("> ").strip()
    except EOFError:
        return Map("normal", 1)
    if choice == "":
        return Map("normal", 1)
    return parse_map_arg(choice)
