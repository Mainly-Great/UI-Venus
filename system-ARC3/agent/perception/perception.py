"""
Layer 1 — Sensory Perception

Converts raw ARC-AGI-3 frame JSON into a dual representation:
  1. Visual: a color-mapped grid image (for optional VLM input)
  2. Symbolic: a list of discovered entities with positions, colors, shapes,
     movement vectors, and relationships.

The symbolic representation is what the reasoning layers consume. It is
the system's "sensory cortex" — it detects *what* is on the screen and
*how it changed* between consecutive frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ARC-AGI-3 uses 16 colors (0-15). This palette maps each to a distinct RGB.
ARC_PALETTE: list[tuple[int, int, int]] = [
    (0, 0, 0),        # 0  — background / void
    (30, 30, 200),    # 1  — blue
    (200, 30, 30),    # 2  — red
    (30, 200, 30),    # 3  — green
    (200, 200, 30),   # 4  — yellow
    (200, 30, 200),   # 5  — magenta
    (30, 200, 200),   # 6  — cyan
    (255, 120, 0),    # 7  — orange
    (120, 0, 255),    # 8  — purple
    (0, 120, 255),    # 9  — light blue
    (255, 200, 150),  # 10 — peach
    (100, 70, 50),    # 11 — brown
    (220, 220, 220),  # 12 — white
    (80, 80, 80),     # 13 — grey
    (180, 180, 120),  # 14 — beige
    (255, 255, 180),  # 15 — light yellow
]

# Semantic color names for symbolic reasoning
COLOR_NAMES: dict[int, str] = {
    0: "void",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "magenta",
    6: "cyan",
    7: "orange",
    8: "purple",
    9: "light_blue",
    10: "peach",
    11: "brown",
    12: "white",
    13: "grey",
    14: "beige",
    15: "light_yellow",
}


@dataclass
class Entity:
    """A connected region of same-colored cells discovered in the grid."""
    eid: int
    color: int
    color_name: str
    cells: list[tuple[int, int]]                      # (x, y) positions
    bbox: tuple[int, int, int, int]                   # min_x, min_y, max_x, max_y
    centroid: tuple[float, float]                     # cx, cy
    area: int                                          # number of cells
    shape_type: str = "unknown"                        # point, rect, line, L, T, other
    prev_centroid: tuple[float, float] | None = None
    movement: tuple[float, float] = (0.0, 0.0)        # delta from previous frame
    is_player: bool = False
    is_dynamic: bool = False                          # changed position across frames
    appeared: bool = False                            # newly appeared this frame
    disappeared: bool = False                         # was present before, now gone
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Entity({self.eid}, {self.color_name}, "
            f"area={self.area}, shape={self.shape_type}, "
            f"center=({self.centroid[0]:.0f},{self.centroid[1]:.0f})"
            f"{', player' if self.is_player else ''}"
            f"{', dynamic' if self.is_dynamic else ''})"
        )


@dataclass
class FrameAnalysis:
    """Complete symbolic description of a single frame."""
    step: int
    grid: np.ndarray                                  # (H, W) int array, values 0-15
    height: int
    width: int
    entities: list[Entity]
    colors_present: set[int]
    player_entity: Entity | None = None
    other_players: list[Entity] = field(default_factory=list)
    static_entities: list[Entity] = field(default_factory=list)
    dynamic_entities: list[Entity] = field(default_factory=list)
    grid_hash: str = ""
    is_anim: bool = False                             # animation frame (non-interactive)
    is_anim_intermediate: bool = False                # intermediate animation frame (skip for rules)
    is_turn_boundary: bool = True                     # True = stable post-action frame (safe for diff)
    state: str = "NOT_FINISHED"                       # WIN / GAME_OVER / NOT_FINISHED
    available_actions: list[str] = field(default_factory=list)
    level: int = 0

    def summary(self) -> str:
        lines = [
            f"Step {self.step} | {self.width}x{self.height} | state={self.state} | level={self.level}",
            f"Colors: {sorted(self.colors_present)}",
            f"Entities: {len(self.entities)} "
            f"(players={1 if self.player_entity else 0}+{len(self.other_players)}, "
            f"dynamic={len(self.dynamic_entities)}, static={len(self.static_entities)})",
        ]
        for e in self.entities:
            lines.append(f"  {e}")
        return "\n".join(lines)


class Perception:
    """
    Sensory perception module — converts raw frame JSON into FrameAnalysis.

    Maintains the previous frame's entity map to track movement and detect
    what appeared / disappeared / changed.

    Animation-Resistant Diff:
      ARC-AGI-3 games emit intermediate animation frames between turns.
      These frames show the player/entity mid-motion and are NOT stable
      game states. Comparing against them produces false movement signals
      and corrupts rule discovery.

      Solution: we track a `_last_turn_frame` — the last stable frame
      that was a turn boundary (had available actions, wasn't an animation
      intermediate). All diffs are computed against this, not against the
      immediately preceding frame. Intermediate animation frames are
      marked `is_anim_intermediate=True` and skipped by RuleDiscovery.
    """

    def __init__(self) -> None:
        self._prev_entities: list[Entity] = []
        self._prev_grid: np.ndarray | None = None
        self._entity_counter: int = 0
        self._step: int = 0
        # Animation-resistant diff: track the last stable turn-boundary frame
        self._last_turn_grid: np.ndarray | None = None
        self._last_turn_entities: list[Entity] = []
        self._prev_frame_dict: dict[str, Any] | None = None

    def reset(self) -> None:
        self._prev_entities = []
        self._prev_grid = None
        self._entity_counter = 0
        self._step = 0
        self._last_turn_grid = None
        self._last_turn_entities = []
        self._prev_frame_dict = None

    def parse_frame(self, frame: dict[str, Any]) -> FrameAnalysis:
        """
        Parse a raw ARC-AGI-3 frame dict into a FrameAnalysis.

        Expected frame structure (from arc-agi toolkit / REST API):
          {
            "grid": [[int, ...], ...],       # 2D list, values 0-15
            "state": "NOT_FINISHED" | "WIN" | "GAME_OVER",
            "available_actions": ["ACTION1", ...],
            "level": int,
            "is_anim": bool (optional)
          }
        """
        self._step += 1

        grid_data = frame.get("grid") or frame.get("frame") or frame
        if isinstance(grid_data, dict):
            grid_data = grid_data.get("grid", grid_data)

        grid = np.array(grid_data, dtype=np.int8)
        if grid.ndim == 3:
            grid = grid[:, :, 0] if grid.shape[2] > 0 else grid.squeeze()

        height, width = grid.shape
        colors_present = set(np.unique(grid).tolist())

        state = frame.get("state", frame.get("game_state", "NOT_FINISHED"))
        available_actions = frame.get("available_actions", [])
        level = frame.get("level", frame.get("current_level", 0))
        is_anim = frame.get("is_anim", False)

        # Detect animation intermediate frames.
        # Heuristics for detecting an intermediate (non-turn-boundary) frame:
        #   1. is_anim flag is True AND available_actions is empty
        #   2. Grid hash matches a previously seen animation-only state
        #   3. No available actions (player can't act = not a real turn)
        is_anim_intermediate = False
        if is_anim and len(available_actions) == 0:
            is_anim_intermediate = True
        elif len(available_actions) == 0 and state == "NOT_FINISHED":
            # No actions available and not won/lost — likely mid-animation
            is_anim_intermediate = True

        # Also detect via grid hash: if this grid appeared as a transient
        # between two stable states, it's intermediate
        if self._prev_frame_dict and not is_anim_intermediate:
            prev_actions = self._prev_frame_dict.get("available_actions", [])
            if len(prev_actions) == 0 and len(available_actions) > 0:
                # Previous frame was intermediate, this one is stable
                is_anim_intermediate = False

        is_turn_boundary = not is_anim_intermediate

        entities = self._detect_entities(grid)
        entities = self._classify_shapes(entities)

        # Animation-resistant diff: compare against last TURN BOUNDARY, not last frame
        # This prevents movement tracking from being polluted by animation frames
        if is_turn_boundary:
            # Diff against the last stable frame
            entities = self._track_movement(entities, self._last_turn_entities)
        else:
            # Intermediate frame — don't update movement tracking
            # Just carry forward previous movement info
            entities = self._track_movement(entities, self._prev_entities)

        entities = self._identify_players(entities)

        grid_hash = self._hash_grid(grid)

        analysis = FrameAnalysis(
            step=self._step,
            grid=grid,
            height=height,
            width=width,
            entities=entities,
            colors_present=colors_present,
            grid_hash=grid_hash,
            is_anim=is_anim,
            is_anim_intermediate=is_anim_intermediate,
            is_turn_boundary=is_turn_boundary,
            state=state,
            available_actions=available_actions,
            level=level,
        )

        analysis.player_entity = next((e for e in entities if e.is_player), None)
        analysis.other_players = [e for e in entities if e.is_player and e is not analysis.player_entity]
        analysis.dynamic_entities = [e for e in entities if e.is_dynamic]
        analysis.static_entities = [e for e in entities if not e.is_dynamic]

        self._prev_entities = entities
        self._prev_grid = grid.copy()
        self._prev_frame_dict = frame

        # Only update turn-boundary state when this is a stable frame
        if is_turn_boundary:
            self._last_turn_grid = grid.copy()
            self._last_turn_entities = entities

        return analysis

    def _detect_entities(self, grid: np.ndarray) -> list[Entity]:
        """Detect connected regions of same-colored cells via flood fill."""
        height, width = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        entities: list[Entity] = []

        for y in range(height):
            for x in range(width):
                if visited[y, x]:
                    continue
                color = int(grid[y, x])
                if color == 0:
                    visited[y, x] = True
                    continue

                cells: list[tuple[int, int]] = []
                stack = [(x, y)]
                while stack:
                    cx, cy = stack.pop()
                    if cx < 0 or cx >= width or cy < 0 or cy >= height:
                        continue
                    if visited[cy, cx] or int(grid[cy, cx]) != color:
                        continue
                    visited[cy, cx] = True
                    cells.append((cx, cy))
                    stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])

                if not cells:
                    continue

                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                eid = self._entity_counter
                self._entity_counter += 1

                entity = Entity(
                    eid=eid,
                    color=color,
                    color_name=COLOR_NAMES.get(color, f"c{color}"),
                    cells=cells,
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    centroid=(sum(xs) / len(xs), sum(ys) / len(ys)),
                    area=len(cells),
                )
                entities.append(entity)

        return entities

    def _classify_shapes(self, entities: list[Entity]) -> list[Entity]:
        """Classify each entity's shape: point, rect, line_h, line_v, L, T, other."""
        for e in entities:
            min_x, min_y, max_x, max_y = e.bbox
            w = max_x - min_x + 1
            h = max_y - min_y + 1

            if e.area == 1:
                e.shape_type = "point"
            elif e.area == w * h:
                if w == 1 and h == 1:
                    e.shape_type = "point"
                elif w == 1:
                    e.shape_type = "line_v"
                elif h == 1:
                    e.shape_type = "line_h"
                else:
                    e.shape_type = "rect"
            elif w == 1:
                e.shape_type = "line_v"
            elif h == 1:
                e.shape_type = "line_h"
            elif e.area <= 3:
                e.shape_type = "L" if e.area == 3 else "other"
            else:
                e.shape_type = "other"

        return entities

    def _track_movement(self, current: list[Entity], previous: list[Entity]) -> list[Entity]:
        """Match entities across frames to detect movement, appearance, disappearance."""
        if not previous:
            for e in current:
                e.appeared = True
            return current

        prev_by_color: dict[int, list[Entity]] = {}
        for pe in previous:
            prev_by_color.setdefault(pe.color, []).append(pe)

        used_prev: set[int] = set()

        for ce in current:
            candidates = [pe for pe in prev_by_color.get(ce.color, []) if pe.eid not in used_prev]
            if not candidates:
                ce.appeared = True
                continue

            best = min(candidates, key=lambda pe: self._centroid_distance(ce.centroid, pe.centroid))
            used_prev.add(best.eid)

            ce.prev_centroid = best.centroid
            ce.movement = (ce.centroid[0] - best.centroid[0], ce.centroid[1] - best.centroid[1])

            if abs(ce.movement[0]) > 0.5 or abs(ce.movement[1]) > 0.5:
                ce.is_dynamic = True

            if best.area != ce.area:
                ce.metadata["area_changed"] = True
                ce.metadata["prev_area"] = best.area

        for pe in previous:
            if pe.eid not in used_prev:
                # Entity disappeared — record as metadata on closest current entity
                ce = min(current, key=lambda c: self._centroid_distance(c.centroid, pe.centroid), default=None)
                if ce:
                    ce.metadata.setdefault("nearby_disappeared", []).append({
                        "color": pe.color_name,
                        "area": pe.area,
                        "centroid": pe.centroid,
                    })

        return current

    def _identify_players(self, entities: list[Entity]) -> list[Entity]:
        """
        Heuristic player identification:
          - Dynamic entities (move between frames) are likely players or autonomous movers
          - Single-cell dynamic entities are most likely player-controlled
          - If no dynamic entities, the smallest non-background entity is a candidate
        """
        dynamic = [e for e in entities if e.is_dynamic]
        if dynamic:
            for e in dynamic:
                if e.area <= 4 and abs(e.movement[0]) <= 1.5 and abs(e.movement[1]) <= 1.5:
                    e.is_player = True
                else:
                    e.metadata["autonomous_mover"] = True

        if not any(e.is_player for e in entities):
            non_bg = [e for e in entities if e.color != 0]
            if non_bg:
                smallest = min(non_bg, key=lambda e: e.area)
                smallest.is_player = True
                smallest.metadata["player_candidate"] = True

        # Mark the first player as the primary player
        players = [e for e in entities if e.is_player]
        if players:
            players[0].metadata["primary_player"] = True

        return entities

    def grid_to_image(self, grid: np.ndarray, scale: int = 8) -> np.ndarray:
        """Convert a grid to an RGB image (for VLM input)."""
        height, width = grid.shape
        img = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                color = ARC_PALETTE[int(grid[y, x]) % 16]
                img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = color
        return img

    def grid_to_ascii(self, grid: np.ndarray) -> str:
        """Compact ASCII rendering of the grid for text-based reasoning."""
        symbols = "0123456789ABCDEF"
        height, width = grid.shape
        lines = []
        for y in range(height):
            row = "".join(symbols[int(grid[y, x]) % 16] for x in range(width))
            lines.append(row)
        return "\n".join(lines)

    @staticmethod
    def _centroid_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _hash_grid(grid: np.ndarray) -> str:
        import hashlib
        return hashlib.md5(grid.tobytes()).hexdigest()[:12]
