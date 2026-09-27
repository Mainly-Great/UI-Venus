"""
Layer 7 — Planner

The decision-making layer. Takes all knowledge from memory, rules, goals,
and the current frame analysis, then selects the best action.

Decision pipeline:
  1. If game-over state -> RESET
  2. If exploration needed -> follow exploration module
  3. If a clear goal exists with high confidence -> plan path to goal
  4. If goal is uncertain -> take information-gathering actions
  5. If model is available -> consult model for recommendation
  6. Fallback -> heuristic action selection

Path planning uses BFS/A* on the grid, respecting discovered obstacles
and hazards. For multi-player games, it plans paths for each player to
a meeting point.
"""

from __future__ import annotations

import logging
from typing import Any
from collections import deque

import numpy as np

logger = logging.getLogger("arc3_planner")

from ..perception.perception import FrameAnalysis, Entity
from ..memory.memory import Memory, MemoryEntry
from ..model.model_client import ModelClient
from ..world_model import WorldModel, PredictedState


class Planner:
    """Selects actions based on accumulated knowledge and current state."""

    def __init__(self, model: ModelClient | None = None) -> None:
        self.model = model
        self.world_model = WorldModel()
        self._action_history: list[str] = []
        self._last_failed_direction: str | None = None
        self._expectation_violation: dict[str, Any] | None = None
        self._model_failed: bool = False
        self._model_skip_until_reset: bool = False

    def reset(self) -> None:
        self._action_history = []
        self._last_failed_direction = None
        self._expectation_violation = None
        self._model_failed = False
        self._model_skip_until_reset = False
        self.world_model.reset()

    def choose(
        self,
        analysis: FrameAnalysis,
        memory: Memory,
        exploration_action: tuple[str, dict[str, int]] | None = None,
    ) -> tuple[str, dict[str, int]]:
        """Choose the next action."""

        # 1. Game over -> must reset
        if analysis.state == "GAME_OVER":
            return "RESET", {}

        # 2. Already won -> reset for next level
        if analysis.state == "WIN":
            return "RESET", {}

        # 3. Exploration (integrated with budget)
        if exploration_action:
            action, data = exploration_action
            # Verify with world model before executing
            pred = self.world_model.predict(analysis, action, data, memory)
            if pred.predicts_game_over and pred.confident:
                # Skip this exploration action — it would kill us
                pass
            else:
                self._action_history.append(action)
                return action, data

        # 4. Goal-directed planning with world-model verification
        best_goal = memory.get_best_goal()
        if best_goal and best_goal.confidence > 0.35:
            planned = self._plan_to_goal(analysis, memory, best_goal)
            if planned:
                action, data = planned
                # Verify: simulate this action before executing
                pred = self.world_model.predict(analysis, action, data, memory)
                if pred.confident and pred.predicts_game_over:
                    # This action would kill us — try an alternative
                    alt = self._find_safe_alternative(analysis, memory, action)
                    if alt:
                        action, data = alt
                self._action_history.append(action)
                return action, data

        # 5. Heuristic fallback — move toward nearest non-hazard entity
        heuristic = self._heuristic_action(analysis, memory)
        if heuristic:
            action, data = heuristic
            # Verify
            pred = self.world_model.predict(analysis, action, data, memory)
            if not (pred.confident and pred.predicts_game_over):
                self._action_history.append(action)
                return action, data

        # 6. Consult model (only when stuck AND model hasn't failed this level)
        if self.model and not self._model_skip_until_reset and self.model.is_available:
            model_action = self._consult_model(analysis, memory)
            if model_action:
                action, data = model_action
                self._action_history.append(action)
                return action, data
            else:
                # Model returned no usable action — skip it for the rest of this level
                self._model_skip_until_reset = True
                logger.warning("[Planner] Model returned no action — skipping model calls until reset")
        elif self.model and not self.model.is_available and not self._model_failed:
            self._model_failed = True
            logger.info("[Planner] Model server not available — running symbolic-only mode")

        # 7. Random valid action (last resort, prefer safe ones)
        return self._safe_random_action(analysis, memory)

    def _plan_to_goal(
        self, analysis: FrameAnalysis, memory: Memory, goal: MemoryEntry
    ) -> tuple[str, dict[str, int]] | None:
        """Plan actions to achieve the best goal hypothesis."""
        goal_text = goal.value.lower()

        if "reach" in goal_text or "bring" in goal_text or "meet" in goal_text:
            return self._plan_reach(analysis, memory, goal)
        if "collect" in goal_text:
            return self._plan_collect(analysis, memory, goal)
        if "refill" in goal_text:
            return self._plan_refill_then_reach(analysis, memory, goal)
        return None

    def _plan_reach(
        self, analysis: FrameAnalysis, memory: Memory, goal: MemoryEntry
    ) -> tuple[str, dict[str, int]] | None:
        """Plan path to reach a target entity."""
        if not analysis.player_entity:
            return None

        player = analysis.player_entity

        # Determine target
        target = self._find_goal_target(analysis, memory, goal)
        if not target:
            return None

        # BFS pathfinding on the grid
        path = self._bfs_path(
            analysis.grid,
            (int(player.centroid[0]), int(player.centroid[1])),
            (int(target.centroid[0]), int(target.centroid[1])),
            memory,
        )

        if path and len(path) >= 2:
            next_pos = path[1]
            return self._action_towards(player, next_pos, memory, analysis.available_actions)

        # No path found — try moving in the general direction
        return self._move_towards(player, target, memory, analysis.available_actions)

    def _plan_collect(
        self, analysis: FrameAnalysis, memory: Memory, goal: MemoryEntry
    ) -> tuple[str, dict[str, int]] | None:
        """Plan to collect all entities of a certain color."""
        if not analysis.player_entity:
            return None

        # Find the color name from the goal
        goal_text = goal.value
        for color_name in memory.entity_dict:
            if color_name in goal_text:
                # Find nearest entity of this color
                targets = [e for e in analysis.entities if e.color_name == color_name]
                if targets:
                    nearest = min(
                        targets,
                        key=lambda e: self._dist(analysis.player_entity.centroid, e.centroid),
                    )
                    path = self._bfs_path(
                        analysis.grid,
                        (int(analysis.player_entity.centroid[0]), int(analysis.player_entity.centroid[1])),
                        (int(nearest.centroid[0]), int(nearest.centroid[1])),
                        memory,
                    )
                    if path and len(path) >= 2:
                        return self._action_towards(
                            analysis.player_entity, path[1], memory, analysis.available_actions
                        )
                    return self._move_towards(
                        analysis.player_entity, nearest, memory, analysis.available_actions
                    )
        return None

    def _plan_refill_then_reach(
        self, analysis: FrameAnalysis, memory: Memory, goal: MemoryEntry
    ) -> tuple[str, dict[str, int]] | None:
        """First go to refill entity, then to the goal target."""
        if not analysis.player_entity:
            return None

        # Find refill entity
        refill = None
        for e in analysis.entities:
            if memory.get_entity_meaning(e.color_name) == "move_refill":
                refill = e
                break

        if refill:
            # Go to refill first
            path = self._bfs_path(
                analysis.grid,
                (int(analysis.player_entity.centroid[0]), int(analysis.player_entity.centroid[1])),
                (int(refill.centroid[0]), int(refill.centroid[1])),
                memory,
            )
            if path and len(path) >= 2:
                return self._action_towards(
                    analysis.player_entity, path[1], memory, analysis.available_actions
                )
            return self._move_towards(
                analysis.player_entity, refill, memory, analysis.available_actions
            )

        # No refill found — proceed to goal
        return self._plan_reach(analysis, memory, goal)

    def _find_goal_target(self, analysis: FrameAnalysis, memory: Memory, goal: MemoryEntry) -> Entity | None:
        """Find the target entity for a reach goal."""
        if not analysis.player_entity:
            return None

        goal_text = goal.value.lower()

        # Meet players
        if "meet" in goal_text or "both" in goal_text or "together" in goal_text:
            other_players = [e for e in analysis.entities if e.is_player and e is not analysis.player_entity]
            if other_players:
                return min(other_players, key=lambda e: self._dist(analysis.player_entity.centroid, e.centroid))

        # Reach a specific color
        for color_name in memory.entity_dict:
            if color_name in goal_text:
                entities = [e for e in analysis.entities if e.color_name == color_name]
                if entities:
                    return min(entities, key=lambda e: self._dist(analysis.player_entity.centroid, e.centroid))

        # Generic: nearest non-hazard entity that's not the player
        for e in analysis.entities:
            if e is analysis.player_entity or e.color == 0:
                continue
            meaning = memory.get_entity_meaning(e.color_name)
            if meaning in ("hazard", "wall/obstacle"):
                continue
            return e

        return None

    def _bfs_path(
        self, grid: np.ndarray, start: tuple[int, int], goal_pos: tuple[int, int], memory: Memory
    ) -> list[tuple[int, int]] | None:
        """BFS pathfinding avoiding obstacles and hazards."""
        h, w = grid.shape
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)

        # Build hazard map from memory
        hazard_colors = set()
        for color_name, entry in memory.entity_dict.items():
            if entry.value in ("hazard", "wall/obstacle"):
                for i, c in enumerate(["void","blue","red","green","yellow","magenta","cyan","orange","purple","light_blue","peach","brown","white","grey","beige","light_yellow"]):
                    if c == color_name:
                        hazard_colors.add(i)

        while queue:
            (x, y), path = queue.popleft()
            if (x, y) == goal_pos:
                return path

            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    continue
                if (nx, ny) in visited:
                    continue
                cell_val = int(grid[ny, nx])
                if cell_val in hazard_colors:
                    continue
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    def _action_towards(
        self, player: Entity, target_pos: tuple[int, int], memory: Memory, available_actions: list[str]
    ) -> tuple[str, dict[str, int]] | None:
        """Choose the action that moves the player toward target_pos."""
        px, py = player.centroid
        tx, ty = target_pos
        dx = tx - px
        dy = ty - py

        # Map direction to action based on memory
        direction = None
        if abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        elif abs(dy) > 0:
            direction = "down" if dy > 0 else "up"

        if direction:
            action = self._direction_to_action(direction, memory, available_actions)
            if action:
                return action, {}

        return None

    def _move_towards(
        self, player: Entity, target: Entity, memory: Memory, available_actions: list[str]
    ) -> tuple[str, dict[str, int]] | None:
        """Move in the general direction of the target."""
        return self._action_towards(
            player,
            (int(target.centroid[0]), int(target.centroid[1])),
            memory,
            available_actions,
        )

    def _direction_to_action(self, direction: str, memory: Memory, available_actions: list[str]) -> str | None:
        """Map a direction to the correct action using learned action effects."""
        # Check memory first
        for action, entry in memory.action_effects.items():
            if direction in entry.value and action in available_actions:
                return action

        # Default mapping (ARC-AGI-3 standard): ACTION1=up, 2=down, 3=left, 4=right
        defaults = {"up": "ACTION1", "down": "ACTION2", "left": "ACTION3", "right": "ACTION4"}
        action = defaults.get(direction)
        if action and action in available_actions:
            return action

        # If default not available, try any action with that direction
        for a in available_actions:
            entry = memory.action_effects.get(a)
            if entry and direction in entry.value:
                return a

        return None

    def _consult_model(self, analysis: FrameAnalysis, memory: Memory) -> tuple[str, dict[str, int]] | None:
        """Ask the model for a recommended action."""
        context = memory.build_context_summary()
        grid_text = self._grid_compact(analysis.grid)

        question = (
            f"Current grid:\n{grid_text}\n\n"
            f"Available actions: {analysis.available_actions}\n"
            f"Player at ({analysis.player_entity.centroid[0]:.0f}, {analysis.player_entity.centroid[1]:.0f})\n"
            "What action should I take? Respond with ONLY the action name "
            "(e.g. ACTION1) and optionally coordinates for ACTION6 as: ACTION6 x y"
        )

        response = self.model.reason(context, question)
        return self._parse_model_action(response, analysis.available_actions)

    def _parse_model_action(self, response: str, available_actions: list[str]) -> tuple[str, dict[str, int]] | None:
        """Parse model response into an action."""
        response = response.strip()

        # Handle "ACTION6 x y" format
        parts = response.split()
        if not parts:
            return None

        action = parts[0].upper()
        if action not in available_actions:
            # Try to find a valid action mentioned
            for a in available_actions:
                if a in response.upper():
                    action = a
                    break
            else:
                return None

        data = {}
        if action == "ACTION6" and len(parts) >= 3:
            try:
                data["x"] = int(parts[1])
                data["y"] = int(parts[2])
            except ValueError:
                pass

        return action, data

    def _heuristic_action(
        self, analysis: FrameAnalysis, memory: Memory
    ) -> tuple[str, dict[str, int]] | None:
        """Fallback: move toward nearest interesting non-hazard entity."""
        if not analysis.player_entity:
            return None

        player = analysis.player_entity
        candidates = [
            e for e in analysis.entities
            if e is not player and e.color != 0
            and memory.get_entity_meaning(e.color_name) not in ("hazard", "wall/obstacle")
        ]

        if candidates:
            nearest = min(candidates, key=lambda e: self._dist(player.centroid, e.centroid))
            return self._move_towards(player, nearest, memory, analysis.available_actions)

        return None

    def _find_safe_alternative(
        self, analysis: FrameAnalysis, memory: Memory, unsafe_action: str
    ) -> tuple[str, dict[str, int]] | None:
        """Find an alternative action that doesn't lead to GAME_OVER."""
        safe_actions = [a for a in analysis.available_actions if a != unsafe_action and a != "RESET"]
        for action in safe_actions:
            pred = self.world_model.predict(analysis, action, {}, memory)
            if not (pred.confident and pred.predicts_game_over):
                return action, {}
        return None

    def _safe_random_action(
        self, analysis: FrameAnalysis, memory: Memory
    ) -> tuple[str, dict[str, int]]:
        """Pick a random action that the world model says is safe."""
        valid = [a for a in analysis.available_actions if a != "RESET" and a != "ACTION6"]
        safe = []
        for action in valid:
            pred = self.world_model.predict(analysis, action, {}, memory)
            if not (pred.confident and pred.predicts_game_over):
                safe.append(action)

        if safe:
            import random
            return random.choice(safe), {}
        if valid:
            import random
            return random.choice(valid), {}
        return "RESET", {}

    def check_expectation_violation(
        self, analysis: FrameAnalysis, last_prediction: PredictedState | None
    ) -> dict[str, Any] | None:
        """Check if the actual outcome violated the world model's prediction."""
        if last_prediction is None or not last_prediction.confident:
            return None
        return self.world_model.detect_expectation_violation(last_prediction, analysis)

    @property
    def last_violation(self) -> dict[str, Any] | None:
        return self._expectation_violation

    @staticmethod
    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _grid_compact(grid: np.ndarray) -> str:
        symbols = "0123456789ABCDEF"
        h, w = grid.shape
        if h > 32 or w > 32:
            grid = grid[::2, ::2]
            h, w = grid.shape
        return "\n".join("".join(symbols[int(grid[y, x]) % 16] for x in range(w)) for y in range(h))
