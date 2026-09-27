"""
World Model — Internal Game Simulator

A lightweight forward model built from discovered rules. It predicts what
will happen if the agent takes an action, WITHOUT executing it in the real
game. This lets the Planner verify plans before committing.

The world model is NOT magic — it's a function that applies the laws
stored in Memory to a grid state. When laws are incomplete, it says so
(predicted=False) so the Planner knows the prediction is unreliable.

Usage:
    wm = WorldModel()
    pred = wm.predict(current_grid, "ACTION1", memory)
    if pred.confident and pred.predicts_win:
        # Safe to execute
    elif pred.predicts_game_over:
        # Avoid this action
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..perception.perception import FrameAnalysis, Entity, COLOR_NAMES
from ..memory.memory import Memory


@dataclass
class PredictedState:
    """Result of a world-model prediction."""
    grid: np.ndarray | None = None            # predicted grid after action
    player_pos: tuple[int, int] | None = None  # predicted player position
    state: str = "NOT_FINISHED"                # predicted game state
    confident: bool = False                    # whether we trust this prediction
    reason: str = ""                           # why/why not confident
    predicts_win: bool = False
    predicts_game_over: bool = False
    entities_removed: list[str] = field(default_factory=list)   # color names removed
    entities_appeared: list[str] = field(default_factory=list)
    movement_direction: str | None = None

    def __repr__(self) -> str:
        flags = []
        if self.predicts_win: flags.append("WIN")
        if self.predicts_game_over: flags.append("GAME_OVER")
        if self.confident: flags.append("confident")
        else: flags.append("uncertain")
        return f"Predicted({self.state}, {', '.join(flags) or 'ok'})"


class WorldModel:
    """
    Predicts game state transitions using discovered rules.

    Confidence rules:
      - If we know the action's effect (confidence > 0.4) → confident
      - If the action is unknown or ambiguous → not confident
      - If a hazard is adjacent and the action moves toward it → predicts GAME_OVER
      - If the goal target is reached → predicts WIN
    """

    def __init__(self) -> None:
        self._prediction_cache: dict[str, PredictedState] = {}

    def reset(self) -> None:
        self._prediction_cache.clear()

    def predict(
        self,
        analysis: FrameAnalysis,
        action: str,
        action_data: dict[str, int],
        memory: Memory,
    ) -> PredictedState:
        """
        Predict the result of taking `action` from the current state.

        Returns a PredictedState. Does NOT modify the real game.
        """
        pred = PredictedState()

        # Can't predict actions we don't understand
        effect = memory.get_action_effect(action)
        if not effect:
            pred.reason = f"Unknown effect for {action}"
            return pred

        if not analysis.player_entity:
            pred.reason = "No player entity to predict for"
            return pred

        player = analysis.player_entity
        px, py = int(player.centroid[0]), int(player.centroid[1])
        pred.grid = analysis.grid.copy()

        # Parse the action's effect
        direction = self._extract_direction(effect)

        if direction and "move" in effect:
            pred.movement_direction = direction
            dx, dy = self._direction_to_delta(direction)
            nx, ny = px + dx, py + dy

            # Check bounds
            h, w = analysis.grid.shape
            if nx < 0 or nx >= w or ny < 0 or ny >= h:
                pred.state = "GAME_OVER"
                pred.predicts_game_over = True
                pred.confident = True
                pred.reason = f"Moving {direction} goes out of bounds"
                return pred

            # Check what's at the target cell
            target_cell = int(analysis.grid[ny, nx])
            target_color_name = COLOR_NAMES.get(target_cell, f"c{target_cell}")
            target_meaning = memory.get_entity_meaning(target_color_name)

            if target_meaning == "hazard":
                pred.state = "GAME_OVER"
                pred.predicts_game_over = True
                pred.confident = True
                pred.reason = f"Moving {direction} into hazard ({target_color_name})"
                return pred

            if target_meaning == "wall/obstacle":
                # Player won't move — stays in place
                pred.player_pos = (px, py)
                pred.confident = True
                pred.reason = f"Blocked by {target_color_name} ({direction})"
                return pred

            # Check for collectible at target
            if target_meaning == "collectible":
                pred.entities_removed.append(target_color_name)
                # Simulate: move player, remove collectible
                # (We don't actually modify the grid in prediction — just note it)

            # Move is safe — predict new position
            pred.player_pos = (nx, ny)
            pred.confident = True

            # Check if reaching the goal target
            best_goal = memory.get_best_goal()
            if best_goal and best_goal.confidence > 0.3:
                goal_text = best_goal.value.lower()

                # Check if target cell matches goal target
                for color_name in memory.entity_dict:
                    if color_name in goal_text and target_color_name == color_name:
                        if target_meaning not in ("hazard", "wall/obstacle"):
                            pred.state = "WIN"
                            pred.predicts_win = True
                            pred.reason = f"Reaching {target_color_name} matches goal: {best_goal.value}"
                            return pred

                # Check for meet_players goal
                if "meet" in goal_text or "together" in goal_text:
                    other_players = [e for e in analysis.entities if e.is_player and e is not player]
                    for op in other_players:
                        if int(op.centroid[0]) == nx and int(op.centroid[1]) == ny:
                            pred.state = "WIN"
                            pred.predicts_win = True
                            pred.reason = "Players would meet at same cell"
                            return pred

        elif "no_effect" in effect or "no_movement" in effect:
            pred.player_pos = (px, py)
            pred.confident = True
            pred.reason = f"{action} has no movement effect"

        # For ACTION6 (click), predict control switch if that rule exists
        if action == "ACTION6":
            if "control_switch" in memory.rules:
                # Predict which entity would become the player
                click_x = action_data.get("x", px)
                click_y = action_data.get("y", py)
                clicked_entity = None
                for e in analysis.entities:
                    if (click_x, click_y) in e.cells:
                        clicked_entity = e
                        break

                if clicked_entity and clicked_entity is not player:
                    pred.reason = f"Control would switch to {clicked_entity.color_name}"
                    pred.confident = True
            else:
                pred.reason = "ACTION6 effect unknown (no control_switch rule)"
                pred.confident = False

        # Check adjacency hazards — even if we don't move, are we next to something deadly?
        if pred.state == "NOT_FINISHED":
            for be in analysis.entities:
                if be is player or be.color == 0:
                    continue
                meaning = memory.get_entity_meaning(be.color_name)
                if meaning == "hazard":
                    if self._are_adjacent_to_cell(player, int(be.centroid[0]), int(be.centroid[1])):
                        if direction:
                            dx, dy = self._direction_to_delta(direction)
                            hx, hy = px + dx, py + dy
                            if int(be.centroid[0]) == hx and int(be.centroid[1]) == hy:
                                pred.state = "GAME_OVER"
                                pred.predicts_game_over = True
                                pred.confident = True
                                pred.reason = f"Would move into hazard {be.color_name}"
                                return pred

        return pred

    def verify_plan(
        self,
        analysis: FrameAnalysis,
        plan: list[tuple[str, dict[str, int]]],
        memory: Memory,
        max_depth: int = 8,
    ) -> dict[str, Any]:
        """
        Simulate a multi-step plan and check if it's safe and goal-achieving.

        Returns a verdict dict:
          {safe: bool, reaches_goal: bool, game_over_at_step: int|None,
           confidence: float, predictions: list[PredictedState]}
        """
        predictions: list[PredictedState] = []
        current_analysis = analysis
        safe = True
        reaches_goal = False
        game_over_step = None
        total_confidence = 1.0

        for i, (action, data) in enumerate(plan[:max_depth]):
            pred = self.predict(current_analysis, action, data, memory)
            predictions.append(pred)

            if not pred.confident:
                total_confidence *= 0.5
            else:
                total_confidence *= 0.9  # confidence decays with depth

            if pred.predicts_game_over:
                safe = False
                game_over_step = i
                break

            if pred.predicts_win:
                reaches_goal = True
                break

            # Advance the simulated state
            if pred.player_pos and current_analysis.player_entity:
                # Create a simplified next analysis with moved player
                current_analysis = self._simulate_step(current_analysis, pred, memory)

        return {
            "safe": safe,
            "reaches_goal": reaches_goal,
            "game_over_at_step": game_over_step,
            "confidence": total_confidence,
            "predictions": predictions,
            "plan_length": len(predictions),
        }

    def _simulate_step(
        self, analysis: FrameAnalysis, pred: PredictedState, memory: Memory
    ) -> FrameAnalysis:
        """Create a lightweight next FrameAnalysis from a prediction."""
        if pred.grid is None:
            return analysis

        new_grid = pred.grid.copy()

        # Move player in grid if we have a new position
        if pred.player_pos and analysis.player_entity:
            old_pos = (int(analysis.player_entity.centroid[0]), int(analysis.player_entity.centroid[1]))
            new_pos = pred.player_pos

            if 0 <= new_pos[0] < new_grid.shape[1] and 0 <= new_pos[1] < new_grid.shape[0]:
                player_color = analysis.player_entity.color
                new_grid[old_pos[1], old_pos[0]] = 0  # clear old position
                new_grid[new_pos[1], new_pos[0]] = player_color

                # Remove collectibles that were collected
                for color_name in pred.entities_removed:
                    for y in range(new_grid.shape[0]):
                        for x in range(new_grid.shape[1]):
                            if COLOR_NAMES.get(int(new_grid[y, x])) == color_name:
                                if (x, y) == new_pos:
                                    new_grid[y, x] = player_color  # player overwrites

        # Create a minimal FrameAnalysis for the simulated state
        from ..perception.perception import Perception
        temp_perc = Perception()
        simulated = temp_perc.parse_frame({
            "grid": new_grid.tolist(),
            "state": pred.state,
            "available_actions": analysis.available_actions,
            "level": analysis.level,
        })

        return simulated

    def detect_expectation_violation(
        self,
        pred: PredictedState,
        actual: FrameAnalysis,
    ) -> dict[str, Any] | None:
        """
        Compare prediction with actual outcome. Returns violation details if
        reality didn't match expectations, or None if consistent.
        """
        violations: list[str] = []

        # Check state prediction
        if pred.confident:
            if pred.predicts_game_over and actual.state != "GAME_OVER":
                violations.append(f"Predicted GAME_OVER but got {actual.state}")
            elif pred.predicts_win and actual.state != "WIN":
                violations.append(f"Predicted WIN but got {actual.state}")
            elif pred.state == "NOT_FINISHED" and actual.state == "GAME_OVER":
                violations.append(f"Predicted safe but got GAME_OVER")

        # Check movement prediction
        if pred.player_pos and pred.confident and actual.player_entity:
            actual_pos = (int(actual.player_entity.centroid[0]), int(actual.player_entity.centroid[1]))
            if pred.player_pos != actual_pos and pred.movement_direction:
                # Player didn't move as predicted — something blocked it we didn't know about
                violations.append(
                    f"Predicted player at {pred.player_pos} but at {actual_pos} "
                    f"(direction {pred.movement_direction})"
                )

        if not violations:
            return None

        return {
            "violations": violations,
            "predicted_state": pred.state,
            "actual_state": actual.state,
            "predicted_pos": pred.player_pos,
            "reason": pred.reason,
        }

    @staticmethod
    def _extract_direction(effect: str) -> str | None:
        for d in ["up", "down", "left", "right"]:
            if d in effect:
                return d
        return None

    @staticmethod
    def _direction_to_delta(direction: str) -> tuple[int, int]:
        return {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]

    @staticmethod
    def _are_adjacent_to_cell(entity: Entity, tx: int, ty: int) -> bool:
        for cx, cy in entity.cells:
            if abs(cx - tx) + abs(cy - ty) == 1:
                return True
        return False
