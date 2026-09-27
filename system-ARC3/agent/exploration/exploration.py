"""
Layer 2 — Integrated Exploration

KEY CHANGE from v1: No separate "exploration phase." Exploration is
woven into gameplay with a MOVE BUDGET.

Design:
  - Each level gets an exploration budget (e.g. 15 moves for level 1,
    fewer for later levels since we carry knowledge forward).
  - During budget, the agent interleaves probing actions with
    goal-directed actions — it doesn't waste all moves probing first.
  - RESET is used strategically: if we hit GAME_OVER during exploration,
    we RESET and try a different action from the same state, not blindly.
  - After budget is exhausted, the agent commits to its best hypothesis
    and plays purely goal-directed.

Exploration strategies:
  1. Probe each unknown action once (interleaved with goal moves)
  2. If stuck (no progress for N moves), try the least-tested action
  3. If all hypotheses fail, use RESET to re-explore from a clean state
  4. ACTION6 gets targeted exploration: click on different entity types
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..perception.perception import FrameAnalysis, Entity
from ..memory.memory import Memory


class Exploration:
    """Integrated exploration with move budget."""

    def __init__(self) -> None:
        self.probed_actions: set[str] = set()
        self.probe_sequence: list[str] = []
        self._steps_this_level: int = 0
        self._exploration_budget: int = 15
        self._stuck_counter: int = 0
        self._last_player_pos: tuple[float, float] | None = None
        self._reset_used_for_exploration: bool = False
        self._actions_since_progress: int = 0

    def reset(self) -> None:
        self.probed_actions = set()
        self.probe_sequence = []
        self._steps_this_level = 0
        self._stuck_counter = 0
        self._last_player_pos = None
        self._reset_used_for_exploration = False
        self._actions_since_progress = 0

    def set_budget(self, budget: int) -> None:
        self._exploration_budget = budget

    def budget_remaining(self) -> int:
        return max(0, self._exploration_budget - self._steps_this_level)

    def is_in_budget(self) -> bool:
        return self._steps_this_level < self._exploration_budget

    def tick(self) -> None:
        """Called every step to track budget consumption."""
        self._steps_this_level += 1

    def recommend_action(
        self,
        analysis: FrameAnalysis,
        memory: Memory,
        available_actions: list[str],
    ) -> tuple[str, dict[str, int]] | None:
        """
        Recommend an exploration action, integrated with gameplay.

        Returns an action only when exploration is worthwhile.
        Returns None when the planner should take over (budget exhausted or
        all actions probed).
        """
        self.tick()

        # After budget — no more exploration
        if not self.is_in_budget():
            return None

        # Skip animation intermediate frames
        if analysis.is_anim_intermediate:
            return None

        simple_actions = [a for a in available_actions if a.startswith("ACTION") and a != "ACTION6"]

        # Priority 1: Probe each unknown action once (interleaved)
        unprobed = [a for a in simple_actions if a not in self.probed_actions]
        if unprobed:
            action = unprobed[0]
            self.probed_actions.add(action)
            self.probe_sequence.append(action)
            return action, {}

        # Priority 2: ACTION6 exploration — try clicking on different entity types
        if "ACTION6" in available_actions and "ACTION6" not in self.probed_actions:
            self.probed_actions.add("ACTION6")
            self.probe_sequence.append("ACTION6")
            target = self._pick_action6_target(analysis, memory)
            return "ACTION6", target

        # Priority 3: If stuck (no progress for 5+ moves), try a RESET to re-explore
        if analysis.player_entity:
            current_pos = analysis.player_entity.centroid
            if self._last_player_pos:
                dist = ((current_pos[0] - self._last_player_pos[0]) ** 2 +
                        (current_pos[1] - self._last_player_pos[1]) ** 2) ** 0.5
                if dist < 1.0:
                    self._actions_since_progress += 1
                else:
                    self._actions_since_progress = 0
            self._last_player_pos = current_pos

            if self._actions_since_progress >= 5 and not self._reset_used_for_exploration:
                self._reset_used_for_exploration = True
                self._actions_since_progress = 0
                return "RESET", {}

        # Priority 4: Re-probe low-confidence actions (but only if budget allows)
        if self.budget_remaining() > 3:
            low_conf = [
                a for a in simple_actions
                if a in memory.action_effects and memory.action_effects[a].confidence < 0.5
            ]
            if low_conf:
                action = low_conf[0]
                self.probe_sequence.append(action)
                return action, {}

        # All explored within budget — let the planner take over
        return None

    def should_use_reset_for_experiment(
        self,
        analysis: FrameAnalysis,
        memory: Memory,
        world_model: Any,
    ) -> tuple[str, dict[str, int]] | None:
        """
        Use RESET strategically to test a hypothesis from a clean state.

        Called when all hypotheses failed or the agent is deeply stuck.
        Returns a RESET action if a focused experiment is warranted.
        """
        # Don't use RESET experiments if we've already used our budget
        if not self.is_in_budget():
            return None

        # Only RESET if we have a specific hypothesis to test
        best_goal = memory.get_best_goal()
        if not best_goal or best_goal.confidence < 0.2:
            return None

        # If we've failed this level multiple times, try a RESET + new approach
        level_fails = sum(1 for f in memory.failed_attempts if f.get("level") == memory.current_level)
        if level_fails >= 2 and not self._reset_used_for_exploration:
            self._reset_used_for_exploration = True
            return "RESET", {}

        return None

    def observe_effect(
        self,
        action: str,
        action_data: dict[str, int],
        before: FrameAnalysis,
        after: FrameAnalysis,
        memory: Memory,
    ) -> str:
        """
        Compare before/after frames to determine what an action did.
        Only called for turn-boundary frames (not animation intermediates).
        """
        effects: list[str] = []

        if before.player_entity and after.player_entity:
            dx = after.player_entity.centroid[0] - before.player_entity.centroid[0]
            dy = after.player_entity.centroid[1] - before.player_entity.centroid[1]

            if abs(dx) < 0.5 and abs(dy) < 0.5:
                effects.append("no_movement")
            else:
                direction = self._direction_from_delta(dx, dy)
                effects.append(f"move_{direction}")
                memory.record_action_effect(
                    action, f"move {direction}",
                    f"Step {after.step}: player moved ({dx:+.0f}, {dy:+.0f})",
                    confidence=0.5,
                )

        appeared = [e for e in after.entities if e.appeared and e.color != 0]
        disappeared = []
        for be in before.entities:
            if be.color == 0:
                continue
            matched = any(ae.color == be.color and not ae.appeared and
                         self._centroid_dist(be.centroid, ae.centroid) < 5
                         for ae in after.entities)
            if not matched:
                disappeared.append(be)

        for e in appeared:
            effects.append(f"appeared:{e.color_name}")

        for e in disappeared:
            effects.append(f"disappeared:{e.color_name}")

        if after.state == "WIN":
            effects.append("WIN")
            memory.record_rule(
                f"win_condition_{action}",
                f"Action {action} contributed to WIN at step {after.step}",
                f"Direct WIN observation", confidence=0.8,
            )
        elif after.state == "GAME_OVER":
            effects.append("GAME_OVER")
            memory.record_failure([action], f"{action} -> GAME_OVER", after.step)

        grid_diff = int(np.sum(before.grid != after.grid))
        if grid_diff > 0 and not effects:
            effects.append(f"grid_changed:{grid_diff}_cells")

        if not effects:
            effects.append("no_effect")
            memory.record_action_effect(
                action, "no_effect",
                f"Step {after.step}: no observable change", confidence=0.4,
            )

        return ", ".join(effects)

    def _pick_action6_target(self, analysis: FrameAnalysis, memory: Memory) -> dict[str, int]:
        """Pick the most informative click target for ACTION6 exploration."""
        if not analysis.player_entity:
            return {"x": analysis.width // 2, "y": analysis.height // 2}

        # Click on the nearest non-player entity we haven't clicked before
        player = analysis.player_entity
        candidates = [e for e in analysis.entities if e is not player and e.color != 0]
        if candidates:
            nearest = min(candidates, key=lambda e: self._centroid_dist(player.centroid, e.centroid))
            return {"x": int(nearest.centroid[0]), "y": int(nearest.centroid[1])}

        return {"x": int(player.centroid[0]), "y": int(player.centroid[1])}

    @staticmethod
    def _direction_from_delta(dx: float, dy: float) -> str:
        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    @staticmethod
    def _centroid_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
