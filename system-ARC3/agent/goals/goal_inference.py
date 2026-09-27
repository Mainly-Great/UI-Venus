"""
Layer 5 — Goal Inference (v2)

Generates, ranks, and tests hypotheses about what the game wants the agent
to achieve. Since ARC-AGI-3 provides no explicit goal, the system must
infer it from observation.

KEY CHANGE from v1: The model is called at exactly 3 decisive moments:

  (a) Game start: Read the first frame and propose a game pattern + goal hypothesis.
      This gives the symbolic reasoning a starting direction.
  (b) All hypotheses failed: When every existing hypothesis has been tested
      and failed (confidence < 0.1), ask the model for an OUT-OF-TEMPLATE
      hypothesis — something the heuristic templates can't generate.
  (c) Expectation violation: When the world model's prediction contradicts
      reality, ask the model to explain the discrepancy and revise the goal.

Outside these 3 points, all reasoning is purely symbolic/programmatic.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..perception.perception import FrameAnalysis
from ..memory.memory import Memory
from ..model.model_client import ModelClient


class GoalInference:
    """Generates and manages goal hypotheses with strategic model use."""

    # Trigger (a): first frame of a game
    TRIGGER_GAME_START = "game_start"
    # Trigger (b): all hypotheses exhausted
    TRIGGER_HYPOTHESES_EXHAUSTED = "hypotheses_exhausted"
    # Trigger (c): prediction contradicted by reality
    TRIGGER_EXPECTATION_VIOLATION = "expectation_violation"

    def __init__(self, model: ModelClient | None = None) -> None:
        self.model = model
        self._hypothesis_counter: int = 0
        self._tested: set[str] = set()
        self._model_called_this_level: bool = False
        self._all_hypotheses_failed: bool = False
        self._first_frame_processed: bool = False

    def reset(self) -> None:
        self._hypothesis_counter = 0
        self._tested = set()
        self._model_called_this_level = False
        self._all_hypotheses_failed = False
        self._first_frame_processed = False

    def generate_hypotheses(
        self, analysis: FrameAnalysis, memory: Memory,
        trigger: str | None = None,
        violation_info: dict[str, Any] | None = None,
    ) -> None:
        """
        Generate goal hypotheses.

        Heuristic hypotheses are always generated. Model hypotheses are only
        generated when a trigger fires (game start, exhaustion, or violation).
        """
        # Always run heuristic hypothesis generation
        self._heuristic_hypotheses(analysis, memory)

        # Check if we should call the model
        should_call_model = False
        if trigger == self.TRIGGER_GAME_START and not self._first_frame_processed:
            should_call_model = True
            self._first_frame_processed = True
        elif trigger == self.TRIGGER_HYPOTHESES_EXHAUSTED and not self._all_hypotheses_failed:
            # All hypotheses have very low confidence
            should_call_model = True
            self._all_hypotheses_failed = True
        elif trigger == self.TRIGGER_EXPECTATION_VIOLATION:
            should_call_model = True

        if should_call_model and self.model and self.model.is_available:
            if trigger == self.TRIGGER_GAME_START:
                self._model_game_start(analysis, memory)
            elif trigger == self.TRIGGER_HYPOTHESES_EXHAUSTED:
                self._model_out_of_template(analysis, memory)
            elif trigger == self.TRIGGER_EXPECTATION_VIOLATION:
                self._model_explain_violation(analysis, memory, violation_info)

    def check_hypotheses_exhausted(self, memory: Memory) -> bool:
        """Check if all existing hypotheses have been tested and failed."""
        if not memory.goal_hypotheses:
            return True
        active = [h for h in memory.goal_hypotheses.values() if h.confidence > 0.1]
        return len(active) == 0

    def _heuristic_hypotheses(self, analysis: FrameAnalysis, memory: Memory) -> None:
        """Generate hypotheses from observable patterns."""
        if not analysis.player_entity:
            return

        player = analysis.player_entity
        non_bg = [e for e in analysis.entities if e.color != 0 and e is not player]

        unique_colors = set()
        for e in non_bg:
            meaning = memory.get_entity_meaning(e.color_name)
            if meaning in ("hazard", "wall/obstacle", "void"):
                continue
            unique_colors.add(e.color_name)

        for color_name in unique_colors:
            hid = f"reach_{color_name}"
            if hid not in memory.goal_hypotheses:
                entities = [e for e in non_bg if e.color_name == color_name]
                if entities:
                    nearest = min(entities, key=lambda e: self._dist(player.centroid, e.centroid))
                    dist = self._dist(player.centroid, nearest.centroid)
                    confidence = max(0.2, 0.5 - dist * 0.01)

                    memory.record_goal_hypothesis(
                        hid,
                        f"Reach the {color_name} entity at ({nearest.centroid[0]:.0f}, {nearest.centroid[1]:.0f})",
                        f"Step {analysis.step}: {color_name} entity present, nearest is {dist:.0f} cells away",
                        confidence=confidence,
                    )

        for color_name in unique_colors:
            count = sum(1 for e in non_bg if e.color_name == color_name)
            if count >= 2:
                hid = f"collect_all_{color_name}"
                if hid not in memory.goal_hypotheses:
                    memory.record_goal_hypothesis(
                        hid,
                        f"Collect all {count} {color_name} entities",
                        f"Step {analysis.step}: found {count} {color_name} entities",
                        confidence=0.2,
                    )

        players = [e for e in analysis.entities if e.is_player]
        if len(players) >= 2:
            hid = "meet_players"
            if hid not in memory.goal_hypotheses:
                memory.record_goal_hypothesis(
                    hid,
                    "Bring both player entities to the same location",
                    f"Step {analysis.step}: {len(players)} players detected",
                    confidence=0.3,
                )

        hazards = [e for e in non_bg if memory.get_entity_meaning(e.color_name) == "hazard"]
        if hazards:
            hid = "avoid_hazards_reach_goal"
            if hid not in memory.goal_hypotheses:
                memory.record_goal_hypothesis(
                    hid,
                    "Navigate past hazards to reach a target",
                    f"Step {analysis.step}: {len(hazards)} hazard entities present",
                    confidence=0.2,
                )

        if any(memory.get_entity_meaning(e.color_name) == "move_refill" for e in non_bg):
            hid = "refill_then_reach"
            if hid not in memory.goal_hypotheses:
                memory.record_goal_hypothesis(
                    hid,
                    "Touch move-refill entity, then reach the target",
                    f"Step {analysis.step}: move_refill entity present",
                    confidence=0.15,
                )

    # ─── Model Intervention (a): Game Start ──────────────────────────────────

    def _model_game_start(self, analysis: FrameAnalysis, memory: Memory) -> None:
        """Ask the model to read the first frame and propose a game pattern."""
        context = self._build_initial_context(analysis)
        grid_text = self._grid_compact(analysis.grid)

        question = (
            f"This is the first frame of a new ARC-AGI-3 game.\n"
            f"Grid ({analysis.width}x{analysis.height}):\n{grid_text}\n\n"
            f"Entities found: {len(analysis.entities)}\n"
            f"Colors present: {sorted(analysis.colors_present)}\n\n"
            "What type of game do you think this is? What is the likely goal?\n"
            "Propose up to 3 goal hypotheses with confidence (0-100%). Format:\n"
            "H1: <description> (XX%)\nH2: <description> (XX%)\nH3: <description> (XX%)"
        )

        response = self.model.reason(context, question)
        self._parse_model_hypotheses(response, analysis.step, memory)

    # ─── Model Intervention (b): All Hypotheses Exhausted ─────────────────────

    def _model_out_of_template(self, analysis: FrameAnalysis, memory: Memory) -> None:
        """Ask the model for a hypothesis outside our heuristic templates."""
        context = memory.build_context_summary()
        grid_text = self._grid_compact(analysis.grid)

        question = (
            f"Current grid:\n{grid_text}\n\n"
            "All previous goal hypotheses have been tested and failed.\n"
            "Propose a NEW goal hypothesis that is DIFFERENT from:\n"
            "- Reach a specific color entity\n"
            "- Collect all entities of a color\n"
            "- Bring players together\n"
            "- Avoid hazards and reach target\n\n"
            "Think outside these templates. What else could the goal be?\n"
            "Format: H1: <description> (XX%)"
        )

        response = self.model.reason(context, question)
        self._parse_model_hypotheses(response, analysis.step, memory)

    # ─── Model Intervention (c): Expectation Violation ────────────────────────

    def _model_explain_violation(
        self, analysis: FrameAnalysis, memory: Memory, violation_info: dict[str, Any] | None
    ) -> None:
        """Ask the model to explain why prediction didn't match reality."""
        context = memory.build_context_summary()
        grid_text = self._grid_compact(analysis.grid)

        violations = violation_info.get("violations", []) if violation_info else ["Unknown violation"]
        violations_text = "\n".join(f"  - {v}" for v in violations)

        question = (
            f"Current grid:\n{grid_text}\n\n"
            f"The world model made a prediction that was WRONG:\n{violations_text}\n\n"
            "What rule or mechanic might explain this discrepancy?\n"
            "Propose a revised goal hypothesis based on this new evidence.\n"
            "Format: H1: <description> (XX%)"
        )

        response = self.model.reason(context, question)
        self._parse_model_hypotheses(response, analysis.step, memory)

        # Also record the violation as evidence
        if violation_info:
            memory.record_rule(
                "expectation_violation",
                f"Prediction failed: {violations[0] if violations else 'unknown'}",
                f"Step {analysis.step}: {violations_text}",
                confidence=0.3,
            )

    def _build_initial_context(self, analysis: FrameAnalysis) -> str:
        """Build context for the first-frame model call."""
        lines = [
            f"=== New Game: First Frame Analysis ===",
            f"Grid size: {analysis.width}x{analysis.height}",
            f"Colors present: {sorted(analysis.colors_present)}",
            f"Entities: {len(analysis.entities)}",
        ]
        for e in analysis.entities:
            lines.append(f"  {e}")
        if analysis.available_actions:
            lines.append(f"Available actions: {analysis.available_actions}")
        return "\n".join(lines)

    def _parse_model_hypotheses(self, response: str, step: int, memory: Memory) -> None:
        """Parse model response into goal hypotheses."""
        for line in response.split("\n"):
            line = line.strip()
            if not line or not line.startswith("H"):
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            desc = parts[1].strip()
            confidence = 0.2

            pct_match = re.search(r'\((\d+)%\)', desc)
            if pct_match:
                confidence = int(pct_match.group(1)) / 100.0
                desc = re.sub(r'\(\d+%\)', '', desc).strip()

            hid = f"model_h{self._hypothesis_counter}"
            self._hypothesis_counter += 1

            memory.record_goal_hypothesis(
                hid, desc,
                f"Model inference at step {step}: {response[:100]}",
                confidence=confidence,
            )

    def update_from_outcome(
        self,
        action: str,
        before: FrameAnalysis,
        after: FrameAnalysis,
        memory: Memory,
    ) -> None:
        """Update goal hypothesis confidences based on action outcomes."""
        best_goal = memory.get_best_goal()
        if not best_goal:
            return

        if after.state == "WIN":
            memory.adjust_goal_confidence(best_goal.key, +0.5, f"WIN achieved! Goal was: {best_goal.value}")
            return

        if after.state == "GAME_OVER":
            memory.adjust_goal_confidence(best_goal.key, -0.1, f"GAME_OVER after {action}")
            return

        if before.player_entity and after.player_entity:
            for hid, entry in memory.goal_hypotheses.items():
                if hid.startswith("reach_"):
                    color_name = hid.replace("reach_", "")
                    target = next(
                        (e for e in after.entities if e.color_name == color_name), None
                    )
                    if target:
                        dist_before = self._dist(before.player_entity.centroid, target.centroid)
                        dist_after = self._dist(after.player_entity.centroid, target.centroid)
                        if dist_after < dist_before:
                            memory.adjust_goal_confidence(hid, +0.05, "Player moved closer to target")
                        elif dist_after > dist_before:
                            memory.adjust_goal_confidence(hid, -0.03, "Player moved away from target")

                if hid.startswith("collect_all_"):
                    color_name = hid.replace("collect_all_", "")
                    count_before = sum(1 for e in before.entities if e.color_name == color_name)
                    count_after = sum(1 for e in after.entities if e.color_name == color_name)
                    if count_after < count_before:
                        memory.adjust_goal_confidence(hid, +0.1, f"Collected a {color_name} ({count_before}->{count_after})")

                if hid == "meet_players":
                    players_before = [e for e in before.entities if e.is_player]
                    players_after = [e for e in after.entities if e.is_player]
                    if len(players_before) >= 2 and len(players_after) >= 2:
                        dist_before = self._dist(players_before[0].centroid, players_before[1].centroid)
                        dist_after = self._dist(players_after[0].centroid, players_after[1].centroid)
                        if dist_after < dist_before:
                            memory.adjust_goal_confidence(hid, +0.08, "Players got closer")

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
