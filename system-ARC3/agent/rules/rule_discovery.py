"""
Layer 3 — Rule Discovery

Infers game laws from observed frame transitions. Three levels:

  Level 1 — Movement laws:    What moves? How? What blocks it?
  Level 2 — Interaction laws:  What happens on contact between entities?
  Level 3 — Win/loss laws:    What conditions cause WIN or GAME_OVER?

Also detects special mechanics:
  - Multi-player control (two entities that must meet)
  - Hidden/switchable player control (clicking to switch which entity you control)
  - Move refill entities (touching a specific entity restores move count)
  - Insufficient moves by design (level requires finding a refill entity)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..perception.perception import FrameAnalysis, Entity
from ..memory.memory import Memory


class RuleDiscovery:
    """Discovers and validates game rules from frame transitions."""

    def __init__(self) -> None:
        self._interaction_log: list[dict[str, Any]] = []
        self._win_observations: list[dict[str, Any]] = []
        self._death_observations: list[dict[str, Any]] = []
        self._multi_player_detected: bool = False
        self._hidden_player_detected: bool = False
        self._move_refill_detected: bool = False

    def reset(self) -> None:
        self._interaction_log = []
        self._win_observations = []
        self._death_observations = []
        self._multi_player_detected = False
        self._hidden_player_detected = False
        self._move_refill_detected = False

    def analyze_transition(
        self,
        action: str,
        action_data: dict[str, int],
        before: FrameAnalysis,
        after: FrameAnalysis,
        memory: Memory,
    ) -> None:
        """Analyze a single frame transition and update rules."""

        # Level 1: Movement laws
        self._infer_movement_laws(action, before, after, memory)

        # Level 2: Interaction laws
        self._infer_interaction_laws(before, after, memory)

        # Level 3: Win/loss laws
        self._infer_win_loss_laws(action, before, after, memory)

        # Special mechanics
        self._detect_multi_player(before, after, memory)
        self._detect_hidden_player(before, after, memory, action)
        self._detect_move_refill(before, after, memory)

    def _infer_movement_laws(
        self, action: str, before: FrameAnalysis, after: FrameAnalysis, memory: Memory
    ) -> None:
        """Infer what blocks movement and how entities move."""
        if not before.player_entity or not after.player_entity:
            return

        dx = after.player_entity.centroid[0] - before.player_entity.centroid[0]
        dy = after.player_entity.centroid[1] - before.player_entity.centroid[1]

        # Check if player tried to move but didn't (blocked)
        expected = memory.get_action_effect(action)
        if expected and "move" in expected and abs(dx) < 0.5 and abs(dy) < 0.5:
            # Player was blocked — find what's in the way
            direction = self._extract_direction(expected)
            if direction:
                blocking_entity = self._find_blocking_entity(before, direction)
                if blocking_entity:
                    memory.record_entity_meaning(
                        blocking_entity.color_name, "wall/obstacle",
                        f"Step {after.step}: player blocked by {blocking_entity.color_name} "
                        f"when moving {direction}",
                        confidence=0.5,
                    )
                    memory.record_rule(
                        f"blocked_by_{blocking_entity.color_name}",
                        f"{blocking_entity.color_name} blocks movement",
                        f"Player couldn't move {direction} into {blocking_entity.color_name}",
                        confidence=0.6,
                    )

        # Detect autonomous movers (entities that move without player action)
        for ae in after.entities:
            if ae.is_dynamic and not ae.is_player and ae.metadata.get("autonomous_mover"):
                memory.record_entity_meaning(
                    ae.color_name, "autonomous_mover",
                    f"Step {after.step}: {ae.color_name} moved without player action",
                    confidence=0.4,
                )

    def _infer_interaction_laws(
        self, before: FrameAnalysis, after: FrameAnalysis, memory: Memory
    ) -> None:
        """Infer what happens when entities touch."""
        if not before.player_entity:
            return

        player = before.player_entity
        # Find entities adjacent to player before the action
        for be in before.entities:
            if be is player or be.color == 0:
                continue
            if self._are_adjacent(player, be):
                # Check what happened to this entity after
                after_entity = self._find_matching_entity(be, after.entities)

                if after_entity is None:
                    # Entity disappeared after being adjacent to player
                    if after.state != "GAME_OVER":
                        memory.record_entity_meaning(
                            be.color_name, "collectible",
                            f"Step {after.step}: {be.color_name} disappeared after player contact",
                            confidence=0.5,
                        )
                        memory.record_rule(
                            f"collect_{be.color_name}",
                            f"Player collects {be.color_name} on contact",
                            f"{be.color_name} vanished after adjacency",
                            confidence=0.5,
                        )
                    else:
                        memory.record_entity_meaning(
                            be.color_name, "hazard",
                            f"Step {after.step}: GAME_OVER after touching {be.color_name}",
                            confidence=0.6,
                        )
                        memory.record_rule(
                            f"hazard_{be.color_name}",
                            f"Touching {be.color_name} causes GAME_OVER",
                            f"Player adjacent to {be.color_name} -> GAME_OVER",
                            confidence=0.7,
                        )
                elif after_entity.color != be.color:
                    # Entity changed color
                    memory.record_entity_meaning(
                        be.color_name, f"transforms_to_{after_entity.color_name}",
                        f"Step {after.step}: {be.color_name} -> {after_entity.color_name}",
                        confidence=0.4,
                    )

    def _infer_win_loss_laws(
        self, action: str, before: FrameAnalysis, after: FrameAnalysis, memory: Memory
    ) -> None:
        """Infer win and loss conditions."""
        if after.state == "WIN":
            obs = {
                "step": after.step,
                "action": action,
                "player_pos": before.player_entity.centroid if before.player_entity else None,
                "entities_before": [(e.color_name, e.centroid) for e in before.entities if e.color != 0],
                "level": after.level,
            }
            self._win_observations.append(obs)

            # Pattern: what was the state just before winning?
            if before.player_entity:
                # Check if player reached a specific entity
                for be in before.entities:
                    if be.color != 0 and be is not before.player_entity:
                        dist = self._centroid_dist(before.player_entity.centroid, be.centroid)
                        if dist < 3.0:
                            memory.record_entity_meaning(
                                be.color_name, "goal_target",
                                f"Step {after.step}: WIN occurred near {be.color_name}",
                                confidence=0.4,
                            )
                            memory.record_rule(
                                f"win_reach_{be.color_name}",
                                f"Reach {be.color_name} to win",
                                f"WIN at step {after.step} when player near {be.color_name}",
                                confidence=0.5,
                            )

            # All collectibles gone?
            collectibles = [e for e in before.entities if memory.get_entity_meaning(e.color_name) == "collectible"]
            if collectibles and after.state == "WIN":
                memory.record_rule(
                    "win_collect_all",
                    "Collect all collectibles to win",
                    f"WIN at step {after.step}, {len(collectibles)} collectibles were present before",
                    confidence=0.4,
                )

        if after.state == "GAME_OVER":
            obs = {
                "step": after.step,
                "action": action,
                "player_pos": before.player_entity.centroid if before.player_entity else None,
                "entities_before": [(e.color_name, e.centroid) for e in before.entities if e.color != 0],
            }
            self._death_observations.append(obs)

            # What killed the player?
            if before.player_entity:
                for be in before.entities:
                    if be.color != 0 and be is not before.player_entity:
                        dist = self._centroid_dist(before.player_entity.centroid, be.centroid)
                        if dist < 3.0:
                            meaning = memory.get_entity_meaning(be.color_name)
                            if meaning != "hazard":
                                memory.record_entity_meaning(
                                    be.color_name, "hazard",
                                    f"Step {after.step}: GAME_OVER near {be.color_name}",
                                    confidence=0.4,
                                )

    def _detect_multi_player(self, before: FrameAnalysis, after: FrameAnalysis, memory: Memory) -> None:
        """Detect if the game has multiple controllable players that need to meet."""
        players = [e for e in before.entities if e.is_player]
        if len(players) >= 2:
            self._multi_player_detected = True
            memory.record_rule(
                "multi_player",
                "Multiple player entities exist — may need to bring them together",
                f"Found {len(players)} player entities at step {before.step}",
                confidence=0.5,
            )

        # Check if two entities are getting closer over frames (convergence pattern)
        if len(players) == 2 and before.player_entity:
            other = [p for p in players if p is not before.player_entity][0]
            dist_before = self._centroid_dist(players[0].centroid, players[1].centroid)

            after_players = [e for e in after.entities if e.is_player]
            if len(after_players) >= 2:
                dist_after = self._centroid_dist(after_players[0].centroid, after_players[1].centroid)
                if dist_after < dist_before:
                    memory.record_rule(
                        "players_converge",
                        "Players need to meet — bring them together",
                        f"Distance decreased {dist_before:.0f} -> {dist_after:.0f}",
                        confidence=0.3,
                    )
                    memory.record_goal_hypothesis(
                        "meet_players",
                        "Bring both player entities to the same location",
                        f"Two players getting closer at step {after.step}",
                        confidence=0.3,
                    )

    def _detect_hidden_player(
        self, before: FrameAnalysis, after: FrameAnalysis, memory: Memory, action: str
    ) -> None:
        """
        Detect hidden/switchable player control:
          - ACTION6 (click) on a non-player entity changes which entity the player controls
          - A previously static entity starts moving after a click
        """
        if action != "ACTION6":
            return

        # Check if the entity that moved after the click is different from the one that moved before
        if before.player_entity and after.player_entity:
            if before.player_entity.color != after.player_entity.color:
                self._hidden_player_detected = True
                memory.record_rule(
                    "control_switch",
                    "ACTION6 switches which entity the player controls",
                    f"Player color changed {before.player_entity.color_name} -> "
                    f"{after.player_entity.color_name} after ACTION6",
                    confidence=0.6,
                )
                memory.record_entity_meaning(
                    after.player_entity.color_name, "switchable_player",
                    f"Step {after.step}: became the active player after ACTION6",
                    confidence=0.5,
                )

    def _detect_move_refill(self, before: FrameAnalysis, after: FrameAnalysis, memory: Memory) -> None:
        """
        Detect move-refill entities: touching a specific entity restores or
        increases the number of available moves/actions.

        Heuristic: if a new action becomes available after player touches an entity,
        that entity is a move-refill.
        """
        new_actions = set(after.available_actions) - set(before.available_actions)
        if new_actions and before.player_entity:
            for be in before.entities:
                if be.color != 0 and be is not before.player_entity:
                    if self._are_adjacent(before.player_entity, be):
                        self._move_refill_detected = True
                        memory.record_entity_meaning(
                            be.color_name, "move_refill",
                            f"Step {after.step}: new actions {new_actions} appeared after "
                            f"touching {be.color_name}",
                            confidence=0.5,
                        )
                        memory.record_rule(
                            "move_refill",
                            f"Touching {be.color_name} restores/refills moves",
                            f"New actions appeared: {new_actions}",
                            confidence=0.5,
                        )

    def _find_blocking_entity(self, frame: FrameAnalysis, direction: str) -> Entity | None:
        """Find entity in the direction the player tried to move."""
        if not frame.player_entity:
            return None
        px, py = frame.player_entity.centroid
        dx, dy = 0, 0
        if direction == "up":
            dy = -1
        elif direction == "down":
            dy = 1
        elif direction == "left":
            dx = -1
        elif direction == "right":
            dx = 1

        target_x, target_y = int(px + dx), int(py + dy)
        for e in frame.entities:
            if (target_x, target_y) in e.cells:
                return e
        return None

    @staticmethod
    def _extract_direction(effect: str) -> str | None:
        for d in ["up", "down", "left", "right"]:
            if d in effect:
                return d
        return None

    @staticmethod
    def _are_adjacent(a: Entity, b: Entity) -> bool:
        for cx, cy in a.cells:
            for ex, ey in b.cells:
                if abs(cx - ex) + abs(cy - ey) == 1:
                    return True
        return False

    @staticmethod
    def _find_matching_entity(before_entity: Entity, after_entities: list[Entity]) -> Entity | None:
        for ae in after_entities:
            if ae.color == before_entity.color and not ae.appeared:
                dist = RuleDiscovery._centroid_dist(before_entity.centroid, ae.centroid)
                if dist < 5.0:
                    return ae
        return None

    @staticmethod
    def _centroid_dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
