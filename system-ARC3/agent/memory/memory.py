"""
Layer 4 — Hierarchical Memory

Three tiers:
  - Short-term:  last N frame analyses (rolling window)
  - Medium-term:  per-game session knowledge — discovered rules, action effects,
                   goal hypotheses, failed attempts, entity dictionary
  - Long-term:   cross-game patterns and strategies that transfer

The medium-term tier is the system's "game dictionary" — a living knowledge base
that accumulates proven facts about each game across levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import json


@dataclass
class MemoryEntry:
    """A single piece of knowledge with provenance (proof/evidence)."""
    key: str
    value: Any
    confidence: float = 0.5              # 0.0 to 1.0
    evidence: list[str] = field(default_factory=list)   # human-readable proofs
    source: str = "observation"          # observation, inference, model, exploration
    step: int = 0                        # when it was learned
    verified: bool = False               # confirmed by multiple observations

    def add_evidence(self, ev: str) -> None:
        self.evidence.append(ev)
        if len(self.evidence) >= 2:
            self.verified = True
            self.confidence = min(1.0, self.confidence + 0.15)

    def __repr__(self) -> str:
        status = "verified" if self.verified else f"conf={self.confidence:.0%}"
        return f"[{self.key}] {self.value} ({status}, {len(self.evidence)} proofs)"


class Memory:
    """
    Hierarchical memory store. One instance per game session.

    Medium-term knowledge is organized into sections:
      - action_effects:  what each ACTION does (direction, interaction type)
      - entity_dict:      per-color/entity meaning (player, wall, collectible, hazard, goal, move_refill)
      - rules:            discovered game laws (movement constraints, win/loss conditions)
      - goal_hypotheses:  what the agent thinks the goal is
      - failed_attempts:  sequences that led to GAME_OVER
      - successful_paths: sequences that led to WIN
    """

    def __init__(self, game_id: str = "unknown") -> None:
        self.game_id = game_id
        self.current_level: int = 0

        # Short-term: rolling window of recent frame analyses
        self.short_term: list[dict[str, Any]] = []
        self._short_term_max = 30

        # Medium-term: the game dictionary
        self.action_effects: dict[str, MemoryEntry] = {}
        self.entity_dict: dict[str, MemoryEntry] = {}        # keyed by color_name
        self.rules: dict[str, MemoryEntry] = {}
        self.goal_hypotheses: dict[str, MemoryEntry] = {}    # keyed by hypothesis ID
        self.failed_attempts: list[dict[str, Any]] = []
        self.successful_paths: list[dict[str, Any]] = []

        # Per-level tracking
        self._level_history: dict[int, dict[str, Any]] = {}

        # Long-term: cross-game patterns
        self.long_term: dict[str, MemoryEntry] = {}

    def push_frame(self, analysis_summary: dict[str, Any]) -> None:
        """Add a frame to short-term memory."""
        self.short_term.append(analysis_summary)
        if len(self.short_term) > self._short_term_max:
            self.short_term.pop(0)

    def record_action_effect(self, action: str, effect: str, evidence: str, confidence: float = 0.5) -> None:
        """Record or update what an action does."""
        if action in self.action_effects:
            entry = self.action_effects[action]
            if effect != entry.value:
                # Contradiction — lower confidence and note it
                entry.confidence = max(0.0, entry.confidence - 0.2)
                entry.add_evidence(f"Contradiction at step {evidence}: was '{entry.value}', now '{effect}'")
            else:
                entry.add_evidence(evidence)
        else:
            self.action_effects[action] = MemoryEntry(
                key=action,
                value=effect,
                confidence=confidence,
                evidence=[evidence],
                source="exploration",
            )

    def record_entity_meaning(self, color_name: str, meaning: str, evidence: str, confidence: float = 0.5) -> None:
        """Record what a color/entity type means in this game."""
        key = color_name
        if key in self.entity_dict:
            entry = self.entity_dict[key]
            if meaning != entry.value:
                entry.confidence = max(0.0, entry.confidence - 0.15)
                entry.add_evidence(f"Conflict: was '{entry.value}', now '{meaning}' — {evidence}")
            else:
                entry.add_evidence(evidence)
        else:
            self.entity_dict[key] = MemoryEntry(
                key=key,
                value=meaning,
                confidence=confidence,
                evidence=[evidence],
                source="observation",
            )

    def record_rule(self, rule_name: str, rule: str, evidence: str, confidence: float = 0.5) -> None:
        """Record a discovered game rule."""
        if rule_name in self.rules:
            self.rules[rule_name].add_evidence(evidence)
        else:
            self.rules[rule_name] = MemoryEntry(
                key=rule_name,
                value=rule,
                confidence=confidence,
                evidence=[evidence],
                source="inference",
            )

    def record_goal_hypothesis(self, hid: str, hypothesis: str, evidence: str, confidence: float = 0.3) -> None:
        """Record or update a goal hypothesis."""
        if hid in self.goal_hypotheses:
            self.goal_hypotheses[hid].add_evidence(evidence)
        else:
            self.goal_hypotheses[hid] = MemoryEntry(
                key=hid,
                value=hypothesis,
                confidence=confidence,
                evidence=[evidence],
                source="inference",
            )

    def adjust_goal_confidence(self, hid: str, delta: float, reason: str) -> None:
        if hid in self.goal_hypotheses:
            entry = self.goal_hypotheses[hid]
            entry.confidence = max(0.0, min(1.0, entry.confidence + delta))
            entry.add_evidence(f"Confidence {'+' if delta > 0 else ''}{delta:.0%}: {reason}")

    def record_failure(self, action_sequence: list[str], reason: str, step: int) -> None:
        self.failed_attempts.append({
            "actions": action_sequence,
            "reason": reason,
            "step": step,
            "level": self.current_level,
        })

    def record_success(self, action_sequence: list[str], level: int, step: int) -> None:
        self.successful_paths.append({
            "actions": action_sequence,
            "level": level,
            "step": step,
        })

    def set_level(self, level: int) -> None:
        if self.current_level != level:
            # Archive previous level
            self._level_history[self.current_level] = {
                "action_effects": {k: {"value": v.value, "confidence": v.confidence} for k, v in self.action_effects.items()},
                "rules": {k: v.value for k, v in self.rules.items()},
                "goal_hypotheses": {k: v.value for k, v in self.goal_hypotheses.items()},
            }
            self.current_level = level

    def get_best_goal(self) -> MemoryEntry | None:
        if not self.goal_hypotheses:
            return None
        return max(self.goal_hypotheses.values(), key=lambda e: e.confidence)

    def get_action_effect(self, action: str) -> str | None:
        entry = self.action_effects.get(action)
        return entry.value if entry and entry.confidence > 0.3 else None

    def get_entity_meaning(self, color_name: str) -> str | None:
        entry = self.entity_dict.get(color_name)
        return entry.value if entry and entry.confidence > 0.3 else None

    def build_context_summary(self) -> str:
        """Build a text summary of everything known — for the model's reasoning prompt."""
        lines = [f"=== Game Knowledge: {self.game_id} (Level {self.current_level}) ==="]

        if self.action_effects:
            lines.append("\n-- Action Effects --")
            for action, entry in sorted(self.action_effects.items()):
                lines.append(f"  {action}: {entry.value} (conf={entry.confidence:.0%}, proofs={len(entry.evidence)})")

        if self.entity_dict:
            lines.append("\n-- Entity Dictionary --")
            for color, entry in sorted(self.entity_dict.items()):
                lines.append(f"  {color}: {entry.value} (conf={entry.confidence:.0%}, proofs={len(entry.evidence)})")

        if self.rules:
            lines.append("\n-- Discovered Rules --")
            for name, entry in sorted(self.rules.items()):
                lines.append(f"  {name}: {entry.value} (conf={entry.confidence:.0%})")

        if self.goal_hypotheses:
            lines.append("\n-- Goal Hypotheses --")
            for hid, entry in sorted(self.goal_hypotheses.items(), key=lambda x: -x[1].confidence):
                lines.append(f"  [{hid}] {entry.value} (conf={entry.confidence:.0%})")

        if self.failed_attempts:
            lines.append(f"\n-- Failed Attempts: {len(self.failed_attempts)} --")
            for fa in self.failed_attempts[-3:]:
                lines.append(f"  Level {fa['level']}: {fa['reason']} (actions: {' '.join(fa['actions'][:10])})")

        if self.successful_paths:
            lines.append(f"\n-- Successful Paths: {len(self.successful_paths)} --")
            for sp in self.successful_paths[-2:]:
                lines.append(f"  Level {sp['level']}: {' '.join(sp['actions'][:15])}")

        return "\n".join(lines)

    def export_state(self) -> str:
        """Export full memory state as JSON (for persistence/debugging)."""
        return json.dumps({
            "game_id": self.game_id,
            "current_level": self.current_level,
            "action_effects": {k: {"value": v.value, "confidence": v.confidence, "evidence": v.evidence} for k, v in self.action_effects.items()},
            "entity_dict": {k: {"value": v.value, "confidence": v.confidence, "evidence": v.evidence} for k, v in self.entity_dict.items()},
            "rules": {k: {"value": v.value, "confidence": v.confidence, "evidence": v.evidence} for k, v in self.rules.items()},
            "goal_hypotheses": {k: {"value": v.value, "confidence": v.confidence, "evidence": v.evidence} for k, v in self.goal_hypotheses.items()},
            "failed_attempts": self.failed_attempts,
            "successful_paths": self.successful_paths,
        }, indent=2, ensure_ascii=False)

    def import_state(self, json_str: str) -> None:
        data = json.loads(json_str)
        self.game_id = data.get("game_id", self.game_id)
        self.current_level = data.get("current_level", 0)
        for section, attr in [("action_effects", "action_effects"), ("entity_dict", "entity_dict"),
                              ("rules", "rules"), ("goal_hypotheses", "goal_hypotheses")]:
            for k, v in data.get(section, {}).items():
                self.__dict__[attr][k] = MemoryEntry(
                    key=k, value=v["value"], confidence=v.get("confidence", 0.5),
                    evidence=v.get("evidence", []), source="imported",
                )
        self.failed_attempts = data.get("failed_attempts", [])
        self.successful_paths = data.get("successful_paths", [])
