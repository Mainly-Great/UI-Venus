"""
JSONL Session Logger

Records every step of every game session as a structured JSONL line.
This is the ground truth for post-session analysis — it captures what
the agent saw, what it knew, what it predicted, what it did, and what
actually happened.

Output format: one JSON object per line, each representing a single step.

Usage in agent:
    logger = JsonlLogger("logs/session_001.jsonl")
    logger.log_step(step=1, game="ls20", level=0, action="ACTION1",
                     action_data={}, state_before="NOT_FINISHED",
                     state_after="NOT_FINISHED",
                     prediction={"confident": True, "predicts_win": False},
                     memory_snapshot={...})
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class JsonlLogger:
    """Appends structured step records to a JSONL file."""

    def __init__(self, path: str = "logs/session.jsonl") -> None:
        self.path = path
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._meta_logged = False

    def log_session_start(self, game_id: str, model_name: str = "none") -> None:
        """Log session metadata at the start."""
        entry = {
            "ts": datetime.now().isoformat(),
            "session_id": self._session_id,
            "type": "session_start",
            "game_id": game_id,
            "model_name": model_name,
        }
        self._write(entry)
        self._meta_logged = True

    def log_step(
        self,
        step: int,
        game_id: str,
        level: int,
        action: str,
        action_data: dict[str, int],
        state_before: str,
        state_after: str,
        prediction: dict[str, Any] | None = None,
        is_anim_intermediate: bool = False,
        player_pos: tuple[float, float] | None = None,
        entities_count: int = 0,
        colors_present: list[int] | None = None,
        available_actions: list[str] | None = None,
        best_goal: str | None = None,
        best_goal_confidence: float = 0.0,
        memory_snapshot: dict[str, Any] | None = None,
        violation: dict[str, Any] | None = None,
        trigger: str | None = None,
    ) -> None:
        """Log a single step."""
        entry = {
            "ts": datetime.now().isoformat(),
            "session_id": self._session_id,
            "type": "step",
            "step": step,
            "game_id": game_id,
            "level": level,
            "action": action,
            "action_data": action_data,
            "state_before": state_before,
            "state_after": state_after,
            "prediction": prediction,
            "is_anim_intermediate": is_anim_intermediate,
            "player_pos": list(player_pos) if player_pos else None,
            "entities_count": entities_count,
            "colors_present": colors_present,
            "available_actions": available_actions,
            "best_goal": best_goal,
            "best_goal_confidence": round(best_goal_confidence, 3),
            "violation": violation,
            "trigger": trigger,
        }

        # Only include memory snapshot if provided (it can be large)
        if memory_snapshot:
            entry["memory"] = {
                "action_effects": {k: {"value": v.value, "conf": round(v.confidence, 2)}
                                   for k, v in memory_snapshot.get("action_effects", {}).items()},
                "entity_dict": {k: {"value": v.value, "conf": round(v.confidence, 2)}
                                for k, v in memory_snapshot.get("entity_dict", {}).items()},
                "rules_count": len(memory_snapshot.get("rules", {})),
                "goals": {k: {"value": v.value, "conf": round(v.confidence, 2)}
                          for k, v in memory_snapshot.get("goal_hypotheses", {}).items()},
                "failed_attempts": memory_snapshot.get("failed_count", 0),
                "successful_paths": memory_snapshot.get("success_count", 0),
            }

        self._write(entry)

    def log_session_end(self, game_id: str, wins: int, losses: int, total_steps: int) -> None:
        """Log session end summary."""
        entry = {
            "ts": datetime.now().isoformat(),
            "session_id": self._session_id,
            "type": "session_end",
            "game_id": game_id,
            "wins": wins,
            "losses": losses,
            "total_steps": total_steps,
        }
        self._write(entry)

    def _write(self, entry: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def close(self) -> None:
        """Flush and close (file is already flushed per write)."""
        pass
