"""
ARC-AGI-3 Sensory-Aware Agent (v2)
===================================

Integrates 8 layers of the sensory-aware reasoning system:

  Layer 1: Perception     — grid → entities, movement, relationships
                             + animation-resistant diff (skips intermediate frames)
  Layer 2: Exploration    — integrated exploration with move budget (no separate phase)
  Layer 3: Rule Discovery — movement, interaction, win/loss laws
  Layer 4: Memory         — hierarchical game dictionary with proofs
  Layer 5: Goal Inference — hypothesis generation; model at 3 decisive points only
  Layer 6: Model Client   — swappable VLM (UI-Venus-2-9B or any OpenAI-compatible)
  Layer 7: Planner        — pathfinding + action selection with world-model verification
  Layer 8: World Model    — internal game simulator for plan verification before execution

The agent implements the two required methods:
  - is_done(frames, latest_frame) -> bool
  - choose_action(frames, latest_frame) -> GameAction
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any

import numpy as np

# --- Layer imports ---
from agent.perception import Perception
from agent.exploration import Exploration
from agent.rules import RuleDiscovery
from agent.memory import Memory
from agent.goals import GoalInference
from agent.planner import Planner
from agent.model import ModelClient
from agent.world_model import WorldModel
from agent.logging import JsonlLogger

logger = logging.getLogger("arc3_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─── Helpers: extract grid from ARC-AGI-3 frame ──────────────────────────────

def _extract_grid(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        grid_data = frame.get("grid") or frame.get("frame") or frame
        if isinstance(grid_data, dict):
            grid_data = grid_data.get("grid", grid_data)
        if isinstance(grid_data, (list, np.ndarray)):
            return np.array(grid_data, dtype=np.int8)
        obs = frame.get("observation", {})
        if isinstance(obs, dict) and "grid" in obs:
            return np.array(obs["grid"], dtype=np.int8)

    for attr in ("grid", "frame", "observation", "state"):
        val = getattr(frame, attr, None)
        if val is not None:
            if isinstance(val, np.ndarray):
                return val.astype(np.int8)
            if isinstance(val, (list, tuple)):
                return np.array(val, dtype=np.int8)
            if isinstance(val, dict) and "grid" in val:
                return np.array(val["grid"], dtype=np.int8)

    if isinstance(frame, (list, np.ndarray)):
        return np.array(frame, dtype=np.int8)

    logger.warning(f"Could not extract grid from frame type: {type(frame)}")
    return np.zeros((1, 1), dtype=np.int8)


def _extract_state(frame: Any) -> str:
    if isinstance(frame, dict):
        state = frame.get("state", frame.get("game_state", "NOT_FINISHED"))
        return str(state)
    for attr in ("state", "game_state", "status"):
        val = getattr(frame, attr, None)
        if val is not None:
            if hasattr(val, "name"):
                return val.name
            return str(val)
    return "NOT_FINISHED"


def _extract_actions(frame: Any) -> list[str]:
    if isinstance(frame, dict):
        actions = frame.get("available_actions", frame.get("actions", []))
        return [str(a) for a in actions] if actions else []
    for attr in ("available_actions", "actions", "action_space"):
        val = getattr(frame, attr, None)
        if val is not None:
            if isinstance(val, (list, tuple)):
                return [str(a) for a in val]
            if hasattr(val, "__iter__"):
                try:
                    return [str(a) for a in val]
                except Exception:
                    pass
    return []


def _extract_level(frame: Any) -> int:
    if isinstance(frame, dict):
        return int(frame.get("level", frame.get("current_level", 0)))
    for attr in ("level", "current_level"):
        val = getattr(frame, attr, None)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return 0


def _frame_to_dict(frame: Any) -> dict[str, Any]:
    grid = _extract_grid(frame)
    return {
        "grid": grid.tolist(),
        "state": _extract_state(frame),
        "available_actions": _extract_actions(frame),
        "level": _extract_level(frame),
    }


def _parse_game_action(action_str: str, data: dict[str, int], available_actions: list[str]):
    try:
        from arcengine import GameAction
        mapping = {
            "RESET": GameAction.RESET,
            "ACTION1": GameAction.ACTION1,
            "ACTION2": GameAction.ACTION2,
            "ACTION3": GameAction.ACTION3,
            "ACTION4": GameAction.ACTION4,
            "ACTION5": GameAction.ACTION5,
            "ACTION6": GameAction.ACTION6,
            "ACTION7": GameAction.ACTION7,
        }
        ga = mapping.get(action_str.upper())
        if ga is not None:
            return ga, data
    except ImportError:
        pass
    return action_str, data


# ─── The Agent ───────────────────────────────────────────────────────────────

class MyAgent:
    """
    Sensory-aware ARC-AGI-3 agent (v2).

    Lifecycle per game:
      1. RESET -> initial frame
      2. Perception analyzes frame (animation-resistant diff)
      3. Goal Inference fires model at game start (trigger a)
      4. Exploration probes actions within move budget (integrated, not separate)
      5. RuleDiscovery infers laws from turn-boundary transitions only
      6. World Model predicts outcome before each action
      7. Planner verifies plan safety, then selects action
      8. On expectation violation: model fires (trigger c)
      9. On all hypotheses exhausted: model fires (trigger b)
     10. Repeat until WIN or done
    """

    def __init__(self, game_id: str = "unknown", **kwargs: Any) -> None:
        self.game_id = game_id

        self.model = ModelClient()
        self.perception = Perception()
        self.exploration = Exploration()
        self.rule_discovery = RuleDiscovery()
        self.memory = Memory(game_id=game_id)
        self.goal_inference = GoalInference(model=self.model)
        self.planner = Planner(model=self.model)
        self.world_model = WorldModel()

        # JSONL logger
        log_path = kwargs.get("log_path", os.environ.get("ARC3_LOG_PATH", "logs/session.jsonl"))
        self.logger_session = JsonlLogger(log_path)
        self.logger_session.log_session_start(game_id, self.model.model_name if hasattr(self.model, "model_name") else "none")

        # State tracking
        self._last_analysis = None
        self._last_action: str | None = None
        self._last_action_data: dict[str, int] = {}
        self._last_prediction = None
        self._step: int = 0
        self._wins: int = 0
        self._losses: int = 0
        self._max_steps: int = 200
        self._reset_count: int = 0
        self._max_resets: int = 5
        self._game_start_processed: bool = False
        self._hypotheses_exhausted_checked: bool = False

    def reset(self) -> None:
        """Reset agent state for a new game or level."""
        self.perception.reset()
        self.exploration.reset()
        self.rule_discovery.reset()
        self.goal_inference.reset()
        self.planner.reset()
        self.world_model.reset()
        self._last_analysis = None
        self._last_action = None
        self._last_action_data = {}
        self._last_prediction = None
        self._step = 0
        self._game_start_processed = False
        self._hypotheses_exhausted_checked = False

    # ─── Required interface methods ──────────────────────────────────────────

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        state = _extract_state(latest_frame)

        if state == "WIN":
            self._wins += 1
            return False

        if state == "GAME_OVER":
            self._losses += 1
            self._reset_count += 1
            if self._reset_count >= self._max_resets:
                logger.info(f"[{self.game_id}] Giving up after {self._reset_count} resets")
                self.logger_session.log_session_end(self.game_id, self._wins, self._losses, self._step)
                return True
            return False

        if self._step >= self._max_steps:
            logger.info(f"[{self.game_id}] Hit max steps ({self._max_steps})")
            self.logger_session.log_session_end(self.game_id, self._wins, self._losses, self._step)
            return True

        return False

    def choose_action(self, frames: list[Any], latest_frame: Any) -> Any:
        self._step += 1

        frame_dict = _frame_to_dict(latest_frame)
        analysis = self.perception.parse_frame(frame_dict)

        # Game over → reset
        if analysis.state == "GAME_OVER":
            self._log_step(analysis, "RESET", {}, "GAME_OVER", "GAME_OVER")
            self.reset()
            self._last_action = "RESET"
            return _parse_game_action("RESET", {}, analysis.available_actions)[0]

        # Win → record success, reset for next level
        if analysis.state == "WIN":
            self.memory.record_success(
                self.planner._action_history[:], analysis.level, self._step
            )
            self._log_step(analysis, "RESET", {}, "WIN", "WIN")
            self.reset()
            self._last_action = "RESET"
            return _parse_game_action("RESET", {}, analysis.available_actions)[0]

        # Track level changes
        if analysis.level != self.memory.current_level:
            self.memory.set_level(analysis.level)
            self.exploration.set_budget(self._exploration_budget_for_level(analysis.level))

        # Store frame in short-term memory
        self.memory.push_frame({
            "step": analysis.step,
            "state": analysis.state,
            "level": analysis.level,
            "player_pos": analysis.player_entity.centroid if analysis.player_entity else None,
            "colors": sorted(analysis.colors_present),
            "is_anim_intermediate": analysis.is_anim_intermediate,
        })

        # ─── Animation-resistant: skip rule discovery on intermediate frames ──
        if analysis.is_anim_intermediate:
            # Still need to choose an action (likely a no-op wait), but don't
            # analyze transitions or update rules from animation frames
            action_str, action_data = self._choose_during_animation(analysis)
            self._last_analysis = analysis
            self._last_action = action_str
            self._last_action_data = action_data
            game_action, data = _parse_game_action(action_str, action_data, analysis.available_actions)
            self._log_step(analysis, action_str, action_data, "NOT_FINISHED", "NOT_FINISHED",
                           is_anim=True)
            return game_action

        # ─── Transition analysis (only on turn-boundary frames) ──────────────
        if self._last_analysis and self._last_action:
            # Check expectation violation from previous prediction
            violation = self.planner.check_expectation_violation(analysis, self._last_prediction)

            self.rule_discovery.analyze_transition(
                self._last_action, self._last_action_data,
                self._last_analysis, analysis, self.memory,
            )
            self.goal_inference.update_from_outcome(
                self._last_action, self._last_analysis, analysis, self.memory,
            )

            # ─── Model trigger (c): expectation violation ────────────────────
            if violation:
                logger.info(f"[{self.game_id}] Expectation violation: {violation['violations'][0]}")
                self.goal_inference.generate_hypotheses(
                    analysis, self.memory,
                    trigger=GoalInference.TRIGGER_EXPECTATION_VIOLATION,
                    violation_info=violation,
                )
            # ─── Model trigger (b): all hypotheses exhausted ──────────────────
            elif self.goal_inference.check_hypotheses_exhausted(self.memory) and not self._hypotheses_exhausted_checked:
                logger.info(f"[{self.game_id}] All hypotheses exhausted — consulting model")
                self._hypotheses_exhausted_checked = True
                self.goal_inference.generate_hypotheses(
                    analysis, self.memory,
                    trigger=GoalInference.TRIGGER_HYPOTHESES_EXHAUSTED,
                )
            # Normal periodic hypothesis refresh (heuristic only, no model)
            elif self._step % 10 == 0:
                self.goal_inference.generate_hypotheses(analysis, self.memory)
        else:
            # ─── Model trigger (a): game start — first frame ──────────────────
            if not self._game_start_processed:
                self._game_start_processed = True
                self.goal_inference.generate_hypotheses(
                    analysis, self.memory,
                    trigger=GoalInference.TRIGGER_GAME_START,
                )
            else:
                self.goal_inference.generate_hypotheses(analysis, self.memory)

        # ─── Integrated exploration (with move budget) ────────────────────────
        exploration_action = self.exploration.recommend_action(
            analysis, self.memory, analysis.available_actions
        )

        # ─── Planner chooses (with world-model verification) ──────────────────
        action_str, action_data = self.planner.choose(
            analysis, self.memory, exploration_action
        )

        # ─── World Model: predict outcome for next step's violation check ────
        self._last_prediction = self.world_model.predict(analysis, action_str, action_data, self.memory)

        # Record for next transition analysis
        self._last_analysis = analysis
        self._last_action = action_str
        self._last_action_data = action_data

        # Convert to framework action
        game_action, data = _parse_game_action(
            action_str, action_data, analysis.available_actions
        )

        # Log step
        best_goal = self.memory.get_best_goal()
        self._log_step(
            analysis, action_str, action_data, analysis.state, analysis.state,
            prediction=self._last_prediction,
            best_goal=best_goal.value if best_goal else None,
            best_goal_confidence=best_goal.confidence if best_goal else 0.0,
        )

        logger.info(
            f"[{self.game_id}] Step {self._step} | {action_str} {action_data} | "
            f"state={analysis.state} | goal={best_goal}"
        )

        return game_action

    # ─── Internal helpers ──────────────────────────────────────────────────────

    def _exploration_budget_for_level(self, level: int) -> int:
        """Higher levels get less exploration budget (we carry knowledge forward)."""
        if level == 0:
            return 15
        return max(5, 15 - level * 2)

    def _choose_during_animation(self, analysis: FrameAnalysis) -> tuple[str, dict[str, int]]:
        """During animation intermediate frames, pick a safe no-op or wait action."""
        # Prefer ACTION5 or ACTION7 if available (often wait/no-op in ARC-AGI-3)
        for safe_action in ("ACTION5", "ACTION7"):
            if safe_action in analysis.available_actions:
                return safe_action, {}
        # If only movement actions are available, pick one the world model says is safe
        for action in analysis.available_actions:
            if action == "RESET":
                continue
            pred = self.world_model.predict(analysis, action, {}, self.memory)
            if not (pred.confident and pred.predicts_game_over):
                return action, {}
        # Last resort
        if analysis.available_actions:
            return analysis.available_actions[0], {}
        return "RESET", {}

    def _log_step(
        self,
        analysis: FrameAnalysis,
        action: str,
        action_data: dict[str, int],
        state_before: str,
        state_after: str,
        prediction: Any = None,
        best_goal: str | None = None,
        best_goal_confidence: float = 0.0,
        is_anim: bool = False,
    ) -> None:
        """Log a step to the JSONL session logger."""
        pred_dict = None
        if prediction:
            pred_dict = {
                "confident": prediction.confident,
                "predicts_win": prediction.predicts_win,
                "predicts_game_over": prediction.predicts_game_over,
                "state": prediction.state,
                "reason": prediction.reason,
            }

        self.logger_session.log_step(
            step=self._step,
            game_id=self.game_id,
            level=analysis.level,
            action=action,
            action_data=action_data,
            state_before=state_before,
            state_after=state_after,
            prediction=pred_dict,
            is_anim_intermediate=is_anim,
            player_pos=analysis.player_entity.centroid if analysis.player_entity else None,
            entities_count=len(analysis.entities),
            colors_present=sorted(analysis.colors_present),
            available_actions=analysis.available_actions,
            best_goal=best_goal,
            best_goal_confidence=best_goal_confidence,
        )


# ─── Convenience for local testing ────────────────────────────────────────────

def create_agent(game_id: str = "unknown", **kwargs: Any) -> MyAgent:
    """Factory function for the Swarm orchestrator."""
    return MyAgent(game_id=game_id, **kwargs)
