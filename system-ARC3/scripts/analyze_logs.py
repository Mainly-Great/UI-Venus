"""
JSONL Log Analyzer

Reads session logs produced by JsonlLogger and produces a summary report:
  - Per-game win/loss/steps statistics
  - Action distribution
  - Prediction accuracy (how often the world model was right)
  - Goal hypothesis evolution (confidence over time)
  - Expectation violations (where the model was surprised)
  - Model intervention triggers (when/why the model was called)
  - Animation intermediate frame ratio
  - Exploration budget usage

Usage:
  python scripts/analyze_logs.py --log logs/session.jsonl
  python scripts/analyze_logs.py --log logs/session.jsonl --report text
  python scripts/analyze_logs.py --log logs/session.jsonl --report json
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


def load_log(path: str) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dicts."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def analyze(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze log entries and return a structured report."""
    sessions: dict[str, dict[str, Any]] = {}
    steps = [e for e in entries if e.get("type") == "step"]

    # Group steps by session
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in steps:
        by_session[s.get("session_id", "unknown")].append(s)

    for sid, session_steps in by_session.items():
        session_info: dict[str, Any] = {
            "session_id": sid,
            "total_steps": len(session_steps),
            "games": {},
        }

        # Group by game
        by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in session_steps:
            by_game[s.get("game_id", "unknown")].append(s)

        for game_id, game_steps in by_game.items():
            game_info = analyze_game(game_steps)
            session_info["games"][game_id] = game_info

        # Session-level meta
        session_starts = [e for e in entries if e.get("type") == "session_start" and e.get("session_id") == sid]
        session_ends = [e for e in entries if e.get("type") == "session_end" and e.get("session_id") == sid]
        if session_starts:
            session_info["model_name"] = session_starts[0].get("model_name", "none")
        if session_ends:
            session_info["wins"] = session_ends[0].get("wins", 0)
            session_info["losses"] = session_ends[0].get("losses", 0)

        sessions[sid] = session_info

    return {"sessions": sessions}


def analyze_game(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze steps for a single game."""
    wins = sum(1 for s in steps if s.get("state_after") == "WIN")
    losses = sum(1 for s in steps if s.get("state_after") == "GAME_OVER")
    total = len(steps)

    # Action distribution
    actions = Counter(s.get("action", "?") for s in steps)

    # Prediction accuracy
    predictions_made = 0
    predictions_correct = 0
    predictions_violated = 0
    for s in steps:
        pred = s.get("prediction")
        if pred and pred.get("confident"):
            predictions_made += 1
            violation = s.get("violation")
            if violation:
                predictions_violated += 1
            else:
                predictions_correct += 1

    pred_accuracy = predictions_correct / predictions_made if predictions_made > 0 else 0.0

    # Animation frames
    anim_frames = sum(1 for s in steps if s.get("is_anim_intermediate", False))
    anim_ratio = anim_frames / total if total > 0 else 0.0

    # Model triggers
    triggers = Counter(s.get("trigger") for s in steps if s.get("trigger"))

    # Goal evolution — track best goal confidence over steps
    goal_confidence_timeline = []
    for s in steps:
        conf = s.get("best_goal_confidence", 0)
        goal = s.get("best_goal", "none")
        goal_confidence_timeline.append({"step": s.get("step", 0), "goal": goal, "conf": conf})

    # Exploration budget (approximation: steps before best_goal confidence > 0.35)
    exploration_steps = 0
    for gc in goal_confidence_timeline:
        if gc["conf"] < 0.35:
            exploration_steps += 1
        else:
            break

    # Violations details
    violations = [s.get("violation") for s in steps if s.get("violation")]

    return {
        "total_steps": total,
        "wins": wins,
        "losses": losses,
        "action_distribution": dict(actions.most_common()),
        "prediction_accuracy": round(pred_accuracy, 3),
        "predictions_made": predictions_made,
        "predictions_violated": predictions_violated,
        "anim_frame_ratio": round(anim_ratio, 3),
        "anim_frames": anim_frames,
        "model_triggers": dict(triggers),
        "exploration_steps": exploration_steps,
        "violations_count": len(violations),
        "violations_sample": violations[:5],
        "goal_confidence_final": goal_confidence_timeline[-1]["conf"] if goal_confidence_timeline else 0,
        "goal_confidence_timeline_len": len(goal_confidence_timeline),
    }


def render_text_report(report: dict[str, Any]) -> str:
    """Render analysis as a human-readable text report."""
    lines = ["=" * 60, "ARC-AGI-3 Session Analysis Report", "=" * 60]

    for sid, session in report["sessions"].items():
        lines.append(f"\n--- Session: {sid} ---")
        lines.append(f"Model: {session.get('model_name', 'none')}")
        lines.append(f"Total steps: {session.get('total_steps', 0)}")
        lines.append(f"Wins: {session.get('wins', 0)} | Losses: {session.get('losses', 0)}")

        for game_id, game in session.get("games", {}).items():
            lines.append(f"\n  Game: {game_id}")
            lines.append(f"    Steps: {game['total_steps']} | Wins: {game['wins']} | Losses: {game['losses']}")
            lines.append(f"    Prediction accuracy: {game['prediction_accuracy']:.1%} ({game['predictions_made']} predictions, {game['predictions_violated']} violations)")
            lines.append(f"    Animation frames: {game['anim_frames']} ({game['anim_frame_ratio']:.1%} of total)")
            lines.append(f"    Exploration steps: {game['exploration_steps']}")
            lines.append(f"    Model triggers: {game['model_triggers'] or 'none'}")
            lines.append(f"    Goal confidence final: {game['goal_confidence_final']:.1%}")
            lines.append(f"    Violations: {game['violations_count']}")
            lines.append(f"    Actions: {game['action_distribution']}")

            if game["violations_sample"]:
                lines.append(f"    Violation samples:")
                for v in game["violations_sample"][:3]:
                    lines.append(f"      - {v}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze ARC-AGI-3 session logs")
    parser.add_argument("--log", type=str, required=True, help="Path to JSONL log file")
    parser.add_argument("--report", type=str, default="text", choices=["text", "json"], help="Report format")
    args = parser.parse_args()

    if not Path(args.log).exists():
        print(f"Log file not found: {args.log}")
        sys.exit(1)

    entries = load_log(args.log)
    report = analyze(entries)

    if args.report == "json":
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render_text_report(report))


if __name__ == "__main__":
    main()
