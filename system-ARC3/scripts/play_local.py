"""
Local play script — runs the agent against real ARC-AGI-3 games.

Usage:
  python scripts/play_local.py --game ls20
  python scripts/play_local.py --game ls20 --steps 100 --render
  python scripts/play_local.py --all
"""

import argparse
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc_agi
from arcengine import GameAction, GameState
from agent.my_agent import MyAgent, _frame_to_dict, _extract_grid, _extract_state, _extract_actions, _parse_game_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("play_local")


def play_game(game_name: str, max_steps: int = 200, render: bool = False) -> dict:
    """Play a single game and return results."""
    arc = arc_agi.Arcade()
    env = arc.make(game_name, render_mode="terminal" if render else None)

    if env is None:
        logger.error(f"Failed to create environment for game: {game_name}")
        return {"game": game_name, "error": "env_creation_failed"}

    agent = MyAgent(game_id=game_name)
    results = {"game": game_name, "levels_won": 0, "steps": 0, "actions": []}

    # Start with RESET
    obs = env.reset() if hasattr(env, "reset") else env.step(GameAction.RESET)
    frames = []

    for step in range(max_steps):
        # Build frame dict from observation
        frame_dict = _frame_to_dict(obs) if obs else {"grid": [[0]], "state": "NOT_FINISHED", "available_actions": [], "level": 0}
        frames.append(frame_dict)

        # Check if done
        if agent.is_done(frames, frame_dict):
            logger.info(f"[{game_name}] Agent decided to stop at step {step}")
            break

        # Choose action
        action = agent.choose_action(frames, frame_dict)
        results["actions"].append(str(action))

        # Execute action
        action_data = {}
        if hasattr(action, "is_complex") and action.is_complex():
            action_data = {"x": 0, "y": 0}  # Agent should provide this

        try:
            obs = env.step(action, data=action_data)
        except Exception as e:
            logger.error(f"[{game_name}] Action failed: {e}")
            break

        # Check game state
        state = _extract_state(obs) if obs else "NOT_FINISHED"
        if state == "WIN":
            results["levels_won"] += 1
            logger.info(f"[{game_name}] Level won at step {step}!")
        elif state == "GAME_OVER":
            logger.info(f"[{game_name}] Game over at step {step}")

        results["steps"] = step + 1

    # Get scorecard
    try:
        scorecard = arc.get_scorecard()
        if scorecard:
            results["score"] = scorecard.score if hasattr(scorecard, "score") else str(scorecard)
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 games locally")
    parser.add_argument("--game", type=str, default="ls20", help="Game name to play")
    parser.add_argument("--all", action="store_true", help="Play all available games")
    parser.add_argument("--steps", type=int, default=200, help="Max steps per game")
    parser.add_argument("--render", action="store_true", help="Render game in terminal")
    args = parser.parse_args()

    if args.all:
        arc = arc_agi.Arcade()
        # List available games
        try:
            games = arc.list_games() if hasattr(arc, "list_games") else ["ls20"]
        except Exception:
            games = ["ls20"]

        all_results = []
        for g in games:
            logger.info(f"Playing game: {g}")
            result = play_game(g, args.steps, args.render)
            all_results.append(result)
            print(f"\n--- {g}: {result.get('levels_won', 0)} levels, {result.get('steps', 0)} steps ---\n")

        print("\n=== Summary ===")
        for r in all_results:
            print(f"  {r['game']}: {r.get('levels_won', 0)} levels, {r.get('steps', 0)} steps")
    else:
        result = play_game(args.game, args.steps, args.render)
        print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
