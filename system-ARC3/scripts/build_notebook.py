"""
Builds a Kaggle submission notebook from the agent code.

This script reads agent/ and packages it into a self-contained notebook
that Kaggle can execute. The notebook:
  1. Installs the arc-agi package from bundled wheels
  2. Writes all agent modules as inline cells
  3. Runs the Swarm orchestrator across all games

Usage:
  python scripts/build_notebook.py
  python scripts/build_notebook.py --output notebooks/submission.ipynb
"""

import argparse
import json
import os
import sys
import base64
from pathlib import Path


def build_notebook(output_path: str = "notebooks/submission.ipynb") -> None:
    """Build a Kaggle submission notebook from the agent source."""

    project_root = Path(__file__).parent.parent
    agent_dir = project_root / "agent"

    # Collect all agent Python files
    agent_files = {}
    for py_file in sorted(agent_dir.rglob("*.py")):
        rel = py_file.relative_to(agent_dir)
        if rel.name == "__init__.py" and rel.parent == Path("."):
            continue
        content = py_file.read_text(encoding="utf-8")
        module_path = str(rel).replace("/", ".").replace(".py", "")
        agent_files[module_path] = content

    # Build notebook cells
    cells = []

    # Cell 1: Install dependencies
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Install arc-agi from bundled wheels\n",
            "import subprocess, sys, os, glob\n",
            "\n",
            "wheel_dir = '/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels'\n",
            "if os.path.exists(wheel_dir):\n",
            "    wheels = glob.glob(os.path.join(wheel_dir, '*.whl'))\n",
            "    for w in wheels:\n",
            "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', w, '-q'])\n",
            "else:\n",
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'arc-agi', '-q'])\n",
            "\n",
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai', 'numpy', '-q'])\n",
            "print('Dependencies installed.')"
        ],
        "outputs": [],
        "execution_count": None,
    })

    # Cell 2: Write agent modules to disk
    write_lines = [
        "import os, pathlib\n",
        "\n",
        "os.makedirs('/kaggle/working/agent', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/perception', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/exploration', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/rules', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/memory', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/goals', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/planner', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/agent/model', exist_ok=True)\n",
        "\n",
    ]

    for module_path, content in agent_files.items():
        file_path = module_path.replace(".", "/") + ".py"
        write_lines.append(f"\n# --- {file_path} ---\n")
        # Use a heredoc-style write to handle multiline content safely
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        write_lines.append(f"pathlib.Path('/kaggle/working/{file_path}').parent.mkdir(parents=True, exist_ok=True)\n")
        write_lines.append(f"pathlib.Path('/kaggle/working/{file_path}').write_bytes(__import__('base64').b64decode('{encoded}'))\n")

    write_lines.append("\nprint('Agent modules written.')\n")

    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": write_lines,
        "outputs": [],
        "execution_count": None,
    })

    # Cell 3: Import and run
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import sys\n",
            "sys.path.insert(0, '/kaggle/working')\n",
            "\n",
            "import arc_agi\n",
            "from arcengine import GameAction, GameState\n",
            "from agent.my_agent import MyAgent, _frame_to_dict, _parse_game_action, _extract_state\n",
            "\n",
            "arc = arc_agi.Arcade(operation_mode=getattr(arc_agi, 'OperationMode', None).COMPETITION if hasattr(arc_agi, 'OperationMode') else None)\n",
            "\n",
            "# List games\n",
            "try:\n",
            "    games = arc.list_games()\n",
            "except:\n",
            "    games = ['ls20']  # fallback\n",
            "\n",
            "print(f'Games: {games}')\n",
            "\n",
            "# Open scorecard\n",
            "try:\n",
            "    scorecard = arc.open_scorecard()\n",
            "except Exception as e:\n",
            "    print(f'Scorecard error: {e}')\n",
            "    scorecard = None\n",
            "\n",
            "# Play each game\n",
            "for game_name in games:\n",
            "    print(f'\\n--- Playing: {game_name} ---')\n",
            "    try:\n",
            "        env = arc.make(game_name)\n",
            "        if env is None:\n",
            "            print(f'Could not create env for {game_name}')\n",
            "            continue\n",
            "        \n",
            "        agent = MyAgent(game_id=game_name)\n",
            "        frames = []\n",
            "        \n",
            "        # Initial reset\n",
            "        obs = env.reset() if hasattr(env, 'reset') else env.step(GameAction.RESET)\n",
            "        \n",
            "        for step in range(200):\n",
            "            frame_dict = _frame_to_dict(obs) if obs else {'grid': [[0]], 'state': 'NOT_FINISHED', 'available_actions': [], 'level': 0}\n",
            "            frames.append(frame_dict)\n",
            "            \n",
            "            if agent.is_done(frames, frame_dict):\n",
            "                print(f'Agent done at step {step}')\n",
            "                break\n",
            "            \n",
            "            action = agent.choose_action(frames, frame_dict)\n",
            "            \n",
            "            action_data = {}\n",
            "            try:\n",
            "                if hasattr(action, 'is_complex') and action.is_complex():\n",
            "                    action_data = {'x': 0, 'y': 0}\n",
            "                obs = env.step(action, data=action_data)\n",
            "            except Exception as e:\n",
            "                print(f'Step error: {e}')\n",
            "                # Try reset\n",
            "                obs = env.reset() if hasattr(env, 'reset') else env.step(GameAction.RESET)\n",
            "                agent.reset()\n",
            "        \n",
            "    except Exception as e:\n",
            "        print(f'Game {game_name} error: {e}')\n",
            "\n",
            "# Close scorecard\n",
            "if scorecard:\n",
            "    try:\n",
            "        arc.close_scorecard()\n",
            "        print(f'Final score: {arc.get_scorecard()}')\n",
            "    except Exception as e:\n",
            "        print(f'Close error: {e}')\n",
            "\n",
            "print('\\nDone.')"
        ],
        "outputs": [],
        "execution_count": None,
    })

    # Build the notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"Notebook written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Kaggle submission notebook")
    parser.add_argument("--output", type=str, default="notebooks/submission.ipynb", help="Output path")
    args = parser.parse_args()

    build_notebook(args.output)


if __name__ == "__main__":
    main()
