.PHONY: play-local build-notebook submit slim test

# Play a game locally
play-local:
	python scripts/play_local.py --game ls20 --render

play-all:
	python scripts/play_local.py --all

# Build the Kaggle notebook
build-notebook:
	python scripts/build_notebook.py --output notebooks/submission.ipynb

# Slim framework deps
slim:
	python scripts/slim_framework.py

# Submit to Kaggle (requires kaggle CLI + token)
submit: build-notebook
	kaggle kernels push -p notebooks/

# Test agent imports
test:
	python -c "from agent.my_agent import MyAgent; print('Agent imports OK')"
	python -c "from agent.perception import Perception; p = Perception(); print('Perception OK')"
	python -c "from agent.memory import Memory; m = Memory(); print('Memory OK')"
	python -c "from agent.planner import Planner; pl = Planner(); print('Planner OK')"
