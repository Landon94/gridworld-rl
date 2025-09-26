# gridworld-rl

## Overview
The projects main purpose is for learning about reinforcement learning. It implements a custom GridWorld environment built on top of OpenAI's gymnassium.

The environment places the agent on a 2D board where its goal it to reach a target position. The agent can move left, right, up, or down and is rewarded based on distance from the target and if it reaches it.

## How to Run

### Create Virtual Environment (Optional)
```
python -m venv venv
source venv/bin/activate
```
### Install Dependencies
```
pip install -r requirements.txt
```
### Training
Training parameters can be changed in train.py
```
python train.py
```
### Visualizer
Can be set to human or ansi in demo.py
```
python demo.py
```
