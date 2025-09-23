import pickle
import time
import numpy as np
from collections import defaultdict
from environment.env import GridWorldEnv
from agent import QAgent


if __name__ == "__main__":

    #Set to ansi or human
    render_mode = "human"

    env = GridWorldEnv(grid=5, render_mode=render_mode)
    agent = QAgent(env, learning_rate=0.0, discount_factor=0.99, epsilon=0.0)

    try:
        with open("q_table.pkl", "rb") as f:
            loaded = pickle.load(f)

        agent.Q_table = defaultdict(
            lambda: np.zeros(agent.env.action_space.n, dtype=np.float32),
            loaded,
        )
        print("---Successfully loaded Q-table---")
    except FileNotFoundError:
        print(f"---[warn] No saved Q-table . Running with an untrained agent.---")

    
    obs, info = env.reset(seed=42)

    if render_mode == "ansi":
        done = False
        while not done:
            action = agent.choose_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            print(env.render())
            time.sleep(1)

            done = terminated or truncated
            obs = next_obs
    elif render_mode == "human":
        done = False
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc

        env.close()

    