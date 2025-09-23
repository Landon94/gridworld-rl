import pickle
from tqdm import tqdm
from environment.env import GridWorldEnv
from agent import QAgent


if __name__ == "__main__":

    #Set to int or custom map
    grid = 5
    
    #Params
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 0.1

    #Training sessions
    n_episodes = 10000
    
    env = GridWorldEnv(grid=grid,render_mode="ansi")
    agent = QAgent(env, learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon)


    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        total_reward=0

        while not done:
            # Agent chooses action
            action = agent.choose_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # Move to next state
            done = terminated or truncated
            total_reward += reward
            obs = next_obs


    with open("q_table.pkl", "wb") as f:
        pickle.dump(dict(agent.Q_table), f)
