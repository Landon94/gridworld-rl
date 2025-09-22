from collections import defaultdict
import numpy as np

class QAgent():
    
    def __init__(self,env,learning_rate,discount_factor,epsilon):

        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

        self.Q_table = defaultdict(lambda: np.zeros(env.action_space.n, dtype=np.float32))
    
    def _obs_to_state(self, obs):
        ax, ay = map(int, obs["agent"])
        tx, ty = map(int, obs["target"])
        return (ax, ay, tx, ty)

    def choose_action(self, obs):
        state = self._obs_to_state(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def update(self,obs,action,reward,terminated,next_obs):
        state = self._obs_to_state(obs)
        next_state = self._obs_to_state(next_obs)

        if terminated:
            future_q = reward
        else:
            future_q = reward + self.discount_factor * np.max(self.Q_table[next_state])

        error = future_q - self.Q_table[state][action]
        
        self.Q_table[state][action] = self.Q_table[state][action] + self.lr * error

