import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps":4}
    
    def __init__(self, grid: int | list[int], render_mode: str | None) -> None:
        
        #Note Custom map feature coming
        self.size = grid


        self.action_space = spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
            "agent": gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32),
            "target": gym.spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int32)
            }
        )

        self._action_to_direction = {
            0: np.array([1,0]), #Right
            1: np.array([-1,0]), #Left
            2: np.array([0,1]), #Up
            3: np.array([0,-1]) #Down
        }

        self.agent_location = None
        self.target = None
        
        self.render_mode = render_mode
        self.fps = self.metadata['render_fps']

    def reset(self, seed: int | None = None, options: dict = None) -> tuple:
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        #Randomly places agent on the board
        self.agent_location = self.rng.integers(0,self.size,size=2,dtype=np.int32)
        
        #Randomly place target location
        self.target = self.rng.integers(0,self.size,size=2,dtype=np.int32)
        
        #If target = agent random places target
        while np.array_equal(self.target, self.agent_location):
            self.target = self.rng.integers(0,self.size,size=2,dtype=np.int32)
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target}

    def step(self, action: int):
        #Convert action into direction tuple
        direction = self._action_to_direction[action]

        # Get new position of agent
        nx = self.agent_location[0] + direction[0]
        ny = self.agent_location[1] + direction[1]

        in_bounds = (0 <= nx < self.size) and (0 <= ny < self.size)

        terminated = False
        reward = 0.0

        if not in_bounds:
            reward = -1.0
        else:
            self.agent_location = np.array([nx, ny], dtype=np.int32)

            terminated = np.array_equal(self.agent_location, self.target)

            distance = np.linalg.norm(self.agent_location - self.target)
            reward = 1.0 if terminated else -0.1 * distance

        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):

        grid = [["_" for _ in range(self.size)] for _ in range(self.size)]
        ax, ay = map(int, self.agent_location)
        tx, ty = map(int, self.target)
        grid[ax][ay] = "A"
        grid[tx][ty] = "T"
        return "\n".join("".join(row) for row in grid)

    