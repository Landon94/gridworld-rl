import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import colors

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["ansi","human"], "render_fps":3}
    
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

        self.fig = None
        self.ax = None
        self.im = None

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
        
        if self.render_mode == "human":
            self._render_frame()

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

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}
    
    def render(self):

        if self.render_mode == "ansi":
            grid = [["_" for _ in range(self.size)] for _ in range(self.size)]
            ax, ay = map(int, self.agent_location)
            tx, ty = map(int, self.target)
            grid[ax][ay] = "A"
            grid[tx][ty] = "T"
            return "\n".join("".join(row) for row in grid)
        elif self.render_mode == "human":
            return None
        else:
            print("Error: Incorrect Render Mode")
            return None

    def _env_grid(self):
        grid = np.zeros((self.size,self.size),dtype=np.int8)
        ax, ay = map(int, self.agent_location)
        tx, ty = map(int, self.target)
        grid[ay, ax] = 2   
        grid[ty, tx] = 3   
        return grid

    
    def _render_init(self):
        if self.fig is not None:
            return
        
        plt.ion()

        grid = self._env_grid()

        # 0=White(space),1=Black(Wall),2=Red(Agent),3=Green(Target)
        cmap = colors.ListedColormap(['white','black','red','green'])
        b_norm = colors.BoundaryNorm([-0.5,0.5,1.5,2.5,3.5],cmap.N)

        self.fig, self.ax = plt.subplots(figsize=(self.size,self.size),tight_layout=True)
        self.im = self.ax.imshow(grid,cmap=cmap,norm=b_norm,interpolation="none",extent=(-0.5, self.size - 0.5, self.size - 0.5, -0.5),)
        self.ax.set_aspect("equal")

        self.ax.set_xticks(np.arange(self.size) - 0.5)
        self.ax.set_yticks(np.arange(self.size) - 0.5)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        self.ax.grid(which="both",color="lightgray", linewidth=0.5)

        plt.pause(0.001)

    def _render_frame(self):
        if self.fig is None:
            self._render_init()

        self.im.set_data(self._env_grid())
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        time.sleep(1/self.fps)
    
    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)    