#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""

    *Name: HUY NGUYEN
    *Figure 3.2 in the book.

    - This is a little bit difference than `bellman_gridworld.py` since it let the agent do the Q-learning with Temporal Difference Learning and the ABILITY to decides its own action with policy function.
    - This approach finds state-value at each cell without knowing pi (Unlike `bellman_gridworld.py`).

"""


import numpy as np


class Environment:
    def __init__(self, grid_height=5, grid_width=5, default_reward=0, outline_grid_reward=-1):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.default_reward = default_reward  # In-line default rewards
        self.outline_grid_reward = outline_grid_reward

    def interact(self, state, action):  # STEP, Return next state, immediate reward
        if state[0] == 0 and state[1] == 1:  # State A -> State A'
            return [4,1], 10
        elif state[0] == 0 and state[1] == 3:  # State B -> State B'
            return [2,3], 5

        next_state = state + action
        if next_state[0] < 0 or next_state[0] >= self.grid_height or next_state[1] < 0 or next_state[1] >= self.grid_width:  # Reach out-line state
            return state, self.outline_grid_reward
        else:  # Default state
            return next_state, self.default_reward

    def get_grid_height(self):
        return self.grid_height

    def get_grid_width(self):
        return self.grid_width


class Agent:
    def __init__(self, grid_height=5, grid_width=5, gamma=0.9, alpha=0.1, init_state=None):
        self.gamma = gamma  # Discount rate
        self.alpha = alpha  # Learning rate
        self.values = np.zeros(shape=(grid_height, grid_width), dtype=np.float64)  # State-value 2D grid
        self.new_values = np.zeros(shape=(grid_height, grid_width), dtype=np.float64) 
        self.actions = np.array([
            [-1,0],  # LEFT
            [1,0],   # RIGHT
            [0,-1],  # UP
            [0,1]    # DOWN
        ], dtype=np.int8)
        self.current_state = init_state

    def policy(self):  # Stochastic policy
        return self.actions[np.random.choice(self.actions.shape[0])]

    def learn(self, reward, next_state): 
        # Bellman Equation + Temporal Difference Learning
        self.values[self.current_state[0], self.current_state[1]] += self.alpha * (reward + self.gamma * self.values[next_state[0], next_state[1]] - self.values[self.current_state[0], self.current_state[1]])

    def get_current_state(self):
        return self.current_state

    def set_current_state(self, state):
        self.current_state = state

    def get_actions(self):
        return self.actions

    def get_values(self):
        return self.values


class Simulator:
    def __init__(self, grid_height=5, grid_width=5, gamma=0.9, alpha=0.1, default_reward=0, outline_grid_reward=-1, init_state=None):
        self.env = Environment(grid_height, grid_width, default_reward, outline_grid_reward)
        self.agent = Agent(grid_height, grid_width, gamma, alpha, init_state)
    
    def simulate(self, num_steps=100, log=False, plot=False):
        for step in range(num_steps):
            current_state = self.agent.get_current_state()
            action = self.agent.policy()
            next_state, reward = self.env.interact(current_state, action)
            self.agent.learn(reward, next_state)
            self.agent.set_current_state(next_state)

        if log:  # Log grid
            print('\t\t----STATE-VALUES-GRID----')
            print(f"{self.agent.get_values()}\n")
        if plot:  # Plot heatmap
            import matplotlib.pyplot as plt
            plt.xticks(ticks=np.arange(self.env.get_grid_height()), labels=np.arange(self.env.get_grid_height()))
            plt.yticks(ticks=np.arange(self.env.get_grid_width()), labels=np.arange(self.env.get_grid_width()))
            plt.imshow(self.agent.get_values(), cmap='Blues', interpolation='none')
            plt.show()

    def get_agent(self):
        return self.agent

    def get_environment(self):
        return self.env

                    
""" SIMULATION """
simulator = Simulator(
    grid_height=5,
    grid_width=5,
    gamma=0.95,  # Discount rate
    alpha=0.1,  # Learning rate
    default_reward=0,
    outline_grid_reward=-1,
    init_state=[0,0],
)

simulator.simulate(
    num_steps=10000, 
    log=True,
    plot=False
)
