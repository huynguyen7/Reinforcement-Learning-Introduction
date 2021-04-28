#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""

    *Name: HUY NGUYEN
    *Figure 3.5 in the book.

    - This implementation uses Monte-Carlo Learning (update at the end of each episode).. However, this implementation does not let the agent decides its own actions.
    - In addition, this approach uniformly updates all the cells (loop i, loop j in the codes) for each action.
    - Importantly, this approach find OPTIMAL optimal state-values if and only if we know the pi(For example, pi=0.25 in this grid).

    - The cells of the grid correspond to the states of the environment. At each cell, four actions are possible: UP,DOWN,LEFT,RIGHT; which deterministically cause the agent to move one cell in the respective direction.
    Actions that would take the agent off the grid leave its location unchanged, but also result in a reward of `outline_grid_reward`. Other actions result in a reward of `default_reward`. Except those that move agent to A and B. From state A at (0,1), all four actions yield a reward of +10 and take the agent to A' at (4,1). From state B at (0,3), all actions yield a reward of +5 and take the agent to B' at (2,3).

"""


import numpy as np
from environment import Environment


class Agent:
    def __init__(self, grid_height=5, grid_width=5,gamma=0.9, error_threshold=10e-2):
        self.gamma = gamma  # Discount rate
        self.error_threshold = error_threshold  # Just to check for convergence
        self.optimal_values = np.zeros(shape=(grid_height, grid_width), dtype=np.float64)  # State-value 2D grid
        self.new_optimal_values = np.zeros(shape=(grid_height, grid_width), dtype=np.float64) 
        self.actions = np.array([
            [-1,0],  # LEFT
            [1,0],   # RIGHT
            [0,-1],  # UP
            [0,1]    # DOWN
        ], dtype=np.int8)
        self.tmp_optimal_values = []

    def update(self): 
        is_converged = np.sum(np.abs(self.optimal_values-self.new_optimal_values)) <= self.error_threshold
        self.optimal_values = self.new_optimal_values
        self.new_optimal_values = np.zeros(shape=self.optimal_values.shape, dtype=np.float64)
        return is_converged

    def step(self, reward, next_state):  # BELLMAN EQUATION FOR UPDATING STATE-VALUE -> EVENTUALLY, IT WILL CONVERGE BASED ON THE LAW OF LARGE NUMBER..
        self.tmp_optimal_values.append(reward + self.gamma * self.optimal_values[next_state[0], next_state[1]])

    def learn(self, current_state):
        self.new_optimal_values[current_state[0], current_state[1]] = max(self.tmp_optimal_values)
        self.tmp_optimal_values = []

    def get_actions(self):
        return self.actions

    def get_optimal_values(self):
        return self.optimal_values

    def get_error_threshold(self):
        return self.error_threshold


class Simulator:
    def __init__(self, grid_height=5, grid_width=5, gamma=0.9, default_reward=0, outline_grid_reward=-1, error_threshold=10e-2):
        self.env = Environment(grid_height, grid_width, default_reward, outline_grid_reward)
        self.agent = Agent(grid_height, grid_width, gamma, error_threshold)
    
    def simulate(self, num_steps=100, log=False, plot=False):
        for step in range(num_steps):
            for i in range(self.env.get_grid_width()):
                for j in range(self.env.get_grid_height()):
                    current_state = [i, j]
                    for action in self.agent.get_actions():
                        next_state, reward = self.env.interact(current_state, action)
                        self.agent.step(reward, next_state)
                    self.agent.learn(current_state)
            is_converged = self.agent.update()
            if is_converged:
                print(f'--> The algorithm converged in {step+1} steps with error rate = {self.agent.get_error_threshold()}')
                break

        if log:  # Log grid
            print('\t\t----STATE-VALUES-GRID----')
            print(f"{self.agent.get_optimal_values()}\n")
        if plot:  # Plot heatmap
            import matplotlib.pyplot as plt
            plt.xticks(ticks=np.arange(self.env.get_grid_height()), labels=np.arange(self.env.get_grid_height()))
            plt.yticks(ticks=np.arange(self.env.get_grid_width()), labels=np.arange(self.env.get_grid_width()))
            plt.imshow(self.agent.get_optimal_values(), cmap='Blues', interpolation='none')
            plt.show()

    def get_agent(self):
        return self.agent

    def get_environment(self):
        return self.env

                    
""" SIMULATION """
simulator = Simulator(
    grid_height=5,
    grid_width=5,
    gamma=0.9,
    default_reward=0,
    outline_grid_reward=-1,
    error_threshold=10e-4
)

simulator.simulate(
    num_steps=10000,
    log=True,  # Set this to true to log grid optimal state-values.
    plot=False  # Set this to True to see heatmap
)
