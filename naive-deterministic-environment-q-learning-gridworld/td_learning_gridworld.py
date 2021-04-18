#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""

    *Name: HUY NGUYEN
    *Figure 3.2 in the book.

    - This is a little bit difference than `bellman_gridworld.py` since it let the agent do the Q-learning with Temporal Difference Learning and the ABILITY to decides its own action with policy function.
    - This approach finds state-value at each cell without knowing pi (Unlike `bellman_gridworld.py`).
    - Using Temporal Difference Learning.

"""


import numpy as np
from environment import Environment


class TDAgent:
    def __init__(self, grid_height=5, grid_width=5, gamma=0.9, alpha=0.1, epsilon=0.1, init_state=None):
        self.gamma = gamma  # Discount rate
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Greedy rate
        self.values = np.zeros(shape=(grid_height, grid_width), dtype=np.float64)  # State-value 2D grid
        self.actions = np.array([
            [-1,0],  # LEFT
            [1,0],   # RIGHT
            [0,-1],  # UP
            [0,1]    # DOWN
        ], dtype=np.int8)
        self.current_state = np.array(init_state)

    def policy(self):  # Stochastic policy
        return self.actions[np.random.choice(self.actions.shape[0])]

    def epsilon_greedy_policy(self):  # Epsilon-greedy based policy
        greed = np.random.uniform()
        if greed <= self.epsilon:
            return self.actions[np.random.choice(self.actions.shape[0])]
        else:
            possible_states = [self.current_state + action for action in self.actions]
            max_state_value = np.NINF  # Negative infinity
            max_i = 0
            for i, state in zip(range(len(possible_states)), possible_states):
                if state[0] < 0 or state[0] >= self.values.shape[0] or state[1] < 0 or state[1] >= self.values.shape[1]:  # Reach out-line state
                    continue
                elif self.values[state[0], state[1]] > max_state_value:
                    max_state_value = self.values[state[0], state[1]]
                    max_i = i
            return self.actions[max_i]


    def learn(self, reward, next_state):  # Temporal Difference Learning
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
    def __init__(self, grid_height=5, grid_width=5, gamma=0.9, alpha=0.1, epsilon=0.1, default_reward=0, outline_grid_reward=-1, init_state=None):
        self.env = Environment(grid_height, grid_width, default_reward, outline_grid_reward)
        self.agent = TDAgent(grid_height, grid_width, gamma, alpha, epsilon, init_state)
    
    def simulate(self, num_steps=100, log=False, plot=False):
        for step in range(num_steps):
            current_state = self.agent.get_current_state()
            action = self.agent.policy()  # This performs better than epsilon-greedy
            #action = self.agent.epsilon_greedy_policy()  # Epsilon-greedy
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
    gamma=0.9,  # Discount rate
    alpha=0.1,  # Learning rate
    epsilon=0.1,  # Greedy rate
    default_reward=0,
    outline_grid_reward=-1,
    init_state=[0,0],
)

simulator.simulate(
    num_steps=1000000, 
    log=True,
    plot=False
)
