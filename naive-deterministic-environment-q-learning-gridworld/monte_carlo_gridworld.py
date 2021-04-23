#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""

    *Name: HUY NGUYEN
    *Figure 3.2 in the book.

    - This is a little bit difference than `bellman_gridworld.py` since it let the agent do the Q-learning with Temporal Difference Learning and the ABILITY to decides its own action with policy function.
    - This approach finds state-value at each cell without knowing pi (Unlike `bellman_gridworld.py`).
    - Using Monte-Carlo Method.

"""


import numpy as np
from environment import Environment


class MCAgent:
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
        self.reset()

    def reset(self):
        self.rewards_history = []
        self.current_state = np.array([np.random.choice(self.values.shape[0]), np.random.choice(self.values.shape[1])], dtype=np.int32)

    def policy(self):  # Stochastic policy -> Used uniform dist
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


    def learn(self, next_state, reward): 
        self.rewards_history.append((next_state, reward))

    def update(self):  # Monte-Carlo Method
        total_expected_return = 0  # G(t)
        visited_states = set()  # Use a hash set to check duplicated states.
        for state, reward in reversed(self.rewards_history):
            hashable_state = str(state)  # Use string since array is not hashable in Python.
            if hashable_state not in visited_states:  # Make sure we won't go to the same visited states.
                visited_states.add(hashable_state)
                total_expected_return = self.gamma*(reward+total_expected_return)
                self.values[state[0], state[1]] += self.alpha*(total_expected_return-self.values[state[0], state[1]])  # MC Update.

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
        self.agent = MCAgent(grid_height, grid_width, gamma, alpha, epsilon, init_state)
    
    def simulate(self, num_episodes=100, num_steps=10, log=False, plot=False):
        for episode in range(num_episodes):
            for step in range(num_steps):
                #action = self.agent.policy()
                action = self.agent.epsilon_greedy_policy()
                next_state, reward = self.env.interact(self.agent.get_current_state(), action)
                self.agent.learn(next_state, reward)
                self.agent.set_current_state(next_state)
            self.agent.update()
            self.agent.reset()  # reset init_state for next episode

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
)

simulator.simulate(
    num_episodes=10000,
    num_steps=10, 
    log=True,
    plot=False
)
