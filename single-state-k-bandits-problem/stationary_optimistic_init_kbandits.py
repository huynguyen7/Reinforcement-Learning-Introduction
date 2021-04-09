#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""
    *SINGLE STATE STATIONARY MULTI-ARMED BANDITS.
    *Name: HUY NGUYEN
    *Source:
        + Reinforcement Learning: An Introduction Second edition  --> Chapter 2
        + https://www.cs.utexas.edu/~pstone/Courses/394Rfall16/resources/8.5.pdf

    *THIS IMPLEMENTATION USES `Epsilon Greedy Method` APPROACH.
    *Just a little modification to the implementation `stationary_kbandits.py`.. since Q are not initialized to 0 all the time. This approach shows that initial assumed Q values increases a fair amount of explorations than setting Q to zeros..
"""

import numpy as np
#import matplotlib.pyplot as plt


class Bandit:  # Action class
    def __init__(self, q, std=1, mean=0):
        self.q = q  # Truth q
        self.noise_std = std
        self.mean = mean

    def reward(self):  # Reward
        return self.q + np.random.randn()*self.noise_std + self.mean

    def get_q(self):
        return self.q


class Agent():  # Agent class
    def __init__(self, k, epsilon, Q_init):
        self.epsilon = epsilon
        self.k = k
        self.Q = np.ones(k, dtype=np.float32)*Q_init  # Estimate/Assumed Q
        self.N = np.zeros(k, dtype=np.float32)  # Number of interaction with respect to Q
        
    def pi(self):  # Policy function -> Return action index
        greed = np.random.rand()  # Uniform dist
        return np.random.choice(range(self.k)) if greed <= self.epsilon else np.argmax(self.Q)
    
    def learn(self, action, reward):  # Update Estimate/Assumed Q and its number of interactions
        self.N[action] += 1
        self.Q[action] += (1/self.N[action]) * (reward-self.Q[action])

    def get_Q(self):
        return self.Q

    def get_N(self):
        return self.N


class Simulator:  # JUST FOR SIMULATING PURPOSES.
    """
        List of k Bandits
        List of k average rewards
        List of k counts for each bandit
        Methods: choose action, reset list
    """

    def __init__(self, k=10, std=1, mean=0, epsilon=0, Q_init=5.0, num_runs=2000, num_steps=1000):
        self.k = k
        self.std = std
        self.mean = mean
        self.epsilon = epsilon
        self.Q_init = Q_init
        self.num_runs = num_runs
        self.num_steps = num_steps

    def simulate(self, log=True, check_convergence=False):
        for run in range(self.num_runs):
            if log:
                rewards_history = np.zeros((self.k,self.num_steps), dtype=np.float32)
            agent = Agent(self.k, self.epsilon, self.Q_init)
            bandits = []

            # Init Environment
            for i in range(self.k):
                q = np.random.randn()*self.std + self.mean  # Gauss data
                bandit = Bandit(q, self.std, self.mean)
                bandits.append(bandit)

            # Greedy Epsilon Agent Learning Process
            for step in range(self.num_steps):
                action = agent.pi()  # Agent acts
                reward = bandits[action].reward()  # Environment sends back reward
                agent.learn(action, reward)  # Agent learns
                
                if log:
                    rewards_history[action, step] = reward
                    if check_convergence:
                        mean_rewards = np.array([rewards_history[i].sum()/agent.get_N()[i] for i in range(self.k) if agent.get_N()[i] != 0])
                        if mean_rewards.shape == agent.get_Q().shape and self.converge(mean_rewards, agent.get_Q()):
                            print(f'The Learning Process converged with {step+1} steps.')
                            break

            if log:
                mean_rewards = np.array([rewards_history[i].sum()/agent.get_N()[i] for i in range(self.k) if agent.get_N()[i] != 0])
                print(f"----RUN-{run+1}--EPSILON-{self.epsilon}----\n*Q_ESTIMATE: {mean_rewards}\n*Q_TRUTH: {[bandit.get_q() for bandit in bandits]}\n")

    def converge(self, mean_rewards, Q):
        return True if not np.any(mean_rewards - Q) else False


""" PARAMS """
epsilons = [0.1]

for epsilon in epsilons:
    simulator = Simulator(
            k=10,  # Number of actions/bandit tasks.
            std=1,  # Used with Gauss dist
            mean=0,  # Used with Gauss dist
            epsilon=epsilon,
            Q_init=5.0,
            num_runs=2,
            num_steps=1000)

    simulator.simulate(
            log=True,
            check_convergence=False)  # Setting this to True will heavily slow down the program because it requires lots of numpy operations.
