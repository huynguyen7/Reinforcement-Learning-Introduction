#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""

    *SINGLE STATE NONSTATIONARY MULTI-ARMED BANDITS.
    *Name: HUY NGUYEN
    *Source:
        + Reinforcement Learning: An Introduction Second edition  --> Chapter 2
    
    *THIS IMPLEMENTATION USES `Upper Confidence Bound` AND `Fixed Learning Rate`.

"""

import warnings
import numpy as np
#import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning) 


class Bandit:  # Action class
    def __init__(self, q, std=1, mean=0):
        self.q = q  # Truth q
        self.noise_std = std
        self.mean = mean

    def reward(self):  # Reward
        return self.q + np.random.randn()*self.noise_std + self.mean

    def get_q(self):
        return self.q

    def update_q(self, alpha):  # Nonstationary q
        self.q = self.q + (2*np.random.randint(0,2)-1)*alpha


class Agent():  # Agent class
    def __init__(self, k, alpha=0.1, c=0.1):
        self.k = k
        self.alpha = alpha  # Learning rate
        self.c = c  # Degree of exploration
        self.Q = np.zeros(k, dtype=np.float64)  # Estimate/Assumed Q
        self.N = np.ones(k, dtype=np.float64)*1e-309  # Number of interaction with respect to Q

    def pi(self, t):  # Policy function -> Return action index, deterministic policy
        return np.argmax(self.Q + self.c*np.sqrt(np.log(t)/self.N))
    
    def learn(self, action, reward):  # Update Estimate/Assumed Q and its number of interactions
        self.N[action] += 1
        self.Q[action] += self.alpha*(reward-self.Q[action])

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

    def __init__(self, k=10, std=1, mean=0, alpha=0.1, c=0.1, num_runs=2000, num_steps=1000):
        self.k = k
        self.std = std
        self.mean = mean
        self.alpha = alpha
        self.c = c
        self.num_runs = num_runs
        self.num_steps = num_steps

    def simulate(self, log=True, check_convergence=False):
        for run in range(self.num_runs):
            if log:
                #rewards_history = np.zeros((self.k,self.num_steps), dtype=np.float64)
                pass
            agent = Agent(self.k, self.alpha, self.c)
            bandits = []

            # Init Environment
            for i in range(self.k):
                q = np.random.randn()*self.std + self.mean  # Gauss data
                bandit = Bandit(q, self.std, self.mean)
                bandits.append(bandit)

            # UCB Agent Learning Process
            for step in range(self.num_steps):
                action = agent.pi(step+1)  # Agent acts
                [bandits[i].update_q(self.alpha) for i in range(self.k)] # Update q / Nonstationary environment
                reward = bandits[action].reward()  # Environment sends back reward
                agent.learn(action, reward)  # Agent learns
                
                if log:
                    #rewards_history[action, step] = reward
                    if check_convergence and self.converge([bandit.get_q() for bandit in bandits], agent.get_Q()):
                        print(f'The Learning Process converged with {step+1} steps.')
                        break

            if log:
                #mean_rewards = np.array([rewards_history[i].sum()/agent.get_N()[i] for i in range(self.k) if agent.get_N()[i] != 0])
                print(f"----RUN-{run+1}--CONFIDENCE_VALUE-{self.c}----\n*Q_ESTIMATE: {agent.get_Q()}\n\n*Q_TRUTH: {[bandit.get_q() for bandit in bandits]}\n\nN: {agent.get_N()}\n")

    def converge(self, q, Q):
        return True if not np.any(q-Q) else False


""" PARAMS """
confidence_values = [0.1]

for c in confidence_values:
    simulator = Simulator(
            k=10,  # Number of actions/bandit tasks.
            std=1,  # Used with Gauss dist
            mean=0,  # Used with Gauss dist
            alpha=0.1,  # Learning rate
            c=3,  # Degree of exploration / Confidence value
            num_runs=1,  # Number of episodes
            num_steps=1000)  # Lifetime

    simulator.simulate(
            log=True,
            check_convergence=False)  # Setting this to True will heavily slow down the program because it requires lots of numpy operations.
