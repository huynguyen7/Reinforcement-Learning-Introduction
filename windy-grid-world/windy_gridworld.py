from Environment import generate_grid_6_5
from tqdm import tqdm
import numpy as np


"""

    *Name: HUY NGUYEN
    *Windy Gridworld (Example 6.5 in the book).
    *Undiscounted finite problem.
    *Using SARSA (On-Policy TD Control) to estimate Optimal Q.

"""


''' PARAMS '''
alpha = 0.5  # Learning rate
gamma = 0.1  # Discounting rate
epsilon = 0.1  # Greedy rate


''' DIRECTIONS '''
ARROWS = ['←','→','↑','↓']
ACTIONS = np.array([
    [0,-1], # LEFT
    [0,1],  # RIGHT
    [-1,0], # UP
    [1,0]   # DOWN
])


''' ENVIRONMENT '''
grid, grid_shape, start, goal = generate_grid_6_5()  # Example 6.5 in the book.


def epsilon_greedy_policy(epsilon=0.1):
    assert epsilon >= 0 and epsilon <= 1, '0 <= epsilon <= 1'

    greed = np.random.uniform(low=0.0, high=1.0)
    if greed <= epsilon:
        return np.random.choice(ACTIONS)
    else:
        pass


def sarsa(num_episodes=10):  # SARSA Learning (On-Policy TD Control).
    assert num_episodes > 0, 'NUMBER OF EPISODES CANNOT BE A NON POSITIVE NUMBER.'

    Q = np.zeros(shape=grid_shape+(ACTIONS.size,), dtype=np.float64)
    #pi = np.zeros(shape=grid_shape, dtype=np.int8)

    for episode in tqdm(range(num_episodes)):
        pass

    return Q


def visualize(Q=None, show=False, save=False):  # Visualize optimal Q and optimal pathway from start to goal position..
    assert Q is not None, 'INVALID INPUT, CANNOT VISUALIZE!'

    import matplotlib.pyplot as plt

    if show:
        plt.show()
    if save:
        plt.savefig('windy_gridworld.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    pass
