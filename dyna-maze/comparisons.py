#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""
    
    JUST FOR COMPARISON WITH DIFFERENCE N.

"""


import matplotlib
import matplotlib.pyplot as plt
from dyna_maze import dyna_q


''' MAIN '''
if __name__ == "__main__":
    ''' PARAMS '''
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discounting rate
    epsilon = 0.1  # Greedy rate 
    num_episodes = 50 
    n = [0,5,50]  # n-step update

    for test_n in n:
        _, num_steps = dyna_q(num_episodes, alpha, gamma, epsilon, test_n)
        plt.plot(num_steps, label=f'n={test_n}')
    plt.grid()
    plt.savefig('examples/comparisons.png')
    plt.close()
