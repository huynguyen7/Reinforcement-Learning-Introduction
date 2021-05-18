"""

    *Name: HUY NGUYEN
    *Windy Gridworld
        

"""


import numpy as np


''' PARAMS '''
alpha = 1e-2  # Learning rate
gamma = 0.99  # Discounting rate
epsilon = 0.1 # Greedy rate


""" DIRECTIONS """
ARROWS = ['←','→','↑','↓']
ACTIONS = np.array([
    [0,-1], # LEFT
    [0,1],  # RIGHT
    [-1,0], # UP
    [1,0]   # DOWN
])



def epsilon_greedy(epsilon):
    greed = np.random.uniform(low=0.0, high=1.0)
    if greed < epsilon:
        return
    else:
        pass


def visualize():
    import matplotlib.pyplot as plt
