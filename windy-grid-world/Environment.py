import numpy as np

''' DIRECTIONS '''
ARROWS = ['←','→','↑','↓']
ACTIONS = np.array([
    [0,-1], # LEFT
    [0,1],  # RIGHT
    [-1,0], # UP
    [1,0]   # DOWN
])


# Generate grid world similar to example 6.5 in the book.
def generate_grid_6_5():
    grid_shape = (7,10)
    grid = np.zeros(shape=grid_shape, dtype=np.int64)
    # Level 1 wind
    grid[:,3:6] = 1
    grid[:,8] = 1
    # Level 2 wind
    grid[:,6:8] = 2

    # Start and goal positions
    start = [3,0]
    goal = [3,7]

    default_reward = -1.0
    terminal_reward = 0.0

    return grid, grid_shape, start, goal, default_reward, terminal_reward
