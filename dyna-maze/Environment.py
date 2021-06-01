import numpy as np


''' DIRECTIONS '''
ARROWS = ['←','→','↑','↓']
ACTIONS = np.array([
    [0,-1], # LEFT
    [0,1],  # RIGHT
    [-1,0], # UP
    [1,0]   # DOWN
])


def generate_maze_8_2():  # Generate maze similar to figure 8.2 in the book
    maze_shape = (6,9)
    maze = np.zeros(shape=maze_shape, dtype=np.int8)

    # Define obstacles, grid with value 1 is obstacle.
    maze[1:3+1,2] = 1
    maze[4,5] = 1
    maze[0:2+1,7] = 1

    start = [2,0]
    goal = [0,8]

    default_reward = 0
    terminal_reward = 1

    maze_max_steps = 10000
    
    return maze, maze_shape, maze_max_steps, start, goal, default_reward, terminal_reward
