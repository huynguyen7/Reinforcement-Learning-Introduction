#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: HUY NGUYEN
    *Windy Gridworld (Example 6.5 in the book).
    *Undiscounted finite problem.
    *Discrete action space.
    *Using SARSA (On-Policy TD Control) to estimate Optimal Q.

"""


from Environment import generate_grid_6_5, ACTIONS, ARROWS
from tqdm import tqdm
import numpy as np


''' ENVIRONMENT '''
grid, grid_shape, start, goal, default_reward, terminal_reward = generate_grid_6_5()  # Example 6.5 in the book.


def interact(s=None, s_prime=None, wind_level=None):  # Return next state, reward
    assert s is not None and s_prime is not None and wind_level is not None

    x = s_prime[0]-wind_level
    y = s_prime[1]

    if x >= grid_shape[0] and y >= grid_shape[1]:
        return (np.array([grid_shape[0]-1,grid_shape[1]-1], dtype=np.int64), default_reward)
    elif x >= grid_shape[0] and y < 0:
        return (np.array([grid_shape[0]-1,0], dtype=np.int64), default_reward)
    elif x < 0 and y >= grid_shape[1]:
        return (np.array([0,grid_shape[1]-1], dtype=np.int64), default_reward)
    elif x < 0 and y < 0:
        return (np.array([0,0], dtype=np.int64), default_reward)
    elif x >= grid_shape[0] or x < 0:
        return (np.array([grid_shape[0]-1,y], dtype=np.int64), default_reward) if x >= grid_shape[0] else (np.array([0,y], dtype=np.int64), default_reward)
    elif y >= grid_shape[1] or y < 0:
        return (np.array([x,grid_shape[1]-1], dtype=np.int64), default_reward) if y >= grid_shape[1] else (np.array([x,0], dtype=np.int64), default_reward)
    elif x == goal[0] and y == goal[1]:
        return (np.array([x,y], dtype=np.int64), terminal_reward)
    else:
        return (np.array([x,y], dtype=np.int64), default_reward)


def random_policy():
    return np.random.randint(0, ACTIONS.shape[0])


def greedy_policy(s=None, Q=None):
    assert s is not None and Q is not None

    state_action_values = Q[s[0],s[1],:]
    max_state_action_values = state_action_values.max()
    return np.random.choice([action for action, state_action_value in enumerate(state_action_values) if state_action_value == max_state_action_values])


def epsilon_greedy_policy(epsilon=0.1, s=None, Q=None):
    assert epsilon >= 0.0 and epsilon <= 1.0, '0 <= epsilon <= 1'
    assert s is not None and Q is not None

    greed = np.random.uniform(low=0.0, high=1.0)
    if greed <= epsilon:  # Explore
        return random_policy()
    else:  # Exploit
        return greedy_policy(s, Q)


def sarsa(num_episodes=10, alpha=0.5, gamma=1.0, epsilon=0.1):  # SARSA Learning (On-Policy TD Control).
    assert num_episodes > 0, 'NUMBER OF EPISODES CANNOT BE A NON POSITIVE NUMBER.'

    Q = np.zeros(shape=grid_shape+(ACTIONS.shape[0],), dtype=np.float64)
    num_steps = []  # Just for plotting purpose!

    for episode in tqdm(range(num_episodes)):
        s = np.array([start[0], start[1]], dtype=np.int64)
        a = epsilon_greedy_policy(epsilon, s, Q)
        step = 0
        while True:
            step += 1
            wind_level = grid[s[0], s[1]]  # Get wind level at the current state.
            (s_prime, reward) = interact(s, s+ACTIONS[a], wind_level)
            a_prime = epsilon_greedy_policy(epsilon, s_prime, Q)
            #print(f's = {s}, a = {a}, s_prime = {s_prime}, a_prime = {a_prime}, wind_level = {wind_level}')

            # SARSA Update for Control problem (Estimate Q).
            Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(reward + gamma*Q[s_prime[0],s_prime[1],a_prime] - Q[s[0],s[1],a])

            # Update the current state and action.
            s = s_prime
            a = a_prime

            if s[0] == goal[0] and s[1] == goal[1]:  # Reach terminal state.
                break
        num_steps.append(step)

    return Q, num_steps


def map_policy_to_arrow(optimal_pi=None):
    assert optimal_pi is not None
    return np.array([list(map(lambda x: ARROWS[x], optimal_pi[i])) for i in range(grid_shape[0])])


def visualize(optimal_pi=None, num_steps=None, num_episodes=1, show=False, save=False):  # Visualize optimal Q and optimal pathway from start to goal position.
    assert optimal_pi is not None and num_steps is not None, 'INVALID INPUT, CANNOT VISUALIZE!'
    assert num_episodes > 0, 'NUMBER OF EPISODES CANNOT BE A NON POSITIVE NUMBER.'

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    matplotlib.rc('font', size=8)

    plt.subplot(2,2,1)
    sns.heatmap(grid)
    plt.title('WINDY GRIDWORLD')

    plt.subplot(2,2,2)
    labels = map_policy_to_arrow(optimal_pi)
    sns.heatmap(optimal_pi, annot=labels, fmt='')
    plt.title('OPTIMAL PI')

    plt.subplot(2,2,3)
    plt.title(f'OPTIMAL PATH FROM {start} TO {goal}')
    optimal_path = np.zeros(grid_shape, dtype=np.int64)
    s = np.array([start[0],start[1]])
    while True:
        optimal_path[s[0],s[1]] = 1
        wind_level = grid[s[0], s[1]]  # Get wind level at the current grid.
        a = greedy_policy(s, Q)
        (s_prime, _) = interact(s, s+ACTIONS[a], wind_level)

        # Update the current state
        s = s_prime

        if s[0] == goal[0] and s[1] == goal[1]:
            break
    optimal_path[goal[0],goal[1]] = 1
    sns.heatmap(optimal_path)

    plt.subplot(2,2,4)
    plt.title(f'NUMBER OF STEPS TO REACH GOAL')
    plt.xlabel('Episodes')
    plt.ylabel('Num steps')
    plt.grid()
    plt.plot(num_steps)

    if show:
        plt.show()
    if save:
        plt.savefig(f'examples/sarsa_{num_episodes}_eps.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    ''' PARAMS '''
    alpha = 0.5  # Learning rate
    gamma = 1.0  # Discounting rate
    epsilon = 0.1  # Greedy rate 
    num_episodes = 200
    Q, num_steps = sarsa(num_episodes, alpha, gamma, epsilon)

    # Get optimal policies through estimated optimal Q with SARSA.
    optimal_pi = np.argmax(Q, axis=-1)

    visualize(optimal_pi, num_steps, num_episodes, save=True)
