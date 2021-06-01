#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *author: Huy Nguyen
    *Dyna Maze (Example 8.1 in the book).
    *Discounted finite problem.
    *Discrete action space.
    *Using Tabular n-step Dyna-Q (On-Policy TD Control) to estimate Optimal Q -> Optimal Policies.

"""


from Environment import generate_maze_8_2, ACTIONS, ARROWS
from tqdm import tqdm
import numpy as np
import random


maze, maze_shape, maze_max_steps, start, goal, default_reward, terminal_reward = generate_maze_8_2()


def interact(s, a):  # Interact with the maze (environemt).
    x = s[0]+a[0]
    y = s[1]+a[1]

    if x < 0 or x >= maze_shape[0] or y < 0 or y >= maze_shape[1] or maze[x, y] == 1:  # Reach obstable or offgrid.
        return default_reward, s
    elif x == goal[0] and y == goal[1]:  # Reach goal!
        return terminal_reward, (x,y)
    else:
        return default_reward, (x,y)


def random_policy():  # Exploratory policy / Behaviour policy.
    return np.random.randint(0, ACTIONS.shape[0])


def greedy_policy(s=None, Q=None):  # Target policy
    assert s is not None and Q is not None

    state_action_values = Q[s[0],s[1],:]
    max_state_action_values = state_action_values.max()
    return np.random.choice([action for action, state_action_value in enumerate(state_action_values) if state_action_value == max_state_action_values])


def epsilon_greedy_policy(epsilon=0.1, s=None, Q=None):
    assert epsilon >= 0.0 and epsilon <= 1.0, '0 <= epsilon <= 1'
    assert s is not None and Q is not None

    greed = np.random.uniform(low=0.0, high=1.0)
    if greed < epsilon:  # Explore
        return random_policy()
    else:  # Exploit
        return greedy_policy(s, Q)


def dyna_q(num_episodes=100, alpha=0.1, gamma=0.95, epsilon=0.1, n=5):  # Return optimal Q, optimal pi.
    assert num_episodes > 0 and n >= 0, 'INVALID INPUT'

    Q = np.zeros(shape=maze_shape+(ACTIONS.shape[0],), dtype=np.float64)
    model = dict()  # Keep track of the latest state-action pair that has been experienced. Simply, it will return last observed (reward, s_prime) tuple.

    num_steps = []  # Just for plotting purpose

    for episode in tqdm(range(num_episodes)):
        s = (start[0],start[1])
        steps_taken = 0
        while True:
            steps_taken += 1
            a = epsilon_greedy_policy(epsilon, s, Q)  # Get action
            reward, s_prime = interact(s, ACTIONS[a])  # Take action (real experience)

            # one-step tabular Q-learning Update (Direct RL).
            Q[s[0],s[1],a] = Q[s[0],s[1],a] + alpha*(reward + gamma*Q[s_prime[0],s_prime[1],:].max() - Q[s[0],s[1],a])

            # Model-learning.
            if model.get(s) is None:
                model[s] = dict()
                
            model[s][a] = (reward, s_prime)

            # n-step tabular Q-learning.
            for step in range(n):
                # Get sample from the model.
                s_ = random.choice(list(model.keys()))
                a_ = random.choice(list(model[s_].keys()))
                (reward_,s_prime_) = model[s_][a_]  # Returns the last-observed next reward and next state (simulated experience).

                # N-step Q-learning Update (Direct RL).
                Q[s_[0],s_[1],a_] = Q[s_[0],s_[1],a_] + alpha*(reward_ + gamma*Q[s_prime_[0],s_prime_[1],:].max() - Q[s_[0],s_[1],a_])

            s = s_prime  # Move to next state
            if (s[0] == goal[0] and s[1] == goal[1]) or steps_taken >= maze_max_steps:
                break

        num_steps.append(steps_taken)

    return Q, num_steps


def map_policy_to_arrow(optimal_pi=None):
    assert optimal_pi is not None
    return np.array([list(map(lambda x: ARROWS[x], optimal_pi[i])) for i in range(maze_shape[0])])


def visualize(Q=None, optimal_pi=None, num_steps=None, n=0, num_episodes=1, alpha=0.1, gamma=0.95, epsilon=0.95, show=False, save=False):
    assert Q is not None and optimal_pi is not None and num_steps is not None

    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    matplotlib.rc('font', size=8)

    plt.subplot(2,1,1)
    plt.plot(num_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Num steps')
    plt.grid()

    plt.subplot(2,1,2)
    labels = map_policy_to_arrow(optimal_pi)
    labels[goal[0],goal[1]] = 'G'  # Plot 'G' as goal.
    for i in range(maze.shape[0]):  # Plot '*' as obstables.
        for j in range(maze.shape[1]):
            if maze[i,j] == 1:
                labels[i,j] = '*'  # Obstable.
    sns.heatmap(optimal_pi, annot=labels, fmt='')
    plt.title('OPTIMAL PI')

    if show:
        plt.show()
    if save:
        plt.savefig(f'examples/{num_episodes}_eps_{alpha}_lr_{gamma}_gamma_{epsilon}_epsilon_{n}_n_dyna_maze.png')
        plt.close()

''' MAIN '''
if __name__ == "__main__":
    ''' PARAMS '''
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discounting rate
    epsilon = 0.1  # Greedy rate 
    num_episodes = 3 
    n = 50  # n-step update

    Q, num_steps = dyna_q(num_episodes, alpha, gamma, epsilon, n)
    optimal_pi = np.argmax(Q, axis=-1)

    visualize(Q, optimal_pi, num_steps, n, num_episodes, \
            alpha, gamma, epsilon, \
            show=False, save=True)
