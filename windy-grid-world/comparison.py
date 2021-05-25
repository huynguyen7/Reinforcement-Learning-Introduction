#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: HUY NGUYEN
    *Windy Gridworld (Example 6.5 in the book).
    *Undiscounted finite problem.
    *Discrete action space.
    *Comparison between SARSA and Q-Learning.

"""

from sarsa_windy_gridworld import sarsa
from q_learning_windy_gridworld import q_learning


def visualize(num_steps_sarsa=None, num_steps_q_learning=None, num_episodes=1, show=False, save=False):  # Visualize optimal Q and optimal pathway from start to goal position.
    assert num_steps_sarsa is not None and num_steps_q_learning is not None, 'INVALID INPUT, CANNOT VISUALIZE!'
    assert num_episodes > 0, 'NUMBER OF EPISODES CANNOT BE A NON POSITIVE NUMBER.'

    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    matplotlib.rc('font', size=8)

    plt.title(f'NUMBER OF STEPS TO REACH GOAL')
    plt.xlabel('Episodes')
    plt.ylabel('Num steps')
    plt.plot(num_steps_sarsa, label='SARSA')
    plt.plot(num_steps_q_learning, label='Q-Learning')
    plt.legend()
    plt.grid()

    if show:
        plt.show()
    if save:
        plt.savefig(f'examples/sarsa_vs_q_learning_{num_episodes}_eps.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    ''' PARAMS '''
    alpha = 0.5  # Learning rate
    gamma = 1.0  # Discounting rate
    epsilon = 0.1  # Greedy rate 
    num_episodes = 200
    _, num_steps_sarsa = sarsa(num_episodes, alpha, gamma, epsilon)
    _, num_steps_q_learning = q_learning(num_episodes, alpha, gamma, epsilon)
    visualize(num_steps_sarsa, num_steps_q_learning, num_episodes, save=True)
