#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: Huy Nguyen
    *Random Walk Problem (Example 6.2 in the book).
    *Comparison between Monte Carlo with Temporal Difference Learning approach.

"""


import numpy as np
from tqdm import tqdm


''' PARAMS '''
alpha = 0.1  # Learning rate.
gamma = 1.0  # Discounting rate.


def pi():  # 0 is LEFT, 1 is RIGHT.
    return np.random.randint(0,1+1)  # Uniform 


def monte_carlo(num_episodes=None):  # MC Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1

    for episode in tqdm(range(num_episodes)):
        s = 3  # Start at the middle.
        history = []
        while True:
            history.append(s)

            action = pi()
            if action == 0:  # LEFT
                s_prime = s-1
            else:  # RIGHT
                s_prime = s+1

            s = s_prime
            # Check if reaching the terminal states.
            if s == 0:
                G = 0.0
                break
            elif s == 6:
                G = 1.0
                break

        for s in history:  # Every-visit update.
            V[s] = V[s] + alpha*(G - V[s])
    return V


def tabular_temporal_difference(num_episodes=None):  # TD(0) Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1

    for episode in tqdm(range(num_episodes)):
        s = 3  # Start at the middle.
        while True:
            action = pi()
            if action == 0:  # LEFT
                s_prime = s-1
            else:  # RIGHT
                s_prime = s+1

            reward = 0  # All rewards are 0
            # TD Update
            V[s] = V[s] + alpha*(reward + gamma*V[s_prime] - V[s])
            
            s = s_prime
            if s == 0 or s == 6:  # Reach terminal states.
                break
    return V


def plot_figure(show=False, save=False, num_episodes=None, truth_V=None, estimated_V_td=None, estimated_V_mc=None):
    assert estimated_V_td is not None and estimated_V_mc is not None and truth_V is not None, 'CANNOT PLOT!'

    import matplotlib.pyplot as plt

    plt.plot(estimated_V_td, label='TD')
    plt.plot(estimated_V_mc, label='MC')
    plt.plot(truth_V, label='TRUTH')
    plt.grid()
    plt.xticks(ticks=np.arange(0,7,1), labels=['0','A','B','C','D','E','1'])
    plt.xlabel('State', fontsize=13)
    plt.ylabel('V', fontsize=18).set_rotation(0)

    plt.legend()
    plt.title(f'TD(0) VS MC WITH {num_episodes} EPS')

    if show:
        plt.show()
    if save:
        plt.savefig('./td_vs_mc.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    # Run TD(0) and MC.
    num_episodes = 100  # Number of episodes
    print('--> RUNNING TD(0)..')
    estimated_V_td = tabular_temporal_difference(num_episodes)
    print('--> RUNNING MC..')
    estimated_V_mc = monte_carlo(num_episodes)

    # Plot
    truth_V = np.linspace(0,1,7)  # Just for comparing purposes.
    plot_figure(show=True, save=False, num_episodes=num_episodes, truth_V=truth_V, estimated_V_td=estimated_V_td, estimated_V_mc=estimated_V_mc)
