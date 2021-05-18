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


def rmse(estimated_V, truth_V):
    return np.sqrt(((estimated_V-truth_V)**2).sum()/5.0)  # Divided by 5 since we only consider [A,B,C,D,E], not accounting terminal states.


def monte_carlo(num_episodes=None, truth_V=None, error_interval=None):  # MC Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1.0
    rmse_mc = []  # Just for plotting purposes

    i = 0  # Used for plotting RMSE.
    for episode in tqdm(range(num_episodes)):
        if i % error_interval == 0:
            rmse_mc.append(rmse(V, truth_V))
        i += 1

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
    rmse_mc.append(rmse(V, truth_V))
    return V, rmse_mc


def tabular_temporal_difference(num_episodes=None, truth_V=None, error_interval=None):  # TD(0) Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1.0
    rmse_td = []  # Just for plotting purposes

    i = 0  # Used for plotting RMSE.
    for episode in tqdm(range(num_episodes)):
        if i % error_interval == 0:
            rmse_td.append(rmse(V, truth_V))
        i += 1

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
    rmse_td.append(rmse(V, truth_V))
    return V, rmse_td


def plot_figure(show=False, save=False, num_episodes=None, truth_V=None, estimated_V_td=None, estimated_V_mc=None, rmse_td=None, rmse_mc=None, error_interval=None):
    assert estimated_V_td is not None and estimated_V_mc is not None \
            and truth_V is not None and rmse_td is not None and rmse_mc is not None, 'CANNOT PLOT!'

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=6)

    plt.subplot(1,2,1)
    plt.plot(estimated_V_td, label='TD')
    plt.plot(estimated_V_mc, label='MC')
    plt.plot(truth_V, label='TRUTH')
    plt.grid()
    plt.xticks(ticks=np.arange(0,7,1), labels=['0','A','B','C','D','E','1'])
    plt.xlabel('State', fontsize=13)
    plt.ylabel('V', fontsize=13).set_rotation(0)
    plt.legend()
    plt.title(f'TD(0) VS MC WITH {num_episodes} EPS')

    plt.subplot(1,2,2)
    plt.plot(rmse_td, label='TD')
    plt.plot(rmse_mc, label='MC')
    plt.grid()
    plt.xlabel('EPISODE', fontsize=13)
    plt.ylabel('RMSE', fontsize=13).set_rotation(0)
    plt.xticks(ticks=np.arange(0,int(num_episodes/error_interval)+1,1, dtype=np.int64), labels=np.linspace(0,num_episodes,int((num_episodes/error_interval)+1), dtype=np.int64))
    plt.legend()
    plt.title(f'RMSE WITH {num_episodes} EPS')

    if show:
        plt.show()
    if save:
        plt.savefig('./td_vs_mc.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    # Just for comparing purposes.
    truth_V = np.linspace(0,1,7)
    error_interval = 10

    # Run TD(0) and MC.
    print(f'ALPHA = {alpha}, GAMMA = {gamma}')
    num_episodes = 100  # Number of episodes
    print('--> RUNNING TD(0)..')
    estimated_V_td, rmse_td = tabular_temporal_difference(num_episodes, truth_V, error_interval)
    print('--> RUNNING MC..')
    estimated_V_mc, rmse_mc = monte_carlo(num_episodes, truth_V, error_interval)

    # Plot
    plot_figure(show=True, save=False, num_episodes=num_episodes, truth_V=truth_V, estimated_V_td=estimated_V_td, estimated_V_mc=estimated_V_mc, rmse_td=rmse_td, rmse_mc=rmse_mc, error_interval=error_interval)
