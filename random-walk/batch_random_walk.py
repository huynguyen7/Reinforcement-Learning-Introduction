#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: Huy Nguyen
    *Random Walk Problem (Example 6.3 in the book).
    *Comparison between Batch Monte Carlo with Batch Temporal Difference Learning approach.
    *In batch form, TD(0) is faster than Monte Carlo methods because it computes the true certainty-equivalence estimate.

"""


import numpy as np
from tqdm import tqdm


np.random.seed(1)  # Deterministic seed.


''' PARAMS '''
alpha = 1e-4  # Learning rate.
gamma = 1.0  # Discounting rate.


def pi():  # 0 is LEFT, 1 is RIGHT.
    return np.random.randint(0,1+1)  # Uniform 


def rmse(estimated_V, truth_V):
    return np.sqrt(((estimated_V-truth_V)**2).sum()/5.0)  # Divided by 5 since we only consider [A,B,C,D,E], not accounting terminal states.


def batch_monte_carlo(num_episodes=100, truth_V=None, error_interval=10, error_threshold=1e-3):  # Batch MC Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1.0
    rmse_mc = []  # Just for plotting purposes
    batch_history = []

    i = 0  # Used for plotting RMSE.
    for episode in tqdm(range(num_episodes)):
        if i % error_interval == 0:
            rmse_mc.append(rmse(V, truth_V))
        i += 1

        s = 3  # Start at the middle.
        batch = []
        while True:
            batch.append(s)
            action = pi()
            if action == 0:  # LEFT
                s_prime = s-1
            else:  # RIGHT
                s_prime = s+1

            s = s_prime
            # Check if reaching the terminal states.
            if s == 0 or s == 6:
                G = 1.0 if s == 6 else 0.0
                break

        batch_history.append(batch)
        while True:  # Keep running until the algorithm converges.
            for batch in batch_history:
                tmp_V = V.copy()  # Deep copy
                for s in batch:
                    # Every-visit MC update.
                    tmp_V[s] = tmp_V[s] + alpha*(G-V[s])
            if np.sum(np.abs(V-tmp_V)) <= error_threshold:  # Check for convergence.
                break
            V = tmp_V

    rmse_mc.append(rmse(V, truth_V))
    return V, rmse_mc


def batch_tabular_temporal_difference(num_episodes=100, truth_V=None, error_interval=10, error_threshold=1e-3):  # Batch TD(0) Update
    assert num_episodes > 0, 'NUM_EPISODES CANNOT BE A NON-POSITIVE NUMBER.'

    V = np.zeros(7, dtype=np.float64)
    V[6] = 1.0
    rmse_td = []  # Just for plotting purposes
    batch_history = []

    i = 0  # Used for plotting RMSE.
    for episode in tqdm(range(num_episodes)):
        if i % error_interval == 0:
            rmse_td.append(rmse(V, truth_V))
        i += 1

        s = 3  # Start at the middle.
        batch = []
        while True:
            action = pi()
            if action == 0:  # LEFT
                s_prime = s-1
            else:  # RIGHT
                s_prime = s+1
            batch.append((s, s_prime))

            s = s_prime
            if s == 0 or s == 6:  # Reach terminal states.
                returns = 1.0 if s == 6 else 0.0
                break
        
        reward = 0  # All rewards are 0
        batch_history.append(batch)
        while True:  # Keep running until the algorithm converges.
            tmp_V = V.copy()  # Deep copy
            for batch in batch_history:
                for (s, s_prime) in batch:
                    # TD Update
                    tmp_V[s] = tmp_V[s] + alpha*(reward + gamma*V[s_prime] - V[s])
            if np.sum(np.abs(V-tmp_V)) <= error_threshold:  # Check for convergence.
                break
            V = tmp_V

    rmse_td.append(rmse(V, truth_V))
    return V, rmse_td


def plot_figure(show=False, save=False, num_episodes=None, truth_V=None, estimated_V_td=None, estimated_V_mc=None, rmse_td=None, rmse_mc=None, error_interval=None):
    assert estimated_V_td is not None and estimated_V_mc is not None \
            and truth_V is not None and rmse_td is not None and rmse_mc is not None, 'CANNOT PLOT!'

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rc('font', size=6)

    plt.subplot(1,2,1)
    plt.plot(estimated_V_td, label='Batch TD')
    plt.plot(estimated_V_mc, label='Batch MC')
    plt.plot(truth_V, label='TRUTH')
    plt.grid()
    plt.xticks(ticks=np.arange(0,7,1), labels=['0','A','B','C','D','E','1'])
    plt.xlabel('State', fontsize=13)
    plt.ylabel('V', fontsize=13).set_rotation(0)
    plt.legend()
    plt.title(f'TD(0) VS MC WITH {num_episodes} EPS')

    plt.subplot(1,2,2)
    plt.plot(rmse_td, label='Batch TD')
    plt.plot(rmse_mc, label='Batch MC')
    plt.grid()
    plt.xlabel('EPISODE', fontsize=13)
    plt.ylabel('RMSE', fontsize=13).set_rotation(0)
    plt.xticks(ticks=np.arange(0,int(num_episodes/error_interval)+1,1, dtype=np.int64), labels=np.linspace(0,num_episodes,int((num_episodes/error_interval)+1), dtype=np.int64))
    plt.legend()
    plt.title(f'RMSE WITH {num_episodes} EPS')

    if show:
        plt.show()
    if save:
        plt.savefig('./batch_td_vs_mc.png')
        plt.close()


''' MAIN '''
if __name__ == "__main__":
    # Just for comparing purposes.
    truth_V = np.linspace(0,1,7)
    error_interval = 10

    # Run TD(0) and MC.
    num_episodes = 100  # Number of episodes
    print(f'ALPHA = {alpha}, GAMMA = {gamma}')
    print('--> RUNNING TD(0)..')
    estimated_V_td, rmse_td = batch_tabular_temporal_difference(num_episodes, truth_V, error_interval)
    print('--> RUNNING MC..')
    estimated_V_mc, rmse_mc = batch_monte_carlo(num_episodes, truth_V, error_interval)

    # Plot
    plot_figure(show=True, save=False, num_episodes=num_episodes, truth_V=truth_V, estimated_V_td=estimated_V_td, estimated_V_mc=estimated_V_mc, rmse_td=rmse_td, rmse_mc=rmse_mc, error_interval=error_interval)
