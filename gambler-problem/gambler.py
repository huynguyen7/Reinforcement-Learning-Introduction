#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: HUY NGUYEN
    *Example 4.3 in the book.
    *VALUE ITERATION -> FIND THE OPTIMAL POLICY THROUGH OPTIMAL STATE VALUES.
    *Undiscounted, episodic, finite MDP problem.
    *If coin~head, the gambler earns money based on their stakes.
    *The state is the gambler’s capital, s ∈ {1, 2, . . . , CAPITAL-1}. The actions are stakes, a ∈ {0, 1, . . . , min(s, CAPITAL − s)}. The reward is zero on all transitions except those on which the gambler reaches his goal (Gambler has `CAPITAL` in the wallet), it is +1.
    *The state-value function then gives the probability of winning from each state. A policy is a mapping from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal. Let p_h denote the probability of the coin coming up heads. If p_h is known, then the entire problem is known and it can be solved, for instance, by value iteration.
    *NOTES: This implementation does not handle if multiple policy have the same values. In another word, we could have many optimal policies for such a state; but this implementation chose to show up to one optimal action.
    -> THE POLICY IS OPTIMAL, BUT NOT UNIQUE!
 
"""


import numpy as np

''' PARAMS '''
gamma = 1  # Discounting rate
alpha = 1e-9  # Error threshold
CAPITAL = 100  # Goal
p_h = 0.7  # Probability of head.
max_floating = 10


def reward(a):
    return 1 if a == CAPITAL else 0


def value_iteration(num_steps=100):
    values = np.zeros(CAPITAL+1)  # Optimal state-values
    for step in range(num_steps):
        new_values = []
        delta = np.NINF  # Negative infinity
        for s in range(1, CAPITAL):
            old_value = values[s]
            '''
                rewards[s+a], rewards[s-a] are immediate rewards.
                V[s+a], V[s-a] are values of the next states.
                Bellman Equation: The EXPECTED VALUE of your action is the sum of immediate rewards and the value of the next state.
            '''
            for a in range(1, min(s, CAPITAL-s)+1):  # Minimum bet is 1, maximum bet is min(s, CAPITAL-s).
                new_values.append(round(p_h*(reward(s+a) + gamma*values[s+a]) + (1-p_h)*(reward(s-a) + gamma*values[s-a]), max_floating))
            values[s] = max(new_values)  # Get optimal state-value
            delta = max(delta, np.abs(values[s]-old_value))

        if delta < alpha:
            print(f'--> The algorithm converged in {step+1} steps with error_threshold = {alpha}.')
            break

    policies = np.zeros(CAPITAL+1, dtype=np.int8)  # Optimal policies
    for s in range(1, CAPITAL):
        new_values = np.zeros(CAPITAL+1)
        for a in range(1, min(s, CAPITAL-s)+1):  # Minimum bet is 1, maximum bet is min(s, CAPITAL-s).
            new_values[a] = round(p_h*(reward(s+a) + gamma*values[s+a]) + (1-p_h)*(reward(s-a) + gamma*values[s-a]), max_floating)
        policies[s] = np.argmax(new_values)  # Get optimal policy

    return values, policies


def visualize(values, policies, log=True, plot=True):
    if values is None or policies is None:
        print('INVALID INPUT.')
        return
    
    if log:
        print('\t\t----VALUES----')
        print(values)
        print('\t\t----POLICIES----')
        print(policies)

    if plot:
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.grid()
        plt.xlabel('Capital')
        plt.ylabel('Value Estimates')
        plt.title('State (Capital) / Optimal Value Estimates')
        plt.plot(np.arange(0,CAPITAL,1), values[:CAPITAL])
        plt.subplot(1,2,2)
        plt.grid()
        plt.xlabel('Capital')
        plt.ylabel('Final Policy')
        plt.title('State (Capital) / Final Policy (Optimal Policy)')
        plt.bar(np.arange(0,CAPITAL,1), policies[:CAPITAL])
        plt.show()

values, policies = value_iteration(num_steps=100)
visualize(
    values,
    policies,
    log=False,
    plot=True)
