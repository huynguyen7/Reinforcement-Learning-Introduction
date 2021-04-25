#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: HUY NGUYEN
    *Policy iteration to approximate optimal pi.
    *Figure 4.1
    *NOTES: This implementation does not handle if multiple policy have the same values. In another word, we could have many optimal policies for such a state; but this implementation chose to show up to one optimal action.

"""


import numpy as np
#import matploblib.pyplot as plt


""" PARAMS """
gamma = 1  # Discounting rate
N = 4
default_rewards = -1
offgrid_rewards = -1
terminal_rewards = 0
alpha = 1e-4  # Error rate

""" DIRECTIONS """
ARROWS = ['←','→','↑','↓']
ACTIONS = np.array([
    [0,-1], # LEFT
    [0,1],  # RIGHT
    [-1,0], # UP
    [1,0]   # DOWN
])
#ACTION_PROB = 0.25


def interact(current_state, next_state):  # Return next_state, reward
    x = next_state[0]
    y = next_state[1]

    if x >= N or x < 0 or y >= N or y < 0:  # OFF-GRID STATES
        return current_state, offgrid_rewards
    elif (x == 0 and y == 0) or (x == N-1 and y == N-1):  # TERMINAL STATES
        return current_state, terminal_rewards
    else:  # DEFAULT STATES
        return next_state, default_rewards

def simulate(num_steps, log=False):
    values = np.zeros(shape=(N, N), dtype=np.float64)
    policies = np.zeros(shape=(N, N), dtype=np.int8)

    for step in range(num_steps):
        # Policy Evaluation/Monte Carlo prediction
        while True:
            new_values = values.copy()  # Deep copy
            theta = np.NINF  # Negative infinity
            for i in range(N):
                for j in range(N):
                    current_state = np.array([i,j])
                    for action in ACTIONS:
                        next_state, reward = interact(current_state, current_state+action)
                        #new_values[i][j] += ACTION_PROB*(reward + gamma*values[next_state[0]][next_state[1]])
                        new_values[i][j] += reward + gamma*values[next_state[0]][next_state[1]]
                    values[i][j] = new_values[i][j]
                    theta = max(theta, np.abs(values-new_values).max())
            if theta < alpha:
                break

        # Policy Improvement/Monte Carlo control
        policy_stable = True
        for i in range(N):
            for j in range(N):
                current_action = policies[i][j]
                current_state = np.array([i,j])
                new_values = []
                for action in ACTIONS:
                    tmp_next_action = current_state+action
                    if tmp_next_action[0] >= N or tmp_next_action[0] < 0 or tmp_next_action[1] >= N or tmp_next_action[1] < 0:  # OFF-GRID STATES
                        new_values.append(np.NINF)
                    else:
                        next_state, reward = interact(current_state, tmp_next_action)
                        #new_values.append(ACTION_PROB*(reward + gamma*values[next_state[0]][next_state[1]]))
                        new_values.append(reward + gamma*values[next_state[0]][next_state[1]])
                policies[i][j] = np.argmax(np.array(new_values))
                if policy_stable and current_action != policies[i][j]:
                    policy_stable = False
                
        if policy_stable:
            break

    if log:
        print(f'--> CONVERGED IN {step+1} steps.')
        #print('----VALUES----')
        #print(values)
        print('----POLICIES----')
        print(policies)
        print('----ARROWS----')
        arrows_policies = [list(map(lambda x: ARROWS[x], policies[i])) for i in range(N)]
        for i in range(N):
            print(arrows_policies[i])
    return values, policies


simulate(
    num_steps=1000,
    log=True
)
