#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    *Name: Huy Nguyen
    *EXTREMELY NAIVE MLE!!
    *Coin Tossing Game.
    *Source: https://en.wikipedia.org/wiki/Maximum_likelihood_estimation

"""


import numpy as np
from tqdm import tqdm

# Calculate Maximum Likelihood of HHT.

num_trials = 1e-8
probs = np.arange(0,1,num_trials)
maximum_likelihood = np.NINF  # Negative infinity
max_H = 0
for prob in tqdm(probs):
    H = prob
    T = 1-prob
    HHT_likelihood = H*H*T
    if HHT_likelihood > maximum_likelihood:
        maximum_likelihood = HHT_likelihood
        max_H = H
print('Maximum Likelihood for HHT:', maximum_likelihood)
print('Best estimated probability for getting head:', max_H)

# Test if assumed prob is the better than the estimation
assumed_H = 0.5
HHT = assumed_H*assumed_H*(1-assumed_H)
print(f'-> Assumed P(H) = {assumed_H}, likelihood = {HHT}')
print(f'-> {HHT >= maximum_likelihood}')
