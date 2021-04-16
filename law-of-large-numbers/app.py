#!/Users/huynguyen/miniforge3/envs/math/bin/python3

"""
    NAME: HUY NGUYEN
    - Experiment with law of large numbers
"""

import numpy as np
import matplotlib.pyplot as plt

""" GENERATE GAUSS DATA """
n = 100000 # Data size
truth_mu = 5.0
variance = 1.5
population_data = np.random.normal(loc=truth_mu, scale=variance, size=n)

""" PARAMS """
error_rate = 10e-4
num_steps = 100
sample_size = 100
increasing_step = 100
estimated_mu = 0
alpha = 0.1  # Learning rate

# Uniformly picking out sample from population dataset to approximate the mean
for step in range(num_steps):
    sample_data = np.array([population_data[np.random.choice(n)] for x in range(sample_size)])
    estimated_mu = sample_data.mean()
    print(f'TRUTH: {truth_mu}; ESTIMATE: {estimated_mu} -> ERROR: {np.abs(truth_mu-estimated_mu)}')
    sample_size += 1000

    if np.abs(estimated_mu-truth_mu) <= error_rate:
        print(f'--> Converged in {step+1} steps with error_rate = {error_rate}.')
        break
