#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    Name: HUY NGUYEN
    *This is an example using Monte Carlo method to calculate integral of sin(x) from 0 to pi. The expected value should be 2.

"""


import numpy as np
#import matplotlib.pyplot as plt

''' PARAMS '''
num_samples = 10000

# Generate uniform data for MC method.
samples = np.sin(np.random.uniform(low=0, high=np.pi, size=num_samples))

# Apply MC method.
integral = ((np.pi-0)/float(num_samples))*samples.sum()

print(f'ANSWER = {integral}')
