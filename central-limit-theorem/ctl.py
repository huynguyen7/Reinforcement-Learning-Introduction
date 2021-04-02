#!/Users/nguyenhuy/opt/miniconda3/envs/LinearAlgebra/bin/python3 -i

import numpy as np
import matplotlib.pyplot as plt

""" PARAMS """ 
N = 100  # Number of sample data.
M = 30  # Number of data in a sample.

data = np.zeros((N,M))
means = np.zeros((N,1))
""" PROOF OF CTL """
for n in range(N):
    data[n] = np.random.exponential(scale=5,size=M)
    means[n] = data[n].mean()

# Plot sample data
plt.hist(data.ravel(),bins=100)
plt.grid()
plt.show()

# Plot sample means
plt.hist(means, bins=30)
plt.grid()
plt.show()
