#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    Name: HUY NGUYEN
    *This is an example using Monte Carlo method to calculate pi number.
    *The idea is generating a large number of samples, then decide the number of data points fit in the circle and the number of data points fit in the square.
    * f(x,y) = x^2 + y^2 = r^2  # Circle func
    * Circle area = pi*R^2
    * Square area = (R*2)^2

    (pi*R^2) / (4*R^2) = circle_area/square_area
    pi = 4*(circle_area/square_area)
    -> pi = 4*(num_points_circle/num_points_square)
    ._____.
    |/   \|
    {  .  }   => The circle should lie inside the square
    |\___/|

"""


import numpy as np
#import matplotlib.pyplot as plt

''' PARAMS '''
num_samples = 100000
r = 1  # radius
plot = False

# Generate uniform data for MC method.
samples = []
for i in range(num_samples):
    samples.append(np.random.uniform(low=-r, high=r, size=2))
samples = np.array(samples)

# Apply MC method.
num_pts_circle = sum(list(map(lambda p: 0 if p[0]**2 + p[1]**2 > r**2 else 1, samples)))  # Applied MapReduce
num_pts_square = num_samples
pi = 4*(num_pts_circle/num_pts_square)

print(f'PI = {pi}')
