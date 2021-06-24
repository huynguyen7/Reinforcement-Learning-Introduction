"""

    @author: Huy Nguyen
    - Calculate Euler number with Monte Carlo.

"""

num_trials = 1e10
e = (1+1/num_trials)**(num_trials)
print('Euler number:', e)
