#!/Users/huynguyen/miniforge3/envs/math/bin/python3


"""

    @author: Huy Nguyen
    *SOURCE: http://www.acme.byu.edu/wp-content/uploads/2016/12/Vol1B-MonteCarlo2-2017.pdf
    *Problem: Find the probability that a randomly chosen variable X from the standard normal distribution is greater than 3. Find P(X>3) given X ~ Standard Norm.
    *Importance sampling is one way to make Monte Carlo simulations converge much faster:
        - We choose a DIFFERENT DISTRIBUTION to sample our points to generate more important points.
        - With our example, we want to choose a distribution that would generate more numbers around 3 to get a more reliable estimate.
        - There is no correct choice for the importance distribution.
        - It may be possible to find the distribution that allows the simulation to converge the fastest, but oftentimes, we don’t need a perfect answer.

"""


import numpy as np
from scipy.stats import norm


np.random.seed(1)  # Deterministic seed, just for testing purpose!


X = 3  # Random variable


def h(x=None):
    assert x is not None
    return 1.0 if x > X else 0.0  # Predicate function


def f(x=None):  # Return pdf of standard norm (Target distribution)
    assert x is not None
    return norm.pdf(x)


# Choose normal distribution (u=1.0, std=3.0) for importance sampling for this example..
def g(x=None):  # Return Return pdf of importance sampling (different from Target distribution)
    assert x is not None
    return norm.pdf(x, loc=1.0,scale=3.0)


def monte_carlo(n=1000):  # Return estimate for P(X>3) with equation (1.2)
    assert n > 0, 'INVALID INPUT'

    samples = np.random.randn(n)
    # P(X>3) = (1/n) * h(x)  with x is the generated data from standard norm.
    h_x = np.array([h(x) for x in samples])
    return (1/n)*h_x.sum()


def importance_sampling(n=1000):  # Return estimate for P(X>3) with equation (1.4)
    assert n > 0, 'INVALID INPUT'

    samples = np.random.normal(loc=1.0, scale=3.0, size=n)
    h_x = np.array([h(x) for x in samples])
    #fraction fX(X) is called the importance gY (X) weight
    return (1/n)*(h_x*f(samples)/g(samples)).sum()


""" MAIN """
if __name__ == "__main__":
    ''' PARAMS '''
    n = 50000  # Number samples
    
    truth_P = 1-norm.cdf(X)  # Using scipy library to get the truth value P(X>3).
    monte_carlo_P = monte_carlo(n)  # Using monte carlo method to estimate P(X>3)
    importance_sampling_P = importance_sampling(n)  # Using importance sampling method to estimate P(X>3)

    # Print out results.
    print('--> NUM SAMPLES: %d' % n)
    print('--> TRUTH P(X>%.1f): %.7f' % (X, truth_P))
    print('--> MC P(X>%.1f): %.7f, error rate: %.3f' % (X, monte_carlo_P, ((monte_carlo_P-truth_P)/truth_P)))
    print('--> IP P(X>%.1f): %.7f, error rate: %.3f' % (X, importance_sampling_P, ((importance_sampling_P-truth_P)/truth_P)))
