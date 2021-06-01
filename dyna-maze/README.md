# DYNA-MAZE
- Using _Dyna Q_ to solve the **Control Problem** using example 8.1:
    - Model Learning and Planning (Using Tabular Model for simulating experience).
    - Estimate optimal Q.
    - Estimate optimal policies based on estimated Q.

- Summary:
    - Non-planning agent (n=0) took 25 episodes to converge.
    - 5-step agent took 5 episodes to converge.
    - 50-step agent took 3 episodes to converge.

- Tuning:
    - alpha = 0.1  (Learning rate)
    - gamma = 0.95  (Discounting rate)
    - epsilon = 0.1  (Exploratory rate)


## Examples
- _one-step tabular Q-learning agent_ with 25 episodes:
![linr](examples/25_eps_0.1_lr_0.95_gamma_0.1_epsilon_0_n_dyna_maze.png)

- _5-step tabular Q-learning agent_ with 5 episodes:
![linr](examples/5_eps_0.1_lr_0.95_gamma_0.1_epsilon_5_n_dyna_maze.png)

- _50-step tabular Q-learning agent_ with 3 episodes:
![linr](examples/3_eps_0.1_lr_0.95_gamma_0.1_epsilon_50_n_dyna_maze.png)


## Comparisons:
- Comparing n=0, n=5, n=50:
![linr](examples/.png)
