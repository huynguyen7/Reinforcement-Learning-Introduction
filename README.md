# REINFORCEMENT-LEARNING ALGORITHMS
This repository contains examples for Reinforcement Learning Algorithms:
- Single-state K-bandits (Basic Monte Carlo with sampling average):
    - Epsilon Greedy vs UCB.
    - Incremental Update vs Sampling Average.
    - Nonstationary vs Stationary Environment.
    - Optimistic Initialization for Exploring Starts.
- Gridworld MDP (Estimating state values with Bellman Equation).
- Gridworld Policy Iteration (Using Policy Iteration to estimate state values for optimal policies)
- Gambler Problem (Using value iteration DP to estimate state values for optimal policies).
- Blackjack (Using MC Prediction to estimate state values, also using Model-Free MC Control to estimate optimal policies):
    - First-Visit vs Every-Visit Update.
    - On-policy vs Off-policy.
    - NOTES: Comparing the difference between `Gambler` vs `Blackjack` problem (Gambler has assumptions about model, but blackjack does not.). `Gambler` already has assumptions about the model, we use DP. `Blackjack` does not have any assumption about the model, we use MC.
- Random Walk(Using Tabular TD Learning TD(0) to approximate State-Values, also a comparison between MC vs TD):
    - TD(0) (aka one-step TD) vs MC.
    - Batch TD(0) vs Batch MC.
    - Batch vs Non-batch update.
- Windy Gridworld (Using SARSA/Q-Learning for Control Problem, one-step TD Update):
    - Estimate optimal Q.
    - Estimate optimal policies from Q.
    - Comparison between SARSA(On-Policy TD Control Method) and Q-Learning(Off-Policy TD Control Method).
- Dyna Maze (Example 8.1, Using Tabular n-step Dyna-Q for Control Problem).


## Resources
- Reinforcement Learning: An Introduction 2nd Edition: [Click Here](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
