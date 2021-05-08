# REINFORCEMENT-LEARNING ALGORITHMS
This repository contains examples for Reinforcement Learning Algorithms:
- Single-state K-bandits (Basic Monte Carlo with sampling average).
- Gridworld MDP (Estimating state values with Bellman Equation).
- Gridworld Policy Iteration (Using Policy Iteration to estimate state values for optimal policies)
- Gambler Problem (Using value iteration DP to estimate state values for optimal policies).
- Blackjack (Using MC Prediction to estimate state values, also using Model-Free MC Control to estimate optimal policies):
    - First-Visit vs Every-Visit Update.
    - On-policy vs Off-policy
    - NOTES: Comparing the difference between `Gambler` vs `Blackjack` problem (Gambler has assumptions about model, but blackjack does not.)
    - Since Gambler already has assumptions about the model, we use DP.
    - Since Blackjack does not have any assumption about the model, we use MC.


## Resources
- Reinforcement Learning: An Introduction 2nd Edition: [Click Here](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
