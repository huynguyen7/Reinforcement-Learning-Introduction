    - In this example we empirically compare the prediction abilities of TD(0) and constant-alpha MC when applied to the following Markov reward process:
     [0,   1/6, 2/6, 3/6, 4/6, 5/6, 1]  # Truth State Values
     [0,   0,   0,   0,   0,   0,   1]  # Init State Values
     [0,   A,   B,   C,   D,   E,   1]  # State Labels
                  ^
                  |
                START
    - A Markov reward process, or MRP, is a Markov decision process without actions. We will often use MRPs when focusing on the prediction problem, in which there is no need to distinguish the dynamics due to the environment from those due to the agent. In this MRP, all episodes start in the center state, C, then proceed either left or right by one state on each step, with equal probability. Episodes terminate either on the extreme left or the extreme right. When an episode terminates on the right, a reward of +1 occurs; all other rewards are zero. For example, a typical episode might consist of the following state-and-reward sequence: (C,0)->(B,0)->(C,0)->(D,0)->(E,1). Because this task is UNDISCOUNTED, the true value of each state is the probability of terminating on the right if starting from that state. Thus, the true value of the center state is V(C) = 0.5. The true values of all the states, A through E, are 1/6, 2/6, 3/6, 4/6, 5/6.
