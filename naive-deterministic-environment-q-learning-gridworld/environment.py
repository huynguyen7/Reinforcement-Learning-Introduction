"""
    - Deterministic environment -> State transitional probability is 1.
"""

class Environment:
    def __init__(self, grid_height=5, grid_width=5, default_reward=0, outline_grid_reward=-1):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.default_reward = default_reward  # In-line default rewards
        self.outline_grid_reward = outline_grid_reward

    def interact(self, state, action):  # STEP, Return next state, immediate reward
        if state[0] == 0 and state[1] == 1:
            return [4,1], 10
        elif state[0] == 0 and state[1] == 3:
            return [2,3], 5

        next_state = state + action
        if next_state[0] < 0 or next_state[0] >= self.grid_height or next_state[1] < 0 or next_state[1] >= self.grid_width:  # Reach out-line state
            return state, self.outline_grid_reward
        else:  # Default state
            return next_state, self.default_reward

    def get_grid_height(self):
        return self.grid_height

    def get_grid_width(self):
        return self.grid_width
