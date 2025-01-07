import numpy as np
import random

class AdEnvironment:
    """
    Custom RL environment for ad simulations.
    """
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.current_state = np.random.rand(state_dim)

    def reset(self):
        """
        Resets the environment to an initial state.
        """
        self.current_state = np.random.rand(self.state_dim)
        return self.current_state

    def step(self, action):
        """
        Simulates a step in the environment given an action.
        """
        reward = np.random.choice([1, -1], p=[0.7, 0.3])  # Example reward logic
        next_state = np.random.rand(self.state_dim)  # Random next state
        done = random.random() < 0.1  # 10% chance of termination
        return next_state, reward, done

