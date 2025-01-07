import torch
import torch.nn as nn

class MuZeroModel(nn.Module):
    """
    MuZero: Dynamic model-based reinforcement learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(MuZeroModel, self).__init__()
        self.representation = nn.Linear(state_dim, hidden_dim)
        self.dynamics = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.prediction = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, action):
        """
        Combines representation, dynamics, and prediction networks.
        """
        latent_state = torch.relu(self.representation(state))
        combined = torch.cat([latent_state, action], dim=-1)
        next_state = torch.relu(self.dynamics(combined))
        action_logits = self.prediction(next_state)
        return next_state, action_logits

