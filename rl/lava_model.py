import torch
import torch.nn as nn
import torch.optim as optim

class LAVAModel(nn.Module):
    """
    LAVA: Latent Action Spaces for Offline Reinforcement Learning.
    """
    def __init__(self, state_dim, action_dim, latent_dim):
        super(LAVAModel, self).__init__()
        self.encoder = nn.Linear(state_dim, latent_dim)
        self.policy = nn.Linear(latent_dim, action_dim)

    def forward(self, state):
        """
        Forward pass to encode state and predict action probabilities.
        """
        latent = torch.relu(self.encoder(state))
        action_logits = self.policy(latent)
        return action_logits

    def select_action(self, state):
        """
        Selects an action based on the policy.
        """
        action_logits = self.forward(state)
        return torch.argmax(action_logits, dim=-1)

    def train_model(self, replay_buffer, num_epochs=50, batch_size=64, learning_rate=1e-3):
        """
        Trains the LAVA model using data from the replay buffer.
        """
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            batch = replay_buffer.sample(batch_size)
            states, actions = batch["states"], batch["actions"]
            optimizer.zero_grad()

            logits = self.forward(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

