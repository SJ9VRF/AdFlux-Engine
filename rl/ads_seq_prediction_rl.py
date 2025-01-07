import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl.lava_model import LAVAModel
from rl.environment import AdEnvironment
from rl.agent import RLAgent


class ReplayBuffer:
    """
    Replay buffer for storing experience tuples.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience tuple to the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Randomly samples a batch of experience tuples.
        """
        batch = np.random.choice(self.buffer, batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.long),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "next_states": torch.tensor(next_states, dtype=torch.float32),
            "dones": torch.tensor(dones, dtype=torch.float32),
        }

    def __len__(self):
        return len(self.buffer)


def generate_synthetic_ad_data(num_sequences=1000, state_dim=10):
    """
    Generates synthetic ad sequence data for RL training.
    Each state represents a user behavior or ad interaction.
    """
    data = np.random.rand(num_sequences, state_dim)
    return data


def train_ad_sequence_rl():
    # Hyperparameters
    state_dim = 10
    action_dim = 5
    latent_dim = 16
    num_steps = 500
    num_epochs = 20
    batch_size = 32
    replay_buffer_capacity = 1000
    learning_rate = 1e-3

    # Initialize synthetic ad environment
    env = AdEnvironment(state_dim=state_dim, action_dim=action_dim)

    # Initialize LAVA model for RL
    model = LAVAModel(state_dim=state_dim, action_dim=action_dim, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
    agent = RLAgent(model=model, environment=env, buffer=replay_buffer)

    # Generate synthetic data
    synthetic_data = generate_synthetic_ad_data(num_sequences=1000, state_dim=state_dim)

    # Training Loop
    print("Collecting experiences...")
    agent.collect_experience(num_steps=num_steps)

    print("Training the model...")
    model.train_model(
        replay_buffer=replay_buffer,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    # Evaluate the model
    print("Evaluating model performance...")
    state = env.reset()
    total_reward = 0
    for _ in range(100):  # Test for 100 steps
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = model.select_action(state_tensor).item()
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            state = env.reset()

    print(f"Total Reward over 100 steps: {total_reward}")


if __name__ == "__main__":
    train_ad_sequence_rl()

