import torch

class RLAgent:
    """
    RL agent to interact with the environment and update policies.
    """
    def __init__(self, model, environment, buffer):
        self.model = model
        self.env = environment
        self.replay_buffer = buffer

    def collect_experience(self, num_steps=100):
        """
        Collects experience by interacting with the environment.
        """
        state = self.env.reset()
        for _ in range(num_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = self.model.select_action(state_tensor).item()
            next_state, reward, done = self.env.step(action)

            # Add experience to the replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            if done:
                state = self.env.reset()

    def train_agent(self, num_epochs=10):
        """
        Trains the agent using the model and replay buffer.
        """
        self.model.train_model(self.replay_buffer, num_epochs=num_epochs)

