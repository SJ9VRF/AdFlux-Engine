import torch
import torch.nn as nn

class GeneratorModel(nn.Module):
    """
    A simple generator network for synthetic data creation.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneratorModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class GenerativeTeachingNetwork:
    """
    Generative Teaching Network (GTN) for synthetic data generation.
    """
    def __init__(self, generator_model):
        self.generator = generator_model

    def generate_data(self, seed_conditions):
        """
        Generates synthetic data based on seed conditions.
        """
        seed_tensor = torch.tensor(seed_conditions, dtype=torch.float32)
        generated_data = self.generator(seed_tensor)
        return generated_data

    def train_generator(self, real_data, num_epochs=100, learning_rate=0.001):
        """
        Trains the generator using real data.
        """
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            noise = torch.randn_like(real_data)
            generated_data = self.generator(noise)
            loss = criterion(generated_data, real_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

