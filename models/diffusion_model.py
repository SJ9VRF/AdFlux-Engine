import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DiffusionScheduler:
    """
    Scheduler for noise levels in the diffusion process.
    """
    def __init__(self, num_steps=1000):
        self.num_steps = num_steps
        self.timesteps = torch.linspace(0, 1, num_steps)

    def get_noise_level(self, step):
        """
        Returns the noise level for a given diffusion step.
        """
        return self.timesteps[step]


class TextDiffusionModel(nn.Module):
    """
    A diffusion-based text generation model using transformer encoders.
    """
    def __init__(self, transformer_name="bert-base-uncased", num_steps=1000):
        super(TextDiffusionModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)
        self.encoder = AutoModel.from_pretrained(transformer_name)
        self.decoder = nn.Linear(self.encoder.config.hidden_size, self.tokenizer.vocab_size)
        self.scheduler = DiffusionScheduler(num_steps=num_steps)

    def add_noise(self, embeddings, noise_level):
        """
        Adds Gaussian noise to the text embeddings based on the noise level.
        """
        noise = torch.randn_like(embeddings) * noise_level
        return embeddings + noise

    def forward(self, input_text, step):
        """
        Forward pass for text generation with diffusion steps.
        """
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        embeddings = self.encoder(**tokens).last_hidden_state
        noise_level = self.scheduler.get_noise_level(step)
        noised_embeddings = self.add_noise(embeddings, noise_level)
        logits = self.decoder(noised_embeddings)
        return logits

    def generate_text(self, seed_text, num_steps=50):
        """
        Generates text using the diffusion process.
        """
        current_text = seed_text
        for step in range(num_steps):
            logits = self.forward(current_text, step)
            predictions = logits.argmax(dim=-1)
            current_text = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
        return current_text

