import numpy as np
from torch.utils.data import Dataset, DataLoader

class SyntheticAdDataset(Dataset):
    """
    Generates synthetic sequences of user interactions for advertisement data.
    Each sequence consists of random integers representing user behavior states.
    """
    def __init__(self, num_sequences=1000, seq_length=10, vocab_size=20):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data, self.labels = self._generate_data()

    def _generate_data(self):
        """
        Generate sequences and their corresponding next-item predictions.
        """
        data = np.random.randint(0, self.vocab_size, (self.num_sequences, self.seq_length))
        labels = data[:, 1:]  # Shifted version of the data for prediction
        return data[:, :-1], labels

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def create_synthetic_data(num_sequences=1000, seq_length=10, vocab_size=20, batch_size=32):
    """
    Create synthetic ad interaction data and return DataLoader objects.
    """
    dataset = SyntheticAdDataset(num_sequences=num_sequences, seq_length=seq_length, vocab_size=vocab_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    # Example usage
    num_sequences = 1000
    seq_length = 10
    vocab_size = 20
    batch_size = 32

    print("Creating synthetic ad data...")
    data_loader = create_synthetic_data(num_sequences, seq_length, vocab_size, batch_size)

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Inputs: {inputs.shape}, Targets: {targets.shape}")
        if batch_idx == 2:  # Limit output for demonstration
            break

