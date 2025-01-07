import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from models.base_model import BaseModel


# Create synthetic advertisement dataset
class SyntheticAdDataset(Dataset):
    def __init__(self, num_sequences=1000, seq_length=10, vocab_size=20):
        self.num_sequences = num_sequences
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.data, self.labels = self.generate_data()

    def generate_data(self):
        """
        Generate synthetic sequences and their next-item predictions.
        """
        data = torch.randint(0, self.vocab_size, (self.num_sequences, self.seq_length))
        labels = data[:, 1:]  # Shifted by one position for prediction
        return data[:, :-1], labels

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Define an LSTM model for sequence prediction
class AdSequenceModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size):
        super(AdSequenceModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # Input shape: (batch_size, seq_length)
        out, _ = self.lstm(x)  # LSTM output shape: (batch_size, seq_length, hidden_size)
        out = self.fc(out)     # Output shape: (batch_size, seq_length, output_size)
        return out


# Main function
def main():
    # Parameters
    vocab_size = 20
    seq_length = 10
    embed_size = 16
    hidden_size = 32
    output_size = vocab_size
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Generate synthetic data
    dataset = SyntheticAdDataset(num_sequences=1000, seq_length=seq_length, vocab_size=vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Split data into training and testing
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = AdSequenceModel(vocab_size, embed_size, hidden_size, output_size)
    base_model = BaseModel()
    base_model.set_model(model)
    base_model.set_optimizer(optim.Adam(model.parameters(), lr=learning_rate))
    base_model.set_criterion(nn.CrossEntropyLoss())

    # Train the model
    print("Starting Training...")
    base_model.train(train_loader, num_epochs=num_epochs)

    # Evaluate the model
    print("Evaluating Model...")
    evaluation_metrics = base_model.evaluate(test_loader)
    print(f"Test Loss: {evaluation_metrics['loss']:.4f}, Test Accuracy: {evaluation_metrics['accuracy']:.4f}")

    # Save the model
    model_path = "ad_sequence_model.pth"
    base_model.save_model(model_path)

    # Load the model and predict
    print("Loading Model...")
    base_model.load_model(model_path, lambda: AdSequenceModel(vocab_size, embed_size, hidden_size, output_size))

    # Predict on a batch of test data
    test_data_iter = iter(test_loader)
    test_inputs, _ = next(test_data_iter)
    predictions = base_model.predict(test_inputs)
    print("Sample Predictions:")
    print(f"Inputs: {test_inputs[0].tolist()}")
    print(f"Predictions: {predictions[0].tolist()}")


if __name__ == "__main__":
    main()

