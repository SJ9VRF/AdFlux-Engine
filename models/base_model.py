import torch
import torch.nn as nn

class BaseModel(nn.Module):
    """
    Base model for extensibility and unified interface.
    Provides a common framework for training, evaluation, and prediction.
    """
    def __init__(self):
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None  # To be defined in derived classes
        self.optimizer = None  # To be defined in derived classes
        self.criterion = None  # To be defined in derived classes

    def set_model(self, model):
        """
        Sets the model to be trained, evaluated, or used for inference.
        """
        self.model = model.to(self.device)

    def set_optimizer(self, optimizer):
        """
        Sets the optimizer for training.
        """
        self.optimizer = optimizer

    def set_criterion(self, criterion):
        """
        Sets the loss function for training.
        """
        self.criterion = criterion

    def train(self, dataloader, num_epochs=10):
        """
        Generic training loop for the model.
        """
        if not self.model or not self.optimizer or not self.criterion:
            raise ValueError("Model, optimizer, and criterion must be set before training.")

        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    def evaluate(self, dataloader):
        """
        Generic evaluation loop for the model.
        """
        if not self.model or not self.criterion:
            raise ValueError("Model and criterion must be set before evaluation.")

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        print(f"Evaluation: Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
        return {"loss": avg_loss, "accuracy": accuracy}

    def predict(self, inputs):
        """
        Generic prediction method for a batch of inputs.
        """
        if not self.model:
            raise ValueError("Model must be set before prediction.")

        self.model.eval()
        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)

        return predicted

    def save_model(self, filepath):
        """
        Saves the trained model to a file.
        """
        if not self.model:
            raise ValueError("Model must be set before saving.")
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath, model_class):
        """
        Loads a model from a file.
        """
        self.model = model_class().to(self.device)
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {filepath}")

