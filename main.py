import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
from config import CONFIG
from utils.utilities import Logger
from data.data_processor import DataProcessor
from models.other_models.cnn_model import CNNModel


def main():
    # Load configuration
    config = CONFIG

    # Initialize logger
    logger = Logger(log_file=config["logging"]["log_file"])
    logger.log("Starting AdFlux Engine...")

    # Load and preprocess data
    data_processor = DataProcessor(data_dir=config["data"]["data_dir"], file_format=config["data"]["file_format"])
    raw_data = data_processor.load_data("ads_data.csv")
    processed_data = data_processor.preprocess_data(raw_data)
    train_data, test_data = data_processor.split_data(processed_data, test_size=config["data"]["test_size"])

    # Prepare PyTorch DataLoader
    train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_tensor[:, :-1], train_tensor[:, -1]), batch_size=config["training"]["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor[:, :-1], test_tensor[:, -1]), batch_size=config["training"]["batch_size"], shuffle=False)

    # Initialize model
    model_config = config["model"]["cnn"]
    model = CNNModel(
        num_channels=model_config["num_channels"],
        kernel_size=model_config["kernel_size"],
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"]
    )

    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # Training loop
    logger.log("Starting training...")
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1))
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        logger.log(f"Epoch [{epoch + 1}/{config['training']['num_epochs']}], Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "cnn_model.pth")
    logger.log("Model training complete and saved.")

    # Evaluate the model
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.unsqueeze(1))
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels.long()).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    logger.log(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

