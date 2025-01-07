from data.synthetic_data import create_synthetic_data
from config import CONFIG

def main():
    # Load synthetic data configuration
    synthetic_config = CONFIG["data"]["synthetic"]

    # Create synthetic ad interaction data
    print("Generating synthetic data...")
    data_loader = create_synthetic_data(
        num_sequences=synthetic_config["num_sequences"],
        seq_length=synthetic_config["seq_length"],
        vocab_size=synthetic_config["vocab_size"],
        batch_size=synthetic_config["batch_size"]
    )

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}: Inputs shape: {inputs.shape}, Targets shape: {targets.shape}")
        if batch_idx == 2:  # Stop after 3 batches for demonstration
            break


if __name__ == "__main__":
    main()
