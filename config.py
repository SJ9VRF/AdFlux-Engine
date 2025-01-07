"""
Configuration settings for the AdFlux Engine.
"""

CONFIG = {
    "data": {
        "synthetic": {
            "num_sequences": 1000,
            "seq_length": 10,
            "vocab_size": 20,
            "batch_size": 32
        }
    },
    "model": {
        "type": "cnn",  # Options: "cnn", "lstm", "gru", etc.
        "cnn": {
            "num_channels": 16,
            "kernel_size": 3,
            "input_dim": 1,
            "output_dim": 10
        }
    },
    "training": {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 0.001
    },
    "logging": {
        "log_file": "adflux.log"
    }
}

