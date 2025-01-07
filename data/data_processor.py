import pandas as pd
import numpy as np
import os

class DataProcessor:
    """
    Handles data loading, preprocessing, and formatting for the AdFlux Engine.
    """
    def __init__(self, data_dir, file_format="csv"):
        self.data_dir = data_dir
        self.file_format = file_format

    def load_data(self, filename):
        """
        Load data from the specified file.
        """
        file_path = os.path.join(self.data_dir, filename)
        if self.file_format == "csv":
            return pd.read_csv(file_path)
        elif self.file_format == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

    def preprocess_data(self, data):
        """
        Preprocess the data (e.g., normalize, encode categorical variables).
        """
        # Example: Normalize numerical columns
        for col in data.select_dtypes(include=[np.number]).columns:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

        # Example: One-hot encode categorical columns
        data = pd.get_dummies(data, drop_first=True)

        return data

    def split_data(self, data, test_size=0.2):
        """
        Split data into training and testing sets.
        """
        train_data = data.sample(frac=1 - test_size, random_state=42)
        test_data = data.drop(train_data.index)
        return train_data, test_data

