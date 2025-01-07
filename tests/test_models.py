import unittest
import torch
from models.other_models.cnn_model import CNNModel


class TestModels(unittest.TestCase):
    """
    Unit tests for AdFlux Engine models.
    """
    def setUp(self):
        self.model = CNNModel(num_channels=16, kernel_size=3, input_dim=1, output_dim=10)
        self.sample_input = torch.randn(32, 1, 28)  # Batch size: 32, Input dim: 28x1

    def test_model_output_shape(self):
        """
        Test if model output shape matches expected dimensions.
        """
        output = self.model(self.sample_input)
        self.assertEqual(output.shape, (32, 10))

    def test_model_forward_pass(self):
        """
        Test if model forward pass works without errors.
        """
        try:
            _ = self.model(self.sample_input)
        except Exception as e:
            self.fail(f"Model forward pass failed with error: {e}")


if __name__ == "__main__":
    unittest.main()

