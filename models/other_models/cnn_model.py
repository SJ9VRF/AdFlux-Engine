import torch.nn as nn

class CNNModel(nn.Module):
    """
    A simple 1D CNN for sequence modeling.
    """
    def __init__(self, num_channels, kernel_size, input_dim, output_dim):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_channels, kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(num_channels, output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

