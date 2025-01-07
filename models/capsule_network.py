import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    """
    A single layer of capsules.
    """
    def __init__(self, num_capsules, in_channels, out_channels, kernel_size=None, stride=None):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride) for _ in range(num_capsules)
        ])

    def forward(self, x):
        outputs = [capsule(x) for capsule in self.capsules]
        outputs = torch.stack(outputs, dim=1)
        return self.squash(outputs)

    @staticmethod
    def squash(tensor, epsilon=1e-7):
        """
        Squashes the capsule output to ensure unit length.
        """
        norm = torch.norm(tensor, dim=-1, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * (tensor / (norm + epsilon))


class CapsuleNetwork(nn.Module):
    """
    A Capsule Network for hierarchical relationship modeling.
    """
    def __init__(self, input_dim, primary_caps_dim, output_caps_dim, num_classes):
        super(CapsuleNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, primary_caps_dim, kernel_size=9, stride=1)
        self.primary_caps = CapsuleLayer(
            num_capsules=primary_caps_dim,
            in_channels=1,
            out_channels=primary_caps_dim,
            kernel_size=9,
            stride=2
        )
        self.output_caps = CapsuleLayer(
            num_capsules=num_classes,
            in_channels=primary_caps_dim,
            out_channels=output_caps_dim
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.primary_caps(x)
        x = self.output_caps(x)
        return x

