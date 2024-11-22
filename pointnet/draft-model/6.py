import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )

    def forward(self, x):
        # Assume x is of shape (num_points, 3) or (batch_size, num_points, 3)
        y = self.mlp(x)
        return y

# Example usage
batch_size = 4
num_points = 100
input_dim = 3

shared_mlp = SharedMLP()
point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
output = shared_mlp(point_cloud)

print(output.shape)  # Expected: (batch_size, num_points, output_dim)
