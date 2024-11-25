import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=1024):
        super(SharedMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # Assume x is of shape (batch_size, num_points, input_dim)
        batch_size, num_points, input_dim = x.shape
        x = x.view(-1, input_dim)  # Flatten for processing
        x = self.mlp(x)
        x = x.view(batch_size, num_points, -1)  # Restore original shape
        return x

# Example usage
batch_size = 4
num_points = 100
input_dim = 3
hidden_dims = [64, 128]
output_dim = 256

shared_mlp = SharedMLP(input_dim, hidden_dims, output_dim)
point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
output = shared_mlp(point_cloud)

print(output.shape)  # Expected: (batch_size, num_points, output_dim)
