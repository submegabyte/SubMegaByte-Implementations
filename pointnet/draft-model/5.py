import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self, dims=[64,128,1024]):
        super(SharedMLP, self).__init__()
        in_dim = 3
        layers = [
            nn.Linear(in_dim,dims[0]),
            nn.ReLU()
        ]
        in_dim = dims[0]
        for dim in dims[1:-1]:
            layers += [
                nn.Linear(in_dim,dim),
                nn.ReLU()
            ]
            in_dim = dim
        layers.append(nn.Linear(in_dim, dims[-1]))
        self.mlp = nn.Sequential(*layers)

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

shared_mlp = SharedMLP()
point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
output = shared_mlp(point_cloud)

print(output.shape)  # Expected: (batch_size, num_points, output_dim)
