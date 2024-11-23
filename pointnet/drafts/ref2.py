import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SharedMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))  # Linear layer for MLP
            layers.append(nn.BatchNorm1d(h_dim))  # Batch Normalization
            layers.append(nn.ReLU())  # ReLU activation
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))  # Final output layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, num_points, input_dim)
        batch_size, num_points, _ = x.size()
        # Reshape to (batch_size * num_points, input_dim) to apply MLP per point
        x = x.view(batch_size * num_points, -1)
        x = self.mlp(x)  # Apply shared MLP to each point
        # Reshape back to (batch_size, num_points, output_dim)
        x = x.view(batch_size, num_points, -1)
        return x

class PointNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(PointNet, self).__init__()
        self.shared_mlp = SharedMLP(input_dim, hidden_dims, output_dim)

    def forward(self, x):
        # x: (batch_size, num_points, input_dim)
        point_features = self.shared_mlp(x)
        # Apply max pooling over points (axis 1)
        global_features = torch.max(point_features, dim=1, keepdim=False)[0]
        return global_features

# Example usage
input_dim = 3  # For 3D points
hidden_dims = [64, 128, 128]  # MLP hidden layers
output_dim = 256  # Output feature dimensions

# Define the model
model = PointNet(input_dim, hidden_dims, output_dim)

# Dummy input: batch_size=2, num_points=1024, input_dim=3
points = torch.rand(2, 1024, 3)

# Forward pass
global_features = model(points)
print(global_features.shape)  # Should output: (batch_size, output_dim)
