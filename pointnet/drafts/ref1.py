## reference

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, num_classes):
        super(PointNet, self).__init__()
        
        # Input transformation network
        self.input_transform = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_features, num_points)
        
        # Transform and global feature aggregation
        x = self.input_transform(x)  # shape: (batch_size, 1024, num_points)
        print(x.shape)
        x = torch.max(x, dim=2, keepdim=False)[0]  # Global max pooling
        print(x.shape)
        
        # Fully connected layers for classification
        x = self.fc_layers(x)  # shape: (batch_size, num_classes)
        return F.log_softmax(x, dim=1)


# Example usage
if __name__ == "__main__":
    num_points = 1024  # Number of points in each point cloud
    num_classes = 10   # Number of classification categories
    
    model = PointNet(num_classes=num_classes)
    print(model)
    
    # Random input tensor (batch_size, num_features, num_points)
    input_tensor = torch.rand((32, 3, num_points))  # Batch size of 32, 3D points
    output = model(input_tensor)
    print("Output shape:", output.shape)  # Expected: (32, num_classes)
