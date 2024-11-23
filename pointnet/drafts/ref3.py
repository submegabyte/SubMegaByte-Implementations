import torch
import torch.nn as nn
import torch.nn.functional as F

class InputTransform(nn.Module):
    def __init__(self):
        super(InputTransform, self).__init__()
        self.mlp1 = nn.Linear(3, 64)  # First linear layer
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization
        self.relu1 = nn.ReLU()  # ReLU activation

        self.mlp2 = nn.Linear(64, 128)  # Second linear layer
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.mlp3 = nn.Linear(128, 1024)  # Third linear layer
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(1024, 512)  # Fully connected layer
        self.bn4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()

        self.fc3 = nn.Linear(256, 9)  # Output for 3x3 transformation matrix

        # Initialize weights to output identity matrix
        nn.init.constant_(self.fc3.weight, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Input shape (B, N, 3) -> reshape to (B*N, 3) for linear layers
        x = x.view(-1, 3)  # Flatten
        x = self.relu1(self.bn1(self.mlp1(x)))
        x = self.relu2(self.bn2(self.mlp2(x)))
        x = self.relu3(self.bn3(self.mlp3(x)))

        # Reshape back to (B, -1)
        x = x.view(batch_size, -1)

        x = self.relu4(self.bn4(self.fc1(x)))
        x = self.relu5(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # No activation on the final layer

        # Output transformation matrix
        x = x.view(batch_size, 3, 3)  # Reshape to (B, 3, 3)
        return x
