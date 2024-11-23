import torch
import torch.nn as nn

class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.fc1 = nn.Linear(k, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, k * k)  # Outputs k*k elements
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        # Reshape to a k x k matrix
        x = x.view(-1, self.k, self.k)

        # Add identity matrix
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        x = x + identity  # Ensure initial output is close to identity

        return x
