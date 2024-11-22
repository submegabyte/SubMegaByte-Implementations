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

class T1(nn.Module):
    def __init__(self):
        super(T1, self).__init__()
        self.mlp = SharedMLP()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
    
    def forward(self, x):

        # (num_points, 3)
        y = self.mlp(x) 
        y = nn.ReLU(y)
        # (num_points, 1024)
        y = torch.max(y, dim=-2)
        y = nn.ReLU(y)
        # (1024)
        y = self.fc1(y)
        y = nn.ReLU(y)
        # (512)
        y = self.fc2(y)
        y = nn.ReLU(y)
        # (256)