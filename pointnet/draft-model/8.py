import torch
import torch.nn as nn

class NormAct(nn.Module):
    def __init__(self, n): # num_features
        super(NormAct, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(n),
            nn.ReLU()
        )
    
    def forward(self, x):
        y = self.layers(x)
        return y

class T1(nn.Module):
    def __init__(self):
        super(T1, self).__init__()
        self.sharedMLP = nn.Sequential(
            nn.Linear(3, 64),
            NormAct(64),
            nn.Linear(64, 128),
            NormAct(128),
            nn.Linear(128, 1024)
        )
        self.norm1 = NormAct(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.norm2 = NormAct(512)
        self.fc2 = nn.Linear(512, 256)
    
    def forward(self, x):

        # (num_points, 3)
        y = self.sharedMLP(x)
        # (num_points, 1024)
        y, _ = torch.max(y, dim=-2)
        # (1024)
        y = self.norm1(y)
        y = self.fc1(y)
        # (512)
        y = self.norm2(y)
        y = self.fc2(y)
        # (256)

# Example usage
batch_size = 4
num_points = 100
input_dim = 3

t1 = T1()
point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
# point_cloud = torch.randn(num_points, input_dim)  # Example input
output = t1(point_cloud)

print(output.shape)  # Expected: (batch_size, 256)

## BatchNorm1d doesn't work without batch size (i.e. on 1D vectors)
## Cannot print-debug when using nn.sequential statements