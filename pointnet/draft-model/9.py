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
        x = self.layers(x)
        return x

class T1(nn.Module):
    def __init__(self):
        super(T1, self).__init__()

        ## shared MLP
        self.mlp1 = nn.Linear(3, 64)
        self.mlpNorm1 = NormAct(64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlpNorm2 = NormAct(128)
        self.mlp3 = nn.Linear(128, 1024)
        self.mlpNorm1 = NormAct(1024)

        ## fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fcNorm1 = NormAct(512)
        self.fc2 = nn.Linear(512, 256)
    
    def forward(self, x):

        # (batch_size, num_points, 3)
        x = self.mlp1(x)
        print(x.shape)
        x = self.mlpNorm1(x)
        # (batch_size, num_points, 64)
        x = self.mlp2(x)
        x = self.mlpNorm2(x)
        # (batch_size, num_points, 128)
        x = self.mlp3(x)
        x = self.mlpNorm3(x)
        # (batch_size, num_points, 1024)
        x, _ = torch.max(x, dim=-2)
        # (batch_size, 1024)
        x = self.fc1(x)
        x = self.fcNorm1(x)
        # (batch_size, 512)
        x = self.fc2(x)
        # (batch_size, 256)

# Example usage
batch_size = 4
num_points = 100
input_dim = 3

t1 = T1()
# point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
# point_cloud = torch.randn(num_points, input_dim)  # Example input
point_cloud = torch.randn(batch_size, num_points, input_dim)
output = t1(point_cloud)

print(output.shape)  # Expected: (batch_size, 256)

## BatchNorm1d doesn't work without batch size (i.e. on 1D vectors)
## Cannot print-debug when using nn.sequential statements

## BatchNorm1d only works for the (batch_size, input_dim, num_points) order