import torch
import torch.nn as nn

## batch norm and 
class NormAct(nn.Module):
    def __init__(self, n): # num_features (points in this case)
        super(NormAct, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(n),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class TNet(nn.Module):
    def __init__(self, n):
        super(TNet, self).__init__()

        ## shared MLP
        self.mlp1 = nn.Linear(3, 64)
        self.mlpNorm1 = NormAct(64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlpNorm2 = NormAct(128)
        self.mlp3 = nn.Linear(128, 1024)
        self.mlpNorm3 = NormAct(1024)

        ## fully connected layers
        self.fc1 = nn.Linear(1024, 512)
        self.fcNorm1 = NormAct(512)
        self.fc2 = nn.Linear(512, 256)
        self.fcNorm2 = NormAct(256)

        ## final output, nxn transformation matrix
        self.n = n
        self.fc3 = nn.Linear(256, self.n**2)
        self.identity = torch.eye(self.n)

        ## Initialize fc3 to output zero matrix
        # nn.init.constant_(self.fc3.weight, 0)
        # nn.init.constant_(self.fc3.bias, 0)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        batch_size = x.shape[0]
        # (batch_size, num_points, 3)
        x = x.view(-1, 3)
        # (batch_size * num_points, 3)
        x = self.mlp1(x)
        x = self.mlpNorm1(x)
        # (batch_size * num_points, 64)
        x = self.mlp2(x)
        x = self.mlpNorm2(x)
        # (batch_size * num_points, 128)
        x = self.mlp3(x)
        x = self.mlpNorm3(x)
        # (batch_size * num_points, 1024)
        x = x.view(batch_size, -1, 1024)
        # (batch_size, num_points, 1024)
        x, _ = torch.max(x, dim=-2)
        # (batch_size, 1024)
        x = self.fc1(x)
        x = self.fcNorm1(x)
        # (batch_size, 512)
        x = self.fc2(x)
        x = self.fcNorm2(x)
        # (batch_size, 256)
        x = self.fc3(x)
        # (batch_size, nxn)
        x = x.view(-1, self.n, self.n)
        # x += self.identity.to(x.device)
        x += self.identity
        # (batch_size, n, n)
        return x

# Example usage
batch_size = 4
num_points = 100
input_dim = 3

t1 = TNet(3)
t2 = TNet(64)
point_cloud = torch.randn(batch_size, num_points, input_dim)  # Example input
# point_cloud = torch.randn(num_points, input_dim)  # Example input
output = t1(point_cloud)

print(output.shape)  # Expected: (batch_size, 3, 3)
print(output[0])