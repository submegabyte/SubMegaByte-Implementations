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

class MLP(nn.Module):
    def __init__(self, *sizes):
        super(MLP, self).__init__()
        self.sizes = sizes

        layers = []
        for i, size in enumerate(sizes[:-1]):
            layers += [
                nn.Linear(size, sizes[i+1]),
                NormAct(sizes[i+1])
            ]
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        x = self.seq(x)
        return x


class TNet(nn.Module):
    def __init__(self, n):
        super(TNet, self).__init__()

        ## shared MLP
        self.mlp = MLP(n, 64, 128, 1024)

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

        ## vector size for each point
        ## in input space, it is 3
        ## in feature space, it is 64
        n = x.shape[-1]

        # (batch_size, num_points, dim)
        x = x.view(-1, n)
        x = self.mlp(x)
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


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()

        ## input transform
        self.tNet1 = TNet(3)

        self.mlp1 = MLP(3, 64, 64)

        ## feature transform
        self.tNet2 = TNet(64)

        self.mlp2 = MLP(64, 64, 128, 1024)

    def forward(self, x):
        batch_size = x.shape[0]
        ## (batch_size, num_points, 3)
        x = x @ self.tNet1(x)
        ## (batch_size, num_points, 3)
        x = x.view(-1, 3)
        x = self.mlp1(x)
        x = x.view(batch_size, -1, 64)
        ## (batch_size, num_points, 64)
        x = x @ self.tNet2(x)
        local_features = x
        ## (batch_size, num_points, 64)
        x = x.view(-1, 64)
        x = self.mlp2(x)
        x = x.view(batch_size, -1, 1024)
        ## (batch_size, num_points, 1024)
        x, _ = torch.max(x, dim=-2)
        global_features = x
        ## (batch_size, 1024)

        return local_features, global_features


class ClassNet(nn.Module):
    def __init__(self, k):
        super(ClassNet, self).__init__()

        self.mlp = MLP(1024, 512, 256, k)

    def forward(self, x):
        ## (batch_size, 1024)
        x = self.mlp(x)
        output_scores = x
        ## (batch_size, k)
        return output_scores

class SegNet(nn.Module):
    def __init__(self, m):
        super(SegNet, self).__init__()

        self.mlp1 = MLP(1088, 512, 256, 128)

        self.m = m
        self.mlp2 = MLP(128, self.m)
    
    def forward(self, x_local, x_global):
        num_points = x_local.shape[1]
        x_global_broadcast = x_global.unsqueeze(1).expand(-1, num_points, -1)

        x = torch.cat((x_local, x_global_broadcast), dim=2)
        ## (batch_size, num_points, 64 + 1024)
        x = x.view(-1, 64 + 1024)
        ## (batch_size * num_points, 64 + 1024)
        x = self.mlp1(x)
        ## (batch_size * num_points, 128)
        point_features = x.view(-1, num_points, 128)

        x = self.mlp2(x)
        ## (batch_size * num_points, m)
        output_scores = x.view(-1, num_points, self.m)

        return point_features, output_scores



## Example usage

batch_size = 4
num_points = 100
input_dim = 3

t1 = TNet(3)
t2 = TNet(64)
point_clouds = torch.randn(batch_size, num_points, input_dim)  # Example input
# point_cloud = torch.randn(num_points, input_dim)  # Example input
output = t1(point_clouds)

# print(output.shape)  # Expected: (batch_size, 3, 3)
# print(output[0])

pointNet = PointNet()
local_features, global_features = pointNet(point_clouds)
print(local_features.shape, global_features.shape)

classNet = ClassNet(10)
global_scores = classNet(global_features)
print(global_scores.shape)

segNet = SegNet(10)
point_features, point_scores = segNet(local_features, global_features)
print(point_features.shape, point_scores.shape)

## need to add dropout layers in the class net

## each point has its own batchnorm at any stage of the network
## the batchnorm is applied to each point over its dimensions,
## as opposed to each dimension having its own batchnorm over all points