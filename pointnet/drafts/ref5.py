import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Custom Dataset for Simulating Point Cloud Data
class PointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=1024, num_classes=10):
        super(PointCloudDataset, self).__init__()
        self.num_samples = num_samples
        self.num_points = num_points
        self.num_classes = num_classes

        # Generate random point clouds (shape: [num_samples, num_points, 3])
        self.data = torch.rand(num_samples, num_points, 3)
        # Generate random labels (shape: [num_samples])
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Each item is a point cloud (1024x3) and its label
        return self.data[idx], self.labels[idx]


# Example PointNet-like Model
class PointNetExample(nn.Module):
    def __init__(self):
        super(PointNetExample, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128, 10)  # Example output for classification

    def forward(self, x):
        # Input shape: [batch_size, num_points, 3]
        x = x.transpose(1, 2)  # Transpose to [batch_size, 3, num_points]
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.max(x, dim=-1)[0]  # Global max pooling
        x = self.fc(x)
        return x


# Initialize Dataset, DataLoader, Model, Optimizer, and Scheduler
train_dataset = PointCloudDataset(num_samples=1000, num_points=1024, num_classes=10)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = PointNetExample()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training Parameters
epochs = 100
initial_momentum = 0.5
final_momentum = 0.99
momentum_step = (final_momentum - initial_momentum) / epochs

# Training Loop
for epoch in range(epochs):
    model.train()
    
    # Update BatchNorm momentum dynamically
    current_momentum = initial_momentum + epoch * momentum_step
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = current_momentum

    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output, target)
        epoch_loss += loss.item()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Loss: {epoch_loss / len(train_loader):.4f}, "
          f"LR: {scheduler.get_last_lr()[0]:.6f}, "
          f"Momentum: {current_momentum:.3f}")
