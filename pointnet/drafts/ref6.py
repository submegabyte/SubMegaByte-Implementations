import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import h5py
import glob

# Custom Dataset for ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, data_dir, num_points=1024, transform=None):
        """
        Args:
            data_dir (string): Directory with all the model data.
            num_points (int): Number of points per point cloud to sample.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.num_points = num_points
        self.transform = transform

        # Load all the data
        self.files = glob.glob(os.path.join(self.data_dir, '*.h5'))
        self.labels = []
        self.points = []

        # Read all h5 files (point cloud data)
        for file in self.files:
            with h5py.File(file, 'r') as f:
                points = f['points'][:]
                label = f['label'][:]
                self.points.append(points)
                self.labels.append(label)

        # Convert lists to numpy arrays
        self.points = np.array(self.points)
        self.labels = np.array(self.labels)

        # Use LabelEncoder to encode labels into integers
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx]
        label = self.labels[idx]

        # Randomly sample num_points points
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]

        points = torch.FloatTensor(points)  # Convert to tensor
        label = torch.LongTensor([label])  # Convert label to tensor

        return points, label


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
        self.fc = nn.Linear(128, 10)  # 10 classes for ModelNet10

    def forward(self, x):
        # Input shape: [batch_size, num_points, 3]
        x = x.transpose(1, 2)  # Transpose to [batch_size, 3, num_points]
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.max(x, dim=-1)[0]  # Global max pooling
        x = self.fc(x)
        return x


# Initialize Dataset, DataLoader, Model, Optimizer, and Scheduler
data_dir = '/path/to/modelnet10'  # Path to ModelNet10 .h5 files
train_dataset = ModelNet10Dataset(data_dir=data_dir, num_points=1024)
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
