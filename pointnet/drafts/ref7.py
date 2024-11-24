import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# Function to read .off files
def read_off(file_path):
    """Read .off file and return vertices."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if lines[0].strip() != 'OFF':
            raise ValueError("Invalid .off file")
        # Extract number of vertices and faces
        parts = lines[1].strip().split()
        num_vertices = int(parts[0])
        # Read vertices
        vertices = []
        for i in range(2, 2 + num_vertices):
            vertex = list(map(float, lines[i].strip().split()))
            vertices.append(vertex)
        return np.array(vertices, dtype=np.float32)


# Custom Dataset for ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024, transform=None):
        """
        Args:
            root_dir (string): Directory containing the ModelNet10 dataset.
            split (string): 'train' or 'test'.
            num_points (int): Number of points per point cloud to sample.
            transform (callable, optional): Optional transform to apply to the point cloud.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points
        self.transform = transform

        # Get all class folders
        self.class_folders = [os.path.join(root_dir, label) for label in os.listdir(root_dir)]
        self.classes = sorted([os.path.basename(label) for label in self.class_folders])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        self.files = []
        for class_folder in self.class_folders:
            folder = os.path.join(class_folder, split)
            for file_name in os.listdir(folder):
                if file_name.endswith('.off'):
                    self.files.append((os.path.join(folder, file_name), self.class_to_idx[os.path.basename(class_folder)]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        points = read_off(file_path)

        # Randomly sample num_points points
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]

        points = torch.FloatTensor(points)  # Convert to tensor
        label = torch.LongTensor([label])  # Convert label to tensor

        # Apply optional transform
        if self.transform:
            points = self.transform(points)

        return points, label


# Example PointNet Model
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
root_dir = 'ModelNet10'  # Path to ModelNet10 directory
train_dataset = ModelNet10Dataset(root_dir=root_dir, split='train', num_points=1024)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = ModelNet10Dataset(root_dir=root_dir, split='test', num_points=1024)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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
        loss = criterion(output, target.squeeze())
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

# Test Loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += (pred == target.squeeze()).sum().item()
        total += target.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
