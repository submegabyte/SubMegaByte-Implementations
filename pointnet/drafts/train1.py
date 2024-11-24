import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model1 import PointNet

## Set the default tensor type to CUDA (float tensors on GPU)
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

## Check if CUDA (GPU support) is available
## from hw5
if torch.cuda.is_available():
    # You can use CUDA
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    # Use CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

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
        # return torch.tensor(vertices, dtype=torch.float32, device=device)

# Custom Dataset for ModelNet10
class ModelNet10Dataset(Dataset):
    def __init__(self, root_dir, split='train', num_points=1024):
        """
        Args:
            root_dir (string): Directory containing the ModelNet10 dataset.
            split (string): 'train' or 'test'.
            num_points (int): Number of points per point cloud to sample.
        """
        self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        # Get all class folders
        self.class_folders = [
            os.path.join(root_dir, label) for label in os.listdir(root_dir)
            if not label.startswith('.') and os.path.isdir(os.path.join(root_dir, label))  # Skip hidden and non-directory files
        ]
        self.classes = sorted([os.path.basename(label) for label in self.class_folders])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        self.files = []
        print("Reading training data files")
        for i, class_folder in enumerate(self.class_folders):
            folder = os.path.join(class_folder, split)
            class_label = os.path.basename(class_folder)
            for file_name in os.listdir(folder):
                if file_name.endswith('.off'):
                    ## Filter files with too few points
                    file_path = os.path.join(folder, file_name)
                    points = read_off(file_path)
                    if points.shape[0] >= self.num_points:
                        self.files.append((os.path.join(folder, file_name), self.class_to_idx[class_label]))
            print(f"{i}, {class_label}, {len(self.files)} files")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        points = read_off(file_path)

        if points.shape[0] > self.num_points:
            ## Randomly sample num_points points
            ## Sample without replacement
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
        # else:
        #     ## Handle cases where there are fewer points than required
        #     ## Sample with replacement
        #     choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        #     points = points[choice, :]

        points = torch.FloatTensor(points)  # Convert to tensor
        label = torch.LongTensor([label])  # Convert label to tensor

        points.to(device)
        label.to(device)

        return points, label

## Initialize Dataset, DataLoader, Model, Optimizer, and Scheduler
root_prefix = os.getenv("HOME") + '/unshared/datasets'
root_dir = root_prefix + '/ModelNet10'  # Path to ModelNet10 directory
train_dataset = ModelNet10Dataset(root_dir=root_dir, split='train', num_points=1024)
# train_dataset.data = train_dataset.data.to(device)
# train_dataset.target = train_dataset.target.to(device)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("Finished reading training data files")

model = PointNet()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Training Parameters
epochs = 100
initial_momentum = 0.5
final_momentum = 0.99
momentum_step = (final_momentum - initial_momentum) / epochs

# Training Loop
print("Beginning Training")
for epoch in range(epochs):
    model.train()
    
    # Update BatchNorm momentum dynamically
    current_momentum = initial_momentum + epoch * momentum_step
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.momentum = current_momentum

    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)

        # print(data.shape, target.shape)

        optimizer.zero_grad()

        # Forward pass
        _, global_features = model(data)

        # Compute loss
        loss = criterion(global_features, target.squeeze())
        epoch_loss += loss.item()

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch} of {epochs}, batch {batch_idx} of {len(train_loader)}")

    # Update learning rate
    scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Loss: {epoch_loss / len(train_loader):.4f}, "
          f"LR: {scheduler.get_last_lr()[0]:.6f}, "
          f"Momentum: {current_momentum:.3f}")

# Save the trained model
model_path = root_prefix + "/pointnet_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
