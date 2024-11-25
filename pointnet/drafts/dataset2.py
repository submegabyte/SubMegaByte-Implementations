import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model19 import PointNet

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

        ## pointclouds
        self.data = []

        ## labels
        self.target = []

        # Collect all file paths and labels
        self.files = []
        print("Reading training data files")
        for i, class_folder in enumerate(self.class_folders):
            folder = os.path.join(class_folder, split)
            class_label = os.path.basename(class_folder)
            class_idx = self.class_to_idx[class_label]
            for file_name in os.listdir(folder):
                if file_name.endswith('.off'):
                    ## Filter files with too few points
                    file_path = os.path.join(folder, file_name)
                    points = read_off(file_path)
                    if points.shape[0] >= self.num_points:
                        self.files.append((os.path.join(folder, file_name), class_idx))

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

                        self.data += [points]
                        self.target += [label]

            print(f"{i}, {class_label}, {len(self.files)} files cumulative")
        
        self.data = torch.stack(self.data)
        self.target = torch.stack(self.target)

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

## Save dataset
torch.save({'data': train_dataset.data, 'target': train_dataset.target}, 'custom_dataset.pth')

## Load dataset
loaded_data = torch.load('custom_dataset.pth')
reloaded_dataset = ModelNet10Dataset(loaded_data['data'], loaded_data['target'])
reloaded_dataloader = DataLoader(reloaded_dataset, batch_size=10)