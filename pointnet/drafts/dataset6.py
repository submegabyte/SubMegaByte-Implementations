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
# try:
#     device
# except NameError:
myfilename = os.path.splitext(os.path.basename(__file__))[0]
if torch.cuda.is_available():
    # You can use CUDA
    device = torch.device("cuda")
    # if __name__ == "main":
    print(f"{myfilename}: CUDA is available. Using GPU.")
else:
    # Use CPU
    device = torch.device("cpu")
    # if __name__ == "main":
    print(f"{myfilename}: CUDA is not available. Using CPU.")
    
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
    def __init__(self, root_dir=None, split='train', num_points=1024):
        """
        Args:
            root_dir (string): Directory containing the ModelNet10 dataset.
            split (string): 'train' or 'test'.
            num_points (int): Number of points per point cloud to sample.
        """
        # self.root_dir = root_dir
        self.split = split
        self.num_points = num_points

        ## pointclouds
        self.data = []

        ## labels
        self.target = []

        if root_dir == None:
            return

        # Get all class folders
        self.class_folders = [
            os.path.join(root_dir, label) for label in os.listdir(root_dir)
            if not label.startswith('.') and os.path.isdir(os.path.join(root_dir, label))  # Skip hidden and non-directory files
        ]
        self.classes = sorted([os.path.basename(label) for label in self.class_folders])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        # self.files = []
        print(f"Reading {split} data files")
        for i, class_folder in enumerate(self.class_folders):
            folder = os.path.join(class_folder, split)
            class_label = os.path.basename(class_folder)
            class_idx = self.class_to_idx[class_label]
            label = class_idx
            for file_name in os.listdir(folder):
                if file_name.endswith('.off'):
                    ## Filter files with too few points
                    file_path = os.path.join(folder, file_name)
                    points = read_off(file_path)
                    if points.shape[0] >= self.num_points:
                        # self.files.append((os.path.join(folder, file_name), class_idx))

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

            print(f"{i}, {class_label}")
        
        self.data = torch.stack(self.data)
        self.target = torch.stack(self.target)

        self.data = self.data.to(device)
        self.target = self.target.to(device)

    def save(self, filepath):
        torch.save({
            # 'split': self.split,
            # 'num_points': self.num_points,
            'data': self.data,
            'target': self.target
        }, filepath)

    def load(self, filepath):
        loaded_data = torch.load(filepath, weights_only=True) # pickle security reasons

        # self.split = loaded_data['split']
        # self.num_points = loaded_data['num_points']
        self.data = loaded_data['data']
        self.target = loaded_data['target']

        self.data = self.data.to(device)
        self.target = self.target.to(device)

    def __len__(self):
        # return len(self.files)
        return self.data.shape[0]

    def __getitem__(self, idx):

        points = self.data[idx]
        label = self.target[idx]

        return points, label


if __name__ == "__main__":

    ## Initialize Dataset, DataLoader, Model, Optimizer, and Scheduler
    root_prefix = os.getenv("HOME") + '/unshared/datasets'
    root_dir = root_prefix + '/ModelNet10'  # Path to ModelNet10 directory
    train_dataset = ModelNet10Dataset(root_dir=root_dir, split='test', num_points=1024)
    # train_dataset.data = train_dataset.data.to(device)
    # train_dataset.target = train_dataset.target.to(device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"Finished reading {train_dataset.split} data files")

    ## Save dataset
    dataset_path = root_prefix + f'/modelnet10_{myfilename}.pth'
    # torch.save({'data': train_dataset.data, 'target': train_dataset.target}, dataset_path)
    train_dataset.save(dataset_path)
    print(f"Dataset saved to f{dataset_path}")

    ## Load dataset
    # loaded_data = torch.load('custom_dataset.pth')
    # reloaded_dataset = ModelNet10Dataset(loaded_data['data'], loaded_data['target'])
    # reloaded_dataloader = DataLoader(reloaded_dataset, batch_size=10)