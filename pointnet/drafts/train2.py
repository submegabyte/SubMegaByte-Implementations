import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model1 import PointNet
from dataset4 import ModelNet10Dataset

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


## Initialize Dataset, DataLoader, Model, Optimizer, and Scheduler
root_prefix = os.getenv("HOME") + '/unshared/datasets'
root_dir = root_prefix + '/ModelNet10'  # Path to ModelNet10 directory
# train_dataset = ModelNet10Dataset(root_dir=root_dir, split='train', num_points=1024)
# train_dataset.data = train_dataset.data.to(device)
# train_dataset.target = train_dataset.target.to(device)
train_dataset = torch.load(root_prefix + '/ModelNet10.pth', weights_only=True)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
print("Finished loading training data ")

print(train_dataset['data'].shape)
print(train_dataset['target'].shape)

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
model_path = root_prefix + f"/pointnet_modelnet10_{myfilename}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
