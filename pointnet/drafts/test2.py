import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model19 import PointNet, ClassNet
from dataset6 import ModelNet10Dataset

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

# train_dataset = torch.load(root_prefix + '/ModelNet10.pth', weights_only=True)

test_dataset = ModelNet10Dataset()
test_dataset.load(root_prefix + '/modelnet10_dataset6.pth')

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print("Finished loading training data ")

# print(train_dataset['data'].shape)
# print(train_dataset['target'].shape)

class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()

        self.pointnet = PointNet()
        self.classnet = ClassNet(10)
    
    def forward(self, x):
        _, x = self.pointnet(x)
        x = self.classnet(x)
        return x

model = TotalNet()
model_path = root_prefix + f"/pointnet_modelnet10_train4.pth"
model.load_state_dict(torch.load(model_path, weights_only=True))
model = model.to(device)
model.eval()

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

## Testing Parameters
# epochs = 1
# initial_momentum = 0.5
# final_momentum = 0.99
# momentum_step = (final_momentum - initial_momentum) / epochs

# Testing Loop
print("Beginning Testing")
# for epoch in range(epochs):
with torch.no_grad():
    
    # Update BatchNorm momentum dynamically
    # current_momentum = initial_momentum + epoch * momentum_step
    # for module in model.modules():
    #     if isinstance(module, nn.BatchNorm1d):
    #         module.momentum = current_momentum

    correct = 0
    total = 0
    test_loss = 0.0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)

        # print(data.shape, target.shape)

        # optimizer.zero_grad()

        # Forward pass
        output_scores = model(data)

        print(output_scores.shape)
        
        _, predicted = torch.max(output_scores, 1)
        
        predicted = predicted.reshape(-1, 1)

        total_current = target.size(0)
        correct_current = (predicted == target).sum().item()

        # print(output_scores.shape, target.shape)
        # print(predicted.shape, target.shape)
        # print(correct_current, total_current)

        total += total_current
        correct += correct_current

        # Compute loss
        loss = criterion(output_scores, target.squeeze())
        test_loss += loss.item()

        # Backward pass and optimizer step
        # loss.backward()
        # optimizer.step()

        print(f"batch {batch_idx} of {len(test_loader)}")

    # Update learning rate
    # scheduler.step()

    accuracy = correct / total
    # print(test_loss)

    print(# f"Epoch {epoch + 1}/{epochs}, "
          f"Loss: {test_loss / len(test_loader):.4f}, "
        #   f"LR: {scheduler.get_last_lr()[0]:.6f}, "
        #   f"Momentum: {current_momentum:.3f}"
        f"Accuracy: {accuracy * 100:.2f}%"
        )

# Save the trained model
# model_path = root_prefix + f"/pointnet_modelnet10_{myfilename}.pth"
# torch.save(model.state_dict(), model_path)
# print(f"Model saved to {model_path}")
