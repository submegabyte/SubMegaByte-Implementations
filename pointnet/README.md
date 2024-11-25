# PointNet

**PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation**

Arxiv - https://arxiv.org/abs/1612.00593

Paper - https://arxiv.org/pdf/1612.00593

The model along with the classification and segmentation heads have been implemented. The classification head has been tested and seems to give 86% accuracy on the ModelNet10 dataset (https://modelnet.cs.princeton.edu/). The paper reports 89% on the ModelNet40 dataset.

The paper reported training with batch size 32 on a GTX 1080. That was about 1.5 GB of VRAM.

On an RTX 3060, you can comfortably run with a batch size of 256 which took about 11 GB of VRAM, just shy of its limit. But the loss convergence suffered. The other training parameters (batchnorm momentum, learning rate, gamma) have to be adjusted accordingly to take advantage of a large batch.

