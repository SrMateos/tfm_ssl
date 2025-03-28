import torch
from monai.losses import SSIMLoss
from torch.nn import L1Loss


class CombinedLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.l1 = L1Loss()
        self.ssim = SSIMLoss(spatial_dims=3)
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.ssim(pred, target)
