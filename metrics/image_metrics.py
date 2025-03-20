from monai.metrics import PSNRMetric
from monai.metrics.regression import SSIMMetric
from torch.nn import L1Loss
import torch

def calculate_metrics(pred, target):
    return {
        "mae": L1Loss()(pred, target).item(),
        "psnr": PSNRMetric(max_val=100)(pred, target).item(),
        "ssim": SSIMMetric(spatial_dims=3)(pred, target).item(),
    }


