import torch
import torch.nn as nn
from focal_frequency_loss import FocalFrequencyLoss as FFL

class FFL3D(nn.Module):
    """Aplica Focal Frequency Loss por cortes en volúmenes (N,C,H,W,D)."""
    def __init__(self, **ffl_kwargs):
        super().__init__()
        self.ffl = FFL(**ffl_kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: (N, C, H, W, D)
        assert pred.dim() == 5 and target.dim() == 5, "FFL3D espera tensores NCHWD"
        n, c, h, w, d = pred.shape
        # Pasamos D al eje batch: (N*D, C, H, W)
        pred2d = pred.permute(0, 4, 1, 2, 3).contiguous().reshape(n * d, c, h, w)
        targ2d = target.permute(0, 4, 1, 2, 3).contiguous().reshape(n * d, c, h, w)
        return self.ffl(pred2d, targ2d)
