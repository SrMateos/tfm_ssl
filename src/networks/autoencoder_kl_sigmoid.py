import torch
from monai.networks.nets import AutoencoderKL

class AutoencoderKLSigmoid(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # El forward de AutoencoderKL devuelve (reconstruction, z_mu, z_sigma)
        reconstruction, z_mu, z_sigma = super().forward(x)
        return self.sigmoid(reconstruction), z_mu, z_sigma

    def reconstruct(self, x):
        # El método reconstruct solo devuelve la reconstrucción
        reconstruction = super().reconstruct(x)
        return self.sigmoid(reconstruction)
