import torch
import torch.nn as nn


class KLLoss(nn.Module):
    """
    Calculates the Kullback-Leibler (KL) divergence loss between the learned
    latent distribution and a standard normal distribution.

    This loss is commonly used in Variational Autoencoders (VAEs) to regularize
    the latent space, encouraging the encoder to produce latent variables
    that follow a known prior distribution (typically a standard normal distribution).

    The formula for the KL divergence between two Gaussian distributions
    (one with mean `mu` and standard deviation `sigma`, and another with mean 0
    and standard deviation 1) is:
        KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    This implementation uses a slightly rearranged but equivalent formula for
    numerical stability and calculates the mean over the batch.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Initializes the KLLoss module.

        Args:
            epsilon (float): A small constant added for numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """
        Calcula la pérdida de divergencia KL.

        Args:
            z_mu (torch.Tensor): La media de la distribución latente.
                                 Shape: (batch_size, latent_dim)
            z_sigma (torch.Tensor): La desviación estándar de la distribución latente.
                                    Shape: (batch_size, latent_dim)

        Returns:
            torch.Tensor: La pérdida de divergencia KL calculada (escalar).
        """
        # Suma sobre las dimensiones latentes
        kl_div_per_sample = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + self.epsilon) - 1,
            dim=list(range(1, len(z_sigma.shape)))
        )

        # Media sobre el lote
        return torch.mean(kl_div_per_sample)
