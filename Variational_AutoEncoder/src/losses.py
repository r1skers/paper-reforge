"""Loss functions for the VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood under a Bernoulli decoder.

    For MNIST we model each pixel as Bernoulli with probability given by
    recon_x. The negative log likelihood is binary cross entropy.
    """

    x = x.view(x.size(0), -1)
    return F.binary_cross_entropy(recon_x, x, reduction="sum")  # ELBO can be interpreted as a sum over the batch, so we sum the reconstruction loss over the batch as well


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q_phi(z | x) || p(z)) for diagonal Gaussian q and standard normal p."""

    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total negative ELBO, reconstruction loss, and KL loss."""

    recon = reconstruction_loss(recon_x, x)
    kl = kl_divergence(mu, logvar)
    return recon + kl, recon, kl
