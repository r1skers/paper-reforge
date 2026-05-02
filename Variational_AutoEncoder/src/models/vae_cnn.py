"""Small convolutional Variational Autoencoder for MNIST."""

from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """Map an MNIST image to the parameters of q_phi(z | x)."""

    def __init__(self, input_channels: int = 1, latent_dim: int = 20) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class CNNDecoder(nn.Module):
    """Map a latent sample z to the parameters of p_theta(x | z)."""

    def __init__(self, latent_dim: int = 20, output_channels: int = 1) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), 64, 7, 7)
        x = self.net(h)
        return x.view(z.size(0), -1)


class CNNVAE(nn.Module):
    """CNN VAE with a diagonal Gaussian approximate posterior."""

    def __init__(self, input_channels: int = 1, input_size: int = 28, latent_dim: int = 20) -> None:
        super().__init__()
        if input_size != 28:
            raise ValueError("CNNVAE currently expects 28x28 MNIST images")
        self.input_channels = input_channels
        self.input_size = input_size
        self.input_dim = input_channels * input_size * input_size
        self.latent_dim = latent_dim
        self.encoder = CNNEncoder(input_channels=input_channels, latent_dim=latent_dim)
        self.decoder = CNNDecoder(latent_dim=latent_dim, output_channels=input_channels)

    def _as_image(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.view(x.size(0), self.input_channels, self.input_size, self.input_size)
        return x

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(self._as_image(x))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
