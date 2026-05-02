"""Model factory helpers."""

from __future__ import annotations

from src.models.vae_cnn import CNNVAE
from src.models.vae_mlp import VAE as MLPVAE


def build_vae(model_cfg: dict) -> MLPVAE | CNNVAE:
    model_type = model_cfg.get("type", "mlp")
    if model_type == "mlp":
        return MLPVAE(
            input_dim=model_cfg["input_dim"],
            hidden_dim=model_cfg["hidden_dim"],
            latent_dim=model_cfg["latent_dim"],
        )
    if model_type == "cnn":
        return CNNVAE(
            input_channels=model_cfg.get("input_channels", 1),
            input_size=model_cfg.get("input_size", 28),
            latent_dim=model_cfg["latent_dim"],
        )
    raise ValueError(f"unknown model type: {model_type}")
