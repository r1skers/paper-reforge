"""Visualize a 2D VAE latent space."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from src.data import get_mnist_loaders
from src.models.vae_mlp import VAE
from src.utils import ensure_dir, resolve_device


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[VAE, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    model = VAE(
        input_dim=model_cfg["input_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        latent_dim=model_cfg["latent_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def plot_latent_scatter(
    model: VAE,
    data_loader,
    device: torch.device,
    output_path: str | Path,
    max_points: int = 5000,
) -> None:
    if model.latent_dim != 2:
        raise ValueError("latent scatter requires latent_dim=2")

    mus = []
    labels = []
    seen = 0
    for x, y in data_loader:
        x = x.to(device)
        x = x.view(x.size(0), model.input_dim)
        mu, _ = model.encode(x)
        mus.append(mu.cpu())
        labels.append(y)
        seen += x.size(0)
        if seen >= max_points:
            break

    z = torch.cat(mus, dim=0)[:max_points]
    y = torch.cat(labels, dim=0)[:max_points]

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=6, alpha=0.75)
    plt.colorbar(scatter, ticks=range(10), label="digit")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.title("MNIST test set encoded by q_phi(z | x)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


@torch.no_grad()
def save_latent_manifold(
    model: VAE,
    device: torch.device,
    output_path: str | Path,
    grid_size: int = 20,
    value_range: float = 3.0,
) -> None:
    if model.latent_dim != 2:
        raise ValueError("latent manifold requires latent_dim=2")

    coords = torch.linspace(-value_range, value_range, grid_size, device=device)
    grid = []
    for y in reversed(coords):
        for x in coords:
            grid.append(torch.stack([x, y]))
    z = torch.stack(grid, dim=0)
    images = model.decode(z).view(-1, 1, 28, 28)
    save_image(images.cpu(), output_path, nrow=grid_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="experiments/mnist_mlp/runs/latent2_e20/checkpoints/latest.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/mnist_mlp/runs/latent2_e20/visualizations",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--range", type=float, default=3.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    model, cfg = load_model(args.checkpoint, device)
    if model.latent_dim != 2:
        raise ValueError(f"expected latent_dim=2, got latent_dim={model.latent_dim}")

    _, test_loader = get_mnist_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=0,
    )

    scatter_path = output_dir / "latent_scatter.png"
    manifold_path = output_dir / "latent_manifold.png"
    plot_latent_scatter(model, test_loader, device, scatter_path, max_points=args.max_points)
    save_latent_manifold(
        model,
        device,
        manifold_path,
        grid_size=args.grid_size,
        value_range=args.range,
    )
    print(f"saved latent scatter to {scatter_path}")
    print(f"saved latent manifold to {manifold_path}")


if __name__ == "__main__":
    main()
