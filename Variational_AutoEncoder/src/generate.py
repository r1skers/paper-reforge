"""Generate samples from a trained VAE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.models import build_vae
from src.utils import ensure_dir, resolve_device


@torch.no_grad()
def generate_samples(
    checkpoint_path: str | Path,
    output_path: str | Path,
    num_samples: int = 64,
    device_name: str = "auto",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    device = resolve_device(device_name)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model_cfg = cfg["model"]

    model = build_vae(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ensure_dir(output_path.parent)
    z = torch.randn(num_samples, model.latent_dim, device=device)
    samples = model.decode(z).view(-1, 1, 28, 28)
    save_image(samples.cpu(), output_path, nrow=8)
    print(f"saved samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="experiments/mnist_mlp/runs/latent20_e20/checkpoints/latest.pt")
    parser.add_argument("--output", default="experiments/mnist_mlp/runs/latent20_e20/samples/generated_from_checkpoint.png")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_samples=args.num_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
