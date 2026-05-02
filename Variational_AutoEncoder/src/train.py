"""Train a minimal VAE on MNIST."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import optim
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import get_mnist_loaders
from src.losses import vae_loss
from src.models import build_vae
from src.utils import ensure_dir, load_config, resolve_device, set_seed


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
    for batch_idx, (x, _) in enumerate(progress, start=1):
        x = x.to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss, recon, kl = vae_loss(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

        if batch_idx % log_interval == 0:
            avg_loss = total_loss / (batch_idx * x.size(0))
            progress.set_postfix(loss=f"{avg_loss:.4f}")

    dataset_size = len(train_loader.dataset)
    return total_loss / dataset_size, total_recon / dataset_size, total_kl / dataset_size


@torch.no_grad()
def evaluate(model: torch.nn.Module, test_loader, device: torch.device) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x, _ in test_loader:
        x = x.to(device)
        recon_x, mu, logvar = model(x)
        loss, recon, kl = vae_loss(recon_x, x, mu, logvar)
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    dataset_size = len(test_loader.dataset)
    return total_loss / dataset_size, total_recon / dataset_size, total_kl / dataset_size


@torch.no_grad()
def save_reconstructions(model: torch.nn.Module, test_loader, device: torch.device, path: Path) -> None:
    model.eval()
    x, _ = next(iter(test_loader))
    x = x.to(device)[:8]
    recon_x, _, _ = model(x)
    comparison = torch.cat([x.view(-1, 1, 28, 28), recon_x.view(-1, 1, 28, 28)])
    save_image(comparison.cpu(), path, nrow=8)


@torch.no_grad()
def save_samples(model: torch.nn.Module, device: torch.device, path: Path, num_samples: int = 64) -> None:
    model.eval()
    z = torch.randn(num_samples, model.latent_dim, device=device)
    samples = model.decode(z).view(-1, 1, 28, 28)
    save_image(samples.cpu(), path, nrow=8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mnist_mlp.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    run_root = ensure_dir(cfg["output"]["run_dir"])
    run_name = cfg["output"].get("run_name", "default")
    run_dir = ensure_dir(run_root / "runs" / run_name)
    sample_dir = ensure_dir(run_dir / "samples")
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    log_dir = ensure_dir(run_dir / "logs")

    train_loader, test_loader = get_mnist_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_vae(cfg["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

    metrics_path = log_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_recon,train_kl,test_loss,test_recon,test_kl\n")

    print(f"device: {device}")
    print(f"run_dir: {run_dir}")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_recon, train_kl = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            log_interval=cfg["train"]["log_interval"],
        )
        test_loss, test_recon, test_kl = evaluate(model, test_loader, device)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} recon {train_recon:.4f} kl {train_kl:.4f} | "
            f"test loss {test_loss:.4f} recon {test_recon:.4f} kl {test_kl:.4f}"
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_recon:.6f},{train_kl:.6f},"
                f"{test_loss:.6f},{test_recon:.6f},{test_kl:.6f}\n"
            )

        save_reconstructions(model, test_loader, device, sample_dir / f"reconstruction_epoch_{epoch:03d}.png")
        save_samples(model, device, sample_dir / f"samples_epoch_{epoch:03d}.png")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
            },
            checkpoint_dir / "latest.pt",
        )


if __name__ == "__main__":
    main()
