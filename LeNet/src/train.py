from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from data import get_mnist_loaders
from models import LeNet5Style
from utils import append_metrics, ensure_dir, load_config, resolve_device, set_seed, write_json, write_shapes


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == y).float().sum().item()


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def run_one_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    desc: str = "epoch",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for x, y in progress:
        x = x.to(device)
        y = y.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = loss_fn(logits, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, y)
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples:.4f}",
        )

    return total_loss / total_samples, total_correct / total_samples


def save_curves(metrics_path: Path, output_path: Path) -> None:
    try:
        import csv
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = []
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    with metrics_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            test_losses.append(float(row["test_loss"]))
            train_accs.append(float(row["train_accuracy"]))
            test_accs.append(float(row["test_accuracy"]))

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax_loss.plot(epochs, train_losses, label="train")
    ax_loss.plot(epochs, test_losses, label="test")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()

    ax_acc.plot(epochs, train_accs, label="train")
    ax_acc.plot(epochs, test_accs, label="test")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def output_dir_from_config(cfg: dict) -> Path:
    output_cfg = cfg["output"]
    run_dir = Path(output_cfg["run_dir"])
    run_name = output_cfg.get("run_name")
    return run_dir / run_name if run_name else run_dir


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.data_root is not None:
        cfg["data"]["root"] = args.data_root
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        cfg["train"]["learning_rate"] = args.learning_rate
    if args.activation is not None:
        cfg["model"]["activation"] = args.activation
    if args.channels is not None:
        cfg["model"]["channels"] = args.channels
    if args.pooling is not None:
        cfg["model"]["pooling"] = args.pooling
    if args.output_dir is not None:
        cfg["output"]["run_dir"] = args.output_dir
        cfg["output"]["run_name"] = ""
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LeNet-5 style CNN on MNIST.")
    parser.add_argument("--config", default="configs/modern.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"])
    parser.add_argument("--channels", choices=["small", "classic", "large"])
    parser.add_argument("--pooling", choices=["maxpool",  "avgpool"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_config(args.config), args)

    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])

    output_dir = ensure_dir(output_dir_from_config(cfg))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = output_dir / "metrics.csv"

    write_json(output_dir / "config.json", cfg | {"resolved_device": str(device)})

    train_loader, test_loader = get_mnist_loaders(
        data_root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = LeNet5Style(
        activation=cfg["model"].get("activation", "relu"),
        channels=cfg["model"].get("channels", "classic"),
        pooling=cfg["model"].get("pooling", "maxpool"),
    ).to(device)
    parameter_counts = count_parameters(model)
    write_json(output_dir / "model_summary.json", parameter_counts)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])

    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x[: cfg["data"]["batch_size"]].to(device)
    with torch.no_grad():
        _, shapes = model(sample_x, return_shapes=True)
    write_shapes(output_dir / "layer_shapes.csv", shapes)

    print(f"device: {device}")
    print(f"output_dir: {output_dir.resolve()}")
    print(
        "parameters: "
        f"total={parameter_counts['total']:,} "
        f"trainable={parameter_counts['trainable']:,}"
    )
    print("layer shapes:")
    for layer, shape in shapes.items():
        print(f"  {layer}: {shape}")

    fields = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    best_test_acc = 0.0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            desc=f"train {epoch:03d}",
        )
        test_loss, test_acc = run_one_epoch(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            desc=f"test {epoch:03d}",
        )

        row = {
            "epoch": epoch,
            "train_loss": f"{train_loss:.6f}",
            "train_accuracy": f"{train_acc:.6f}",
            "test_loss": f"{test_loss:.6f}",
            "test_accuracy": f"{test_acc:.6f}",
        }
        append_metrics(metrics_path, row, fields)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "test_accuracy": test_acc,
            "config": cfg,
        }
        torch.save(state, checkpoint_dir / "latest.pt")
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(state, checkpoint_dir / "best.pt")

    save_curves(metrics_path, output_dir / "training_curves.png")


if __name__ == "__main__":
    main()
