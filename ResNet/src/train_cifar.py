from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from models import count_parameters, plain_cifar, resnet_cifar


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_model(name: str, depth: int) -> nn.Module:
    if name == "resnet":
        return resnet_cifar(depth)
    if name == "plain":
        return plain_cifar(depth)
    raise ValueError(f"Unknown model: {name}")


def build_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def build_fake_loaders(batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.FakeData(
        size=512,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform,
    )
    test_set = datasets.FakeData(
        size=256,
        image_size=(3, 32, 32),
        num_classes=10,
        transform=transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    progress: bool = True,
    desc: str = "",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)
        batches = itertools.islice(loader, max_batches)
    else:
        batches = iter(loader)

    if progress:
        batches = tqdm(batches, total=total_batches, desc=desc, leave=False)

    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, targets)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

        if progress:
            batches.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

    return total_loss / total, correct / total


def write_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(path: Path, rows: list[dict[str, float]]) -> None:
    epochs = [row["epoch"] for row in rows]
    train_acc = [row["train_acc"] for row in rows]
    test_acc = [row["test_acc"] for row in rows]
    train_loss = [row["train_loss"] for row in rows]
    test_loss = [row["test_loss"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, test_loss, label="test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train")
    axes[1].plot(epochs, test_acc, label="test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "plain"], default="resnet")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="*", default=[10, 15])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/scratch"))
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.depth).to(device)
    if args.fake_data:
        train_loader, test_loader = build_fake_loaders(args.batch_size, args.num_workers)
    else:
        train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "device": str(device),
        "parameters": count_parameters(model),
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, default=str)

    rows: list[dict[str, float]] = []
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            args.max_train_batches,
            not args.no_progress,
            f"train {epoch}/{args.epochs}",
        )
        test_loss, test_acc = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            None,
            args.max_test_batches,
            not args.no_progress,
            f"test  {epoch}/{args.epochs}",
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        rows.append(row)
        write_metrics(args.output_dir / "metrics.csv", rows)
        plot_metrics(args.output_dir / "training_curves.png", rows)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.output_dir / "best.pt")
        torch.save(model.state_dict(), args.output_dir / "latest.pt")

        print(
            f"epoch {epoch:03d} "
            f"lr {lr:.5f} "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} "
            f"test loss {test_loss:.4f} acc {test_acc:.4f}"
        )

    print(f"best test acc: {best_acc:.4f}")
    print(f"outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
