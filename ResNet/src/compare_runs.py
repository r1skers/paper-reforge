from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_RUNS = {
    "plain20": Path("experiments/trial/plain20_e10"),
    "plain56": Path("experiments/trial/plain56_e10"),
    "resnet20": Path("experiments/trial/resnet20_e10"),
    "resnet56": Path("experiments/trial/resnet56_e10"),
}


def read_metrics(path: Path) -> list[dict[str, float]]:
    with (path / "metrics.csv").open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [
            {
                "epoch": float(row["epoch"]),
                "lr": float(row["lr"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "test_loss": float(row["test_loss"]),
                "test_acc": float(row["test_acc"]),
            }
            for row in reader
        ]


def summarize(name: str, metrics: list[dict[str, float]]) -> dict[str, str]:
    best = max(metrics, key=lambda row: row["test_acc"])
    last = metrics[-1]
    return {
        "run": name,
        "best_epoch": str(int(best["epoch"])),
        "best_train_acc": f"{best['train_acc']:.4f}",
        "best_test_acc": f"{best['test_acc']:.4f}",
        "last_epoch": str(int(last["epoch"])),
        "last_train_acc": f"{last['train_acc']:.4f}",
        "last_test_acc": f"{last['test_acc']:.4f}",
    }


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run",
                "best_epoch",
                "best_train_acc",
                "best_test_acc",
                "last_epoch",
                "last_train_acc",
                "last_test_acc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(path: Path, all_metrics: dict[str, list[dict[str, float]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    styles = {
        "plain20": {"color": "#9a3412", "linestyle": "--"},
        "plain56": {"color": "#c2410c", "linestyle": "-"},
        "resnet20": {"color": "#1d4ed8", "linestyle": "--"},
        "resnet56": {"color": "#2563eb", "linestyle": "-"},
    }

    panels = [
        ("train_loss", "Train loss", axes[0, 0]),
        ("test_loss", "Test loss", axes[0, 1]),
        ("train_acc", "Train accuracy", axes[1, 0]),
        ("test_acc", "Test accuracy", axes[1, 1]),
    ]

    for metric_name, title, axis in panels:
        for run_name, rows in all_metrics.items():
            epochs = [row["epoch"] for row in rows]
            values = [row[metric_name] for row in rows]
            axis.plot(epochs, values, label=run_name, **styles[run_name])
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)

    axes[0, 0].set_ylabel("loss")
    axes[0, 1].set_ylabel("loss")
    axes[1, 0].set_ylabel("accuracy")
    axes[1, 1].set_ylabel("accuracy")
    axes[0, 1].legend(loc="best")

    fig.suptitle("CIFAR-10 plain vs residual networks, 10 epochs")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/trial"))
    args = parser.parse_args()

    all_metrics = {name: read_metrics(path) for name, path in DEFAULT_RUNS.items()}
    summary_rows = [summarize(name, metrics) for name, metrics in all_metrics.items()]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    figure_path = args.output_dir / "comparison_curves.png"

    write_summary(summary_path, summary_rows)
    plot_comparison(figure_path, all_metrics)

    print(f"wrote: {summary_path}")
    print(f"wrote: {figure_path}")
    for row in summary_rows:
        print(
            f"{row['run']}: best test acc {row['best_test_acc']} "
            f"at epoch {row['best_epoch']}"
        )


if __name__ == "__main__":
    main()

