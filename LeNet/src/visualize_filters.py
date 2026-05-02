from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data import get_mnist_loaders
from models import LeNet5Style
from utils import ensure_dir, resolve_device


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / (image_max - image_min + 1e-8)


def unnormalize_mnist(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu() * MNIST_STD + MNIST_MEAN).clamp(0.0, 1.0)


def flatten_axes(axes) -> list:
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return [axes]


def output_dir_from_checkpoint(checkpoint_path: Path, output_dir: str | None) -> Path:
    if output_dir is not None:
        return ensure_dir(output_dir)
    if checkpoint_path.parent.name == "checkpoints":
        return ensure_dir(checkpoint_path.parent.parent / "visualizations")
    return ensure_dir(checkpoint_path.parent / "visualizations")


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> LeNet5Style:
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    model = LeNet5Style(
        activation=model_cfg.get("activation", "relu"),
        channels=model_cfg.get("channels", "classic"),
        pooling=model_cfg.get("pooling", "maxpool"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_c1_filters(model: LeNet5Style, output_path: Path) -> None:
    filters = model.c1.weight.detach().cpu()
    num_filters = filters.size(0)
    cols = min(8, num_filters)
    rows = math.ceil(num_filters / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axes = flatten_axes(axes)

    for index, ax in enumerate(axes):
        ax.axis("off")
        if index >= num_filters:
            continue
        image = normalize_image(filters[index, 0])
        ax.imshow(image, cmap="gray")
        ax.set_title(f"C1-{index}", fontsize=8)

    fig.suptitle("C1 learned filters", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect_feature_maps(model: LeNet5Style, x: torch.Tensor) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        c1 = model.activation(model.c1(x))
        s2 = model.s2(c1)
        c3 = model.activation(model.c3(s2))
        logits = model(x)
    return {
        "C1": c1,
        "S2": s2,
        "C3": c3,
        "logits": logits,
    }


def plot_map_grid(
    fig: plt.Figure,
    panel: tuple[float, float, float, float],
    maps: torch.Tensor,
    title: str,
    max_maps: int,
) -> None:
    maps = maps.detach().cpu()
    num_maps = min(max_maps, maps.size(0))
    cols = min(4, num_maps)
    rows = math.ceil(num_maps / cols)
    left, bottom, width, height = panel
    title_height = 0.08 * height
    gap = 0.018
    cell_gap_x = 0.018
    cell_gap_y = 0.035
    grid_bottom = bottom
    grid_height = height - title_height - gap
    cell_width = (width - cell_gap_x * (cols - 1)) / cols
    cell_height = (grid_height - cell_gap_y * (rows - 1)) / rows

    fig.text(left, bottom + height - title_height * 0.6, title, fontsize=13, weight="bold")

    for index in range(rows * cols):
        row = index // cols
        col = index % cols
        x0 = left + col * (cell_width + cell_gap_x)
        y0 = grid_bottom + (rows - row - 1) * (cell_height + cell_gap_y)
        ax = fig.add_axes([x0, y0, cell_width, cell_height])
        ax.axis("off")
        if index >= num_maps:
            continue
        ax.imshow(normalize_image(maps[index]), cmap="viridis")
        ax.set_title(str(index), fontsize=7, pad=2)


def save_feature_maps(
    model: LeNet5Style,
    images: torch.Tensor,
    labels: torch.Tensor,
    output_dir: Path,
    max_maps: int,
) -> None:
    features = collect_feature_maps(model, images)
    predictions = features["logits"].argmax(dim=1).detach().cpu()

    for sample_index in range(images.size(0)):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Intermediate activations", fontsize=15)

        fig.text(
            0.07,
            0.86,
            f"input | true={labels[sample_index].item()} pred={predictions[sample_index].item()}",
            fontsize=13,
            weight="bold",
        )

        ax_input = fig.add_axes([0.11, 0.58, 0.28, 0.28])
        ax_input.imshow(unnormalize_mnist(images[sample_index, 0]), cmap="gray")
        ax_input.axis("off")

        plot_map_grid(fig, (0.55, 0.53, 0.38, 0.34), features["C1"][sample_index], "C1 feature maps", max_maps)
        plot_map_grid(fig, (0.07, 0.08, 0.38, 0.34), features["S2"][sample_index], "S2 pooled maps", max_maps)
        plot_map_grid(fig, (0.55, 0.08, 0.38, 0.34), features["C3"][sample_index], "C3 feature maps", max_maps)

        fig.savefig(
            output_dir
            / (
                f"feature_maps_sample_{sample_index}"
                f"_true_{labels[sample_index].item()}"
                f"_pred_{predictions[sample_index].item()}.png"
            ),
            dpi=180,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize C1 filters and feature maps.")
    parser.add_argument(
        "--checkpoint",
        default="experiments/lenet5_modern/relu_large_maxpool_e10/checkpoints/best.pt",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-feature-maps", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = output_dir_from_checkpoint(checkpoint_path, args.output_dir)
    device = resolve_device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model_from_checkpoint(checkpoint, device)

    save_c1_filters(model, output_dir / "c1_filters.png")

    _, test_loader = get_mnist_loaders(
        data_root=args.data_root,
        batch_size=args.num_samples,
        num_workers=args.num_workers,
    )
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    save_feature_maps(model, images, labels, output_dir, args.max_feature_maps)

    print(f"checkpoint: {checkpoint_path.resolve()}")
    print(f"output_dir: {output_dir.resolve()}")
    print("saved: c1_filters.png")
    print(f"saved: {args.num_samples} feature map figure(s)")


if __name__ == "__main__":
    main()
