from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from models import AlexNetPaper, AlexNetTorchvisionStyle, count_parameters


def shape_text(tensor: torch.Tensor) -> str:
    return "x".join(str(dim) for dim in tensor.shape)


def inspect_sequential(
    rows: list[dict[str, str]],
    prefix: str,
    module: nn.Sequential,
    x: torch.Tensor,
) -> torch.Tensor:
    for index, layer in enumerate(module):
        x = layer(x)
        rows.append(
            {
                "stage": f"{prefix}.{index:02d}",
                "layer": layer.__class__.__name__,
                "output_shape": shape_text(x),
            }
        )
    return x


def inspect_model(model_name: str, input_size: int) -> list[dict[str, str]]:
    if model_name == "paper":
        model = AlexNetPaper()
    elif model_name == "torchvision":
        model = AlexNetTorchvisionStyle()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.eval()
    x = torch.zeros(1, 3, input_size, input_size)
    rows: list[dict[str, str]] = [
        {
            "stage": "input",
            "layer": "Input",
            "output_shape": shape_text(x),
        }
    ]

    with torch.no_grad():
        x = inspect_sequential(rows, "features", model.features, x)
        if hasattr(model, "avgpool"):
            x = model.avgpool(x)
            rows.append(
                {
                    "stage": "avgpool",
                    "layer": "AdaptiveAvgPool2d",
                    "output_shape": shape_text(x),
                }
            )
        x = torch.flatten(x, 1)
        rows.append(
            {
                "stage": "flatten",
                "layer": "Flatten",
                "output_shape": shape_text(x),
            }
        )
        x = inspect_sequential(rows, "classifier", model.classifier, x)

    rows.append(
        {
            "stage": "parameters",
            "layer": model.__class__.__name__,
            "output_shape": str(count_parameters(model)),
        }
    )
    return rows


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["stage", "layer", "output_shape"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print AlexNet layer shapes using a dummy input tensor."
    )
    parser.add_argument(
        "--model",
        choices=["paper", "torchvision"],
        default="paper",
        help="paper uses the classic 227 input path; torchvision uses 224-friendly conv1 padding.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Square image size. Defaults to 227 for paper, 224 for torchvision.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/layer_shapes.csv"),
        help="Where to save the shape table.",
    )
    args = parser.parse_args()

    input_size = args.input_size
    if input_size is None:
        input_size = 227 if args.model == "paper" else 224

    rows = inspect_model(args.model, input_size)
    width_stage = max(len(row["stage"]) for row in rows)
    width_layer = max(len(row["layer"]) for row in rows)

    for row in rows:
        print(
            f"{row['stage']:<{width_stage}}  "
            f"{row['layer']:<{width_layer}}  "
            f"{row['output_shape']}"
        )

    write_csv(rows, args.csv)
    print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
