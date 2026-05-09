from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from models import count_parameters, plain_cifar, resnet_cifar


def build_model(name: str, depth: int) -> nn.Module:
    if name == "resnet":
        return resnet_cifar(depth)
    if name == "plain":
        return plain_cifar(depth)
    raise ValueError(f"Unknown model: {name}")


def inspect_shapes(model: nn.Module, input_shape: tuple[int, int, int, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    hooks = []

    def register(name: str, module: nn.Module) -> None:
        if len(list(module.children())) > 0:
            return

        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                rows.append(
                    {
                        "layer": name,
                        "type": module.__class__.__name__,
                        "output_shape": "x".join(str(dim) for dim in output.shape),
                    }
                )

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name:
            register(name, module)

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape)
        model(dummy)

    for hook in hooks:
        hook.remove()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "plain"], default="resnet")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/layer_shapes.csv"))
    args = parser.parse_args()

    model = build_model(args.model, args.depth)
    rows = inspect_shapes(model, (1, 3, 32, 32))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["layer", "type", "output_shape"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"model: {args.model}-{args.depth}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()

