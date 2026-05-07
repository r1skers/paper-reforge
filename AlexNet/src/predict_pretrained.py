from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import AlexNet_Weights, alexnet


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def predict(image_path: Path, topk: int) -> list[tuple[str, float]]:
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)
    model.eval()

    preprocess = weights.transforms()
    image = preprocess(load_image(image_path)).unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        probabilities = logits.softmax(dim=1)[0]
        scores, indices = probabilities.topk(topk)

    categories = weights.meta["categories"]
    return [(categories[index], score.item()) for score, index in zip(scores, indices)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ImageNet top-k prediction with torchvision's pretrained AlexNet."
    )
    parser.add_argument("image", type=Path, help="Path to a local image.")
    parser.add_argument("--topk", type=int, default=5, help="Number of classes to print.")
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    for rank, (label, score) in enumerate(predict(args.image, args.topk), start=1):
        print(f"{rank:>2}. {label:<30} {score:.4f}")


if __name__ == "__main__":
    main()
