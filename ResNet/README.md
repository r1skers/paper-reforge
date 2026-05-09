# ResNet

Study folder for ResNet / deep residual learning.

## Paper

- **Title:** Deep Residual Learning for Image Recognition
- **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Venue:** CVPR 2016
- **arXiv:** https://arxiv.org/abs/1512.03385
- **PDF:** `resnet_he_2015.pdf`

## Reading Focus

1. Why simply increasing depth causes the degradation problem.
2. What residual learning changes compared with a plain deep network.
3. How identity shortcut connections work.
4. The difference between BasicBlock and Bottleneck.
5. Why ResNet becomes the natural next step after VGG.

## Current Takeaway

ResNet answers the question left open by VGG:

```text
VGG: deeper plain CNNs are useful.
ResNet: very deep CNNs need residual connections to remain trainable.
```

## Reproduction Target

Use CIFAR-10 rather than ImageNet. The goal is to reproduce the paper's main
trend, not the full ImageNet result:

```text
plain network gets harder to optimize when made deeper.
residual network keeps benefiting from depth.
```

Primary comparison:

```text
plain-20 vs resnet-20
plain-56 vs resnet-56
```

## Setup

```powershell
pip install -r requirements.txt
```

## Inspect Shapes

```powershell
python .\src\inspect_shapes.py --model resnet --depth 20
```

This writes:

```text
outputs/layer_shapes.csv
```

## Train CIFAR-10

Quick smoke run:

```powershell
python .\src\train_cifar.py --model resnet --depth 20 --epochs 1 --fake-data --max-train-batches 2 --max-test-batches 1 --output-dir experiments\smoke\resnet20
```

Training shows a batch progress bar by default. Add `--no-progress` for quieter
logs.

Main comparison runs:

```powershell
python .\src\train_cifar.py --model plain --depth 20 --epochs 100 --output-dir experiments\plain20
python .\src\train_cifar.py --model resnet --depth 20 --epochs 100 --output-dir experiments\resnet20
python .\src\train_cifar.py --model plain --depth 56 --epochs 100 --output-dir experiments\plain56
python .\src\train_cifar.py --model resnet --depth 56 --epochs 100 --output-dir experiments\resnet56
```

Each run writes:

```text
config.json
metrics.csv
training_curves.png
best.pt
latest.pt
```
