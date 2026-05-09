# CIFAR-10 Residual Learning Experiment

## Purpose

This experiment checks the central ResNet claim on a small, local setup:

```text
Increasing the depth of a plain CNN can make optimization worse.
Residual connections make deeper networks easier to train.
```

The goal is not to reproduce the full ImageNet result. The goal is to reproduce the core phenomenon behind ResNet:

```text
plain network degradation
vs
residual network stability
```

## Compared Models

The comparison uses CIFAR-style networks:

```text
plain-20
plain-56
resnet-20
resnet-56
```

The plain and residual models use the same basic stage layout:

```text
32x32, 16 channels
-> 16x16, 32 channels
-> 8x8, 64 channels
-> global average pooling
-> linear classifier
```

The key difference is the block:

```text
PlainBlock:
output = conv_branch(x)

BasicBlock:
output = conv_branch(x) + shortcut(x)
```

## Training Setup

Dataset:

```text
CIFAR-10
```

Training settings:

```text
epochs = 10
batch size = 128
optimizer = SGD
learning rate = 0.1
momentum = 0.9
weight decay = 1e-4
augmentation = random crop + horizontal flip
```

This is a short mechanism-checking run. Since the default milestones are `[10, 15]`, these 10-epoch runs mostly stay at learning rate `0.1`. The results should be interpreted as early training evidence, not final benchmark accuracy.

## Results

Summary:

| Run | Best epoch | Best train acc | Best test acc |
| --- | ---: | ---: | ---: |
| plain-20 | 9 | 79.18% | 75.13% |
| plain-56 | 9 | 44.12% | 43.20% |
| resnet-20 | 9 | 82.37% | 80.52% |
| resnet-56 | 10 | 83.53% | 81.57% |

Generated files:

```text
experiments/trial/summary.csv
experiments/trial/comparison_curves.png
```

## Main Observation

The most important comparison is:

```text
plain-20 best test acc: 75.13%
plain-56 best test acc: 43.20%
```

Making the plain network deeper made it much worse.

This is not overfitting. In overfitting, the deeper model would usually have high training accuracy and low test accuracy:

```text
overfitting:
train acc high
test acc low
```

But plain-56 has low training accuracy and low test accuracy:

```text
plain-56 best train acc: 44.12%
plain-56 best test acc: 43.20%
```

So the problem is not that the model memorized the training set and failed to generalize. The problem is that the deeper plain network did not optimize well enough to fit the training set.

This is the degradation problem.

## Residual Connection Effect

The residual models behave very differently:

```text
resnet-20 best test acc: 80.52%
resnet-56 best test acc: 81.57%
```

The deeper residual network does not collapse. It slightly improves over resnet-20 in this short run.

This supports the main ResNet idea:

```text
Depth is useful, but plain deep networks are hard to optimize.
Residual connections make depth easier to use.
```

## Intuition

In a plain network, every block must directly transform the input feature map into a new feature map:

```text
x -> H(x)
```

If the best behavior for some extra layers is to preserve the existing feature map, the plain block has to learn an identity mapping:

```text
H(x) = x
```

This is not necessarily easy for a stack of convolution, batch normalization, and ReLU layers.

In a residual block:

```text
H(x) = F(x) + x
```

If the best behavior is close to identity, the residual branch only needs to learn:

```text
F(x) = 0
```

So the block can preserve useful features more easily and focus on learning corrections.

## Current Limitations

This is still a small experiment:

- only one seed,
- only 10 epochs,
- no long learning-rate decay phase,
- local CPU training,
- no final benchmark tuning.

The result is strong enough as a mechanism check, but not enough as a polished benchmark reproduction.

## Next Steps

Possible next steps:

1. Run longer training for resnet-20 and resnet-56.
2. Add multiple seeds.
3. Add a report table to the README.
4. Use the comparison curves in the ResNet blog note.
5. Continue reading the paper's experiment section with this result in mind.

