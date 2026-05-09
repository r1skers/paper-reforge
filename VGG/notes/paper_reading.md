# Very Deep Convolutional Networks for Large-Scale Image Recognition

Paper:

- Title: Very Deep Convolutional Networks for Large-Scale Image Recognition
- Authors: Karen Simonyan, Andrew Zisserman
- Venue: ICLR 2015
- arXiv: https://arxiv.org/abs/1409.1556
- Local PDF: `../vgg_simonyan_zisserman_2014.pdf`

## Position in the Roadmap

VGG is not the final target. Its job in the learning route is to connect AlexNet and ResNet.

```text
AlexNet:
CNNs become effective at ImageNet scale.

VGG:
Depth becomes the main controlled variable.
Small 3x3 convolutions make deep plain CNNs regular and easy to compare.

ResNet:
Plain networks cannot simply keep getting deeper.
Residual learning makes very deep networks trainable.
```

The main thing to take from VGG is not a new module. It is the design principle:

```text
Use a simple, regular architecture.
Control most design choices.
Then test how much depth matters.
```

## Abstract

The abstract asks one central question:

```text
How important is ConvNet depth for large-scale image recognition?
```

The paper's answer is:

```text
Depth matters a lot.
```

To test this cleanly, the authors use very small `3x3` convolution filters throughout the network and increase the number of weight layers up to 16 and 19. The resulting models achieve strong ImageNet performance and become useful visual feature extractors for other tasks.

The important reading point is that VGG is not trying many unrelated tricks at once. It makes the network more regular and pushes depth as the main variable.

## Introduction

The introduction starts from the post-AlexNet context. CNNs had already become successful on large-scale image recognition, helped by:

- large datasets such as ImageNet,
- stronger GPU-based training,
- better CNN architectures.

Many earlier improvements changed details such as the first convolution filter size, stride, or multi-scale image processing. VGG chooses a more direct focus:

```text
What happens if we mainly increase network depth?
```

This is why the paper feels simple. It basically says:

```text
Use 3x3 convolutions.
Use 2x2 max pooling.
Keep the design regular.
Make the network deeper.
Measure the effect.
```

That simplicity is the point. Compared with AlexNet, VGG replaces a more irregular large-filter architecture with a repeated block-style design.

## Core Architecture

The input image size is:

```text
224 x 224 x 3
```

The standard convolution setting is:

```text
kernel size = 3x3
stride = 1
padding = 1
```

This keeps the spatial size unchanged:

```text
output = (input + 2 * padding - kernel_size) / stride + 1
       = (224 + 2 * 1 - 3) / 1 + 1
       = 224
```

So in VGG, convolution usually changes channel count but does not change height or width.

The max pooling setting is:

```text
kernel size = 2x2
stride = 2
```

Pooling halves the spatial size:

```text
224 -> 112 -> 56 -> 28 -> 14 -> 7
```

The classifier has three fully connected layers:

```text
4096 -> 4096 -> 1000
```

The final `1000` corresponds to the 1000 ImageNet classes.

## VGG-16 Shape Route

The famous VGG-16 is configuration D:

```text
64, 64, M
128, 128, M
256, 256, 256, M
512, 512, 512, M
512, 512, 512, M
FC 4096
FC 4096
FC 1000
```

`M` means max pooling.

The shape progression is:

```text
input:        224 x 224 x 3

conv1_1:      224 x 224 x 64
conv1_2:      224 x 224 x 64
pool1:        112 x 112 x 64

conv2_1:      112 x 112 x 128
conv2_2:      112 x 112 x 128
pool2:        56 x 56 x 128

conv3_1:      56 x 56 x 256
conv3_2:      56 x 56 x 256
conv3_3:      56 x 56 x 256
pool3:        28 x 28 x 256

conv4_1:      28 x 28 x 512
conv4_2:      28 x 28 x 512
conv4_3:      28 x 28 x 512
pool4:        14 x 14 x 512

conv5_1:      14 x 14 x 512
conv5_2:      14 x 14 x 512
conv5_3:      14 x 14 x 512
pool5:        7 x 7 x 512

flatten:      7 * 7 * 512 = 25088
fc6:          4096
fc7:          4096
fc8:          1000
```

The CNN pattern is:

```text
Spatial resolution decreases.
Channel dimension increases.
```

Or:

```text
224 -> 112 -> 56 -> 28 -> 14 -> 7
3 -> 64 -> 128 -> 256 -> 512 -> 512
```

This is the standard CNN tradeoff:

```text
early layers: more spatial detail
late layers: richer semantic representation
```

## Why Repeated 3x3 Convolutions Matter

The paper's Section 2.3 explains why VGG prefers multiple small filters instead of one large filter.

Stacking small filters increases the effective receptive field:

```text
two 3x3 conv layers roughly cover a 5x5 receptive field
three 3x3 conv layers roughly cover a 7x7 receptive field
```

This is not exactly the same as a single large convolution, but it gives a comparable receptive field while adding more nonlinearities.

If input and output channel counts are both `C`, then:

```text
one 7x7 conv:
49C^2 parameters

three 3x3 conv layers:
3 * 9C^2 = 27C^2 parameters
```

So repeated `3x3` layers are:

- parameter-efficient compared with large filters,
- deeper,
- equipped with more ReLU nonlinearities,
- easier to arrange into regular blocks.

This is the real architectural idea behind VGG.

## Configurations A to E

The configurations are not unrelated models. They are a controlled depth series.

```text
A: 11 weight layers
B: 13 weight layers
C: 16 weight layers, includes some 1x1 conv layers
D: 16 weight layers, the common VGG-16
E: 19 weight layers, the common VGG-19
```

The term `weight layers` means:

```text
convolution layers + fully connected layers
```

It does not count:

```text
ReLU
pooling
softmax
```

Therefore:

```text
VGG-16 = 13 conv layers + 3 fully connected layers
VGG-19 = 16 conv layers + 3 fully connected layers
```

## Parameter Count Intuition

The convolution parameter formula is:

```text
kernel_h * kernel_w * input_channels * output_channels + output_channels
```

Examples:

```text
conv1_1 = 3 * 3 * 3 * 64 + 64
        = 1,792

conv1_2 = 3 * 3 * 64 * 64 + 64
        = 36,928

conv4_1 = 3 * 3 * 256 * 512 + 512
        = 1,180,160
```

However, the largest parameter cost is in the fully connected classifier.

```text
fc6 = 25088 * 4096 + 4096
    = 102,764,544
```

So VGG is regular and strong, but very heavy. This is one reason later CNNs moved toward more efficient classifier heads and global average pooling.

## Training Framework

The training setup is a classic pre-BatchNorm ImageNet recipe:

```text
loss: CrossEntropyLoss / softmax classification loss
optimizer: SGD
batch size: 256
momentum: 0.9
weight decay: 5e-4
dropout: 0.5
initial learning rate: 0.01
learning rate decay: divide by 10 when validation accuracy saturates
```

Original VGG does not use BatchNorm. BatchNorm became common later. VGG training relies on:

- SGD with momentum,
- weight decay,
- dropout in the classifier,
- learning rate decay,
- careful initialization,
- sometimes initializing deeper networks from shallower ones.

Data augmentation includes:

- random `224 x 224` crop,
- random horizontal flip,
- RGB mean subtraction,
- scale jittering.

The paper uses `S` for the training scale. The image is resized so its smaller side equals `S`, then a `224 x 224` crop is sampled.

Two training-scale settings:

```text
single-scale training: fixed S, such as 256 or 384
multi-scale training: sample S from a range such as [256, 512]
```

## Testing Framework

At test time, VGG does more than a simple center crop.

The fully connected layers can be converted into convolution layers:

```text
fc6 -> 7x7 conv
fc7 -> 1x1 conv
fc8 -> 1x1 conv
```

This lets the model process a larger image in a fully convolutional way and produce a spatial score map. The scores are spatially averaged to get image-level class scores.

The test result can also average predictions from:

- the original image,
- the horizontally flipped image,
- multiple test scales.

This improves accuracy but increases computation.

## What to Remember Before ResNet

VGG proves that deeper plain CNNs can produce stronger visual representations, but it also exposes two limitations:

1. VGG is computationally and parametrically heavy.
2. Plain networks cannot be made arbitrarily deep without optimization problems.

This leads naturally to ResNet.

The ResNet question is:

```text
If VGG shows depth is useful, why do deeper plain networks eventually train worse?
```

The ResNet answer is:

```text
Use residual connections so layers learn residual functions instead of complete mappings.
```

## Closure Summary

VGG can be summarized in one sentence:

```text
VGG turns CNN architecture into a regular stack of small 3x3 convolutions and shows that depth is a major driver of ImageNet performance.
```

Minimum understanding checkpoint:

- I can explain why VGG uses repeated `3x3` convolutions.
- I can write the VGG-16 block structure from memory.
- I can hand-calculate the spatial shape route from `224 x 224` to `7 x 7`.
- I know why VGG has many parameters.
- I know why VGG naturally leads to ResNet.
