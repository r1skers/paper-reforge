# VGG

Study folder for VGG / very deep convolutional networks.

## Paper

- **Title:** Very Deep Convolutional Networks for Large-Scale Image Recognition
- **Authors:** Karen Simonyan, Andrew Zisserman
- **Venue:** ICLR 2015
- **arXiv:** https://arxiv.org/abs/1409.1556
- **PDF:** `vgg_simonyan_zisserman_2014.pdf`

## Reading Focus

1. Why VGG replaces large convolution filters with repeated `3x3` filters.
2. How depth changes from configurations A to E.
3. Why VGG-16 and VGG-19 became common visual backbones.
4. How the shape progression compares with AlexNet.
5. Why the fully connected classifier still dominates parameter count.

## Notes

- [Paper reading note](notes/paper_reading.md)

## Current Takeaway

VGG is the bridge from AlexNet to ResNet:

```text
AlexNet: large-scale CNNs work on ImageNet.
VGG: deeper plain CNNs with repeated 3x3 convolutions learn stronger visual features.
ResNet: going deeper needs residual connections to solve the degradation problem.
```
