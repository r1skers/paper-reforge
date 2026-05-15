"""
V3 — Data loaders.

Two datasets supported:
    - MNIST     : 28×28 grayscale, 10 digit classes — smoke-test target
    - CIFAR-10  : 32×32 RGB,        10 object classes — real ViT target

Both return (train_loader, test_loader) with the same calling convention.

ViT-friendly patch_size suggestions:
    MNIST     (28×28) :  patch_size=7  → 16 tokens
                          patch_size=4  → 49 tokens
    CIFAR-10  (32×32) :  patch_size=4  → 64 tokens
                          patch_size=8  → 16 tokens

Normalization stats are the standard dataset-level mean/std (computed once
on the train split and reused everywhere).

Run
---
    python data.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms


# Paper-reforge convention: put downloaded datasets under ViT/data/.
_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Dataset normalization stats — standard values reused everywhere.
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


def get_mnist_loaders(batch_size=128, num_workers=0, root=_DATA_ROOT):
    """
    Build train / test DataLoaders for MNIST.

    Parameters
    ----------
    batch_size  : int
    num_workers : int — keep 0 on Windows / single-CPU machines to avoid
                        multiprocessing overhead and weird startup issues
    root        : path-like — where to download / cache the dataset

    Returns
    -------
    train_loader, test_loader : torch.utils.data.DataLoader
    """
    # ------------------------------------------------------------------ #
    # TODO 1 — Build the transform pipelines.                             #
    #                                                                     #
    # Train and test both use:                                            #
    #   - transforms.ToTensor()      → (0..1) float, shape (1, 28, 28)    #
    #   - transforms.Normalize(MNIST_MEAN, MNIST_STD)                     #
    #                                                                     #
    # For MNIST we DON'T augment (no flip, no crop) — digits have a       #
    # canonical orientation, augmenting hurts.                            #
    #                                                                     #
    #   tf = transforms.Compose([                                          #
    #       transforms.ToTensor(),                                         #
    #       transforms.Normalize(MNIST_MEAN, MNIST_STD),                   #
    #   ])                                                                 #
    # ------------------------------------------------------------------ #
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    # ------------------------------------------------------------------ #
    # TODO 2 — Build datasets and loaders.                                #
    #                                                                     #
    #   train_set = torchvision.datasets.MNIST(                            #
    #       root=root, train=True,  download=True, transform=tf)          #
    #   test_set  = torchvision.datasets.MNIST(                            #
    #       root=root, train=False, download=True, transform=tf)          #
    #                                                                     #
    #   train_loader = DataLoader(train_set, batch_size=batch_size,        #
    #                             shuffle=True,  num_workers=num_workers,  #
    #                             pin_memory=False)                        #
    #   test_loader  = DataLoader(test_set,  batch_size=batch_size,        #
    #                             shuffle=False, num_workers=num_workers,  #
    #                             pin_memory=False)                        #
    #   return train_loader, test_loader                                   #
    # ------------------------------------------------------------------ #
    train_set = torchvision.datasets.MNIST(
        root=root, train=True,  download=True, transform=tf
    )
    test_set  = torchvision.datasets.MNIST(
        root=root, train=False, download=True, transform=tf
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, num_workers=0, root=_DATA_ROOT,
                        augment_train=True):
    """
    Build train / test DataLoaders for CIFAR-10.

    Parameters
    ----------
    batch_size    : int
    num_workers   : int
    root          : path-like
    augment_train : bool — if True, train pipeline adds RandomCrop +
                           RandomHorizontalFlip (standard CIFAR recipe).
                           Test pipeline is always deterministic.

    Returns
    -------
    train_loader, test_loader
    """
    # ------------------------------------------------------------------ #
    # TODO 3 — Build train transform (with optional augmentation).        #
    #                                                                     #
    #   norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)            #
    #                                                                     #
    #   if augment_train:                                                  #
    #       train_tf = transforms.Compose([                                #
    #           transforms.RandomCrop(32, padding=4),                      #
    #           transforms.RandomHorizontalFlip(),                         #
    #           transforms.ToTensor(),                                     #
    #           norm,                                                       #
    #       ])                                                              #
    #   else:                                                               #
    #       train_tf = transforms.Compose([                                 #
    #           transforms.ToTensor(),                                      #
    #           norm,                                                        #
    #       ])                                                               #
    #                                                                       #
    #   test_tf = transforms.Compose([                                      #
    #       transforms.ToTensor(),                                          #
    #       norm,                                                            #
    #   ])                                                                   #
    # ------------------------------------------------------------------ #
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if augment_train:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            norm,
        ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])
    # ------------------------------------------------------------------ #
    # TODO 4 — Build datasets and loaders. Same pattern as MNIST but use  #
    # torchvision.datasets.CIFAR10 and pass test_tf to the test dataset.  #
    # ------------------------------------------------------------------ #

    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True,  download=True, transform=train_tf
    )
    test_set  = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=test_tf
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------- #
# Driver: quick sanity                                                         #
# ---------------------------------------------------------------------------- #


def main():
    print("Loading MNIST ...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128, num_workers=0)
    x, y = next(iter(train_loader))
    print(f"MNIST train batch:  x.shape={tuple(x.shape)}   y.shape={tuple(y.shape)}")
    print(f"  x.min={x.min().item():.3f}   x.max={x.max().item():.3f}")
    print(f"  unique labels in this batch: {sorted(set(y.tolist()))}")
    print(f"  len(train_set)={len(train_loader.dataset)}  len(test_set)={len(test_loader.dataset)}")


if __name__ == "__main__":
    main()
