# dataset_cifar.py
import os
import torchvision.transforms as T
from torchvision.datasets import CIFAR10


def get_cifar10_dataset(root=None):
    """Training dataset: random horizontal flip + normalize to [-1, 1]."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return CIFAR10(root=root, train=True, download=True, transform=transform)


def get_cifar10_eval_dataset(root=None):
    """Evaluation dataset for FID: images in [0, 1] (no normalization)."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")
    return CIFAR10(root=root, train=True, download=True, transform=T.ToTensor())
