# dataset_cifar.py
import os
import torchvision.transforms as T
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10


def _maybe_subset(dataset, class_indices):
    if not class_indices:
        return dataset
    class_set = set(class_indices)
    indices = [i for i, label in enumerate(dataset.targets) if label in class_set]
    return Subset(dataset, indices)


def get_cifar10_dataset(root=None, class_indices=None, color_jitter=0.0):
    """Training dataset: random horizontal flip + optional color jitter, normalized to [-1, 1]."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")
    extra = []
    if color_jitter > 0.0:
        extra.append(T.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                   saturation=color_jitter, hue=color_jitter / 4))
    transform = T.Compose([T.RandomHorizontalFlip(), *extra,
                           T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return _maybe_subset(CIFAR10(root=root, train=True, download=True, transform=transform), class_indices)


def get_cifar10_eval_dataset(root=None, class_indices=None):
    """Evaluation dataset for FID: images in [0, 1] (no normalization)."""
    if root is None:
        root = os.environ.get("CIFAR10_PATH", "./data")
    return _maybe_subset(CIFAR10(root=root, train=True, download=True, transform=T.ToTensor()), class_indices)
