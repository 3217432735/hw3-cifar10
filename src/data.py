from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("CIFAR10_ROOT", r"D:\asya-ai\data"))
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class CIFAR10Part(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool,
        transform: transforms.Compose,
        indices: list[int] | None = None,
    ) -> None:
        self.dataset = datasets.CIFAR10(
            root=str(root), train=train, download=True, transform=None
        )
        self.indices = list(range(len(self.dataset))) if indices is None else indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, target = self.dataset[self.indices[index]]
        return self.transform(image), target


def build_transform(aug: str = "basic", train: bool = True) -> transforms.Compose:
    normalize = transforms.Normalize(MEAN, STD)
    if not train:
        return transforms.Compose([transforms.ToTensor(), normalize])
    if aug == "none":
        return transforms.Compose([transforms.ToTensor(), normalize])
    if aug == "basic":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    if aug == "strong":
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"
                ),
            ]
        )
    raise ValueError(f"Unknown augmentation: {aug}")


def split_indices(seed: int = 42, val_size: int = 5000):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(50000, generator=generator).tolist()
    return permutation[:-val_size], permutation[-val_size:]


def make_loaders(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    aug: str = "basic",
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
):
    train_indices, val_indices = split_indices(seed)
    train_set = CIFAR10Part(data_root, True, build_transform(aug, True), train_indices)
    val_set = CIFAR10Part(data_root, True, build_transform(aug, False), val_indices)
    test_set = CIFAR10Part(data_root, False, build_transform(aug, False))

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = make_loaders(num_workers=0)
    print(
        f"train={len(train_loader.dataset)}, "
        f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )
