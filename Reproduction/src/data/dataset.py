from __future__ import annotations

from pathlib import Path

from torchvision.datasets import ImageFolder

from Reproduction.src.data.transforms import build_transforms


def create_imagefolder(data_dir: str | Path, image_size: int, augmentation: dict, train: bool) -> ImageFolder:
    transform = build_transforms(image_size=image_size, augmentation=augmentation, train=train)
    return ImageFolder(root=str(data_dir), transform=transform)
