from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, augmentation: dict, train: bool):
    ops = [transforms.Resize((image_size, image_size))]
    if train and augmentation.get("enabled", True):
        if augmentation.get("horizontal_flip", True):
            ops.append(transforms.RandomHorizontalFlip())
        if augmentation.get("vertical_flip", True):
            ops.append(transforms.RandomVerticalFlip())
        ops.append(transforms.RandomRotation(degrees=(0, augmentation.get("rotation_degrees", 45))))
        ops.append(
            transforms.ColorJitter(
                brightness=(augmentation.get("brightness_min", 0.3), augmentation.get("brightness_max", 1.0))
            )
        )
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)
