from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .utils import CLASS_NAMES, CLASS_TO_INDEX, RAW_TO_CANONICAL, ensure_dir


PathLike = Union[str, Path]
IMG_SIZE = (227, 227)


@dataclass
class DatasetConfig:
    data_dir: PathLike
    batch_size: int = 32
    num_workers: int = 0
    augmentation: bool = False
    seed: int = 42
    img_size: Tuple[int, int] = IMG_SIZE
    rotation_mode: str = "positive"


class CrackDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        augmentation: bool,
        img_size: Tuple[int, int],
        rotation_mode: str = "positive",
    ) -> None:
        self.samples = list(samples)
        self.transform = build_transform(
            augmentation=augmentation,
            rotation_mode=rotation_mode,
        )
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError("Failed to read image: {0}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image)
        image_tensor = self.transform(image)
        return image_tensor, label, str(image_path)


def build_transform(
    augmentation: bool,
    rotation_mode: str,
) -> transforms.Compose:
    ops = []
    if augmentation:
        if rotation_mode not in {"positive", "symmetric"}:
            raise ValueError("Unsupported rotation mode: {0}".format(rotation_mode))
        ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=(0.3, 1.0)),
                PositiveRandomRotation(45)
                if rotation_mode == "positive"
                else transforms.RandomRotation(45),
            ]
        )
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class PositiveRandomRotation:
    def __init__(self, max_degrees: float) -> None:
        self.max_degrees = max_degrees

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = float(transforms.RandomRotation.get_params([0.0, self.max_degrees]))
        return TF.rotate(image, angle)


def create_dataloaders(config: DatasetConfig):
    data_dir = Path(config.data_dir)
    train_samples = collect_samples(data_dir / "train")
    val_samples = collect_samples(data_dir / "val")
    test_samples = collect_samples(data_dir / "test")

    train_dataset = CrackDataset(
        train_samples,
        augmentation=config.augmentation,
        img_size=config.img_size,
        rotation_mode=config.rotation_mode,
    )
    val_dataset = CrackDataset(
        val_samples,
        augmentation=False,
        img_size=config.img_size,
        rotation_mode=config.rotation_mode,
    )
    test_dataset = CrackDataset(
        test_samples,
        augmentation=False,
        img_size=config.img_size,
        rotation_mode=config.rotation_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, test_loader


def create_test_loader(
    test_dir: PathLike,
    batch_size: int,
    num_workers: int,
    img_size: Tuple[int, int] = IMG_SIZE,
):
    samples = collect_samples(test_dir)
    dataset = CrackDataset(
        samples,
        augmentation=False,
        img_size=img_size,
        rotation_mode="positive",
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def collect_samples(split_dir: PathLike) -> List[Tuple[Path, int]]:
    split_dir = Path(split_dir)
    samples = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError("Missing class directory: {0}".format(class_dir))
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                samples.append((image_path, CLASS_TO_INDEX[class_name]))
    if not samples:
        raise ValueError("No images found in {0}".format(split_dir))
    return samples


def prepare_split_dataset(
    raw_dir: PathLike,
    output_dir: PathLike,
    seed: int = 42,
    train_size: int = 500,
    val_size: int = 100,
    test_size: int = 100,
) -> Path:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    class_files = _collect_raw_class_files(raw_dir)

    total_expected = train_size + val_size + test_size
    total_actual = sum(len(files) for files in class_files.values())
    if total_expected != total_actual:
        raise ValueError(
            "Requested split total {0} does not match dataset size {1}".format(
                total_expected, total_actual
            )
        )

    train_per_class = train_size // 2
    val_per_class = val_size // 2
    test_per_class = test_size // 2

    if output_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(str(split_dir))

    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            ensure_dir(output_dir / split / class_name)

    for class_name, files in class_files.items():
        train_files, remainder = train_test_split(
            files,
            train_size=train_per_class,
            random_state=seed,
            shuffle=True,
        )
        val_files, test_files = train_test_split(
            remainder,
            train_size=val_per_class,
            test_size=test_per_class,
            random_state=seed,
            shuffle=True,
        )
        _copy_files(train_files, output_dir / "train" / class_name)
        _copy_files(val_files, output_dir / "val" / class_name)
        _copy_files(test_files, output_dir / "test" / class_name)

    return output_dir


def prepare_dataset(
    raw_dir: PathLike,
    output_dir: PathLike,
    seed: int = 42,
    train_size: int = 500,
    val_size: int = 100,
    test_size: int = 100,
) -> Path:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if has_split_structure(raw_dir):
        prepare_presplit_dataset(raw_dir, output_dir)
        return output_dir

    prepare_split_dataset(
        raw_dir=raw_dir,
        output_dir=output_dir,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    return output_dir


def backup_raw_dataset(
    source_dir: PathLike,
    backup_dir: PathLike,
    overwrite: bool = False,
) -> Path:
    source_dir = Path(source_dir)
    backup_dir = Path(backup_dir)

    if not source_dir.exists():
        raise FileNotFoundError("Raw dataset directory does not exist: {0}".format(source_dir))

    if backup_dir.exists():
        if overwrite:
            shutil.rmtree(str(backup_dir))
        else:
            return backup_dir

    shutil.copytree(str(source_dir), str(backup_dir))
    return backup_dir


def has_split_structure(data_dir: PathLike) -> bool:
    data_dir = Path(data_dir)
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.exists():
            return False
        if not _split_has_supported_classes(split_dir):
            return False
    return True


def prepare_presplit_dataset(raw_dir: PathLike, output_dir: PathLike) -> Path:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    if output_dir.exists():
        for split in ("train", "val", "test"):
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(str(split_dir))

    for split in ("train", "val", "test"):
        source_split_dir = raw_dir / split
        for source_class_dir in source_split_dir.iterdir():
            if not source_class_dir.is_dir():
                continue
            mapped_name = RAW_TO_CANONICAL.get(source_class_dir.name)
            if mapped_name is None:
                continue
            destination = output_dir / split / mapped_name
            ensure_dir(destination)
            files = [
                path
                for path in sorted(source_class_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            _copy_files(files, destination)
    _validate_processed_structure(output_dir)
    return output_dir


def _collect_raw_class_files(raw_dir: Path) -> dict:
    class_files = {"non-crack": [], "crack": []}
    for child in raw_dir.iterdir():
        if not child.is_dir():
            continue
        class_name = RAW_TO_CANONICAL.get(child.name)
        if class_name is None:
            continue
        files = sorted(
            [
                path
                for path in child.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        class_files[class_name].extend(files)

    for class_name in CLASS_NAMES:
        if not class_files[class_name]:
            raise FileNotFoundError(
                "Could not find raw files for class {0} under {1}".format(class_name, raw_dir)
            )
    return class_files


def _split_has_supported_classes(split_dir: Path) -> bool:
    present = set()
    for child in split_dir.iterdir():
        if child.is_dir() and RAW_TO_CANONICAL.get(child.name) is not None:
            present.add(RAW_TO_CANONICAL[child.name])
    return set(CLASS_NAMES).issubset(present)


def _validate_processed_structure(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        for class_name in CLASS_NAMES:
            class_dir = output_dir / split / class_name
            if not class_dir.exists():
                raise FileNotFoundError("Missing processed class directory: {0}".format(class_dir))


def _copy_files(files: Sequence[Path], destination: Path) -> None:
    for source in files:
        shutil.copy2(str(source), str(destination / source.name))
