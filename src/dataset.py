"""
数据集模块：PyTorch DataLoader
- 训练集：支持数据增强
- 验证/测试集：只做 ToTensor（自动 /255）
- target_size=(227, 227), batch_size=32
"""

from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_train_loader(
    data_dir: str,
    batch_size: int = 32,
    augmentation: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """构建训练集 DataLoader"""
    if augmentation:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=(0.3, 1.0)),
            transforms.ToTensor(),  # 自动归一化到 [0, 1]（即 /255）
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
        ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset.class_to_idx, len(dataset)


def get_val_test_loader(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """构建验证集/测试集 DataLoader（不做增强）"""
    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, dataset.class_to_idx, len(dataset)
