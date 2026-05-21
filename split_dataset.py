"""
数据划分脚本：将原始数据集划分为 train/val/test
按照论文设定：
- Train: 500 (250 crack, 250 non-crack)
- Validation: 100 (50 crack, 50 non-crack)
- Test: 100 (50 crack, 50 non-crack)
使用分层抽样 (stratified sampling)
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(
    source_dir: str,
    output_dir: str,
    train_ratio: float = 500 / 700,
    val_ratio: float = 100 / 700,
    test_ratio: float = 100 / 700,
    seed: int = 42,
):
    """
    划分数据集为 train/val/test

    Args:
        source_dir: 原始数据集目录，包含 Positive/ 和 Negative/ 子目录
        output_dir: 输出目录，将创建 train/, val/, test/ 子目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    random.seed(seed)

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # 类别映射：Positive -> crack, Negative -> non-crack
    class_mapping = {
        "Positive": "crack",
        "Negative": "non-crack",
    }

    # 创建输出目录结构
    for split in ["train", "val", "test"]:
        for class_name in class_mapping.values():
            (output_path / split / class_name).mkdir(parents=True, exist_ok=True)

    # 对每个类别进行分层抽样
    for original_class, mapped_class in class_mapping.items():
        original_class_path = source_path / original_class
        if not original_class_path.exists():
            print(f"警告: {original_class_path} 不存在，跳过")
            continue

        # 获取所有图像文件
        image_files = list(original_class_path.glob("*.jpg")) + list(
            original_class_path.glob("*.png")
        )
        image_files.sort()  # 确保顺序一致

        print(f"\n处理类别: {original_class} -> {mapped_class}")
        print(f"  总图像数: {len(image_files)}")

        # 打乱顺序
        random.shuffle(image_files)

        # 计算划分点
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)  # 250
        n_val = int(n_total * val_ratio)  # 50
        # 剩余给 test

        train_files = image_files[:n_train]
        val_files = image_files[n_train : n_train + n_val]
        test_files = image_files[n_train + n_val :]

        print(f"  训练集: {len(train_files)}")
        print(f"  验证集: {len(val_files)}")
        print(f"  测试集: {len(test_files)}")

        # 复制文件
        def copy_files(file_list, split_name):
            for img_path in file_list:
                dest = output_path / split_name / mapped_class / img_path.name
                shutil.copy2(img_path, dest)

        copy_files(train_files, "train")
        copy_files(val_files, "val")
        copy_files(test_files, "test")

    print("\n数据划分完成！")
    print(f"输出目录: {output_path}")


if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    SOURCE_DIR = os.path.join(PROJECT_ROOT, "dataset")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    split_dataset(SOURCE_DIR, OUTPUT_DIR, seed=42)
