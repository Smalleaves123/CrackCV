"""
训练脚本（PyTorch）
包含完整训练循环、Early Stopping、TensorBoard 支持
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.tensorboard import SummaryWriter

from src.dataset import get_train_loader, get_val_test_loader
from src.models import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def train_model(
    data_dir: str,
    backbone: str,
    train_backbone: bool = False,
    augmentation: bool = True,
    batch_size: int = 32,
    lr: float = 1e-5,
    max_epochs: int = 200,
    patience: int = 20,
    output_dir: str = None,
    seed: int = 42,
):
    """训练模型"""
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 设置输出目录
    if output_dir is None:
        strategy_name = "frozen_aug" if not train_backbone and augmentation else "other"
        output_dir = os.path.join("outputs", f"{backbone}_{strategy_name}")

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    print(f"\n{'=' * 60}")
    print(f"训练配置:")
    print(f"  Backbone: {backbone}")
    print(f"  训练 backbone: {train_backbone}")
    print(f"  数据增强: {augmentation}")
    print(f"  Batch size: {batch_size}")
    print(f"  学习率: {lr}")
    print(f"  最大 epochs: {max_epochs}")
    print(f"  Early stopping patience: {patience}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}\n")

    # 构建 DataLoader
    print("加载训练集...")
    train_loader, class_to_idx, train_size = get_train_loader(
        train_dir, batch_size=batch_size, augmentation=augmentation
    )
    print(f"  训练集样本数: {train_size}")
    print(f"  类别索引: {class_to_idx}")

    print("加载验证集...")
    val_loader, _, val_size = get_val_test_loader(val_dir, batch_size=batch_size)
    print(f"  验证集样本数: {val_size}\n")

    # 构建模型
    print("构建模型...")
    model, optimizer, criterion, num_trainable = build_model(
        backbone_name=backbone,
        num_classes=2,
        train_backbone=train_backbone,
        learning_rate=lr,
        device=device,
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Early Stopping 状态
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    print(f"\n开始训练...")
    print(f"{'─' * 60}")

    for epoch in range(1, max_epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 记录历史
        history["loss"].append(train_loss)
        history["accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # 打印进度
        print(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early Stopping 检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "backbone": backbone,
                    "class_to_idx": class_to_idx,
                },
                os.path.join(output_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    writer.close()

    print(f"{'─' * 60}")
    print(f"\n训练完成！")
    print(f"  最佳验证 accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"  最佳模型保存在: {os.path.join(output_dir, 'best_model.pt')}")

    # 保存训练历史
    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  训练历史保存在: {history_path}")

    # 绘制训练曲线
    curves_path = os.path.join(output_dir, "training_curves.png")
    plot_training_curves(history, curves_path)
    print(f"  训练曲线保存在: {curves_path}")

    print(f"\n查看 TensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/logs")
    print(f"  然后打开浏览器访问 http://localhost:6006")

    return model, history


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["accuracy"], label="Train Accuracy")
    axes[1].plot(history["val_accuracy"], label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="训练砖砌体裂缝分类模型 (PyTorch)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"C:\Users\24214\Desktop\CrackCV\data\processed",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vgg16",
        choices=SUPPORTED_MODELS,
    )
    parser.add_argument("--train-backbone", type=str, default="false")
    parser.add_argument("--augmentation", type=str, default="true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_model(
        data_dir=args.data_dir,
        backbone=args.backbone,
        train_backbone=args.train_backbone.lower() == "true",
        augmentation=args.augmentation.lower() == "true",
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
