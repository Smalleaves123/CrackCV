"""
评估脚本（PyTorch）
加载最佳模型，在测试集上计算各项指标
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    confusion_matrix,
)
import torch

from src.dataset import get_val_test_loader
from src.models import build_model


@torch.no_grad()
def predict(model, loader, device):
    """对整个数据集进行预测"""
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def evaluate_model(
    model_path: str,
    backbone: str,
    test_dir: str,
    output_dir: str = None,
    batch_size: int = 32,
):
    """评估模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), "eval")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"评估配置:")
    print(f"  模型路径: {model_path}")
    print(f"  Backbone: {backbone}")
    print(f"  测试集目录: {test_dir}")
    print(f"  设备: {device}")
    print(f"{'=' * 60}\n")

    # 加载 checkpoint
    print("加载模型...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # 构建模型结构
    model, _, _, _ = build_model(
        backbone_name=backbone,
        num_classes=2,
        train_backbone=True,  # 评估时不影响，只用于加载权重
        learning_rate=1e-5,
        device=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    class_to_idx = checkpoint.get("class_to_idx", {"crack": 0, "non-crack": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 加载测试集
    print("加载测试集...")
    test_loader, _, test_size = get_val_test_loader(test_dir, batch_size=batch_size)
    print(f"  测试集样本数: {test_size}")
    print(f"  类别索引: {class_to_idx}")

    # 预测
    print("\n进行预测...")
    y_true, y_pred = predict(model, test_loader, device)

    # 类别顺序
    class_names = [idx_to_class.get(0, "non-crack"), idx_to_class.get(1, "crack")]
    print(f"  类别顺序: {class_names}")

    # 计算指标
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    jaccard = jaccard_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": float(acc),
        "f1_score_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "jaccard_macro": float(jaccard),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "test_samples": int(test_size),
    }

    # 打印结果
    print(f"\n{'=' * 40}")
    print(f"测试集评估结果:")
    print(f"{'=' * 40}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Jaccard:   {jaccard:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"  Rows: actual [{class_names[0]}, {class_names[1]}]")
    print(f"  Cols: predicted [{class_names[0]}, {class_names[1]}]")
    print(f"{'=' * 40}\n")

    # 保存 metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"保存指标到: {metrics_path}")

    # 保存 confusion matrix CSV
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    with open(cm_csv_path, "w") as f:
        f.write(f",predicted_{class_names[0]},predicted_{class_names[1]}\n")
        f.write(f"actual_{class_names[0]},{cm[0, 0]},{cm[0, 1]}\n")
        f.write(f"actual_{class_names[1]},{cm[1, 0]},{cm[1, 1]}\n")
    print(f"保存混淆矩阵 CSV 到: {cm_csv_path}")

    # 绘制混淆矩阵图
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, cm_plot_path)
    print(f"保存混淆矩阵图到: {cm_plot_path}")

    return metrics


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Pred: {name}" for name in class_names],
        yticklabels=[f"True: {name}" for name in class_names],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="评估模型 (PyTorch)")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--backbone", type=str, required=True, help="需要指定 backbone 名"
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=r"C:\Users\24214\Desktop\CrackCV\data\processed\test",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        backbone=args.backbone,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
