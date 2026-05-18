"""
主训练脚本（PyTorch）：运行 frozen backbone + data augmentation 策略

修改 BACKBONE 变量选择要训练的模型，然后运行此脚本
"""

import os
# 解决 Windows 下 OpenMP 运行时冲突（numpy/torch/timm 链接了不同版本的 libiomp5md.dll）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.train import train_model
from src.evaluate import evaluate_model


# ============================================================
# 在这里选择要训练的模型（取消注释你想用的那个）
# ============================================================
#BACKBONE = "vgg16"
#BACKBONE = "vgg19"
#BACKBONE = "mobilenetv2"
#BACKBONE = "inceptionresnetv2"
#BACKBONE = "inceptionv3"
BACKBONE = "xception"
# ============================================================

DATA_DIR = r"C:\Users\24214\Desktop\frozen_aug\data\processed"
OUTPUTS_DIR = r"C:\Users\24214\Desktop\frozen_aug\outputs"

TRAIN_CONFIG = {
    "train_backbone": False,
    "augmentation": True,
    "batch_size": 32,
    "lr": 1e-5,
    "max_epochs": 200,
    "patience": 20,
    "seed": 42,
}


def main():
    strategy_name = "frozen_aug"
    output_dir = os.path.join(OUTPUTS_DIR, f"{BACKBONE}_{strategy_name}")

    print(f"\n{'=' * 80}")
    print(f"砖砌体裂缝检测 - 策略 3: 冻结卷积基 + 数据增强 (PyTorch)")
    print(f"{'=' * 80}")
    print(f"  模型: {BACKBONE}")
    print(f"  训练 backbone: {TRAIN_CONFIG['train_backbone']}")
    print(f"  数据增强: {TRAIN_CONFIG['augmentation']}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 80}\n")

    # 训练
    model, history = train_model(
        data_dir=DATA_DIR,
        backbone=BACKBONE,
        output_dir=output_dir,
        **TRAIN_CONFIG,
    )

    # 评估
    model_path = os.path.join(output_dir, "best_model.pt")
    eval_output_dir = os.path.join(output_dir, "eval")

    metrics = evaluate_model(
        model_path=model_path,
        backbone=BACKBONE,
        test_dir=os.path.join(DATA_DIR, "test"),
        output_dir=eval_output_dir,
    )

    print(f"\n{'=' * 80}")
    print(f"实验完成: {BACKBONE}")
    print(f"  最佳验证 accuracy: {max(history['val_accuracy']):.4f}")
    print(f"  测试 accuracy: {metrics['accuracy']:.4f}")
    print(f"  测试 F1-score: {metrics['f1_score_macro']:.4f}")
    print(f"{'=' * 80}")

    print(f"\n查看 TensorBoard:")
    print(f"  tensorboard --logdir {OUTPUTS_DIR}")
    print(f"  然后打开浏览器访问 http://localhost:6006")


if __name__ == "__main__":
    main()
