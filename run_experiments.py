import os
import sys
import argparse

# 解决 Windows 下 OpenMP 运行时冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.train import train_model
from src.evaluate import evaluate_model

BACKBONES = [
    "vgg16",
    "vgg19",
    "mobilenetv2",
    "inceptionresnetv2",
    "inceptionv3",
    "xception",
]

STRATEGIES = [
    {"name": "e2e_aug", "train_backbone": True, "augmentation": True},
    {"name": "e2e_noaug", "train_backbone": True, "augmentation": False},
    {"name": "frozen_noaug", "train_backbone": False, "augmentation": False},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true", help="Run with 1 epoch for testing")
    args = parser.parse_args()

    max_epochs = 1 if args.quick_test else 200
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")

    for strategy in STRATEGIES:
        for backbone in BACKBONES:
            strategy_name = strategy["name"]
            output_dir = os.path.join(outputs_dir, f"{backbone}_{strategy_name}")
            
            print(f"\n{'=' * 80}")
            print(f"Running Strategy: {strategy_name} | Backbone: {backbone}")
            print(f"{'=' * 80}")
            
            # Train
            model, history = train_model(
                data_dir=data_dir,
                backbone=backbone,
                train_backbone=strategy["train_backbone"],
                augmentation=strategy["augmentation"],
                batch_size=32,
                lr=1e-5,
                max_epochs=max_epochs,
                patience=20 if not args.quick_test else 1,
                output_dir=output_dir,
                seed=42,
            )
            
            # Evaluate
            model_path = os.path.join(output_dir, "best_model.pt")
            eval_output_dir = os.path.join(output_dir, "eval")
            
            try:
                evaluate_model(
                    model_path=model_path,
                    backbone=backbone,
                    test_dir=os.path.join(data_dir, "test"),
                    output_dir=eval_output_dir,
                )
            except Exception as e:
                print(f"Evaluation failed for {backbone} - {strategy_name}: {e}")

if __name__ == "__main__":
    main()
