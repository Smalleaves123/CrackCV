from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score

from .dataset import DatasetConfig, create_dataloaders, create_test_loader
from .models import build_model
from .utils import CLASS_NAMES, ensure_dir, save_json, select_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crack classification model.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = select_device()
    output_dir = ensure_dir(args.output_dir)

    checkpoint = torch.load(args.model_path, map_location=device)
    model = build_model(
        backbone_name=checkpoint["backbone_name"],
        train_backbone=checkpoint.get("train_backbone", True),
        use_pretrained=checkpoint.get("use_pretrained", True),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    if args.test_dir:
        test_loader = create_test_loader(
            test_dir=args.test_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        _, _, test_loader = create_dataloaders(
            DatasetConfig(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                augmentation=False,
                seed=args.seed,
                rotation_mode="positive",
            )
        )

    y_true, y_pred, probabilities, paths = predict(model, test_loader, device)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    save_json(metrics, output_dir / "metrics.json")

    prediction_df = pd.DataFrame(
        {
            "filepath": paths,
            "y_true": y_true,
            "y_pred": y_pred,
            "prob_non_crack": probabilities[:, 0],
            "prob_crack": probabilities[:, 1],
        }
    )
    prediction_df.to_csv(str(output_dir / "predictions.csv"), index=False)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        str(output_dir / "confusion_matrix.csv")
    )
    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")


def predict(model, data_loader, device):
    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in data_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_targets.extend(labels.numpy().tolist())
            all_predictions.extend(preds.tolist())
            all_probabilities.append(probs)
            all_paths.extend(paths)

    probabilities = np.concatenate(all_probabilities, axis=0)
    return np.array(all_targets), np.array(all_predictions), probabilities, all_paths


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=30)
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
