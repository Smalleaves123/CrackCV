from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from Reproduction.src.data.dataset import create_imagefolder
from Reproduction.src.models.model_factory import create_model_from_config
from Reproduction.src.utils.checkpoints import load_checkpoint
from Reproduction.src.utils.config import resolve_config_path
from Reproduction.src.utils.metrics import compute_classification_metrics
from Reproduction.src.utils.plotting import plot_confusion_matrix


def _build_eval_config(config: dict) -> dict:
    eval_config = {
        **config,
        "model": {**config["model"]},
        "weights": {**config.get("weights", {})},
    }
    eval_config["weights"]["pretrained"] = False
    eval_config["weights"]["offline_mode"] = True
    return eval_config


def evaluate(config: dict, checkpoint_path: str, split: str = "test") -> dict:
    device = "cuda" if config["project"].get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(resolve_config_path(config, checkpoint_path), map_location=device)
    model = create_model_from_config(_build_eval_config(config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    dataset = create_imagefolder(
        data_dir=resolve_config_path(config, config["data"][f"{split}_dir"]),
        image_size=config["data"]["image_size"],
        augmentation={**config["augmentation"], "enabled": False},
        train=False,
    )
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    y_true, y_pred, rows = [], [], []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            for idx in range(labels.size(0)):
                image_path, _ = dataset.samples[batch_idx * loader.batch_size + idx]
                rows.append(
                    {
                        "image_path": image_path,
                        "true_label": int(labels[idx].item()),
                        "pred_label": int(preds[idx].item()),
                        "prob_non_crack": float(probs[idx, 0].item()),
                        "prob_crack": float(probs[idx, 1].item()),
                        "correct": bool(preds[idx].item() == labels[idx].item()),
                    }
                )
    metrics = compute_classification_metrics(y_true, y_pred)
    run_dir = resolve_config_path(config, config["runtime"]["run_dir"])
    metrics_dir = run_dir / "metrics"
    figures_dir = run_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (metrics_dir / "classification_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(metrics["classification_report"])
    pd.DataFrame(rows).to_csv(metrics_dir / "predictions.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"], config["data"]["class_names"], figures_dir / "confusion_matrix.png")
    return metrics
