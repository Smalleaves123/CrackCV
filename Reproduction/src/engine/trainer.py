from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from Reproduction.src.data.dataset import create_imagefolder
from Reproduction.src.models.model_factory import create_model_from_config
from Reproduction.src.utils.amp import autocast_context, build_grad_scaler
from Reproduction.src.utils.checkpoints import save_checkpoint
from Reproduction.src.utils.config import resolve_config_path, save_config
from Reproduction.src.utils.metrics import compute_classification_metrics
from Reproduction.src.utils.plotting import plot_train_val_curve
from Reproduction.src.utils.seed import set_seed


class Trainer:
    def __init__(self, config: dict) -> None:
        self.config = config
        set_seed(config["project"]["seed"])
        requested_device = config["project"].get("device", "cpu")
        self.device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
        self.run_dir = resolve_config_path(config, config["runtime"]["run_dir"])
        if self.run_dir is None:
            raise RuntimeError("runtime.run_dir must be configured")
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.figures_dir = self.run_dir / "figures"
        self.logs_dir = self.run_dir / "logs"
        for directory in [self.checkpoint_dir, self.metrics_dir, self.figures_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.use_amp = bool(self.config["training"].get("amp", False) and self.device == "cuda")
        self.scaler = build_grad_scaler(enabled=self.use_amp)
        self.best_metric = float("-inf")
        self.history: list[dict] = []

    def _build_model(self):
        return create_model_from_config(self.config)

    def _build_optimizer(self):
        params = [param for param in self.model.parameters() if param.requires_grad]
        lr = self.config["training"]["learning_rate"]
        wd = self.config["training"]["weight_decay"]
        if self.config["training"]["optimizer"].lower() == "sgd":
            return SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        return Adam(params, lr=lr, weight_decay=wd)

    def _create_loader(self, split: str, train: bool) -> DataLoader:
        dataset = create_imagefolder(
            data_dir=resolve_config_path(self.config, self.config["data"][f"{split}_dir"]),
            image_size=self.config["data"]["image_size"],
            augmentation=self.config["augmentation"],
            train=train,
        )
        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=train,
            num_workers=self.config["data"]["num_workers"],
        )

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, dict]:
        self.model.train(train)
        total_loss = 0.0
        y_true, y_pred = [], []
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.set_grad_enabled(train):
                with autocast_context(enabled=self.use_amp):
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = self.criterion(outputs, labels)
                if train:
                    self.optimizer.zero_grad()
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        self.optimizer.step()
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
        metrics = compute_classification_metrics(y_true, y_pred)
        avg_loss = total_loss / max(len(loader.dataset), 1)
        return avg_loss, metrics

    def train_one_epoch(self, epoch: int, loader: DataLoader) -> tuple[float, dict]:
        del epoch
        return self._run_epoch(loader, train=True)

    def validate(self, epoch: int, loader: DataLoader) -> tuple[float, dict]:
        del epoch
        return self._run_epoch(loader, train=False)

    def fit(self) -> dict:
        train_loader = self._create_loader("train", train=True)
        val_loader = self._create_loader("val", train=False)
        epochs = self.config["training"]["epochs"]
        patience = self.config["training"]["early_stopping"]["patience"]
        stale_epochs = 0
        save_config(self.config, self.logs_dir / "config.yaml")
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_metrics = self.train_one_epoch(epoch, train_loader)
            val_loss, val_metrics = self.validate(epoch, val_loader)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1_score"],
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1_score"],
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch_time": time.time() - start,
            }
            self.history.append(row)
            pd.DataFrame(self.history).to_csv(self.logs_dir / "train_log.csv", index=False)
            is_best = val_metrics["accuracy"] > self.best_metric
            if is_best:
                self.best_metric = val_metrics["accuracy"]
                stale_epochs = 0
            else:
                stale_epochs += 1
            self.save_checkpoint(epoch=epoch, is_best=is_best)
            with (self.metrics_dir / "val_metrics.json").open("w", encoding="utf-8") as handle:
                json.dump(val_metrics, handle, indent=2)
            if stale_epochs >= patience:
                break
        self._finalize_plots()
        return {"best_val_accuracy": self.best_metric, "epochs_ran": len(self.history)}

    def save_checkpoint(self, epoch: int, is_best: bool) -> None:
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
            "class_names": self.config["data"]["class_names"],
        }
        save_checkpoint(payload, self.checkpoint_dir / "last.pt")
        if is_best:
            save_checkpoint(payload, self.checkpoint_dir / "best.pt")

    def _finalize_plots(self) -> None:
        csv_path = self.logs_dir / "train_log.csv"
        plot_train_val_curve(
            csv_path,
            "epoch",
            "train_loss",
            "val_loss",
            self.figures_dir / "loss_curve.png",
            "Train vs Val Loss",
            "Loss",
        )
        plot_train_val_curve(
            csv_path,
            "epoch",
            "train_accuracy",
            "val_accuracy",
            self.figures_dir / "accuracy_curve.png",
            "Train vs Val Accuracy",
            "Accuracy",
        )
        plot_train_val_curve(
            csv_path,
            "epoch",
            "train_f1",
            "val_f1",
            self.figures_dir / "f1_curve.png",
            "Train vs Val F1",
            "F1 Score",
        )
