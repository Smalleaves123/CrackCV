from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from .dataset import DatasetConfig, backup_raw_dataset, create_dataloaders, prepare_dataset
from .models import build_model
from .utils import CLASS_NAMES, ensure_dir, save_json, select_device, set_global_seed


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train brickwork crack classifier with PyTorch.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--raw-data-dir", default="dataset")
    parser.add_argument("--raw-backup-dir", default="data/raw_backup")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--train-backbone", type=parse_bool, default=True)
    parser.add_argument("--augmentation", type=parse_bool, default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rotation-mode", choices=["positive", "symmetric"], default="positive")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-data-if-missing", type=parse_bool, default=True)
    parser.add_argument("--refresh-raw-backup", type=parse_bool, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    device = select_device()

    data_dir = Path(args.data_dir)
    backup_dir = Path(args.raw_backup_dir)
    if args.prepare_data_if_missing:
        backup_raw_dataset(
            source_dir=args.raw_data_dir,
            backup_dir=backup_dir,
            overwrite=args.refresh_raw_backup,
        )
        if args.refresh_raw_backup or not _is_prepared_dataset(data_dir):
            prepare_dataset(backup_dir, data_dir, seed=args.seed)

    train_loader, val_loader, _ = create_dataloaders(
        DatasetConfig(
            data_dir=data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augmentation=args.augmentation,
            seed=args.seed,
            rotation_mode=args.rotation_mode,
        )
    )

    model = build_model(
        backbone_name=args.backbone,
        train_backbone=args.train_backbone,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    output_dir = ensure_dir(args.output_dir)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        checkpoint_path=output_dir / "best_model.pt",
        backbone_name=args.backbone,
        train_backbone=args.train_backbone,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "history.csv", index=False)
    _plot_history(history_df, output_dir / "training_curves.png")
    save_json(
        {
            "framework": "pytorch",
            "raw_data_dir": str(Path(args.raw_data_dir)),
            "raw_backup_dir": str(backup_dir),
            "processed_data_dir": str(data_dir),
            "backbone": args.backbone,
            "train_backbone": args.train_backbone,
            "augmentation": args.augmentation,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "rotation_mode": args.rotation_mode,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "seed": args.seed,
            "device": str(device),
            "class_names": CLASS_NAMES,
        },
        output_dir / "train_config.json",
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    max_epochs: int,
    patience: int,
    checkpoint_path: Path,
    backbone_name: str,
    train_backbone: bool,
) -> List[Dict[str, float]]:
    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )
        val_loss, val_accuracy = _run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        print(
            "epoch={0} train_loss={1:.4f} train_acc={2:.4f} val_loss={3:.4f} val_acc={4:.4f}".format(
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                backbone_name=backbone_name,
                train_backbone=train_backbone,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered at epoch {0}".format(epoch))
            break

    return history


def _run_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    training: bool,
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels, _paths in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += images.size(0)

    avg_loss = total_loss / total_examples
    avg_accuracy = total_correct / total_examples
    return avg_loss, avg_accuracy


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    backbone_name: str,
    train_backbone: bool,
) -> None:
    ensure_dir(checkpoint_path.parent)
    torch.save(
        {
            "backbone_name": backbone_name,
            "train_backbone": train_backbone,
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "input_size": [3, 227, 227],
        },
        checkpoint_path,
    )


def _plot_history(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="train_accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def _is_prepared_dataset(data_dir: Path) -> bool:
    return all(
        (data_dir / split / class_name).exists()
        for split in ("train", "val", "test")
        for class_name in CLASS_NAMES
    )


if __name__ == "__main__":
    main()
