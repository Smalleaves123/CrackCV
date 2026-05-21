from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd


def plot_curve(csv_path: str | Path, x: str, y: str, out_path: str | Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or y not in df.columns:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(df[x], df[y], marker="o")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_train_val_curve(
    csv_path: str | Path,
    x: str,
    train_y: str,
    val_y: str,
    out_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or train_y not in df.columns or val_y not in df.columns:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(df[x], df[train_y], label="Train", linewidth=2.0)
    plt.plot(df[x], df[val_y], label="Val", linewidth=2.0)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], out_path: str | Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
