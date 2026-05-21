from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")

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


def plot_confusion_matrix(matrix: list[list[int]], class_names: list[str], out_path: str | Path) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            plt.text(j, i, str(value), ha="center", va="center")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
