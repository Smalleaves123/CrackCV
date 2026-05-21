from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT
from Reproduction.src.utils.plotting import plot_confusion_matrix
from Reproduction.src.utils.experiment import MODEL_PARAMETER_MILLIONS, REPRODUCTION_MODELS, STRATEGIES


def _load_metrics(metrics_path: Path) -> dict:
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _collect_rows() -> list[dict]:
    rows: list[dict] = []
    for model_name in REPRODUCTION_MODELS:
        for strategy in STRATEGIES:
            run_dir = ROOT / "Reproduction" / "results" / model_name / strategy
            val_path = run_dir / "metrics" / "val_metrics.json"
            test_path = run_dir / "metrics" / "test_metrics.json"
            log_path = run_dir / "logs" / "train_log.csv"
            best_val = _load_metrics(val_path) if val_path.exists() else None
            test_metrics = _load_metrics(test_path) if test_path.exists() else None
            epochs_ran = None
            if log_path.exists():
                log_df = pd.read_csv(log_path)
                if not log_df.empty and "epoch" in log_df.columns:
                    epochs_ran = int(log_df["epoch"].max())
            rows.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "params_m": MODEL_PARAMETER_MILLIONS.get(model_name),
                    "epochs_ran": epochs_ran,
                    "val_accuracy": None if best_val is None else best_val.get("accuracy"),
                    "val_precision": None if best_val is None else best_val.get("precision"),
                    "val_recall": None if best_val is None else best_val.get("recall"),
                    "val_f1_score": None if best_val is None else best_val.get("f1_score"),
                    "val_jaccard_index": None if best_val is None else best_val.get("jaccard_index"),
                    "val_cm_tn": None if best_val is None else best_val.get("confusion_matrix", [[None, None], [None, None]])[0][0],
                    "val_cm_fp": None if best_val is None else best_val.get("confusion_matrix", [[None, None], [None, None]])[0][1],
                    "val_cm_fn": None if best_val is None else best_val.get("confusion_matrix", [[None, None], [None, None]])[1][0],
                    "val_cm_tp": None if best_val is None else best_val.get("confusion_matrix", [[None, None], [None, None]])[1][1],
                    "test_accuracy": None if test_metrics is None else test_metrics.get("accuracy"),
                    "test_precision": None if test_metrics is None else test_metrics.get("precision"),
                    "test_recall": None if test_metrics is None else test_metrics.get("recall"),
                    "test_f1_score": None if test_metrics is None else test_metrics.get("f1_score"),
                    "test_jaccard_index": None if test_metrics is None else test_metrics.get("jaccard_index"),
                    "test_cm_tn": None if test_metrics is None else test_metrics.get("confusion_matrix", [[None, None], [None, None]])[0][0],
                    "test_cm_fp": None if test_metrics is None else test_metrics.get("confusion_matrix", [[None, None], [None, None]])[0][1],
                    "test_cm_fn": None if test_metrics is None else test_metrics.get("confusion_matrix", [[None, None], [None, None]])[1][0],
                    "test_cm_tp": None if test_metrics is None else test_metrics.get("confusion_matrix", [[None, None], [None, None]])[1][1],
                    "run_dir": str(run_dir),
                }
            )
    return rows


def _save_summary_tables(df: pd.DataFrame) -> None:
    tables_dir = ROOT / "report_assets" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables_dir / "reproduction_summary.csv", index=False)
    completed = df.dropna(subset=["test_accuracy"]).copy()
    completed.to_csv(tables_dir / "reproduction_completed_runs.csv", index=False)
    if completed.empty:
        return
    best_by_model = completed.sort_values(["model", "test_accuracy", "test_f1_score"], ascending=[True, False, False])
    best_by_model = best_by_model.groupby("model", as_index=False).first()
    best_by_model.to_csv(tables_dir / "reproduction_best_by_model.csv", index=False)


def _plot_grouped_bar(df: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    completed = df.dropna(subset=[metric]).copy()
    if completed.empty:
        return
    pivot = completed.pivot(index="model", columns="strategy", values=metric)
    pivot = pivot.reindex(REPRODUCTION_MODELS)
    ax = pivot.plot(kind="bar", figsize=(12, 5))
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    figures_dir = ROOT / "report_assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / filename, dpi=180)
    plt.close()


def _plot_parameter_vs_accuracy(df: pd.DataFrame) -> None:
    completed = df.dropna(subset=["test_accuracy", "params_m"]).copy()
    if completed.empty:
        return
    best_by_model = completed.sort_values(["model", "test_accuracy"], ascending=[True, False]).groupby("model", as_index=False).first()
    plt.figure(figsize=(7, 4.5))
    plt.scatter(best_by_model["params_m"], best_by_model["test_accuracy"])
    for _, row in best_by_model.iterrows():
        plt.text(row["params_m"], row["test_accuracy"], row["model"])
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Test Accuracy")
    plt.title("Parameter Count vs Best Test Accuracy")
    plt.tight_layout()
    figures_dir = ROOT / "report_assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "parameter_vs_accuracy.png", dpi=180)
    plt.close()


def _plot_best_strategy(df: pd.DataFrame) -> None:
    completed = df.dropna(subset=["test_accuracy"]).copy()
    if completed.empty:
        return
    best_by_model = completed.sort_values(["model", "test_accuracy", "test_f1_score"], ascending=[True, False, False])
    best_by_model = best_by_model.groupby("model", as_index=False).first()
    plt.figure(figsize=(9, 4.5))
    plt.bar(best_by_model["model"], best_by_model["test_accuracy"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Test Accuracy")
    plt.title("Best Strategy Per Model")
    for idx, row in best_by_model.iterrows():
        plt.text(idx, row["test_accuracy"], row["strategy"], rotation=90, va="bottom", ha="center", fontsize=8)
    plt.tight_layout()
    figures_dir = ROOT / "report_assets" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "best_strategy_per_model.png", dpi=180)
    plt.close()


def _export_confusion_matrix_panels(df: pd.DataFrame) -> None:
    figures_dir = ROOT / "report_assets" / "figures" / "confusion_matrices"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        model_name = row["model"]
        strategy = row["strategy"]
        if pd.notna(row["test_cm_tn"]):
            matrix = [
                [int(row["test_cm_tn"]), int(row["test_cm_fp"])],
                [int(row["test_cm_fn"]), int(row["test_cm_tp"])],
            ]
            plot_confusion_matrix(
                matrix,
                ["non_crack", "crack"],
                figures_dir / f"{model_name}__{strategy}__test_confusion_matrix.png",
            )
        if pd.notna(row["val_cm_tn"]):
            matrix = [
                [int(row["val_cm_tn"]), int(row["val_cm_fp"])],
                [int(row["val_cm_fn"]), int(row["val_cm_tp"])],
            ]
            plot_confusion_matrix(
                matrix,
                ["non_crack", "crack"],
                figures_dir / f"{model_name}__{strategy}__val_confusion_matrix.png",
            )


def main() -> None:
    df = pd.DataFrame(_collect_rows())
    _save_summary_tables(df)
    if df.empty:
        return
    _export_confusion_matrix_panels(df)
    _plot_grouped_bar(df, "test_accuracy", "model_accuracy_comparison.png", "Test Accuracy by Model and Strategy")
    _plot_grouped_bar(df, "test_f1_score", "model_f1_comparison.png", "Test F1 Score by Model and Strategy")
    _plot_parameter_vs_accuracy(df)
    _plot_best_strategy(df)


if __name__ == "__main__":
    main()
