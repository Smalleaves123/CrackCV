from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from common import ROOT
from Reproduction.src.utils.plotting import plot_confusion_matrix


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_results_root(input_path: Path) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if input_path.is_dir():
        return input_path, None
    if input_path.suffix.lower() != ".zip":
        raise ValueError("Input must be a results directory or a .zip archive")
    temp_dir = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(input_path, "r") as archive:
        archive.extractall(temp_dir.name)
    extracted_root = Path(temp_dir.name)
    candidates = [path for path in extracted_root.rglob("results") if path.is_dir()]
    if candidates:
        return candidates[0], temp_dir
    return extracted_root, temp_dir


def _flatten_cm(prefix: str, metrics: dict | None) -> dict:
    if metrics is None or "confusion_matrix" not in metrics:
        return {
            f"{prefix}_cm_tn": None,
            f"{prefix}_cm_fp": None,
            f"{prefix}_cm_fn": None,
            f"{prefix}_cm_tp": None,
        }
    matrix = metrics["confusion_matrix"]
    return {
        f"{prefix}_cm_tn": matrix[0][0],
        f"{prefix}_cm_fp": matrix[0][1],
        f"{prefix}_cm_fn": matrix[1][0],
        f"{prefix}_cm_tp": matrix[1][1],
    }


def _collect_rows(results_root: Path) -> list[dict]:
    rows: list[dict] = []
    for config_path in sorted(results_root.glob("*/*/logs/config.yaml")):
        strategy = config_path.parents[1].name
        model_name = config_path.parents[2].name
        run_dir = config_path.parents[1]
        val_metrics = _load_json(run_dir / "metrics" / "val_metrics.json")
        test_metrics = _load_json(run_dir / "metrics" / "test_metrics.json")
        log_path = run_dir / "logs" / "train_log.csv"
        epochs_ran = None
        if log_path.exists():
            log_df = pd.read_csv(log_path)
            if not log_df.empty and "epoch" in log_df.columns:
                epochs_ran = int(log_df["epoch"].max())
        rows.append(
            {
                "model": model_name,
                "strategy": strategy,
                "epochs_ran": epochs_ran,
                "val_accuracy": None if val_metrics is None else val_metrics.get("accuracy"),
                "val_precision": None if val_metrics is None else val_metrics.get("precision"),
                "val_recall": None if val_metrics is None else val_metrics.get("recall"),
                "val_f1_score": None if val_metrics is None else val_metrics.get("f1_score"),
                "val_jaccard_index": None if val_metrics is None else val_metrics.get("jaccard_index"),
                **_flatten_cm("val", val_metrics),
                "test_accuracy": None if test_metrics is None else test_metrics.get("accuracy"),
                "test_precision": None if test_metrics is None else test_metrics.get("precision"),
                "test_recall": None if test_metrics is None else test_metrics.get("recall"),
                "test_f1_score": None if test_metrics is None else test_metrics.get("f1_score"),
                "test_jaccard_index": None if test_metrics is None else test_metrics.get("jaccard_index"),
                **_flatten_cm("test", test_metrics),
                "source_run_dir": str(run_dir),
            }
        )
    return rows


def _export_confusion_matrices(df: pd.DataFrame, output_dir: Path) -> None:
    cm_dir = output_dir / "confusion_matrices"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for _, row in df.iterrows():
        base = f"{row['model']}__{row['strategy']}"
        if pd.notna(row["test_cm_tn"]):
            plot_confusion_matrix(
                [
                    [int(row["test_cm_tn"]), int(row["test_cm_fp"])],
                    [int(row["test_cm_fn"]), int(row["test_cm_tp"])],
                ],
                ["non_crack", "crack"],
                cm_dir / f"{base}__test.png",
            )
        if pd.notna(row["val_cm_tn"]):
            plot_confusion_matrix(
                [
                    [int(row["val_cm_tn"]), int(row["val_cm_fp"])],
                    [int(row["val_cm_fn"]), int(row["val_cm_tp"])],
                ],
                ["non_crack", "crack"],
                cm_dir / f"{base}__val.png",
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a results directory or zipped results archive")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "report_assets" / "shared_results_merge"),
        help="Where to write merged tables and confusion matrices",
    )
    parser.add_argument("--copy-source", action="store_true", help="Copy the input results tree into the output directory")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_root, temp_dir = _resolve_results_root(input_path)
    try:
        rows = _collect_rows(results_root)
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "merged_results_summary.csv", index=False)
        (output_dir / "merged_results_summary.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not df.empty:
            _export_confusion_matrices(df, output_dir)
        if args.copy_source:
            copy_target = output_dir / "source_results"
            if copy_target.exists():
                shutil.rmtree(copy_target)
            shutil.copytree(results_root, copy_target)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
