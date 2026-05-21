from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT
from Reproduction.src.engine.evaluator import evaluate
from Reproduction.src.utils.config import load_config
from Reproduction.src.utils.experiment import REPRODUCTION_MODELS, STRATEGIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="*", default=None)
    args = parser.parse_args()
    results = []
    models = args.models or REPRODUCTION_MODELS
    strategies = args.strategies or list(STRATEGIES.keys())
    for model_name in models:
        for strategy in strategies:
            config_path = ROOT / "Reproduction" / "results" / model_name / strategy / "logs" / "config.yaml"
            if not config_path.exists():
                continue
            config = load_config(config_path)
            checkpoint = Path(config["runtime"]["run_dir"]) / "checkpoints" / "best.pt"
            if not checkpoint.exists():
                continue
            metrics = evaluate(config, str(checkpoint), split=args.split)
            results.append(
                {
                    "model": model_name,
                    "strategy": strategy,
                    "split": args.split,
                    "run_dir": config["runtime"]["run_dir"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                    "jaccard_index": metrics["jaccard_index"],
                }
            )
    output_path = ROOT / "Reproduction" / "results" / f"{args.split}_evaluation_summary.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
