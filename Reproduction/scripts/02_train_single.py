from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT
from Reproduction.src.engine.trainer import Trainer
from Reproduction.src.utils.config import attach_project_root, load_config
from Reproduction.src.utils.experiment import STRATEGIES, apply_runtime, apply_strategy, apply_training_overrides


def infer_run_dir(config: dict) -> str:
    return str(ROOT / "Reproduction" / "results" / config["model"]["name"] / config["runtime"]["strategy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    base_path = ROOT / "Reproduction" / "configs" / "base.yaml"
    config = attach_project_root(load_config(args.config, base_path=base_path), ROOT)
    config = apply_training_overrides(config, epochs=args.epochs, offline=args.offline)
    config = apply_strategy(config, args.strategy)
    config = apply_runtime(config, infer_run_dir(config))
    Path(config["runtime"]["run_dir"]).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config)
    result = trainer.fit()
    print(
        json.dumps(
            {
                "model": config["model"]["name"],
                "strategy": args.strategy,
                "freeze_mode": config["model"]["freeze_mode"],
                "augmentation_enabled": config["augmentation"]["enabled"],
                **result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
