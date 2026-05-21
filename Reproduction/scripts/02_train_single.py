from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT
from Reproduction.src.engine.trainer import Trainer
from Reproduction.src.utils.config import load_config


def infer_run_dir(config: dict) -> str:
    return str(ROOT / "Reproduction" / "results" / config["model"]["name"] / config["runtime"]["strategy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    base_path = ROOT / "Reproduction" / "configs" / "base.yaml"
    config = load_config(args.config, base_path=base_path)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.offline:
        config["weights"]["offline_mode"] = True
        config["weights"]["pretrained"] = False
    if args.strategy:
        config["runtime"]["strategy"] = args.strategy
    config["runtime"]["run_dir"] = infer_run_dir(config)
    Path(config["runtime"]["run_dir"]).mkdir(parents=True, exist_ok=True)
    trainer = Trainer(config)
    print(json.dumps(trainer.fit(), indent=2))


if __name__ == "__main__":
    main()
