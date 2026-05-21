from __future__ import annotations

import argparse
import json

from common import ROOT
from Reproduction.src.engine.trainer import Trainer
from Reproduction.src.utils.config import load_config
from Reproduction.src.utils.experiment import REPRODUCTION_MODELS, STRATEGIES, apply_runtime, apply_strategy, apply_training_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="*", default=None)
    args = parser.parse_args()
    base_path = ROOT / "Reproduction" / "configs" / "base.yaml"
    completed = []
    count = 0
    models = args.models or REPRODUCTION_MODELS
    strategies = args.strategies or list(STRATEGIES.keys())
    for model_name in models:
        model_config = ROOT / "Reproduction" / "configs" / f"{model_name}.yaml"
        for strategy in strategies:
            if args.limit is not None and count >= args.limit:
                print(json.dumps(completed, indent=2))
                return
            config = load_config(model_config, base_path=base_path)
            config = apply_training_overrides(config, epochs=args.epochs, offline=args.offline)
            config = apply_strategy(config, strategy)
            config = apply_runtime(config, ROOT / "Reproduction" / "results" / model_name / strategy)
            trainer = Trainer(config)
            result = trainer.fit()
            completed.append({"model": model_name, "strategy": strategy, **result})
            count += 1
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
