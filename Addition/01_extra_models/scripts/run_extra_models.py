from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Reproduction.src.engine.trainer import Trainer
from Reproduction.src.utils.config import load_config
from Reproduction.src.utils.experiment import EXTRA_MODELS, STRATEGIES, apply_runtime, apply_strategy, apply_training_overrides


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--strategies", nargs="*", default=None)
    args = parser.parse_args()
    base_config = ROOT / "Reproduction" / "configs" / "base.yaml"
    completed = []
    models = args.models or EXTRA_MODELS
    strategies = args.strategies or list(STRATEGIES.keys())
    for model_name in models:
        config_path = ROOT / "Addition" / "01_extra_models" / "configs" / f"{model_name}.yaml"
        for strategy in strategies:
            config = load_config(config_path, base_path=base_config)
            config = apply_training_overrides(config, epochs=args.epochs, offline=args.offline)
            config = apply_strategy(config, strategy)
            config = apply_runtime(config, ROOT / "Addition" / "01_extra_models" / "results" / model_name / strategy)
            trainer = Trainer(config)
            completed.append({"model": model_name, "strategy": strategy, **trainer.fit()})
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
