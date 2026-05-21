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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    base_config = ROOT / "Reproduction" / "configs" / "base.yaml"
    configs = {
        "small_cnn_from_scratch": ROOT / "Addition" / "03_train_from_scratch" / "configs" / "small_cnn.yaml",
        "mobilenetv2_pretrained": ROOT / "Addition" / "03_train_from_scratch" / "configs" / "mobilenetv2_pretrained.yaml",
    }
    completed = []
    for tag, config_path in configs.items():
        config = load_config(config_path, base_path=base_config)
        config["training"]["epochs"] = args.epochs
        if args.offline:
            config["weights"]["offline_mode"] = True
            if config["model"]["name"] != "small_cnn":
                config["weights"]["pretrained"] = False
        config["runtime"]["strategy"] = tag
        config["runtime"]["run_dir"] = str(ROOT / "Addition" / "03_train_from_scratch" / "results" / tag)
        trainer = Trainer(config)
        completed.append({"experiment": tag, **trainer.fit()})
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
