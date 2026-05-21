from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Reproduction.src.engine.trainer import Trainer
from Reproduction.src.utils.config import attach_project_root, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "Addition" / "02_hyperparameter_study" / "configs" / "mobilenetv2_lr_sweep.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    study_config = attach_project_root(load_config(args.config), ROOT)
    base_config = attach_project_root(load_config(ROOT / "Reproduction" / "configs" / "base.yaml"), ROOT)
    completed = []
    for learning_rate in study_config["study"]["learning_rates"]:
        for batch_size in study_config["study"]["batch_sizes"]:
            config = deepcopy(base_config)
            config["model"]["name"] = study_config["study"]["model_name"]
            config["model"]["freeze_mode"] = study_config["study"]["freeze_mode"]
            config["augmentation"]["enabled"] = study_config["study"]["augmentation_enabled"]
            config["training"]["epochs"] = args.epochs
            config["training"]["learning_rate"] = learning_rate
            config["training"]["batch_size"] = batch_size
            config["weights"]["offline_mode"] = args.offline
            if args.offline:
                config["weights"]["pretrained"] = False
            tag = f"lr_{learning_rate}_bs_{batch_size}"
            config["runtime"]["strategy"] = tag
            config["runtime"]["run_dir"] = str(ROOT / "Addition" / "02_hyperparameter_study" / "results" / tag)
            trainer = Trainer(config)
            completed.append(
                {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    **trainer.fit(),
                }
            )
    summary_path = ROOT / "Addition" / "02_hyperparameter_study" / "results" / "study_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
