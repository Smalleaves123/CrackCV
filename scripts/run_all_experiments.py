from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BACKBONES = [
    "vgg16",
    "vgg19",
    "mobilenetv2",
    "inceptionresnetv2",
    "inceptionv3",
    "xception",
]

STRATEGIES = [
    ("e2e_aug", True, True),
    ("e2e_no_aug", True, False),
    ("frozen_aug", False, True),
    ("frozen_no_aug", False, False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all crack detection experiments.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--raw-data-dir", default="dataset")
    parser.add_argument("--raw-backup-dir", default="data/raw_backup")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rotation-mode", choices=["positive", "symmetric"], default="positive")
    parser.add_argument("--use-pretrained", choices=["true", "false"], default="true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for backbone in BACKBONES:
        for strategy_name, train_backbone, augmentation in STRATEGIES:
            experiment_dir = Path(args.output_root) / "{0}_{1}".format(backbone, strategy_name)
            train_cmd = [
                sys.executable,
                "-m",
                "src.train",
                "--data-dir",
                args.data_dir,
                "--raw-data-dir",
                args.raw_data_dir,
                "--raw-backup-dir",
                args.raw_backup_dir,
                "--backbone",
                backbone,
                "--train-backbone",
                str(train_backbone).lower(),
                "--augmentation",
                str(augmentation).lower(),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--rotation-mode",
                args.rotation_mode,
                "--use-pretrained",
                args.use_pretrained,
                "--lr",
                str(args.lr),
                "--max-epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--seed",
                str(args.seed),
                "--refresh-raw-backup",
                "false",
                "--output-dir",
                str(experiment_dir),
            ]
            subprocess.run(train_cmd, check=True)

            eval_cmd = [
                sys.executable,
                "-m",
                "src.evaluate",
                "--model-path",
                str(experiment_dir / "best_model.pt"),
                "--data-dir",
                args.data_dir,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(experiment_dir),
            ]
            subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
