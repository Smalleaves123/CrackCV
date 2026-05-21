from __future__ import annotations

from copy import deepcopy
from pathlib import Path


REPRODUCTION_MODELS = [
    "vgg16",
    "vgg19",
    "mobilenetv2",
    "inceptionv3",
    "inception_resnet_v2",
    "xception",
]

EXTRA_MODELS = ["resnet50", "efficientnet_b0"]

STRATEGIES: dict[str, dict[str, object]] = {
    "full_finetune_aug": {"freeze_mode": "full_finetune", "augmentation_enabled": True},
    "full_finetune_no_aug": {"freeze_mode": "full_finetune", "augmentation_enabled": False},
    "linear_probe_aug": {"freeze_mode": "linear_probe", "augmentation_enabled": True},
    "linear_probe_no_aug": {"freeze_mode": "linear_probe", "augmentation_enabled": False},
}

MODEL_PARAMETER_MILLIONS = {
    "vgg16": 138.0,
    "vgg19": 144.0,
    "mobilenetv2": 3.5,
    "inceptionv3": 27.0,
    "inception_resnet_v2": 56.0,
    "xception": 23.0,
    "resnet50": 25.6,
    "efficientnet_b0": 5.3,
    "small_cnn": 0.02,
}


def clone_config(config: dict) -> dict:
    return deepcopy(config)


def apply_strategy(config: dict, strategy_name: str) -> dict:
    strategy = STRATEGIES[strategy_name]
    config["model"]["freeze_mode"] = str(strategy["freeze_mode"])
    config["augmentation"]["enabled"] = bool(strategy["augmentation_enabled"])
    config["runtime"]["strategy"] = strategy_name
    return config


def apply_runtime(config: dict, run_dir: str | Path) -> dict:
    config["runtime"]["run_dir"] = str(Path(run_dir))
    return config


def apply_training_overrides(config: dict, epochs: int | None = None, offline: bool | None = None) -> dict:
    if epochs is not None:
        config["training"]["epochs"] = epochs
    if offline is not None:
        config["weights"]["offline_mode"] = offline
        if offline:
            config["weights"]["pretrained"] = False
    return config
