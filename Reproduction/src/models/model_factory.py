from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torchvision import models

from Reproduction.src.models.model_utils import SmallCNN, freeze_all, unfreeze_last_n_children, unfreeze_module
from Reproduction.src.utils.config import resolve_config_path

try:
    import timm
except ImportError:
    timm = None


TORCHVISION_BUILDERS: dict[str, tuple[Callable, str]] = {
    "vgg16": (models.vgg16, "VGG16_Weights"),
    "vgg19": (models.vgg19, "VGG19_Weights"),
    "mobilenetv2": (models.mobilenet_v2, "MobileNet_V2_Weights"),
    "inceptionv3": (models.inception_v3, "Inception_V3_Weights"),
    "resnet50": (models.resnet50, "ResNet50_Weights"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights"),
}

TIMM_BUILDERS = {
    "xception": "xception",
    "inception_resnet_v2": "inception_resnet_v2",
}


@dataclass
class ModelLoadResult:
    model: nn.Module
    init_source: str
    warning: str | None = None


def _load_torchvision_weights_enum(weights_name: str):
    return getattr(models, weights_name).DEFAULT


def _create_torchvision_model(model_name: str, pretrained: bool, allow_random_init: bool) -> ModelLoadResult:
    builder, weights_name = TORCHVISION_BUILDERS[model_name]
    if not pretrained:
        return ModelLoadResult(model=builder(weights=None), init_source="random_init")
    try:
        return ModelLoadResult(
            model=builder(weights=_load_torchvision_weights_enum(weights_name)),
            init_source="torchvision_pretrained",
        )
    except Exception as exc:
        if allow_random_init:
            return ModelLoadResult(
                model=builder(weights=None),
                init_source="random_init_fallback",
                warning=f"Falling back to random init for {model_name}: {exc}",
            )
        raise RuntimeError(
            f"Failed to load pretrained weights for {model_name}. "
            f"Set weights.local_weights_path, enable allow_random_init_when_download_fails, or use offline mode."
        ) from exc


def _create_timm_model(model_name: str, pretrained: bool, allow_random_init: bool) -> ModelLoadResult:
    if timm is None:
        raise RuntimeError(
            f"Model '{model_name}' requires the timm package, but timm is not installed. "
            "Install timm or switch to a torchvision-backed model."
        )
    timm_name = TIMM_BUILDERS[model_name]
    if not pretrained:
        return ModelLoadResult(model=timm.create_model(timm_name, pretrained=False), init_source="random_init")
    try:
        return ModelLoadResult(
            model=timm.create_model(timm_name, pretrained=True),
            init_source="timm_pretrained",
        )
    except Exception as exc:
        if allow_random_init:
            return ModelLoadResult(
                model=timm.create_model(timm_name, pretrained=False),
                init_source="random_init_fallback",
                warning=f"Falling back to random init for {model_name}: {exc}",
            )
        raise RuntimeError(
            f"Failed to load pretrained weights for {model_name}. "
            f"Use local weights, fix network access, or enable allow_random_init_when_download_fails."
        ) from exc


def _replace_classifier_torchvision(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "mobilenetv2":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name in {"resnet50", "inceptionv3"}:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if model_name == "inceptionv3" and getattr(model, "AuxLogits", None) is not None:
            aux_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(aux_features, num_classes)
    elif model_name == "efficientnet_b0":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _replace_classifier_generic(model: nn.Module, num_classes: int) -> nn.Module:
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        elif isinstance(classifier, nn.Sequential) and isinstance(classifier[-1], nn.Linear):
            classifier[-1] = nn.Linear(classifier[-1].in_features, num_classes)
        return model
    raise RuntimeError("Unable to replace classifier head for model")


def _load_local_weights(model: nn.Module, local_weights_path: str) -> None:
    state = torch.load(local_weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)


def _unfreeze_classification_head(model: nn.Module) -> None:
    updated = False
    if hasattr(model, "get_classifier"):
        classifier_ref = model.get_classifier()
        if isinstance(classifier_ref, str):
            updated = unfreeze_module(getattr(model, classifier_ref, None)) or updated
        else:
            updated = unfreeze_module(classifier_ref) or updated
    for name in ["classifier", "fc", "head", "classif", "last_linear", "AuxLogits"]:
        updated = unfreeze_module(getattr(model, name, None)) or updated
    if not updated:
        raise RuntimeError(
            "Unable to identify a trainable classification head for freeze_mode=linear_probe. "
            "Add the model-specific head name to model_factory._unfreeze_classification_head."
        )


def _apply_freeze_mode(model: nn.Module, freeze_backbone: bool, partial_unfreeze: str | None) -> None:
    if freeze_backbone:
        freeze_all(model)
        _unfreeze_classification_head(model)
    if partial_unfreeze:
        freeze_all(model)
        _unfreeze_classification_head(model)
        unfreeze_last_n_children(model, 2)


def create_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    local_weights_path: str | None = None,
    freeze_backbone: bool = False,
    partial_unfreeze: str | None = None,
    allow_random_init_when_download_fails: bool = False,
) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "small_cnn":
        model = SmallCNN(num_classes=num_classes)
    elif model_name in TORCHVISION_BUILDERS:
        loaded = _create_torchvision_model(
            model_name=model_name,
            pretrained=pretrained,
            allow_random_init=allow_random_init_when_download_fails,
        )
        model = _replace_classifier_torchvision(loaded.model, model_name, num_classes)
    elif model_name in TIMM_BUILDERS:
        loaded = _create_timm_model(
            model_name=model_name,
            pretrained=pretrained,
            allow_random_init=allow_random_init_when_download_fails,
        )
        model = _replace_classifier_generic(loaded.model, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    if local_weights_path:
        path = Path(local_weights_path)
        if not path.exists():
            raise FileNotFoundError(f"Local weights file not found: {path}")
        _load_local_weights(model, str(path))
    _apply_freeze_mode(model, freeze_backbone=freeze_backbone, partial_unfreeze=partial_unfreeze)
    return model


def create_model_from_config(config: dict) -> nn.Module:
    weights_cfg = config.get("weights", {})
    model_cfg = config["model"]
    freeze_mode = model_cfg.get("freeze_mode", "full_finetune")
    offline_mode = weights_cfg.get("offline_mode", False)
    local_weights_path = weights_cfg.get("local_weights_path")
    resolved_local_weights = resolve_config_path(config, local_weights_path) if local_weights_path else None
    return create_model(
        model_name=model_cfg["name"],
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=bool(weights_cfg.get("pretrained", True) and not offline_mode),
        local_weights_path=None if resolved_local_weights is None else str(resolved_local_weights),
        freeze_backbone=freeze_mode == "linear_probe",
        partial_unfreeze="last_blocks" if freeze_mode == "partial_finetune" else None,
        allow_random_init_when_download_fails=weights_cfg.get("allow_random_init_when_download_fails", False),
    )
