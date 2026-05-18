from __future__ import annotations

from typing import Dict, Tuple

import timm
import torch
from torch import nn


TIMM_MODELS: Dict[str, str] = {
    "vgg16": "vgg16",
    "vgg19": "vgg19",
    "mobilenetv2": "mobilenetv2_100",
    "inceptionresnetv2": "inception_resnet_v2",
    "inceptionv3": "inception_v3",
    "xception": "xception",
}


class CrackClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        train_backbone: bool = True,
        input_size: Tuple[int, int, int] = (3, 227, 227),
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if backbone_name not in TIMM_MODELS:
            raise ValueError("Unsupported backbone: {0}".format(backbone_name))

        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            TIMM_MODELS[backbone_name],
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        self._set_backbone_trainable(train_backbone)

        feature_shape = self._infer_feature_shape(input_size)
        if backbone_name.startswith("vgg"):
            flattened_dim = int(feature_shape[0] * feature_shape[1] * feature_shape[2])
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flattened_dim, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
        else:
            channels = int(feature_shape[0])
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(channels, num_classes),
            )

    def _set_backbone_trainable(self, train_backbone: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = train_backbone

    def _infer_feature_shape(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            features = self.backbone.forward_features(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            if features.ndim != 4:
                raise ValueError("Expected 4D feature maps, got shape {0}".format(tuple(features.shape)))
            return int(features.shape[1]), int(features.shape[2]), int(features.shape[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return self.head(features)


def build_model(
    backbone_name: str,
    train_backbone: bool = True,
    input_size: Tuple[int, int, int] = (3, 227, 227),
    num_classes: int = 2,
) -> CrackClassifier:
    return CrackClassifier(
        backbone_name=backbone_name,
        train_backbone=train_backbone,
        input_size=input_size,
        num_classes=num_classes,
    )


def find_last_4d_layer(model: nn.Module, input_size: Tuple[int, int, int] = (3, 227, 227)):
    captured = []
    hooks = []

    def hook_factory(name: str):
        def hook(_module, _inputs, output):
            tensor = output[-1] if isinstance(output, (list, tuple)) else output
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
                captured.append((name, _module))
        return hook

    for name, module in model.backbone.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook_factory(name)))

    was_training = model.backbone.training
    model.backbone.eval()
    device = next(model.backbone.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros(1, *input_size, device=device)
        model.backbone.forward_features(dummy)

    for handle in hooks:
        handle.remove()
    if was_training:
        model.backbone.train()

    if not captured:
        raise ValueError("No 4D feature layer found for Grad-CAM.")
    return captured[-1]
