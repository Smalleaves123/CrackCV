from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torchvision import models


SUPPORTED_BACKBONES = {
    "vgg16",
    "vgg19",
    "mobilenetv2",
    "inceptionresnetv2",
    "inceptionv3",
    "xception",
}


class InceptionV3Backbone(nn.Module):
    def __init__(self, use_pretrained: bool) -> None:
        super().__init__()
        weights = models.Inception_V3_Weights.DEFAULT if use_pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=False)

        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.maxpool1 = model.maxpool1
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.maxpool2 = model.maxpool2
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        return x


class CrackClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        train_backbone: bool = True,
        input_size: Tuple[int, int, int] = (3, 227, 227),
        num_classes: int = 2,
        use_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                "Unsupported backbone: {0}. Supported backbones: {1}".format(
                    backbone_name, ", ".join(sorted(SUPPORTED_BACKBONES))
                )
            )

        self.backbone_name = backbone_name
        self.use_pretrained = use_pretrained
        self.backbone = _build_backbone(backbone_name, use_pretrained=use_pretrained)
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
        was_training = self.backbone.training
        self.backbone.eval()
        device = next(self.backbone.parameters()).device
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size, device=device)
            features = self.backbone(dummy)
            if features.ndim != 4:
                raise ValueError("Expected 4D feature maps, got shape {0}".format(tuple(features.shape)))
            shape = (int(features.shape[1]), int(features.shape[2]), int(features.shape[3]))
        if was_training:
            self.backbone.train()
        return shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


def _build_backbone(backbone_name: str, use_pretrained: bool) -> nn.Module:
    if backbone_name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if use_pretrained else None
        return models.vgg16(weights=weights).features
    if backbone_name == "vgg19":
        weights = models.VGG19_Weights.DEFAULT if use_pretrained else None
        return models.vgg19(weights=weights).features
    if backbone_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        return models.mobilenet_v2(weights=weights).features
    if backbone_name == "inceptionresnetv2":
        return _build_pretrainedmodels_backbone(
            model_name="inceptionresnetv2",
            use_pretrained=use_pretrained,
        )
    if backbone_name == "inceptionv3":
        return InceptionV3Backbone(use_pretrained=use_pretrained)
    if backbone_name == "xception":
        return _build_pretrainedmodels_backbone(
            model_name="xception",
            use_pretrained=use_pretrained,
        )
    raise ValueError("Unsupported backbone: {0}".format(backbone_name))


def _build_pretrainedmodels_backbone(model_name: str, use_pretrained: bool) -> nn.Module:
    try:
        import pretrainedmodels
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Backbone {0} requires the 'pretrainedmodels' package. "
            "Install it from requirements.txt before training.".format(model_name)
        ) from exc

    pretrained = "imagenet" if use_pretrained else None
    model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
    return PretrainedModelsBackbone(model)


class PretrainedModelsBackbone(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model.features(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if not isinstance(features, torch.Tensor):
            raise ValueError("Expected tensor features from pretrainedmodels backbone.")
        return features


def build_model(
    backbone_name: str,
    train_backbone: bool = True,
    input_size: Tuple[int, int, int] = (3, 227, 227),
    num_classes: int = 2,
    use_pretrained: bool = True,
) -> CrackClassifier:
    return CrackClassifier(
        backbone_name=backbone_name,
        train_backbone=train_backbone,
        input_size=input_size,
        num_classes=num_classes,
        use_pretrained=use_pretrained,
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
        model.backbone(dummy)

    for handle in hooks:
        handle.remove()
    if was_training:
        model.backbone.train()

    if not captured:
        raise ValueError("No 4D feature layer found for Grad-CAM.")
    return captured[-1]
