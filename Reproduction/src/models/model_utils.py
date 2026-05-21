from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def freeze_all(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_last_n_children(module: nn.Module, num_children: int) -> None:
    children = [child for child in module.children()]
    for child in children[-num_children:]:
        for param in child.parameters():
            param.requires_grad = True


def unfreeze_module(module: nn.Module | None) -> bool:
    if module is None or not isinstance(module, nn.Module):
        return False
    updated = False
    for param in module.parameters():
        param.requires_grad = True
        updated = True
    return updated
