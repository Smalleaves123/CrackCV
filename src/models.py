"""
模型构建模块（PyTorch）
支持 6 个 backbone：
- vgg16, vgg19: Flatten -> Dense(4096) -> Dense(4096) -> Dense(2)
- 其他模型: GAP -> Dense(2)

VGG 使用 torchvision，InceptionResNetV2/Xception 使用 timm
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TORCH_HOME"] = "E:\\torch_cache"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models

SUPPORTED_MODELS = [
    "vgg16",
    "vgg19",
    "mobilenetv2",
    "inceptionresnetv2",
    "inceptionv3",
    "xception",
]

# 需要 timm 的模型
TIMM_MODELS = {"inceptionresnetv2", "xception"}

# VGG Flatten 后的维度：7*7*512 = 25088
VGG_FLATTEN_DIM = 25088


def _build_vgg_classifier(num_classes):
    """VGG 专用分类头：Flatten + 4096 + 4096 + num_classes"""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(VGG_FLATTEN_DIM, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, num_classes),
    )


def _get_trainable_params(model, backbone_name):
    """获取需要训练的参数的生成器"""
    if backbone_name in ("vgg16", "vgg19"):
        # 只训练 classifier
        return list(model.classifier.parameters())
    elif backbone_name in ("mobilenetv2", "inceptionv3"):
        # 只训练 fc/classifier 头
        if backbone_name == "mobilenetv2":
            return list(model.classifier.parameters())
        else:
            return list(model.fc.parameters())
    else:
        # timm 模型：只训练最后的分类头
        # timm 模型分类头通常叫 'head' 或 'fc' 或 'logits'
        head_params = []
        for name, param in model.named_parameters():
            if any(k in name for k in ["head", "fc", "logits", "classifier"]):
                # 确保不是 backbone 中的某个中间 classifier
                if name.split(".")[0] in ["head", "fc", "logits", "classifier"]:
                    if hasattr(model, name.split(".")[0]):
                        head_params.append(param)
        return head_params


def build_model(
    backbone_name: str,
    num_classes: int = 2,
    train_backbone: bool = False,
    learning_rate: float = 1e-5,
    device: str = "cuda",
):
    """
    构建模型、优化器和损失函数

    Returns:
        model, optimizer, criterion, num_trainable_params
    """
    backbone_name = backbone_name.lower()

    if backbone_name not in SUPPORTED_MODELS:
        raise ValueError(f"不支持的模型: {backbone_name}，支持: {SUPPORTED_MODELS}")

    # ======================== 创建 backbone ========================
    if backbone_name == "vgg16":
        model = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        # 替换分类头
        model.classifier = _build_vgg_classifier(num_classes)
        if not train_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    elif backbone_name == "vgg19":
        model = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
        model.classifier = _build_vgg_classifier(num_classes)
        if not train_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    elif backbone_name == "mobilenetv2":
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # MobileNetV2 forward: x = features(x) -> avgpool -> flatten -> classifier(x)
        # classifier 默认是 [Dropout, Linear(1280, 1000)]
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
        if not train_backbone:
            for param in model.features.parameters():
                param.requires_grad = False

    elif backbone_name == "inceptionv3":
        model = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
        )
        # 移除辅助分类器
        model.aux_logits = False
        model.AuxLogits = None

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if not train_backbone:
            # 冻结除 fc 外的所有参数
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

    elif backbone_name == "inceptionresnetv2":
        import timm

        model = timm.create_model(
            "inception_resnet_v2", pretrained=True, num_classes=num_classes
        )
        if not train_backbone:
            # 全部冻结
            for param in model.parameters():
                param.requires_grad = False
            # 解冻分类头（timm 模型分类头在 head 或 last_linear 或 get_classifier()）
            if hasattr(model, "get_classifier"):
                for param in model.get_classifier().parameters():
                    param.requires_grad = True
            elif hasattr(model, "head"):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, "last_linear"):
                for param in model.last_linear.parameters():
                    param.requires_grad = True

    elif backbone_name == "xception":
        import timm

        model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
        if not train_backbone:
            for param in model.parameters():
                param.requires_grad = False
            if hasattr(model, "get_classifier"):
                for param in model.get_classifier().parameters():
                    param.requires_grad = True
            elif hasattr(model, "head"):
                for param in model.head.parameters():
                    param.requires_grad = True
            elif hasattr(model, "last_linear"):
                for param in model.last_linear.parameters():
                    param.requires_grad = True

    # ======================== 移动到设备 ========================
    model = model.to(device)

    # ======================== 优化器和损失 ========================
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    num_total = sum(p.numel() for p in model.parameters())

    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"  总参数量: {num_total:,} ({num_total / 1e6:.2f}M)")
    print(f"  可训练参数: {num_trainable:,} ({num_trainable / 1e6:.2f}M)")
    print(f"  设备: {device}")

    return model, optimizer, criterion, num_trainable
