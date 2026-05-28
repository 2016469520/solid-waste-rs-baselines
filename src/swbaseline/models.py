from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _torchvision_models():
    try:
        import torchvision.models as tvm
    except Exception as exc:
        raise ImportError("torchvision is required for this baseline model") from exc
    return tvm


def _get_weights(tvm, model_name: str, pretrained: bool):
    if not pretrained:
        return None
    weight_map = {
        "resnet18": tvm.ResNet18_Weights.DEFAULT,
        "resnet50": tvm.ResNet50_Weights.DEFAULT,
        "mobilenet_v3_small": tvm.MobileNet_V3_Small_Weights.DEFAULT,
        "efficientnet_b0": tvm.EfficientNet_B0_Weights.DEFAULT,
        "convnext_tiny": tvm.ConvNeXt_Tiny_Weights.DEFAULT,
    }
    return weight_map[model_name]


def _expand_conv_weight(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv
    new_conv = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        old = conv.weight.data
        if in_channels < old.shape[1]:
            new_conv.weight.copy_(old[:, :in_channels])
        else:
            new_conv.weight[:, : old.shape[1]].copy_(old)
            extra = old.mean(dim=1, keepdim=True).repeat(1, in_channels - old.shape[1], 1, 1)
            new_conv.weight[:, old.shape[1] :].copy_(extra)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)
    return new_conv


def _adapt_first_conv(model: nn.Module, model_name: str, in_channels: int) -> None:
    if model_name.startswith("resnet"):
        model.conv1 = _expand_conv_weight(model.conv1, in_channels)
    elif model_name == "mobilenet_v3_small":
        model.features[0][0] = _expand_conv_weight(model.features[0][0], in_channels)
    elif model_name == "efficientnet_b0":
        model.features[0][0] = _expand_conv_weight(model.features[0][0], in_channels)
    elif model_name == "convnext_tiny":
        model.features[0][0] = _expand_conv_weight(model.features[0][0], in_channels)
    else:
        raise ValueError(f"Unsupported first-conv adaptation for {model_name}")


def _replace_classifier(model: nn.Module, model_name: str, num_classes: int, dropout: float) -> None:
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name in {"mobilenet_v3_small", "efficientnet_b0"}:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_tiny":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier replacement for {model_name}")


def create_model(
    name: str,
    in_channels: int = 4,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    name = name.lower()
    if name == "small_cnn":
        return SmallCNN(in_channels=in_channels, num_classes=num_classes, dropout=dropout)

    tvm = _torchvision_models()
    if name not in {"resnet18", "resnet50", "mobilenet_v3_small", "efficientnet_b0", "convnext_tiny"}:
        raise ValueError(f"Unknown model '{name}'")
    constructor: Callable = getattr(tvm, name)
    model = constructor(weights=_get_weights(tvm, name, pretrained))
    _adapt_first_conv(model, name, in_channels)
    _replace_classifier(model, name, num_classes, dropout)
    return model

