"""Image and clinical encoders."""

from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Torchvision CNN backbone reduced to a fixed-width embedding.

    The classifier head is replaced with a projection to ``embedding_dim`` so
    every model in the comparison shares an identical fusion interface. The final
    convolutional block is exposed as :attr:`feature_layer` for Grad-CAM.

    Args:
        backbone: Torchvision model name (``resnet18``, ``resnet50``, ``densenet121``,
            ``efficientnet_b0``).
        pretrained: Load ImageNet weights.
        embedding_dim: Width of the output embedding.
        dropout: Dropout applied to the embedding.
        in_channels: 1 for raw grayscale, 3 to reuse ImageNet stems.
    """

    SUPPORTED = ("resnet18", "resnet34", "resnet50", "densenet121", "efficientnet_b0")

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if backbone not in self.SUPPORTED:
            raise ValueError(f"unsupported backbone {backbone!r}; choose from {self.SUPPORTED}")
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        net, feature_dim = self._build_backbone(backbone, pretrained)
        self.backbone = net
        self.backbone_feature_dim = feature_dim

        if in_channels != 3:
            self._adapt_input_channels(in_channels)

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """Return a backbone truncated to its pooled feature map, and its width."""
        from torchvision import models

        weights = "DEFAULT" if pretrained else None
        if name.startswith("resnet"):
            net = getattr(models, name)(weights=weights)
            feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            return net, feature_dim
        if name == "densenet121":
            net = models.densenet121(weights=weights)
            feature_dim = net.classifier.in_features
            net.classifier = nn.Identity()
            return net, feature_dim
        if name == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            feature_dim = net.classifier[1].in_features
            net.classifier = nn.Identity()
            return net, feature_dim
        raise ValueError(f"unhandled backbone {name!r}")

    def _adapt_input_channels(self, in_channels: int) -> None:
        """Average pretrained RGB stem weights down to ``in_channels``."""
        if self.backbone_name.startswith("resnet"):
            old = self.backbone.conv1
        elif self.backbone_name == "densenet121":
            old = self.backbone.features.conv0
        else:
            old = self.backbone.features[0][0]

        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1))
        if self.backbone_name.startswith("resnet"):
            self.backbone.conv1 = new
        elif self.backbone_name == "densenet121":
            self.backbone.features.conv0 = new
        else:
            self.backbone.features[0][0] = new

    @property
    def feature_layer(self) -> nn.Module:
        """Last convolutional block, the Grad-CAM attachment point."""
        if self.backbone_name.startswith("resnet"):
            return self.backbone.layer4
        if self.backbone_name == "densenet121":
            return self.backbone.features.denseblock4
        return self.backbone.features[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into ``(batch, embedding_dim)``."""
        features = self.backbone(images)
        return self.projection(features)


class ClinicalEncoder(nn.Module):
    """Small MLP over standardized clinical features and missingness indicators.

    Args:
        input_dim: Width of the clinical feature vector (features + indicators).
        hidden_dims: Widths of the hidden layers.
        embedding_dim: Width of the output embedding.
        dropout: Dropout between layers; also enables MC dropout at inference.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        dims = [input_dim, *(hidden_dims or [64, 32])]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], embedding_dim))
        layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """Encode clinical features into ``(batch, embedding_dim)``."""
        return self.network(clinical)
