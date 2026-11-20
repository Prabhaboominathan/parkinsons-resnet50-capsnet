"""Hybrid ResNet50 + CapsNet classifier (paper Eq. 5-12).

ResNet50 (up to layer4) -> feature maps Z -> primary capsules (Eq. 7)
-> class capsules with dynamic routing (Eq. 8-10) -> capsule length = class score (Eq. 11).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def squash(s: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Squashing non-linearity: short vectors -> ~0, long vectors -> length ~1."""
    sq_norm = (s ** 2).sum(dim=dim, keepdim=True)
    return (sq_norm / (1.0 + sq_norm)) * s / torch.sqrt(sq_norm + eps)


class ResNet50CapsNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False,
                 caps_types: int = 32, caps_dim: int = 8, class_caps_dim: int = 16,
                 routing_iters: int = 3, img_size: int = 224):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])      # drop avgpool + fc
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.caps_types, self.caps_dim = caps_types, caps_dim
        self.num_classes, self.class_caps_dim = num_classes, class_caps_dim
        self.routing_iters = routing_iters

        # Feature maps -> primary capsules (u_i = W_i Z)
        self.primary = nn.Conv2d(2048, caps_types * caps_dim, kernel_size=1)

        fmap = img_size // 32
        self.num_primary = fmap * fmap * caps_types
        # One transformation matrix per (primary capsule, class capsule) pair
        self.W = nn.Parameter(0.01 * torch.randn(self.num_primary, num_classes, class_caps_dim, caps_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                                           # (B, 2048, H/32, W/32)
        u = self.primary(feats)                                            # (B, T*D, h, w)
        B = u.size(0)
        u = u.view(B, self.caps_types, self.caps_dim, -1)                  # (B, T, D, h*w)
        u = u.permute(0, 3, 1, 2).reshape(B, -1, self.caps_dim)            # (B, N, D)
        u = squash(u)

        u_hat = torch.einsum("bni,ncdi->bncd", u, self.W)                  # (B, N, C, D_out)

        b = torch.zeros(B, self.num_primary, self.num_classes, device=x.device)
        for it in range(self.routing_iters):
            c = F.softmax(b, dim=2)                                        # coupling coefficients
            s = (c.unsqueeze(-1) * u_hat).sum(dim=1)                       # (B, C, D_out)
            v = squash(s)                                                  # class capsule outputs
            if it < self.routing_iters - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1)               # agreement update
        return v.norm(dim=-1)                                              # (B, C) capsule lengths


def margin_loss(lengths: torch.Tensor, targets: torch.Tensor, m_pos: float = 0.9,
                m_neg: float = 0.1, lam: float = 0.5) -> torch.Tensor:
    """Margin loss (paper Eq. 12)."""
    t = F.one_hot(targets, lengths.size(1)).float()
    loss = t * F.relu(m_pos - lengths) ** 2 + lam * (1 - t) * F.relu(lengths - m_neg) ** 2
    return loss.sum(dim=1).mean()
