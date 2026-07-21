"""Baseline classifiers for the etiology SOTA comparison.

All take one patient = (B, V=3, T=16, 1, 224, 224) and output 2-class logits.
Kept intentionally simple; the point is a fair reference, not tuning.
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def _gray3(x):                                   # (N,1,H,W) -> (N,3,H,W)
    return x.repeat(1, 3, 1, 1)


def _resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    try:
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        net = resnet18(weights=None)
    feat = nn.Sequential(*list(net.children())[:-1])   # -> (N,512,1,1)
    return feat


class ResNetMV(nn.Module):
    """2D ResNet-18 per view on the mid-cycle frame; average-pool the views."""
    def __init__(self):
        super().__init__(); self.feat = _resnet18(); self.fc = nn.Linear(512, 2)

    def forward(self, views):
        B, V, T = views.shape[:3]
        x = views[:, :, T // 2, 0]                       # (B,V,H,W)
        x = _gray3(x.reshape(B * V, 1, C.IMG_SIZE, C.IMG_SIZE))
        f = self.feat(x).flatten(1).reshape(B, V, -1).mean(1)
        return self.fc(f)


class MVCNN(nn.Module):
    """Classical multi-view CNN: shared per-view ResNet on the mid-frame, fused by
    view pooling (max)."""
    def __init__(self):
        super().__init__(); self.feat = _resnet18(); self.fc = nn.Linear(512, 2)

    def forward(self, views):
        B, V, T = views.shape[:3]
        x = _gray3(views[:, :, T // 2, 0].reshape(B * V, 1, C.IMG_SIZE, C.IMG_SIZE))
        f = self.feat(x).flatten(1).reshape(B, V, -1).max(1).values      # view pooling
        return self.fc(f)


class SAMIL(nn.Module):
    """Attention-based multiple-instance multi-view fusion: per-view/per-frame
    instances pooled by a gated-attention MIL head."""
    def __init__(self, n_frames=4):
        super().__init__(); self.feat = _resnet18()
        self.att = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Linear(512, 2); self.n_frames = n_frames

    def forward(self, views):
        B, V, T = views.shape[:3]
        idx = torch.linspace(0, T - 1, self.n_frames).long()
        x = views[:, :, idx, 0]                          # (B,V,nf,H,W)
        N = V * self.n_frames
        x = _gray3(x.reshape(B * N, 1, C.IMG_SIZE, C.IMG_SIZE))
        f = self.feat(x).flatten(1).reshape(B, N, -1)    # (B,N,512)
        a = torch.softmax(self.att(f), dim=1)            # (B,N,1)
        return self.fc((a * f).sum(1))


class SwinMV(nn.Module):
    """Swin Transformer (Swin-T, ImageNet-pretrained) per view on the mid-frame; mean views."""
    def __init__(self):
        super().__init__()
        from torchvision.models import swin_t, Swin_T_Weights
        try:
            net = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        except Exception:
            net = swin_t(weights=None)
        net.head = nn.Identity(); self.net = net; self.fc = nn.Linear(768, 2)

    def forward(self, views):
        B, V, T = views.shape[:3]
        x = _gray3(views[:, :, T // 2, 0].reshape(B * V, 1, C.IMG_SIZE, C.IMG_SIZE))
        f = self.net(x).reshape(B, V, -1).mean(1)
        return self.fc(f)


class ConvNeXtMV(nn.Module):
    """ConvNeXt-T (ImageNet-pretrained) per view on the mid-frame; mean views."""
    def __init__(self):
        super().__init__()
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
        try:
            net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        except Exception:
            net = convnext_tiny(weights=None)
        self.features = net.features; self.avgpool = net.avgpool; self.fc = nn.Linear(768, 2)

    def forward(self, views):
        B, V, T = views.shape[:3]
        x = _gray3(views[:, :, T // 2, 0].reshape(B * V, 1, C.IMG_SIZE, C.IMG_SIZE))
        f = self.avgpool(self.features(x)).flatten(1).reshape(B, V, -1).mean(1)
        return self.fc(f)


def build(name):
    return {"resnet": ResNetMV, "mvcnn": MVCNN, "samil": SAMIL,
            "swin": SwinMV, "convnext": ConvNeXtMV}[name]()
