"""The three official-train CelebA black boxes (Attractive/Male, 4-way head: [0:2] Attractive, [2:4] Male) for the
baselines in this folder. Selected with env CARDS_ARCH in {resnet18 (default), vit, convnext}. The ViT-B/16 and
ConvNeXt-Tiny loaders match main's src/cards/models/backbones.py (_load_celeba_official_train_{vit,convnext}_native);
checkpoints come from trained_models_new/celeba/ (override with CARDS_CKPT_DIR). Run from the repo root.

feats(model, x) returns (pooled (N, d), maps (N, d, 7, 7)), where pooled is exactly the input of head(model):
  resnet18: layer4 maps; pooled = spatial mean.
  vit:      final encoder tokens (after encoder.ln); pooled = CLS token; maps = the 14x14 patch tokens average-pooled
            2x2 to 7x7, so all three models share a 49-patch grid.
  convnext: features.7 maps; pooled = avgpool -> LayerNorm2d -> flatten.
"""
import os
from pathlib import Path

import torch
import torch.nn.functional as Fn
from torch import nn
from torchvision.models import convnext_tiny, resnet18, vit_b_16

ARCH = os.environ.get("CARDS_ARCH", "resnet18")
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
CKPT = {"resnet18": "resnet18_official_train_attractive_male.pt",
        "vit": "vit_b_16_official_train_attractive_male.pt",
        "convnext": "convnext_tiny_official_train_attractive_male.pt"}
if ARCH not in CKPT:
    raise ValueError(f"CARDS_ARCH must be one of {sorted(CKPT)}, got {ARCH!r}")
SUFFIX = {"resnet18": "", "vit": "_vit", "convnext": "_convnext"}[ARCH]  # results-file suffix used by main's pipeline
FEAT_CACHE = Path(os.environ.get("CARDS_FEAT_CACHE", f"results/baselines/feat_cache{SUFFIX}"))


def load(arch: str = ARCH) -> nn.Module:
    if arch == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 4)
    elif arch == "vit":
        m = vit_b_16(weights=None)
        m.heads = nn.Linear(m.hidden_dim, 4)
    else:
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 4)
    m.load_state_dict(torch.load(CKPT_DIR / CKPT[arch], map_location="cpu"))
    return m.eval()


def head(m: nn.Module, arch: str = ARCH) -> nn.Linear:
    return {"resnet18": lambda: m.fc, "vit": lambda: m.heads, "convnext": lambda: m.classifier[2]}[arch]()


def feats(m: nn.Module, x: torch.Tensor, arch: str = ARCH) -> tuple[torch.Tensor, torch.Tensor]:
    if arch == "resnet18":
        f = m.layer4(m.layer3(m.layer2(m.layer1(m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        return f.mean((2, 3)), f
    if arch == "vit":
        t = m.encoder(torch.cat([m.class_token.expand(len(x), -1, -1), m._process_input(x)], 1))
        g = t[:, 1:].transpose(1, 2).reshape(len(x), m.hidden_dim, 14, 14)
        return t[:, 0], Fn.avg_pool2d(g, 2)
    f = m.features(x)
    return m.classifier[1](m.classifier[0](m.avgpool(f))), f
