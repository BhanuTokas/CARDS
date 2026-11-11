"""Train a multi-label classifier on 20 COCO-Stuff classes.

Supports three architectures via --arch:
  resnet18       (default) -- 512-dim, torchvision ResNet18
  vit_b_16                 -- 768-dim, torchvision ViT-B/16
  convnext_tiny            -- 768-dim, torchvision ConvNeXt-Tiny

All three are initialised from ImageNet pretrained weights, the head is
replaced with nn.Linear(?, 20), and trained with BCE loss (multi-label).
Checkpoint path: trained_models_new/coco_stuff/{arch}_coco_stuff.pt

Usage (on SOL):
    cd /path/to/CARDS
    uv run python scripts/coco_stuff/build/train_coco_stuff_classifier.py
    uv run python scripts/coco_stuff/build/train_coco_stuff_classifier.py --arch vit_b_16
    uv run python scripts/coco_stuff/build/train_coco_stuff_classifier.py --arch convnext_tiny
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import json
import time

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    resnet18,
    vit_b_16,
)

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES

COCO_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS              = 100
EARLY_STOP_PATIENCE = 10
BATCH_SIZE          = 64
LR                  = 1e-4
NUM_WORKERS         = 4

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
])


class CocoStuffDataset(Dataset):
    """COCO multi-label dataset for the 20 co-occurrence-bias classes."""

    def __init__(self, root: Path, split: str, transform):
        ann_path = root / "annotations" / f"instances_{split}2017.json"
        with open(ann_path) as f:
            data = json.load(f)

        cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
        cat_id_to_idx: dict[int, int] = {}
        for idx, cls in enumerate(COCO_STUFF_20_CLASSES):
            cid = cat_name_to_id.get(cls)
            if cid is not None:
                cat_id_to_idx[cid] = idx
        target_cat_ids = set(cat_id_to_idx)

        image_to_classes: dict[int, set[int]] = {}
        for ann in data["annotations"]:
            cid = ann["category_id"]
            if cid not in target_cat_ids:
                continue
            iid = ann["image_id"]
            image_to_classes.setdefault(iid, set()).add(cat_id_to_idx[cid])

        image_dir = root / f"{split}2017"
        self.samples: list[tuple[Path, torch.Tensor]] = []
        for img_info in data["images"]:
            iid = img_info["id"]
            if iid not in image_to_classes:
                continue
            path = image_dir / img_info["file_name"]
            label = torch.zeros(len(COCO_STUFF_20_CLASSES), dtype=torch.float32)
            for cls_idx in image_to_classes[iid]:
                label[cls_idx] = 1.0
            self.samples.append((path, label))

        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def build_model(arch: str) -> nn.Module:
    n_cls = len(COCO_STUFF_20_CLASSES)
    if arch == "resnet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, n_cls)
    elif arch == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, n_cls)
    elif arch == "convnext_tiny":
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, n_cls)
    else:
        raise ValueError(f"Unknown arch: {arch!r}. Choose resnet18, vit_b_16, or convnext_tiny.")
    return model


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", default="resnet18",
        choices=["resnet18", "vit_b_16", "convnext_tiny"],
        help="Backbone architecture to train (default: resnet18)",
    )
    args = parser.parse_args()

    ckpt_path = Path(f"trained_models_new/coco_stuff/{args.arch}_coco_stuff.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"arch={args.arch}  checkpoint={ckpt_path}", flush=True)

    train_ds = CocoStuffDataset(COCO_ROOT, "train", TRAIN_TRANSFORM)
    val_ds   = CocoStuffDataset(COCO_ROOT, "val",   VAL_TRANSFORM)
    print(f"train: {len(train_ds)} images  val: {len(val_ds)} images", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model     = build_model(args.arch).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss    = float("inf")
    patience_counter = 0
    for epoch in range(1, EPOCHS + 1):
        t0         = time.time()
        train_loss = run_epoch(model, train_loader, criterion, optimizer, DEVICE, train=True)
        val_loss   = run_epoch(model, val_loader,   criterion, optimizer, DEVICE, train=False)
        elapsed    = time.time() - t0
        print(
            f"epoch {epoch:>2d}/{EPOCHS}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  ({elapsed:.1f}s)", flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved checkpoint to {ckpt_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {EARLY_STOP_PATIENCE} epochs).",
                    flush=True,
                )
                break

    print(f"Done. Best val_loss={best_val_loss:.4f}. Checkpoint at {ckpt_path}")


if __name__ == "__main__":
    main()
