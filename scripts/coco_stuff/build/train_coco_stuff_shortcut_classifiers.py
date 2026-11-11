"""Train 11 ResNet18 classifiers on COCO-Stuff with shortcut injection rates 0-100%.

80/20 split of COCO train2017 (fixed seed). For each rate in {0,10,...,100}%,
trains a ResNet18 with per-class colored 10x10px blocks injected into training images.

Shortcut design (one block per COCO_STUFF_20_CLASSES):
  20 unique HSV-colored blocks arranged in a 4x5 grid at the top-left corner of
  the 224x224 image. Injected after RandomResizedCrop + RandomHorizontalFlip,
  before ToTensor + Normalize. For each image, each present class independently
  gets its block with probability `rate`.

Accuracy gap (per model):
  Evaluate on clean test-split (rate=0%) and 100%-shortcut test-split.
  Gap = mean_shortcut_acc - mean_clean_acc (mean binary accuracy across 20 classes).

Outputs (results/coco_stuff/shortcut/):
  shortcut_model_accuracy.csv
  train_test_split.json
Outputs (trained_models_new/coco_stuff/shortcut/):
  resnet18_coco_stuff_shortcut_{rate_pct}.pt   (11 checkpoints)
"""

from __future__ import annotations

import colorsys
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from cards.data.coco_stuff import COCO_STUFF_20_CLASSES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT  = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
OUT_DIR    = Path("results/coco_stuff/shortcut")
CKPT_DIR   = Path("trained_models_new/coco_stuff/shortcut")
SPLIT_FILE = OUT_DIR / "train_test_split.json"

# ---------------------------------------------------------------------------
# Training hyper-params
# ---------------------------------------------------------------------------
SEED        = 42
TEST_FRAC   = 0.20
RATES_PCT   = list(range(0, 101, 10))
BATCH_SIZE  = 64
MAX_EPOCHS  = 50
PATIENCE    = 5
LR          = 1e-4
NUM_WORKERS = 4
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Shortcut design
# ---------------------------------------------------------------------------
N_CLASSES        = len(COCO_STUFF_20_CLASSES)
SHORTCUT_SIZE    = 10
SHORTCUT_MARGIN  = 4
SHORTCUT_SPACING = 14   # 10px block + 4px gap; 4x5 grid -> max x=70, y=56, all < 224

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _hsv_palette(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.9, 0.9)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


SHORTCUT_COLORS: list[tuple[int, int, int]] = _hsv_palette(N_CLASSES)

# 4 rows x 5 cols = 20 non-overlapping positions
SHORTCUT_POSITIONS: list[tuple[int, int]] = [
    (SHORTCUT_MARGIN + col * SHORTCUT_SPACING,
     SHORTCUT_MARGIN + row * SHORTCUT_SPACING)
    for row in range(4) for col in range(5)
]


def inject_shortcuts(image: Image.Image, class_flags: np.ndarray) -> Image.Image:
    """Stamp colored blocks for each class where class_flags[i] is True."""
    image = image.copy()
    draw  = ImageDraw.Draw(image)
    for cls_idx, flag in enumerate(class_flags):
        if not flag:
            continue
        x, y  = SHORTCUT_POSITIONS[cls_idx]
        color = SHORTCUT_COLORS[cls_idx]
        draw.rectangle(
            [x, y, x + SHORTCUT_SIZE - 1, y + SHORTCUT_SIZE - 1],
            fill=color,
        )
    return image


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ShortcutCocoDataset(Dataset):
    """COCO multi-label dataset with per-class shortcut injection.

    inject_flags[i, j] = True means: if class j is present in image i, stamp its block.
    Injection happens after crop/flip (on the final 224x224 PIL image).
    """

    def __init__(
        self,
        samples: list[tuple[Path, np.ndarray]],
        rate: float,
        seed: int,
        augment: bool,
    ) -> None:
        self.samples = samples
        rng = np.random.default_rng(seed)
        self.inject_flags = rng.random((len(samples), N_CLASSES)) < rate

        geo = [transforms.RandomResizedCrop(224)] if augment else [transforms.Resize((224, 224))]
        if augment:
            geo.append(transforms.RandomHorizontalFlip())
        self.pre_transform  = transforms.Compose(geo)
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.pre_transform(image)

        # inject only for present classes (label > 0.5) that are flagged
        class_flags = (label > 0.5) & self.inject_flags[idx]
        if class_flags.any():
            image = inject_shortcuts(image, class_flags)

        return self.post_transform(image), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(coco_root: Path, split: str) -> list[tuple[Path, np.ndarray]]:
    """Load COCO images with >= 1 of the 20 target classes for train2017 or val2017."""
    ann_path = coco_root / "annotations" / f"instances_{split}.json"
    with open(ann_path) as f:
        data = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    cat_id_to_idx  = {
        cat_name_to_id[cls]: idx
        for idx, cls in enumerate(COCO_STUFF_20_CLASSES)
        if cls in cat_name_to_id
    }
    target_cat_ids = set(cat_id_to_idx)

    image_to_classes: dict[int, set[int]] = {}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid not in target_cat_ids:
            continue
        image_to_classes.setdefault(ann["image_id"], set()).add(cat_id_to_idx[cid])

    image_dir = coco_root / split
    samples: list[tuple[Path, np.ndarray]] = []
    for img_info in sorted(data["images"], key=lambda x: x["id"]):
        iid = img_info["id"]
        if iid not in image_to_classes:
            continue
        label = np.zeros(N_CLASSES, dtype=np.float32)
        for cls_idx in image_to_classes[iid]:
            label[cls_idx] = 1.0
        samples.append((image_dir / img_info["file_name"], label))
    return samples


@torch.no_grad()
def _eval_per_class_acc(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """Binary accuracy per class (logit > 0 as prediction)."""
    correct = np.zeros(N_CLASSES, dtype=np.float64)
    total   = 0
    for images, labels in loader:
        logits = model(images.to(DEVICE)).cpu().numpy()
        preds  = (logits > 0).astype(float)
        correct += (preds == labels.numpy()).sum(axis=0)
        total   += images.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading COCO train2017 annotations...", flush=True)
    all_samples = _load_samples(COCO_ROOT, "train2017")
    print(f"Total train2017 images with >= 1 target class: {len(all_samples)}", flush=True)

    print("Loading COCO val2017 annotations...", flush=True)
    val2017_samples = _load_samples(COCO_ROOT, "val2017")
    print(f"Total val2017 images with >= 1 target class: {len(val2017_samples)}", flush=True)

    # 80/20 split of train2017
    rng     = np.random.default_rng(SEED)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    n_test        = int(len(all_samples) * TEST_FRAC)
    test_samples  = [all_samples[i] for i in indices[:n_test]]
    train_samples = [all_samples[i] for i in indices[n_test:]]
    print(f"Train-split: {len(train_samples)}  Test-split: {len(test_samples)}", flush=True)

    with open(SPLIT_FILE, "w") as f:
        json.dump({
            "train": [str(p) for p, _ in train_samples],
            "test":  [str(p) for p, _ in test_samples],
        }, f)
    print(f"Saved split -> {SPLIT_FILE}", flush=True)

    # Fixed test loaders for accuracy gap
    test_clean_loader = DataLoader(
        ShortcutCocoDataset(test_samples, rate=0.0, seed=0, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )
    test_shortcut_loader = DataLoader(
        ShortcutCocoDataset(test_samples, rate=1.0, seed=SEED + 999, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    acc_rows: list[dict] = []

    for rate_pct in RATES_PCT:
        rate = rate_pct / 100.0
        print(f"\n{'=' * 60}", flush=True)
        print(f"Training rate={rate_pct}%", flush=True)

        train_loader = DataLoader(
            ShortcutCocoDataset(train_samples, rate=rate, seed=SEED + rate_pct, augment=True),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True,
        )
        val_loader = DataLoader(
            ShortcutCocoDataset(val2017_samples, rate=rate, seed=SEED + rate_pct + 1000, augment=False),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        )

        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        model = model.to(DEVICE)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        ckpt_path      = CKPT_DIR / f"resnet18_coco_stuff_shortcut_{rate_pct}.pt"
        best_val_loss  = float("inf")
        best_state     = None
        patience_count = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    val_loss += criterion(model(images), labels).item() * images.size(0)
            val_loss /= len(val_loader.dataset)

            print(
                f"  epoch {epoch:>2d}/{MAX_EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
                torch.save(best_state, ckpt_path)
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})", flush=True)
                    break

        model.load_state_dict(best_state)
        model.eval()

        per_class_clean    = _eval_per_class_acc(model, test_clean_loader)
        per_class_shortcut = _eval_per_class_acc(model, test_shortcut_loader)
        mean_clean    = float(per_class_clean.mean())
        mean_shortcut = float(per_class_shortcut.mean())
        mean_gap      = mean_shortcut - mean_clean

        print(
            f"  [rate={rate_pct}%] mean_clean={mean_clean:.4f}  "
            f"mean_shortcut={mean_shortcut:.4f}  gap={mean_gap:+.4f}",
            flush=True,
        )

        row: dict = {
            "rate_pct":         rate_pct,
            "mean_clean_acc":   mean_clean,
            "mean_shortcut_acc": mean_shortcut,
            "mean_gap":         mean_gap,
        }
        for cls_idx, cls_name in enumerate(COCO_STUFF_20_CLASSES):
            row[f"clean_{cls_name}"]    = float(per_class_clean[cls_idx])
            row[f"shortcut_{cls_name}"] = float(per_class_shortcut[cls_idx])
        acc_rows.append(row)

    # Save accuracy CSV
    acc_csv = OUT_DIR / "shortcut_model_accuracy.csv"
    with open(acc_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(acc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(acc_rows)
    print(f"\nSaved -> {acc_csv}", flush=True)

    print("\n=== Accuracy Gap Summary ===", flush=True)
    for row in acc_rows:
        print(
            f"  rate={row['rate_pct']:>3d}%  "
            f"clean={row['mean_clean_acc']:.4f}  "
            f"shortcut={row['mean_shortcut_acc']:.4f}  "
            f"gap={row['mean_gap']:+.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
