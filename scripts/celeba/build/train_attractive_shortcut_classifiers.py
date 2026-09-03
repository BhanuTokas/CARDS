"""Shortcut-learning validation experiment for the masking hybrid,
prompted directly ("The idea is to introduce shortcuts to images ... we
train models on datasets with 0%, 33%, 67% and 100% shortcuts in the
training dataset. Then we evaluate the predicted attribution of the
top-5 concept for each model and expect to see a steady decline.").

Two DISTINCT shortcuts, one per class of a single binary target
(Attractive -- not the joint 2-task head every other CelebA checkpoint
in this track uses, deliberately simpler here since the shortcut only
needs to correlate with one label): a 10x10px solid-color block, fixed
top-left corner (4px margin) -- magenta (255,0,255) for Attractive=1,
cyan (0,255,255) for Attractive=0. "Strength" X% (0/33/67/100) means: X%
of Attractive=1 train images get the magenta block, X% of Attractive=0
images get the cyan block, independently per class; the remaining
(100-X)% of each class is untouched. At X=100%, the shortcut alone
perfectly predicts the label; at X=0%, no shortcut exists (clean
baseline).

Injection happens AFTER resize/flip/color-jitter, directly on the final
224x224 frame, immediately before ToTensor/Normalize -- NOT earlier in
the pipeline. Two concrete correctness reasons: RandomHorizontalFlip
would otherwise sometimes relocate a pre-flip stamp to the wrong corner
(breaking the "always exactly top-left" cue), and ColorJitter would
distort the shortcut's own exact color per-image (weakening how
reliable/clean the cue is at each strength level, muddying the
strength-vs-reliance relationship the whole experiment depends on).

Same 30,000 CelebA-HQ images, same 85/15 split (`split_celebamask_hq`,
same seed) as every other classifier in this track -- injection is a
per-sample dataset-level transform, not a change to which photos are
used or how they're split. Trains all 4 rates in one script (same
architecture/hyperparameters as train_attractive_young_classifier.py,
just a single 2-way head instead of the 2-task 4-way one, since this
experiment only needs one binary target).

Val accuracy is reported BOTH on the same-rate-injected val split (used
for early stopping, matching train's own distribution) AND on a clean
(rate=0) version of the SAME val images (a free diagnostic -- if the
model leans on the shortcut, a growing gap between these two numbers as
rate increases is a direct, independent signal of shortcut reliance,
before any concept-attribution evaluation even runs).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import TARGET_CLASSES, load_attribute_labels, load_attribute_names

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
CKPT_DIR = Path("trained_models_new/celeba")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_FRACTION = 0.15
BATCH_SIZE = 64
MAX_EPOCHS = 15
PATIENCE = 3
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
IMG_SIZE = 224
TARGET_TASK = "Attractive"

RATES = [0.0, 0.33, 0.67, 1.0]  # 0%, 33%, 67%, 100%
SHORTCUT_SIZE = 10
SHORTCUT_MARGIN = 4
SHORTCUT_COLOR_POSITIVE = (255, 0, 255)  # magenta, Attractive=1
SHORTCUT_COLOR_NEGATIVE = (0, 255, 255)  # cyan, Attractive=0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def inject_shortcut(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [SHORTCUT_MARGIN, SHORTCUT_MARGIN, SHORTCUT_MARGIN + SHORTCUT_SIZE - 1, SHORTCUT_MARGIN + SHORTCUT_SIZE - 1],
        fill=color,
    )
    return image


class ShortcutDataset(Dataset):
    """`rate` independently controls what fraction of EACH class gets its
    own class-specific shortcut stamped on -- decided once per image at
    construction time (seeded, reproducible), not re-rolled per epoch."""

    def __init__(self, image_paths: list[Path], labels: np.ndarray, rate: float, seed: int, augment: bool):
        self.image_paths = image_paths
        self.labels = labels  # (N,) 0/1, Attractive only
        rng = np.random.default_rng(seed)
        self.inject_flags = rng.random(len(image_paths)) < rate

        geometry: list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            geometry += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        self.pre_transform = transforms.Compose(geometry)
        self.post_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.pre_transform(image)  # resize (+flip/jitter) FIRST
        if self.inject_flags[idx]:
            color = SHORTCUT_COLOR_POSITIVE if self.labels[idx] == 1 else SHORTCUT_COLOR_NEGATIVE
            image = inject_shortcut(image, color)  # shortcut stamped on the FINAL frame, unjittered
        x = self.post_transform(image)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # single binary target, not the 2-task 4-way head
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.shape[0]
    return correct / total


def train_one_rate(rate: float, image_paths, labels, train_idx, val_idx) -> dict:
    rate_pct = round(rate * 100)
    print(f"\n=== training rate={rate_pct}% ===", flush=True)

    train_ds = ShortcutDataset([image_paths[i] for i in train_idx], labels[train_idx], rate, seed=SEED, augment=True)
    val_ds = ShortcutDataset([image_paths[i] for i in val_idx], labels[val_idx], rate, seed=SEED + 1, augment=False)
    clean_val_ds = ShortcutDataset([image_paths[i] for i in val_idx], labels[val_idx], rate=0.0, seed=0, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    clean_val_loader = DataLoader(clean_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(DEVICE)
    ce = nn.CrossEntropyLoss()
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ])

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    ckpt_path = CKPT_DIR / f"resnet18_attractive_shortcut_{rate_pct}.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.shape[0]
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss_sum += ce(model(x), y).item() * x.shape[0]
        val_loss = val_loss_sum / len(val_ds)
        val_acc = evaluate(model, val_loader)

        print(f"[rate={rate_pct}% epoch {epoch:>2d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(best_state, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.", flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)

    same_rate_acc = evaluate(model, val_loader)
    clean_acc = evaluate(model, clean_val_loader)
    print(f"\n=== rate={rate_pct}% final: same-rate val_acc={same_rate_acc:.4f}  clean val_acc={clean_acc:.4f}  "
          f"gap={same_rate_acc - clean_acc:+.4f} ===", flush=True)

    run_info = {
        "rate": rate, "rate_pct": rate_pct, "same_rate_val_acc": same_rate_acc, "clean_val_acc": clean_acc,
        "best_val_loss": best_val_loss, "n_train": len(train_idx), "n_val": len(val_idx), "seed": SEED,
    }
    with open(CKPT_DIR / f"run_info_shortcut_{rate_pct}.pkl", "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)
    return run_info


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CelebAMask-HQ image paths + attribute labels...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_idx = attr_names.index(TARGET_TASK)

    indices = sorted(image_paths_by_idx)
    image_paths = [image_paths_by_idx[i] for i in indices]
    labels = np.array([attr_labels_by_file[f"{i}.jpg"][target_idx] for i in indices])
    print(f"{len(image_paths)} images. {TARGET_TASK} positive rate: {labels.mean():.3f}", flush=True)

    # SAME split as every other checkpoint in this track (same seed AND
    # same joint-(Attractive,Young) stratification -- split_celebamask_hq
    # is hardcoded for exactly 2 target indices, `labels[:, 0]`/`labels[:,
    # 1]`, so it must be called with TARGET_CLASSES, not just [target_idx],
    # even though only Attractive's own label is used for training below).
    split_target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, split_target_indices, seed=SEED, val_fraction=VAL_FRACTION)
    hq_to_pos = {hq_idx: pos for pos, hq_idx in enumerate(indices)}
    train_idx = np.array([hq_to_pos[i] for i in train_hq])
    val_idx = np.array([hq_to_pos[i] for i in val_hq])
    print(f"train: {len(train_idx)}  val: {len(val_idx)}", flush=True)

    all_run_info = {}
    for rate in RATES:
        all_run_info[rate] = train_one_rate(rate, image_paths, labels, train_idx, val_idx)

    print("\n=== summary ===", flush=True)
    for rate in RATES:
        info = all_run_info[rate]
        print(f"  rate={info['rate_pct']:>3d}%  same-rate_acc={info['same_rate_val_acc']:.4f}  "
              f"clean_acc={info['clean_val_acc']:.4f}  gap={info['same_rate_val_acc'] - info['clean_val_acc']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
