"""Extends Phase 1's own classifier (train_attractive_young_classifier.py)
with a 3rd task, Male, prompted directly ("Can we extend to add Male to
our analysis?" -> "Extend existing checkpoint" -- retrain the SAME
ResNet18 architecture/data/split with a 3rd task head, not a separate
model).

Deliberately saved to a NEW checkpoint path
(resnet18_attractive_young_male.pt), NOT overwriting the original
resnet18_attractive_young.pt: retraining produces genuinely different
weights even at identical architecture/data, and the original
checkpoint underlies every historical result in this entire CelebA
track (v1-v115g) -- overwriting it would silently invalidate all of
that. This script's own Attractive/Young accuracy numbers are a fresh,
independent measurement, not assumed identical to the original run's.

Critically, the 85/15 TRAIN/VAL SPLIT itself is reproduced bit-for-bit
via `split_celebamask_hq` called with the SAME 2-class target_indices
(Attractive, Young only, NOT Male) as the original -- this keeps every
image's train/val membership identical to the original run, so the
existing ground-truth generation scripts (which assume a specific
held-out val set) stay valid for Male too without needing their own
Male-specific split logic. Male's own labels are attached as a 3rd
column for the LOSS/accuracy computation only, never fed into the
stratification key.

Same go/no-go gate as the original (>70% val accuracy), now required
on ALL THREE tasks.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)

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
TASKS = ["Attractive", "Young", "Male"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ThreeTaskDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: np.ndarray, transform):
        self.image_paths = image_paths
        self.labels = labels  # (N, 3) bool: [:,0]=Attractive [:,1]=Young [:,2]=Male
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # (3,) 0/1
        return x, y


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 6)  # [0:2]=Attractive [2:4]=Young [4:6]=Male
    return model


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ce: nn.Module) -> torch.Tensor:
    return (ce(logits[:, 0:2], targets[:, 0]) + ce(logits[:, 2:4], targets[:, 1])
            + ce(logits[:, 4:6], targets[:, 2]))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    correct = np.zeros(3)
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = [logits[:, 0:2].argmax(dim=1), logits[:, 2:4].argmax(dim=1), logits[:, 4:6].argmax(dim=1)]
        for i, pred in enumerate(preds):
            correct[i] += (pred == y[:, i]).sum().item()
        total += x.shape[0]
    return {"attractive_acc": correct[0] / total, "young_acc": correct[1] / total,
            "male_acc": correct[2] / total, "n": total}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CelebAMask-HQ image paths + attribute labels...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    task_indices = [attr_names.index(t) for t in TASKS]  # [Attractive, Young, Male]
    split_indices = [attr_names.index(t) for t in TARGET_CLASSES]  # [Attractive, Young] ONLY -- split reproducibility

    indices = sorted(image_paths_by_idx)
    image_paths = [image_paths_by_idx[i] for i in indices]
    labels = np.array([attr_labels_by_file[f"{i}.jpg"][task_indices] for i in indices])  # (30000, 3)
    print(f"{len(image_paths)} images, label shape {labels.shape}", flush=True)
    print(f"Attractive rate: {labels[:, 0].mean():.3f}  Young rate: {labels[:, 1].mean():.3f}  "
          f"Male rate: {labels[:, 2].mean():.3f}", flush=True)

    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, split_indices, seed=SEED, val_fraction=VAL_FRACTION)
    hq_to_pos = {hq_idx: pos for pos, hq_idx in enumerate(indices)}
    train_idx = np.array([hq_to_pos[i] for i in train_hq])
    val_idx = np.array([hq_to_pos[i] for i in val_hq])
    print(f"train: {len(train_idx)}  val: {len(val_idx)}  (SAME split as the original 2-task classifier)", flush=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = ThreeTaskDataset([image_paths[i] for i in train_idx], labels[train_idx], train_transform)
    val_ds = ThreeTaskDataset([image_paths[i] for i in val_idx], labels[val_idx], val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(DEVICE)
    ce = nn.CrossEntropyLoss()
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ])

    ckpt_path = CKPT_DIR / "resnet18_attractive_young_male.pt"
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y, ce)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.shape[0]
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss_sum += compute_loss(model(x), y, ce).item() * x.shape[0]
        val_loss = val_loss_sum / len(val_ds)
        val_metrics = evaluate(model, val_loader)

        print(f"[epoch {epoch:>2d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_attractive_acc={val_metrics['attractive_acc']:.4f}  val_young_acc={val_metrics['young_acc']:.4f}  "
              f"val_male_acc={val_metrics['male_acc']:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(best_state, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).", flush=True)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    print(f"\nConfirmed checkpoint saved at {ckpt_path}", flush=True)

    final_metrics = evaluate(model, val_loader)
    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    train_metrics = evaluate(model, train_eval_loader)

    print("\n=== Final (best val_loss checkpoint) ===", flush=True)
    print(f"train: attractive_acc={train_metrics['attractive_acc']:.4f}  young_acc={train_metrics['young_acc']:.4f}  "
          f"male_acc={train_metrics['male_acc']:.4f}", flush=True)
    print(f"val:   attractive_acc={final_metrics['attractive_acc']:.4f}  young_acc={final_metrics['young_acc']:.4f}  "
          f"male_acc={final_metrics['male_acc']:.4f}", flush=True)
    gate_pass = (final_metrics["attractive_acc"] > 0.70 and final_metrics["young_acc"] > 0.70
                 and final_metrics["male_acc"] > 0.70)
    print(f"go/no-go gate (>70% val on all THREE tasks): {'PASS' if gate_pass else 'FAIL'}", flush=True)

    run_info = {
        "train_attractive_acc": train_metrics["attractive_acc"], "train_young_acc": train_metrics["young_acc"],
        "train_male_acc": train_metrics["male_acc"],
        "val_attractive_acc": final_metrics["attractive_acc"], "val_young_acc": final_metrics["young_acc"],
        "val_male_acc": final_metrics["male_acc"],
        "best_val_loss": best_val_loss, "n_train": len(train_idx), "n_val": len(val_idx),
        "gate_pass": gate_pass, "seed": SEED,
    }
    with open(CKPT_DIR / "run_info_male.pkl", "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved run_info to {CKPT_DIR / 'run_info_male.pkl'}", flush=True)


if __name__ == "__main__":
    main()
