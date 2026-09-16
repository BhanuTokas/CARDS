"""Resolution-mismatch follow-up to train_attractive_young_classifier.py
(notes/celeba_correlation_investigation.md), prompted directly ("Can we
train a classifier on the low resolution CelebA data" -> "My bad I meant
on the low resolution, original CelebA data" -> "Can we not simply
resize the images in validation set of CelebA-HQ and use that for
ground truth?").

Identical to the original in every respect -- same 30,000 CelebA-HQ
images, same 85/15 split (`split_celebamask_hq`, same seed/stratum),
same architecture/hyperparameters/early-stopping -- EXCEPT both
transforms now degrade to standard (non-HQ) CelebA's own native
img_align_celeba resolution (178x218) BEFORE the final 224x224 resize.
This permanently destroys the fine detail CelebA-HQ's own super-
resolution pipeline added back to its 1024x1024 crops, so the trained
classifier only ever sees information a genuinely low-resolution photo
would carry -- the same degrade step cards.models.backbones.
_celeba_lowres_preprocess applies at inference time, so training and
eval stay consistent.

Rationale: evaluating the ORIGINAL (HQ-resolution-trained) classifier on
standard CelebA's own low-res official val images showed a real, not
fully explained, drop in ground-truth correlation (Attractive rho 0.615
-> 0.472, sign 84.6% -> 65.4% n.s.) relative to CelebA-HQ's own val
pool. This checkpoint isolates whether that drop is a genuine train/eval
resolution mismatch (this classifier, trained AND evaluated at low
resolution throughout, should recover most of the gap if so) or
something else entirely (if the gap persists here too).

Saves to trained_models_new/celeba/resnet18_attractive_young_lowres.pt
-- a SEPARATE checkpoint from the original, which is left untouched.
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
# Standard (non-HQ) CelebA's own native img_align_celeba resolution
# (width, height) -- matches cards.models.backbones._STANDARD_CELEBA_SIZE,
# duplicated here (not imported) since this script predates that module
# addition and stays a standalone, directly-runnable script like every
# other scripts/celeba/build/*.py file.
STANDARD_CELEBA_SIZE = (178, 218)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class AttractiveYoungDataset(Dataset):
    def __init__(self, image_paths: list[Path], labels: np.ndarray, transform):
        self.image_paths = image_paths
        self.labels = labels  # (N, 2) bool: [:, 0]=Attractive, [:, 1]=Young
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # (2,) 0/1
        return x, y


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 4)  # [0:2]=Attractive, [2:4]=Young
    return model


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ce: nn.Module) -> torch.Tensor:
    loss_attractive = ce(logits[:, 0:2], targets[:, 0])
    loss_young = ce(logits[:, 2:4], targets[:, 1])
    return loss_attractive + loss_young


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    correct = np.zeros(2)
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        pred_attractive = logits[:, 0:2].argmax(dim=1)
        pred_young = logits[:, 2:4].argmax(dim=1)
        correct[0] += (pred_attractive == y[:, 0]).sum().item()
        correct[1] += (pred_young == y[:, 1]).sum().item()
        total += x.shape[0]
    return {"attractive_acc": correct[0] / total, "young_acc": correct[1] / total, "n": total}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CelebAMask-HQ image paths + attribute labels...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]  # [Attractive_idx, Young_idx]

    indices = sorted(image_paths_by_idx)
    image_paths = [image_paths_by_idx[i] for i in indices]
    labels = np.array([attr_labels_by_file[f"{i}.jpg"][target_indices] for i in indices])  # (30000, 2)
    print(f"{len(image_paths)} images, label shape {labels.shape}", flush=True)
    print(f"Attractive positive rate: {labels[:, 0].mean():.3f}  Young positive rate: {labels[:, 1].mean():.3f}", flush=True)

    # SAME split as the original checkpoint -- same seed/stratum/
    # val_fraction -- so the held-out val boundary (and every existing
    # real-mask ground truth pairing) stays valid; only the transforms
    # below differ.
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices, seed=SEED, val_fraction=VAL_FRACTION)
    hq_to_pos = {hq_idx: pos for pos, hq_idx in enumerate(indices)}
    train_idx = np.array([hq_to_pos[i] for i in train_hq])
    val_idx = np.array([hq_to_pos[i] for i in val_hq])
    print(f"train: {len(train_idx)}  val: {len(val_idx)}", flush=True)

    degrade_h, degrade_w = STANDARD_CELEBA_SIZE[1], STANDARD_CELEBA_SIZE[0]
    train_transform = transforms.Compose([
        transforms.Resize((degrade_h, degrade_w)),  # degrade to standard CelebA's native res FIRST
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((degrade_h, degrade_w)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = AttractiveYoungDataset([image_paths[i] for i in train_idx], labels[train_idx], train_transform)
    val_ds = AttractiveYoungDataset([image_paths[i] for i in val_idx], labels[val_idx], val_transform)
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

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    ckpt_path = CKPT_DIR / "resnet18_attractive_young_lowres.pt"

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
              f"val_attractive_acc={val_metrics['attractive_acc']:.4f}  val_young_acc={val_metrics['young_acc']:.4f}",
              flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            torch.save(best_state, ckpt_path)  # persist on every improvement, see original script's own note
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
    print(f"train: attractive_acc={train_metrics['attractive_acc']:.4f}  young_acc={train_metrics['young_acc']:.4f}", flush=True)
    print(f"val:   attractive_acc={final_metrics['attractive_acc']:.4f}  young_acc={final_metrics['young_acc']:.4f}", flush=True)
    gate_pass = final_metrics["attractive_acc"] > 0.70 and final_metrics["young_acc"] > 0.70
    print(f"go/no-go gate (>70% val on both tasks): {'PASS' if gate_pass else 'FAIL'}", flush=True)

    run_info = {
        "train_attractive_acc": train_metrics["attractive_acc"], "train_young_acc": train_metrics["young_acc"],
        "val_attractive_acc": final_metrics["attractive_acc"], "val_young_acc": final_metrics["young_acc"],
        "best_val_loss": best_val_loss, "n_train": len(train_idx), "n_val": len(val_idx),
        "gate_pass": gate_pass, "seed": SEED,
    }
    with open(CKPT_DIR / "run_info_lowres.pkl", "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved run_info to {CKPT_DIR / 'run_info_lowres.pkl'}", flush=True)


if __name__ == "__main__":
    main()
