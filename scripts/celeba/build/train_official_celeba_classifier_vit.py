"""ViT-B/16 counterpart of `train_official_celeba_classifier.py` --
SAME data (official CelebA train, 162,770 images), SAME 2-task 4-way-
logit head convention ([0:2]=Attractive, [2:4]=Male), SAME training
recipe (AdamW, backbone/head LR split, early-stop-on-val-loss, >70%
val-acc-both-tasks gate) -- only the backbone architecture changes,
prompted directly ("I want to do the main CelebA experiment with ViT
and one other model as the black box model" -> "Official-train arc").
See `train_official_celeba_classifier_convnext.py` for the other new
architecture.

`model.heads` (torchvision's own Sequential(head: Linear(768,1000)))
is replaced wholesale with a single `nn.Linear(768, 4)` -- ViT's own
`forward()` just calls `self.heads(x)`, so this is a drop-in swap, not
a partial-Sequential edit the way ConvNeXt's classifier needs.
Backbone/head param split keys off the "heads." name prefix (this
architecture's own single classification module), matching the ResNet
script's own "fc." split by direct analogy, not copied verbatim (ViT
has no "fc" attribute).

HPC-portable from the start (CELEBA_ROOT/CARDS_CKPT_DIR/CARDS_RESULTS_DIR/
CARDS_NUM_WORKERS env-var overrides) -- the original ResNet18 script
predates this project's HPC-portability convention and is still
hardcoded; not retrofitted here since it wasn't asked for.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
NUM_WORKERS = int(os.environ.get("CARDS_NUM_WORKERS", "4"))
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_EPOCHS = 15
PATIENCE = 3
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
IMG_SIZE = 224
TASKS = ["Attractive", "Male"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_attr_names(path: Path) -> list[str]:
    lines = Path(path).read_text().splitlines()
    return lines[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        filename, values = parts[0], parts[1:]
        result[filename] = np.array([v == "1" for v in values], dtype=bool)
    return result


def load_partition(path: Path) -> dict[str, str]:
    partition = {}
    with open(path) as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    return partition


class TwoTaskDataset(Dataset):
    def __init__(self, filenames: list[str], labels: np.ndarray, transform):
        self.filenames = filenames
        self.labels = labels  # (N, 2) bool: [:,0]=Attractive [:,1]=Male
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(CELEBA_ROOT / "img_align_celeba" / self.filenames[idx]).convert("RGB")
        x = self.transform(image)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def build_model() -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Linear(model.hidden_dim, 4)  # [0:2]=Attractive [2:4]=Male
    return model


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, ce: nn.Module) -> torch.Tensor:
    return ce(logits[:, 0:2], targets[:, 0]) + ce(logits[:, 2:4], targets[:, 1])


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    correct = np.zeros(2)
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        preds = [logits[:, 0:2].argmax(dim=1), logits[:, 2:4].argmax(dim=1)]
        for i, pred in enumerate(preds):
            correct[i] += (pred == y[:, i]).sum().item()
        total += x.shape[0]
    return {"attractive_acc": correct[0] / total, "male_acc": correct[1] / total, "n": total}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CELEBA_ROOT={CELEBA_ROOT}  CKPT_DIR={CKPT_DIR}  DEVICE={DEVICE}", flush=True)
    print("Loading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    partition = load_partition(CELEBA_ROOT / "list_eval_partition.txt")
    task_indices = [attr_names.index(t) for t in TASKS]

    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    print(f"official train: {len(train_files)}  official val: {len(val_files)}", flush=True)

    train_labels = np.array([attr_labels[f][task_indices] for f in train_files])
    val_labels = np.array([attr_labels[f][task_indices] for f in val_files])
    print(f"train Attractive rate={train_labels[:, 0].mean():.3f} Male rate={train_labels[:, 1].mean():.3f}", flush=True)
    print(f"val   Attractive rate={val_labels[:, 0].mean():.3f} Male rate={val_labels[:, 1].mean():.3f}", flush=True)

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

    train_ds = TwoTaskDataset(train_files, train_labels, train_transform)
    val_ds = TwoTaskDataset(val_files, val_labels, val_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model().to(DEVICE)
    ce = nn.CrossEntropyLoss()
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("heads.")]
    head_params = list(model.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ])

    ckpt_path = CKPT_DIR / "vit_b_16_official_train_attractive_male.pt"
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = compute_loss(logits, y, ce)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.shape[0]
            if (batch_idx + 1) % 500 == 0:
                print(f"  [epoch {epoch}] batch [{batch_idx + 1}/{n_batches}] running_loss={train_loss_sum / ((batch_idx + 1) * BATCH_SIZE):.4f}", flush=True)
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
              f"val_attractive_acc={val_metrics['attractive_acc']:.4f}  val_male_acc={val_metrics['male_acc']:.4f}",
              flush=True)

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
    print("\n=== Final (best val_loss checkpoint) ===", flush=True)
    print(f"val: attractive_acc={final_metrics['attractive_acc']:.4f}  male_acc={final_metrics['male_acc']:.4f}", flush=True)
    gate_pass = final_metrics["attractive_acc"] > 0.70 and final_metrics["male_acc"] > 0.70
    print(f"go/no-go gate (>70% val on both tasks): {'PASS' if gate_pass else 'FAIL'}", flush=True)

    run_info = {
        "val_attractive_acc": final_metrics["attractive_acc"], "val_male_acc": final_metrics["male_acc"],
        "best_val_loss": best_val_loss, "n_train": len(train_files), "n_val": len(val_files),
        "gate_pass": gate_pass, "seed": SEED,
    }
    with open(CKPT_DIR / "run_info_official_train_vit.pkl", "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved run_info to {CKPT_DIR / 'run_info_official_train_vit.pkl'}", flush=True)


if __name__ == "__main__":
    main()
