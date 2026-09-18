"""Male companion to `train_attractive_shortcut_classifiers_official_
train.py` (SAME design, script docstring, and reasoning -- see that
file for the full write-up), prompted directly in the same request
("I needed Male and Attractive. Can you write the scripts and I can
run it on HPC in the interest of compute?"). Only TARGET_TASK and the
output checkpoint/info filenames differ; everything else (shortcut
mechanism, official-train data source, 11-rate 10%-step sweep,
architecture/hyperparameters, HPC-portable env-var overrides) is
identical between the two scripts, by design -- two independent,
single-target (2-way head) experiments, not a joint 2-task shortcut
(keeping each task's own shortcut-reliance diagnostic clean and
unconflated with the other task, same reasoning the original Attractive-
only script gave for not using the project's usual joint head).

Shortcut: 10x10px solid-color block, top-left corner (4px margin) --
magenta (255,0,255) for Male=1, cyan (0,255,255) for Male=0.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CELEBA_ROOT = Path(os.environ.get("CELEBA_ROOT", r"C:\Users\btokas\Projects\Datasets\CelebA\celeba"))
RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MAX_EPOCHS = 15
PATIENCE = 3
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
IMG_SIZE = 224
TARGET_TASK = "Male"
NUM_WORKERS = int(os.environ.get("CARDS_NUM_WORKERS", "4"))

RATES = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]  # 0, 10, 20, ..., 100%
SHORTCUT_SIZE = 10
SHORTCUT_MARGIN = 4
SHORTCUT_COLOR_POSITIVE = (255, 0, 255)  # magenta, target=1
SHORTCUT_COLOR_NEGATIVE = (0, 255, 255)  # cyan, target=0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_attr_names(path: Path) -> list[str]:
    return Path(path).read_text().splitlines()[1].split()


def load_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


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

    def __init__(self, filenames: list[str], labels: np.ndarray, rate: float, seed: int, augment: bool):
        self.filenames = filenames
        self.labels = labels  # (N,) 0/1, TARGET_TASK only
        rng = np.random.default_rng(seed)
        self.inject_flags = rng.random(len(filenames)) < rate

        geometry: list = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            geometry += [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        self.pre_transform = transforms.Compose(geometry)
        self.post_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(CELEBA_ROOT / "img_align_celeba" / self.filenames[idx]).convert("RGB")
        image = self.pre_transform(image)  # resize (+flip/jitter) FIRST
        if self.inject_flags[idx]:
            color = SHORTCUT_COLOR_POSITIVE if self.labels[idx] == 1 else SHORTCUT_COLOR_NEGATIVE
            image = inject_shortcut(image, color)  # shortcut stamped on the FINAL frame, unjittered
        x = self.post_transform(image)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)  # single binary target, not a joint 2-task head
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


def train_one_rate(rate: float, train_files, train_labels, val_files, val_labels) -> dict:
    rate_pct = round(rate * 100)
    ckpt_path = CKPT_DIR / f"resnet18_official_train_male_shortcut_{rate_pct}.pt"
    info_path = CKPT_DIR / f"run_info_official_train_male_shortcut_{rate_pct}.pkl"
    if ckpt_path.exists() and info_path.exists():
        print(f"\n=== rate={rate_pct}% already trained, skipping ===", flush=True)
        with open(info_path, "rb") as f:
            return pickle.load(f)

    print(f"\n=== training rate={rate_pct}% ===", flush=True)

    train_ds = ShortcutDataset(train_files, train_labels, rate, seed=SEED, augment=True)
    val_ds = ShortcutDataset(val_files, val_labels, rate, seed=SEED + 1, augment=False)
    clean_val_ds = ShortcutDataset(val_files, val_labels, rate=0.0, seed=0, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
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

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = len(train_loader)
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x.shape[0]
            if (batch_idx + 1) % 500 == 0:
                print(f"  [rate={rate_pct}% epoch {epoch}] batch [{batch_idx + 1}/{n_batches}]", flush=True)
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
        "best_val_loss": best_val_loss, "n_train": len(train_files), "n_val": len(val_files), "seed": SEED,
    }
    with open(info_path, "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved checkpoint to {ckpt_path}", flush=True)
    return run_info


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CELEBA_ROOT={CELEBA_ROOT}  CKPT_DIR={CKPT_DIR}  DEVICE={DEVICE}", flush=True)
    print("Loading official CelebA metadata...", flush=True)
    attr_names = load_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    target_idx = attr_names.index(TARGET_TASK)
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part

    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    train_labels = np.array([attr_labels[f][target_idx] for f in train_files])
    val_labels = np.array([attr_labels[f][target_idx] for f in val_files])
    print(f"train: {len(train_files)}  val: {len(val_files)}  "
          f"{TARGET_TASK} positive rate: train={train_labels.mean():.3f} val={val_labels.mean():.3f}", flush=True)

    all_run_info = {}
    for rate in RATES:
        all_run_info[rate] = train_one_rate(rate, train_files, train_labels, val_files, val_labels)

    print("\n=== summary ===", flush=True)
    for rate in RATES:
        info = all_run_info[rate]
        print(f"  rate={info['rate_pct']:>3d}%  same-rate_acc={info['same_rate_val_acc']:.4f}  "
              f"clean_acc={info['clean_val_acc']:.4f}  gap={info['same_rate_val_acc'] - info['clean_val_acc']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
