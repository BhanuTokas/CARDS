"""Noisy-shortcut variant of `train_attractive_shortcut_classifiers_
official_train.py` -- prompted directly ("current Shortcut experiment
is not able to cause the model to actually rely on the Shortcut...if we
added noise to the image before adding the shortcut, would that be
helpful?" -> "I was thinking pixel noise" -> "I would like atleast 20%
drop in clean test vs shortcut test accuracy").

**Why this doesn't introduce a second exploitable shortcut** (confirmed
directly against the original script's own `ShortcutDataset`, not
assumed): `inject_flags` is drawn at the SAME rate independently within
EACH class (`rng.random(len(filenames)) < rate` over the whole set,
color chosen per-image from the true label) -- so shortcut/noise
PRESENCE is class-balanced (P(noise|positive) == P(noise|negative) ==
rate), only the COLOR carries the class signal. Adding pixel noise only
to the stamped subset can't hand the model an alternative class-
predictive cue; it just degrades the real face signal specifically
where the perfectly-predictive color patch is available, which should
raise the marginal incentive to use that patch.

**Additive Gaussian pixel noise**, applied to the PIL image (uint8,
0-255) BEFORE the shortcut rectangle is stamped -- "noise to the image,
THEN shortcut on top" per direct instruction, so the 10x10 color patch
itself always stays a clean, noise-free solid color (a noisy patch
would blur the color signal itself, confounding the manipulation this
script is testing). `NOISE_STD` in 0-255 pixel units, per-pixel i.i.d.,
independent across channels, clipped to [0,255].

**Pilot scope, decided directly**: rate=80% only for this first pass
(the stronger of the two requested rates, "70/80%"), Attractive only,
sweeping NOISE_STD_VALUES to find one clearing the >=20-point gap
target (`same_rate_val_acc - clean_val_acc`) -- rate=70% and additional
noise levels are a cheap follow-up once a working noise level is found,
not swept blindly upfront.
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
TARGET_TASK = "Attractive"
NUM_WORKERS = int(os.environ.get("CARDS_NUM_WORKERS", "4"))

RATES = [0.8]  # pilot: rate=80% only, per direct instruction
NOISE_STD_VALUES = [25.0, 50.0, 100.0]  # 0-255 pixel scale, to sweep
SHORTCUT_SIZE = 10
SHORTCUT_MARGIN = 4
SHORTCUT_COLOR_POSITIVE = (255, 0, 255)  # magenta, target=1
SHORTCUT_COLOR_NEGATIVE = (0, 255, 255)  # cyan, target=0
GAP_TARGET = 0.20

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


def add_pixel_noise(image: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    arr = np.asarray(image, dtype=np.float32)
    noise = rng.normal(0.0, std, size=arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def inject_shortcut(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [SHORTCUT_MARGIN, SHORTCUT_MARGIN, SHORTCUT_MARGIN + SHORTCUT_SIZE - 1, SHORTCUT_MARGIN + SHORTCUT_SIZE - 1],
        fill=color,
    )
    return image


class NoisyShortcutDataset(Dataset):
    """SAME `inject_flags` design as the original ShortcutDataset (rate
    applied independently per class, seeded, reproducible) -- the only
    addition is `noise_std`: when > 0, Gaussian pixel noise is stamped
    onto the SAME inject_flags==True subset, BEFORE the shortcut
    rectangle, using a per-image RNG seeded off the base seed + index
    (deterministic, not re-rolled per epoch)."""

    def __init__(self, filenames: list[str], labels: np.ndarray, rate: float, noise_std: float, seed: int, augment: bool):
        self.filenames = filenames
        self.labels = labels
        self.noise_std = noise_std
        self.seed = seed
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
            if self.noise_std > 0:
                img_rng = np.random.default_rng(self.seed + 1_000_000 + idx)
                image = add_pixel_noise(image, self.noise_std, img_rng)  # noise BEFORE the shortcut patch
            color = SHORTCUT_COLOR_POSITIVE if self.labels[idx] == 1 else SHORTCUT_COLOR_NEGATIVE
            image = inject_shortcut(image, color)  # clean, noise-free color patch stamped on top
        x = self.post_transform(image)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def build_model() -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
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


def train_one_config(rate: float, noise_std: float, train_files, train_labels, val_files, val_labels) -> dict:
    rate_pct = round(rate * 100)
    noise_tag = int(noise_std)
    ckpt_path = CKPT_DIR / f"resnet18_official_train_attractive_shortcut_noisy{noise_tag}_{rate_pct}.pt"
    info_path = CKPT_DIR / f"run_info_official_train_attractive_shortcut_noisy{noise_tag}_{rate_pct}.pkl"
    if ckpt_path.exists() and info_path.exists():
        print(f"\n=== rate={rate_pct}% noise_std={noise_std} already trained, skipping ===", flush=True)
        with open(info_path, "rb") as f:
            return pickle.load(f)

    print(f"\n=== training rate={rate_pct}% noise_std={noise_std} ===", flush=True)

    train_ds = NoisyShortcutDataset(train_files, train_labels, rate, noise_std, seed=SEED, augment=True)
    val_ds = NoisyShortcutDataset(val_files, val_labels, rate, noise_std, seed=SEED + 1, augment=False)
    clean_val_ds = NoisyShortcutDataset(val_files, val_labels, rate=0.0, noise_std=0.0, seed=0, augment=False)
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
                print(f"  [rate={rate_pct}% noise={noise_std} epoch {epoch}] batch [{batch_idx + 1}/{n_batches}]", flush=True)
        train_loss = train_loss_sum / len(train_ds)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss_sum += ce(model(x), y).item() * x.shape[0]
        val_loss = val_loss_sum / len(val_ds)
        val_acc = evaluate(model, val_loader)

        print(f"[rate={rate_pct}% noise={noise_std} epoch {epoch:>2d}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}", flush=True)

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
    gap = same_rate_acc - clean_acc
    hit_target = "YES" if gap >= GAP_TARGET else "no"
    print(f"\n=== rate={rate_pct}% noise_std={noise_std} final: same-rate val_acc={same_rate_acc:.4f}  "
          f"clean val_acc={clean_acc:.4f}  gap={gap:+.4f}  hit_20pt_target={hit_target} ===", flush=True)

    run_info = {
        "rate": rate, "rate_pct": rate_pct, "noise_std": noise_std,
        "same_rate_val_acc": same_rate_acc, "clean_val_acc": clean_acc, "gap": gap,
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
        for noise_std in NOISE_STD_VALUES:
            all_run_info[(rate, noise_std)] = train_one_config(rate, noise_std, train_files, train_labels, val_files, val_labels)

    print("\n=== pilot summary ===", flush=True)
    for rate in RATES:
        for noise_std in NOISE_STD_VALUES:
            info = all_run_info[(rate, noise_std)]
            hit = "YES" if info["gap"] >= GAP_TARGET else "no"
            print(f"  rate={info['rate_pct']:>3d}%  noise_std={noise_std:>5.0f}  "
                  f"same-rate_acc={info['same_rate_val_acc']:.4f}  clean_acc={info['clean_val_acc']:.4f}  "
                  f"gap={info['gap']:+.4f}  hit_20pt_target={hit}", flush=True)


if __name__ == "__main__":
    main()
