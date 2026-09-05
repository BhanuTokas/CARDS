"""Phase 1 of the CelebA plan: trains this track's own black-box model
from scratch (no off-the-shelf pretrained CelebA classifier exists
anywhere -- confirmed via two independent web-research passes during
planning). A ResNet18 fine-tuned as TWO independent binary classification
tasks (Attractive, Young) on CelebAMask-HQ's own 30,000 images.

Output head shape is deliberately 2 separate 2-way softmax blocks
(logits[0:2] = not-Attractive/Attractive, logits[2:4] = not-Young/Young),
not raw sigmoids -- lets each task present as a clean (N,2)-logit
`MultiClassModel` later (cards.validation.broden_faithfulness's own
protocol), so Phase 4's masking-based faithfulness ground truth can call
`compute_faithfulness` completely unmodified per task, exactly like
CUB's own `Resnet18CubAdapter` pattern.

Split is a fresh 85/15 stratified split of the 30K CelebAMask-HQ images
computed directly from their own attribute labels, jointly stratified on
(Attractive, Young) so class balance stays consistent between train/val
-- NOT inherited from standard CelebA's own 202K-scale train/val/test
partition file, which wasn't designed for this 30K HQ subset.

This is the one genuine go/no-go gate in the whole CelebA plan: unlike
every other track in this investigation, there is no fallback pretrained
checkpoint if this doesn't reach a usable accuracy (>70% val on both
tasks, per the plan's own reasoned threshold -- majority-class baselines
are ~51% Attractive / ~78% Young, so anything near those numbers is a
real failure, not noise).
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

    # split_celebamask_hq is the canonical split -- also used by
    # cards.data.datasets.load_celeba, so the retrieval pool / faithfulness
    # ground truth later draw only from images this classifier never
    # trained on. Re-expressed here as positions into `indices` (this
    # function's own local ordering) rather than raw hq indices.
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices, seed=SEED, val_fraction=VAL_FRACTION)
    hq_to_pos = {hq_idx: pos for pos, hq_idx in enumerate(indices)}
    train_idx = np.array([hq_to_pos[i] for i in train_hq])
    val_idx = np.array([hq_to_pos[i] for i in val_hq])
    print(f"train: {len(train_idx)}  val: {len(val_idx)}", flush=True)

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
                logits = model(x)
                val_loss_sum += compute_loss(logits, y, ce).item() * x.shape[0]
        val_loss = val_loss_sum / len(val_ds)
        val_metrics = evaluate(model, val_loader)

        print(f"[epoch {epoch:>2d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_attractive_acc={val_metrics['attractive_acc']:.4f}  val_young_acc={val_metrics['young_acc']:.4f}",
              flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            # Persist to disk on every improvement, not just held in memory
            # until the run finishes -- a crash at any later point (e.g.
            # the final-evaluation flake this replaces) no longer loses a
            # fully-trained model that only ever lived in RAM.
            torch.save(best_state, CKPT_DIR / "resnet18_attractive_young.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs).", flush=True)
                break

    model.load_state_dict(best_state)

    # Already saved on disk per-epoch above (whenever val_loss improved) --
    # this re-save is just a final confirmation, not the first write. A
    # prior run crashed silently (no Python traceback, likely a Windows
    # multiprocessing DataLoader worker-spawn flake) inside a freshly-
    # constructed num_workers=4 DataLoader built for the final train-set
    # evaluation below, losing the entire trained model since it had only
    # ever lived in memory until the very last line -- the per-epoch save
    # above is the actual fix; this line is now just redundant insurance.
    ckpt_path = CKPT_DIR / "resnet18_attractive_young.pt"
    torch.save(best_state, ckpt_path)
    print(f"\nConfirmed checkpoint saved at {ckpt_path}", flush=True)

    final_metrics = evaluate(model, val_loader)
    # num_workers=0 here (not 4, unlike train_loader/val_loader above) --
    # this exact call, with num_workers=4, is what crashed the prior run;
    # the training loop's own loaders ran for 7 full epochs without issue,
    # so the flake is specific to constructing a NEW multi-worker loader
    # this late in the process, not DataLoaders in general.
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
    with open(CKPT_DIR / "run_info.pkl", "wb") as f:
        pickle.dump(run_info, f)
    print(f"Saved run_info to {CKPT_DIR / 'run_info.pkl'}", flush=True)


if __name__ == "__main__":
    main()
