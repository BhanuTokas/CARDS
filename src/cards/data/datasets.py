"""Dataset loaders: CIFAR-10/100, CUB, CCE MetaDataset, Broden.

See Section 5 of the design doc for the role each dataset plays.

load_cifar / load_cub / load_metadataset all return a flat list of
(image_path, label) pairs -- a generic pool for cards.retrieval.pool.
CandidatePool to encode and retrieve from.

Broden is different: the local copy at ../Datasets/broden_concepts is
already split per-concept into ground-truth positives/negatives (from prior
NetDissect-style processing -- image-level labels already resolved, not raw
segmentation masks needing a coverage-threshold conversion). load_broden
therefore returns ground truth for a single concept directly, for the
retrieval-purity validation check (design doc Section 2, item 5), rather
than a pool to retrieve from.
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_CIFAR_VARIANTS = {"cifar10": CIFAR10, "cifar100": CIFAR100}


def load_cifar(
    root: Path,
    variant: str = "cifar10",
    split: str = "val",
) -> list[tuple[Path, int]]:
    """Materializes torchvision's CIFAR-10/100 (binary batches, no native
    image files) to `root/<split>/<class_name>/<index>.png` on first call,
    then returns (path, label) pairs read back from that materialized copy.
    CIFAR has no official validation split: `split="val"` maps to the
    10k-image test partition, `split="train"` to the 50k-image training
    partition. Labels are re-derived from the alphabetically sorted
    materialized class directories, independent of CIFAR's internal class
    ordering, so they stay consistent between separate train/val calls.
    """
    if variant not in _CIFAR_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_CIFAR_VARIANTS)}, got {variant!r}")
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    root = Path(root)
    materialized_dir = root / split

    if not materialized_dir.exists():
        dataset_cls = _CIFAR_VARIANTS[variant]
        dataset = dataset_cls(root=str(root / "_raw"), train=(split == "train"), download=True)
        for index, (image, label) in enumerate(dataset):
            class_dir = materialized_dir / dataset.classes[label]
            class_dir.mkdir(parents=True, exist_ok=True)
            image.save(class_dir / f"{index}.png")

    class_dirs = sorted((p for p in materialized_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return [
        (path, label)
        for label, class_dir in enumerate(class_dirs)
        for path in sorted(class_dir.glob("*.png"))
    ]


def load_cub(root: Path, split: str = "val") -> list[tuple[Path, int]]:
    """Standard CUB-200-2011 layout: images.txt maps image id -> relative
    path, image_class_labels.txt maps id -> 1-indexed class id,
    train_test_split.txt marks each id as train (1) or test (0). CUB has no
    separate validation partition; `split="val"` maps to the test
    partition. Labels are zero-indexed (class id - 1).
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    root = Path(root)
    want_train = 1 if split == "train" else 0

    image_paths: dict[str, Path] = {}
    for line in (root / "images.txt").read_text().splitlines():
        image_id, relative_path = line.split(maxsplit=1)
        image_paths[image_id] = root / "images" / relative_path

    labels: dict[str, int] = {}
    for line in (root / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        labels[image_id] = int(class_id) - 1

    split_flags: dict[str, int] = {}
    for line in (root / "train_test_split.txt").read_text().splitlines():
        image_id, is_train = line.split()
        split_flags[image_id] = int(is_train)

    return sorted(
        (image_paths[image_id], labels[image_id])
        for image_id in image_paths
        if split_flags[image_id] == want_train
    )


def load_metadataset(
    root: Path,
    scenario: str,
    split: str = "val",
) -> list[tuple[Path, int]]:
    """CCE's MetaDataset benchmark: one of 20 spurious-correlation scenarios
    (e.g. "dog_snow"). No copy of this dataset exists locally yet (see the
    design doc's open checklist item on whether CCE released their 20
    checkpoints), so this assumes an ImageFolder-style layout --
    `root/<scenario>/<split>/<class_name>/*` -- consistent with the other
    loaders' materialized layout. Update this once the real release format
    is confirmed.
    """
    scenario_dir = Path(root) / scenario / split
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"no MetaDataset scenario directory at {scenario_dir}")

    class_dirs = sorted((p for p in scenario_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return [
        (path, label)
        for label, class_dir in enumerate(class_dirs)
        for path in sorted(class_dir.iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def list_broden_concepts(root: Path) -> list[str]:
    """Concept names available in the local Broden copy -- one subdirectory
    per concept, each already split into positives/ and negatives/."""
    root = Path(root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "positives").is_dir() and (p / "negatives").is_dir()
    )


def load_broden(root: Path, concept: str) -> tuple[list[Path], list[Path]]:
    """Ground-truth (positives, negatives) image paths for `concept`, for
    the retrieval-purity validation check (design doc Section 2, item 5) --
    not a CandidatePool source. The local copy at ../Datasets/broden_concepts
    is already image-level labeled per concept, so there's no pixel-mask
    coverage-threshold conversion to do here.
    """
    concept_dir = Path(root) / concept
    if not concept_dir.is_dir():
        raise ValueError(f"unknown Broden concept {concept!r} (no directory at {concept_dir})")

    def _list_images(subdir: Path) -> list[Path]:
        return sorted(p for p in subdir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    return _list_images(concept_dir / "positives"), _list_images(concept_dir / "negatives")
