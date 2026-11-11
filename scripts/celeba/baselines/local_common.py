"""Shared pieces of the local (per-image) attribution runs: the records, image paths and sharding.

env CARDS_LOCAL_SET picks the records:
  present (default): the (image, concept, task, delta_p) rows of the clean ground truth
      (results/celeba_official_train[_vit|_convnext]_faithfulness_{CARDS_LOCAL_GT}.csv, default non_overlapping),
      grouped into (image, concept) pairs in file order exactly as
      scripts/celeba/analysis/local_attribution_comparison_celeba_official_train.py groups them, so pair indices
      (and the seeds derived from them) match that run.
  absent: the concept-absent images of the paper's local figures
      (results/local_attribution_celeba_official_train_absent_pairs.csv; ground truth 0 by construction).
Images are looked up by file name under $CELEBA_HQ_ROOT/CelebA-HQ-img, so CSVs written on another machine work.

Sharding: env SHARD / NSHARDS (default 0 / 1) split the work round-robin; LOCAL_LIMIT=n keeps the first n items of
the shard (smoke runs)."""
import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/celeba/run")
from run_cards_celeba_masking_hybrid_official_val_zscore import CELEBA_HQ_ROOT

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import SUFFIX

RESULTS = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
GT_VERSION = os.environ.get("CARDS_LOCAL_GT", "non_overlapping")
LOCAL_SET = os.environ.get("CARDS_LOCAL_SET", "present")
ABSENT_CSV = RESULTS / "local_attribution_celeba_official_train_absent_pairs.csv"
POS = {"Attractive": 1, "Male": 3}  # positive-class logit per task in the 4-way head
SEED = 42  # seed of the existing local run (mask fills)


def image_name(path: str) -> str:
    return re.split(r"[\\/]", path)[-1]


def gt_pairs() -> list[tuple[str, str, dict[str, float]]]:
    """[(image file name, concept, {task: ground truth})] in file order, for CARDS_LOCAL_SET."""
    if LOCAL_SET == "present":
        src, col = RESULTS / f"celeba_official_train{SUFFIX}_faithfulness_{GT_VERSION}.csv", "delta_p"
    elif LOCAL_SET == "absent":
        src, col = ABSENT_CSV, "gt_delta_p"
    else:
        raise ValueError(f"CARDS_LOCAL_SET must be present or absent, got {LOCAL_SET!r}")
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    with open(src, newline="") as f:
        for r in csv.DictReader(f):
            pairs.setdefault((image_name(r["image"]), r["concept_name"]), {})[r["target_task"]] = float(r[col])
    return [(n, c, t) for (n, c), t in pairs.items()]


def hq_path(name: str) -> Path:
    return CELEBA_HQ_ROOT / "CelebA-HQ-img" / name


def shard(items: list) -> list[tuple[int, object]]:
    """(global index, item) for this shard."""
    s, n = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    out = [(i, x) for i, x in enumerate(items) if i % n == s]
    limit = int(os.environ.get("LOCAL_LIMIT", "0"))
    return out[:limit] if limit else out


def shard_tag() -> str:
    return f"{LOCAL_SET}_{int(os.environ.get('SHARD', '0')):02d}"
