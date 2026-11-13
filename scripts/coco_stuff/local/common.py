"""Shared pieces of the COCO-Stuff local (per-image) attribution runs for the ResNet18 black box.

Records (written by make_records.py to $COCO_LOCAL_OUT/records.csv): one row per (set, image, concept, class), where
  set = present: the concept was detected in the image (a coco_stuff_faithfulness.csv row exists) and the class is in
        the image (COCO instance annotations); ground truth = that row's delta_p / delta_logit;
  set = absent:  the class is in the image but the concept was not detected; ground truth 0 by construction.
Images are stored by file name and read from $COCO_ROOT/train2017, so the CSV works on any machine.

Sharding: env SHARD / NSHARDS (default 0 / 1) split the work round-robin; LOCAL_LIMIT=n keeps the first n items of the
shard (smoke runs). Run everything from the repo root."""
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, "src"); sys.path.insert(0, "scripts/coco_stuff/baselines")
from coco_common import BRODEN_ROOT, COCO_ROOT, DROPPED

from cards.data.datasets import list_broden_concepts

OUT = Path(os.environ.get("COCO_LOCAL_OUT", "results/coco_stuff_local/resnet18"))
RECORDS = OUT / "records.csv"
SEED = 42


def all_concepts() -> list[str]:
    """The concept bank the COCO CARDS run orthogonalizes over (every non-dropped Broden concept, sorted)."""
    return [c for c in list_broden_concepts(BRODEN_ROOT) if c not in DROPPED]


def image_path(name: str) -> Path:
    return COCO_ROOT / "train2017" / name


def records(only_set: str | None = None) -> list[dict]:
    with open(RECORDS, newline="") as f:
        return [r for r in csv.DictReader(f) if only_set is None or r["set"] == only_set]


def groups(recs: list[dict], by=("image", "concept_name")) -> list[tuple[tuple, list[dict]]]:
    """Records grouped by the given keys, in file order."""
    out: dict[tuple, list[dict]] = {}
    for r in recs:
        out.setdefault(tuple(r[k] for k in by), []).append(r)
    return list(out.items())


def shard(items: list) -> list[tuple[int, object]]:
    """(global index, item) for this shard."""
    s, n = int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARDS", "1"))
    out = [(i, x) for i, x in enumerate(items) if i % n == s]
    limit = int(os.environ.get("LOCAL_LIMIT", "0"))
    return out[:limit] if limit else out


def shard_tag() -> str:
    return f"{int(os.environ.get('SHARD', '0')):02d}"


BASE_COLS = ["set", "image", "concept_name", "class_name", "gt_delta_p", "gt_delta_logit"]


def base(r: dict) -> list:
    return [r[k] for k in BASE_COLS]
