"""Normalization ablation for the masking hybrid's LOCAL attribution
score, prompted directly ("Will type of normalization impact the rho
value?" -> "Can you run a normalization ablation for the hybrid
technique?"), following up on `local_attribution_comparison_celeba.py`'s
own pooled rho (computed on entirely raw, unnormalized per-image deltas
-- masking_hybrid mode already skips Step 7's own `variance_normalize`
by design, see `cards.attribution.normalization`'s own docstring: its
present-vs-absent formulas don't transfer to same-image paired deltas).

Reuses the already-saved `results/local_attribution_celeba_pairs.csv`
directly -- no model recompute needed, pure re-aggregation. Tests
whether per-concept normalization (which IS a non-uniform transform
across the pooled n~2500 sample, unlike a single global rescaling) shifts
the pooled Spearman rho, since different concepts' raw hybrid deltas may
sit on different natural scales that a single pooled ranking doesn't
account for.

Four variants, all applied to hybrid_score only (baseline/tcav columns
untouched):
  - raw: unchanged (the number already reported).
  - global_zscore: (x - mean_all) / std_all -- a UNIFORM monotonic
    transform across the whole pooled array. Included as a sanity check:
    Spearman rho is invariant to any monotonic transform applied
    uniformly, so this MUST come out identical to raw if computed
    correctly -- confirms the mechanism, not a candidate fix.
  - per_concept_zscore: (x - mean_within_concept) / std_within_concept.
  - per_concept_minmax: (x - min_within_concept) / (max_within_concept -
    min_within_concept), each concept independently rescaled to [0, 1].
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = Path("results")
EPSILON = 1e-8


def load_rows() -> list[dict]:
    rows = []
    with open(RESULTS_DIR / "local_attribution_celeba_pairs.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "concept_name": row["concept_name"],
                "target_task": row["target_task"],
                "gt_delta_p": float(row["gt_delta_p"]),
                "hybrid_score": float(row["hybrid_score"]),
            })
    return rows


def per_concept_zscore(rows: list[dict]) -> list[float]:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_concept[r["concept_name"]].append(r["hybrid_score"])
    stats = {c: (np.mean(v), np.std(v)) for c, v in by_concept.items()}
    out = []
    for r in rows:
        mean, std = stats[r["concept_name"]]
        out.append((r["hybrid_score"] - mean) / std if std > EPSILON else 0.0)
    return out


def per_concept_minmax(rows: list[dict]) -> list[float]:
    by_concept: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_concept[r["concept_name"]].append(r["hybrid_score"])
    stats = {c: (min(v), max(v)) for c, v in by_concept.items()}
    out = []
    for r in rows:
        lo, hi = stats[r["concept_name"]]
        out.append((r["hybrid_score"] - lo) / (hi - lo) if (hi - lo) > EPSILON else 0.0)
    return out


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} scored rows.\n", flush=True)

    raw = [r["hybrid_score"] for r in rows]
    all_mean, all_std = np.mean(raw), np.std(raw)
    global_z = [(v - all_mean) / all_std for v in raw]
    pc_z = per_concept_zscore(rows)
    pc_minmax = per_concept_minmax(rows)

    variants = {
        "raw": raw,
        "global_zscore": global_z,
        "per_concept_zscore": pc_z,
        "per_concept_minmax": pc_minmax,
    }

    for variant_name, scores in variants.items():
        print(f"=== {variant_name} ===", flush=True)
        for task_name in ["Attractive", "Young"]:
            gt = [r["gt_delta_p"] for r in rows if r["target_task"] == task_name]
            sc = [s for r, s in zip(rows, scores) if r["target_task"] == task_name]
            rho, p = spearmanr(gt, sc)
            print(f"  [{task_name}] n={len(gt)} rho={rho:+.4f} p={p:.4g}", flush=True)
        print()


if __name__ == "__main__":
    main()
