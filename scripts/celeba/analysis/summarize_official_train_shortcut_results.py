"""Reads the `run_info_official_train_{task}_shortcut_{rate}.pkl` files
written by `train_attractive_shortcut_classifiers_official_train.py` /
`train_male_shortcut_classifiers_official_train.py` and prints the
same-rate-vs-clean val-accuracy gap per rate, per task -- the core
shortcut-reliance diagnostic, without needing GPU or re-running anything.
CPU-only, cheap -- run directly (no sbatch needed), on whichever
filesystem the 22 checkpoints/pickles actually live on (set CARDS_CKPT_DIR
if not the local default).
"""

from __future__ import annotations

import csv
import os
import pickle
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("CARDS_RESULTS_DIR", "results"))
CKPT_DIR = Path(os.environ.get("CARDS_CKPT_DIR", "trained_models_new/celeba"))
TASKS = ["Attractive", "Male"]
RATES_PCT = list(range(0, 101, 10))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    all_rows = []  # (task, rate_pct, same_rate_val_acc, clean_val_acc, gap, n_train, n_val)
    missing = []

    for task in TASKS:
        task_lower = task.lower()
        print(f"\n=== {task} ===")
        for rate_pct in RATES_PCT:
            info_path = CKPT_DIR / f"run_info_official_train_{task_lower}_shortcut_{rate_pct}.pkl"
            if not info_path.exists():
                missing.append(str(info_path))
                print(f"  rate={rate_pct:>3d}%  MISSING ({info_path})")
                continue
            with open(info_path, "rb") as f:
                info = pickle.load(f)
            gap = info["same_rate_val_acc"] - info["clean_val_acc"]
            all_rows.append((task, rate_pct, info["same_rate_val_acc"], info["clean_val_acc"], gap,
                              info["n_train"], info["n_val"]))
            print(f"  rate={rate_pct:>3d}%  same-rate_acc={info['same_rate_val_acc']:.4f}  "
                  f"clean_acc={info['clean_val_acc']:.4f}  gap={gap:+.4f}")

    if missing:
        print(f"\n{len(missing)} run_info file(s) not found -- rerun/resync those rates before trusting this summary:")
        for m in missing:
            print(f"  {m}")

    out_path = RESULTS_DIR / "celeba_official_train_shortcut_accuracy_gap_summary.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", "rate_pct", "same_rate_val_acc", "clean_val_acc", "gap", "n_train", "n_val"])
        writer.writerows(all_rows)
    print(f"\nSaved {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
