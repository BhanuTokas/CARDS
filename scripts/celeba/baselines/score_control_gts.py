# ruff: noqa: SIM115
# One-shot research script: bare open() on small CSVs is intentional.

"""Score every ResNet18 method against the control-adjusted blur ground truths: delta_p minus the random-mask control
under the permissive (original random_delta_p_mean), strict and strict_excl controls. Wraps
scripts/celeba/analysis/score_methods_against_strict_control_official_train.py without changing it: adds the methods
given in env EXTRA ("label|csv|column;...", CSVs with task, concept_name columns) and the permissive control, and
writes to argv[1] instead of the repo's results CSV. Needs post_hoc_cbm on PYTHONPATH. Run from the repo root."""
import csv
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "scripts/celeba/analysis")
import score_methods_against_strict_control_official_train as S

_base = S.load_methods


def load_methods():
    m = _base()
    for spec in filter(None, os.environ.get("EXTRA", "").split(";")):
        label, path, column = spec.split("|")
        for r in csv.DictReader(open(path)):
            m[label][(r["task"], r["concept_name"])] = float(r[column])
    return m


def main():
    S.load_methods = load_methods
    rows = S.read_csv(S.IN_CSV)
    for r in rows:
        r["permissive_n_valid"] = "5"  # the original control always has its 5 draws
    tmp = Path(tempfile.mkdtemp()) / "controls.csv"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S.IN_CSV, S.OUT_CSV = tmp, Path(sys.argv[1])
    S.CONTROLS = ("permissive", "strict", "strict_excl")
    S.N_BOOT = 2000
    S.main()


if __name__ == "__main__":
    main()
