"""Scores every official-train method against the strict-control ground truth
from `run_celeba_official_train_strict_control.py`, ResNet18, clean records.

Ground truth per concept is the mean of `delta_p - control` over records with
a valid control. `delta_p` is scored on the same records, so any change comes
from the control rather than from which images are included. A concept-blind
baseline (the control's own effect, used to predict `delta_p`) shows how much
of the raw ground truth any same-shaped mask can explain.

Inclusion rules, set before the full run: a record needs at least
MIN_VALID_DRAWS control draws, and a concept at least MIN_VALID_RECORDS such
records. Concepts failing this have no measurable control and are listed, not
scored. `strict` is the primary control; `strict_excl` is a sensitivity check.

"PCBM conventional" reads the local checkpoints, which reproduce the older
v123 numbers rather than the model behind the 09-20 comparison CSVs.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, "../post_hoc_cbm")

RESULTS_DIR = Path("results")
IN_CSV = RESULTS_DIR / "celeba_official_train_faithfulness_non_overlapping_strict_control.csv"
OUT_CSV = RESULTS_DIR / "score_methods_against_strict_control_official_train.csv"
PCBM_DIR = Path("trained_models_new/celeba_full/celeba_official_train_attractive_male")
TASKS = ["Attractive", "Male"]
CONTROLS = ["strict", "strict_excl"]
MIN_VALID_DRAWS = 3
MIN_VALID_RECORDS = 30
N_BOOT = 5000
SEED = 0


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_methods() -> dict[str, dict[tuple[str, str], float]]:
    methods: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for r in read_csv(RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_raw_scores.csv"):
        methods["CARDS"][(r["task"], r["concept_name"])] = float(r["hybrid_raw_score"])
    for r in read_csv(RESULTS_DIR / "cards_celeba_masking_hybrid_official_train_cohens_d_raw.csv"):
        methods["CARDS (Cohen's d)"][(r["task"], r["concept_name"])] = float(r["cohens_d"])
    for r in read_csv(RESULTS_DIR / "tcav_celeba_official_train_scores.csv"):
        methods["TCAV"][(r["target_task"], r["concept_name"])] = float(r["mean_sign_count"])
    for task in TASKS:
        ckpt = PCBM_DIR / f"pcbm_celeba_full__celeba_official_train_attractive_male__{task.lower()}__surrogate__seed_42__linear.ckpt"
        layer = torch.load(ckpt, weights_only=False)
        weight = layer.classifier.weight.detach().cpu().numpy()
        for i, name in enumerate(layer.names):
            methods["PCBM conventional (local ckpt)"][(task, name)] = float(weight[0, i])
    for label, stem in (("PCBM CLIP-SigLIP", "siglip"), ("PCBM CLIP-RN50", "clip_rn50")):
        for r in read_csv(RESULTS_DIR / f"pcbm_clip_concepts_celeba_official_train_{stem}_scores.csv"):
            methods[label][(r["target_task"], r["concept_name"])] = float(r["weight"])
    return methods


def concept_table(rows: list[dict], task: str, control: str) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    """concept -> (mean delta_p, mean control, mean adjusted), plus excluded concepts."""
    by_concept: dict[str, list[tuple[float, float]]] = defaultdict(list)
    concepts = set()
    for r in rows:
        if r["target_task"] != task:
            continue
        concepts.add(r["concept_name"])
        if int(r[f"{control}_n_valid"]) >= MIN_VALID_DRAWS:
            by_concept[r["concept_name"]].append((float(r["delta_p"]), float(r[f"{control}_random_mean"])))
    kept, dropped = {}, []
    for c in sorted(concepts):
        if len(by_concept[c]) >= MIN_VALID_RECORDS:
            a = np.array(by_concept[c])
            kept[c] = (float(a[:, 0].mean()), float(a[:, 1].mean()), float((a[:, 0] - a[:, 1]).mean()))
        else:
            dropped.append(c)
    return kept, dropped


def main():
    rows = read_csv(IN_CSV)
    diffs = [abs(float(r["delta_p"]) - float(r["delta_p_stored"])) for r in rows]
    print(f"{len(rows)} records; max |delta_p - stored| = {max(diffs):.2e}")

    methods = load_methods()
    rng = np.random.default_rng(SEED)
    out = []
    for control in CONTROLS:
        tables = {t: concept_table(rows, t, control) for t in TASKS}
        print(f"\n=== control={control}")
        for t in TASKS:
            kept, dropped = tables[t]
            print(f" {t}: {len(kept)} concepts scored; no valid control: {dropped}")
            cs = sorted(kept)
            dp, ctrl, adj = (np.array([kept[c][i] for c in cs]) for i in range(3))
            baseline = spearmanr(ctrl, dp)[0]
            print(f"   concept-blind baseline rho(control, delta_p)={baseline:+.3f}  rho(delta_p, adjusted)={spearmanr(dp, adj)[0]:+.3f}")
            out.append((control, t, "Concept-blind (control effect)", len(cs), baseline, "", spearmanr(ctrl, adj)[0]))

        print(f" {'method':32s}{'A dp':>7s}{'A adj':>7s}{'M dp':>7s}{'M adj':>7s}{'pool dp':>9s}{'pool adj':>9s}")
        arrays = {}
        for m, scores in methods.items():
            vals = []
            for t in TASKS:
                kept = tables[t][0]
                cs = sorted(kept)
                x = np.array([scores[(t, c)] for c in cs])
                dp, adj = np.array([kept[c][0] for c in cs]), np.array([kept[c][2] for c in cs])
                arrays[(m, t)] = (x, adj)
                rho_dp, rho_adj = spearmanr(x, dp)[0], spearmanr(x, adj)[0]
                vals += [rho_dp, rho_adj]
                out.append((control, t, m, len(cs), rho_dp, rho_adj, ""))
            print(f" {m:32s}{vals[0]:+7.2f}{vals[1]:+7.2f}{vals[2]:+7.2f}{vals[3]:+7.2f}"
                  f"{(vals[0] + vals[2]) / 2:+9.2f}{(vals[1] + vals[3]) / 2:+9.2f}")

        print(f" bootstrap pooled adjusted rho ({N_BOOT} concept resamples)")
        boot = defaultdict(list)
        for _ in range(N_BOOT):
            per = defaultdict(list)
            for t in TASKS:
                n = len(arrays[("CARDS", t)][0])
                i = rng.integers(0, n, n)
                for m in methods:
                    x, adj = arrays[(m, t)]
                    per[m].append(spearmanr(x[i], adj[i])[0])
            for m in methods:
                boot[m].append(np.mean(per[m]))
        print(f"   CARDS 95% CI [{np.nanpercentile(boot['CARDS'], 2.5):+.2f},{np.nanpercentile(boot['CARDS'], 97.5):+.2f}]")
        for m in methods:
            if m != "CARDS":
                d = np.array(boot["CARDS"]) - np.array(boot[m])
                print(f"   CARDS minus {m:32s} [{np.nanpercentile(d, 2.5):+.2f},{np.nanpercentile(d, 97.5):+.2f}]")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["control", "task", "method", "n_concepts", "rho_delta_p", "rho_adjusted", "rho_control_vs_adjusted"])
        writer.writerows(out)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
