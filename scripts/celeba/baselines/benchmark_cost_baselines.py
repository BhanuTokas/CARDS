"""Wall-clock cost of the newer CelebA baselines (ConceptSAE, ConceptSHAP, EAC) at full scale, for the computational
cost table next to results/computational_cost_benchmark*_full_scale.csv (HnS, TCAV, PCBM).

Each phase runs the SAME script that produced the main-table scores, unchanged, as a subprocess, so the timed work is
exactly the reported configuration (official-train ResNet18 by default; env CARDS_ARCH picks the black box):
  features      extract_feats.py   pooled features of all official train + val images, and layer4 maps of a fixed 40k
                                   of train + val (shared by ConceptSAE and ConceptSHAP; counted once for each)
  ConceptSAE    concept_sae.py     TopK SAE fit (60 epochs), label naming and scoring
  ConceptSHAP   concept_shap.py    concept fit (10 epochs), 300-permutation Shapley values and label naming
  EAC           celeba_eac.py      per image: SAM segments, 2,500 masked black-box queries, surrogate fit, 50k coalitions
To time the full cost, everything starts cold: features go to a fresh $CARDS_FEAT_CACHE and every output to a fresh run
folder (EAC would otherwise resume from its per-image CSV).
The paper's cost table splits each method into reusable (black-box-independent) and black-box-dependent cost.
ConceptSAE and ConceptSHAP are entirely black-box-dependent (their features come from the model). For EAC, SAM
segmentation does not depend on the model: EAC runs twice, first with an empty SAM cache (total), then on the same
images with that cache filled (black-box-dependent part); reusable = the difference.

argv[1] = SAM checkpoint, argv[2] = run folder (must not exist). Needs segment-anything on PYTHONPATH.
Writes <run folder>/computational_cost_benchmark_baselines_full_scale.csv (method, phase, seconds) in the same format
as the existing benchmarks, plus each phase's log. Run from the repo root."""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

B = Path(__file__).parent


def main():
    ckpt, run = sys.argv[1], Path(sys.argv[2])
    if run.exists():
        raise SystemExit(f"{run} exists; use a fresh folder so nothing is reused from an earlier run")
    run.mkdir(parents=True)
    env = dict(os.environ, CARDS_FEAT_CACHE=str(run / "feat_cache"), CARDS_SAM_CACHE=str(run / "sam_cache"))
    env.pop("EAC_LIMIT", None)
    phases = [("features", ["extract_feats.py"]),
              ("ConceptSAE", ["concept_sae.py", str(run / "sae.csv")]),
              ("ConceptSHAP", ["concept_shap.py", str(run / "cshap.csv")]),
              ("EAC_cold", ["celeba_eac.py", ckpt, str(run / "eac.csv")]),             # empty SAM cache: SAM + rest
              ("EAC_warm", ["celeba_eac.py", ckpt, str(run / "eac_warm.csv")])]       # same images, SAM cached
    sec = {}
    for name, (script, *args) in phases:
        print(f"== {name}: {script} {' '.join(args)}", flush=True)
        t = time.perf_counter()
        with open(run / f"{name}.log", "w") as log:
            subprocess.run([sys.executable, "-u", str(B / script), *args], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        sec[name] = time.perf_counter() - t
        print(f"   {sec[name]:.1f} s", flush=True)
    sam = sec["EAC_cold"] - sec["EAC_warm"]
    # method, phase, seconds; phases named as in the paper's split (reusable / black_box_dependent)
    rows = [("ConceptSAE", "feature_extraction (shared with ConceptSHAP)", sec["features"]),
            ("ConceptSAE", "sae_fit_naming_scoring", sec["ConceptSAE"]),
            ("ConceptSAE", "reusable", 0.0),
            ("ConceptSAE", "black_box_dependent", sec["features"] + sec["ConceptSAE"]),
            ("ConceptSAE", "TOTAL", sec["features"] + sec["ConceptSAE"]),
            ("ConceptSHAP", "feature_extraction (shared with ConceptSAE)", sec["features"]),
            ("ConceptSHAP", "concept_fit_shapley_naming", sec["ConceptSHAP"]),
            ("ConceptSHAP", "reusable", 0.0),
            ("ConceptSHAP", "black_box_dependent", sec["features"] + sec["ConceptSHAP"]),
            ("ConceptSHAP", "TOTAL", sec["features"] + sec["ConceptSHAP"]),
            ("EAC", "sam_segmentation", sam),
            ("EAC", "queries_surrogate_shapley", sec["EAC_warm"]),
            ("EAC", "reusable", sam),
            ("EAC", "black_box_dependent", sec["EAC_warm"]),
            ("EAC", "TOTAL", sec["EAC_cold"])]
    out = run / "computational_cost_benchmark_baselines_full_scale.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "phase", "seconds"])
        w.writerows(rows)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
