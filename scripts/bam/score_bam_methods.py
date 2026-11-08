"""Score HnS, TCAV and PCBM on BAM's model-contrast test.

Every method gives S[model, class, concept]. Two tests:

  A. Class-matched rank. For each model and class, rank the 20 concepts by S.
     The class's own concept (dog for obj/dog, bedroom for scene/bedroom) should
     come first. Scale-free, so valid for every method.

  B. Model contrast (BAM's MCS, at concept level). U[model, concept] is the
     effect of a concept on the class the model predicts for images containing
     it: S weighted by how often that model predicts each class on held-out
     images with the concept. Averaging over all classes would not work, since
     probability changes, and roughly elastic-net weights, sum to zero over
     classes. Object concepts should have U[obj] > U[scene], and scene concepts
     the reverse. Only meaningful where S shares a scale across the two models.

HnS's S[m, k, c] is the mean change in class k's output over the K images
retrieved for concept c, from logits stored by store_bam_outputs.py. Its U is
computed directly: each image's change in its own predicted class.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_hns_bam_masks import CONCEPTS  # noqa: E402
from sample_bam_concept_sets import OBJECTS, SCENES  # noqa: E402

MODELS = {"obj": OBJECTS, "scene": SCENES}
FAMILY = {c: "obj" for c in OBJECTS} | {c: "scene" for c in SCENES}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_logits(path: Path) -> dict:
    data = np.load(path)
    return dict(zip(data["files"], data["logits"]))


def prediction_weights(bam_root: Path) -> dict:
    """pi[model][concept] = share of each class among the model's predictions on
    held-out images that contain the concept."""
    with open(bam_root / "val" / "manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pi = {}
    for model in MODELS:
        logits = load_logits(bam_root / "logits" / f"{model}_composite.npz")
        pi[model] = {}
        for concept in CONCEPTS:
            column = "object" if FAMILY[concept] == "obj" else "scene"
            preds = [logits[r["file"]].argmax() for r in rows if r[column] == concept]
            pi[model][concept] = np.bincount(preds, minlength=10) / len(preds)
    return pi


def hns_scores(bam_root: Path, variant: str, measure: str) -> tuple:
    """S[(model, class, concept)] and U[(model, concept)] for HnS."""
    with open(bam_root / "hns" / variant / "manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    S, U = {}, {}
    for model, classes in MODELS.items():
        original = load_logits(bam_root / "logits" / f"{model}_composite.npz")
        masked = load_logits(bam_root / "hns" / variant / f"{model}_masked.npz")
        by_concept = defaultdict(list)
        for row in rows:
            a = original[row["file"]][None]
            b = masked[row["masked_file"]][None]
            if measure == "prob":
                a, b = softmax(a), softmax(b)
            own = int(original[row["file"]].argmax())
            by_concept[row["concept"]].append(((a - b)[0], own))
        for concept, items in by_concept.items():
            deltas = np.stack([d for d, _ in items])
            for k, cls in enumerate(classes):
                S[model, cls, concept] = float(deltas[:, k].mean())
            U[model, concept] = float(np.mean([d[own] for d, own in items]))
    return S, U


def weighted_u(S: dict, pi: dict) -> dict:
    U = {}
    for model, classes in MODELS.items():
        for concept in CONCEPTS:
            U[model, concept] = float(sum(pi[model][concept][k] * S[model, cls, concept]
                                          for k, cls in enumerate(classes)))
    return U


def rank_test(S: dict) -> dict:
    ranks = {}
    for model, classes in MODELS.items():
        for cls in classes:
            scores = np.array([S[model, cls, c] for c in CONCEPTS])
            order = np.argsort(-scores, kind="stable")
            ranks[model, cls] = int(np.where(np.array(CONCEPTS)[order] == cls)[0][0]) + 1
    return ranks


def contrast_test(U: dict) -> dict:
    out = {}
    for concept in CONCEPTS:
        own = FAMILY[concept]
        other = "scene" if own == "obj" else "obj"
        out[concept] = U[own, concept] - U[other, concept]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    args = parser.parse_args()
    pi = prediction_weights(args.bam_root)

    methods = {}
    for variant in sorted(p.name for p in (args.bam_root / "hns").iterdir() if p.is_dir()):
        if not (args.bam_root / "hns" / variant / "obj_masked.npz").exists():
            continue
        for measure in ("prob", "logit"):
            methods[f"HnS ({variant}, delta {measure})"] = hns_scores(args.bam_root, variant, measure) + (True,)

    tcav_path = args.results_dir / "bam_tcav_scores.csv"
    if tcav_path.exists():
        with open(tcav_path, newline="") as handle:
            S = {(r["model"], r["class"], r["concept"]): float(r["mean_sign_count"]) for r in csv.DictReader(handle)}
        methods["TCAV (sign count)"] = (S, weighted_u(S, pi), True)

    pcbm_path = args.results_dir / "bam_pcbm_scores.csv"
    if pcbm_path.exists():
        with open(pcbm_path, newline="") as handle:
            pcbm_rows = list(csv.DictReader(handle))
        for variant in sorted({r["variant"] for r in pcbm_rows}):
            S = {(r["model"], r["class"], r["concept"]): float(r["weight"])
                 for r in pcbm_rows if r["variant"] == variant}
            # The own-model variant's features differ between the two models.
            methods[f"PCBM ({variant})"] = (S, weighted_u(S, pi), variant != "own")

    summary, detail = [], []
    for name, (S, U, comparable) in methods.items():
        ranks = rank_test(S)
        contrast = contrast_test(U)
        for model in MODELS:
            r = [ranks[model, cls] for cls in MODELS[model]]
            summary.append({"method": name, "test": f"rank, {model} model",
                            "value": f"top-1 {sum(x == 1 for x in r)}/10, mean rank {np.mean(r):.1f}"})
        values = np.array(list(contrast.values()))
        summary.append({"method": name, "test": "contrast" + ("" if comparable else " (scales differ)"),
                        "value": f"{int((values > 0).sum())}/20 correct, mean {values.mean():+.3f}"})
        for concept in CONCEPTS:
            detail.append({"method": name, "concept": concept, "family": FAMILY[concept],
                           "u_obj": U["obj", concept], "u_scene": U["scene", concept],
                           "contrast": contrast[concept],
                           "rank_in_own_class": ranks[FAMILY[concept], concept]})

    for row in summary:
        print(f"{row['method']:<36s} {row['test']:<28s} {row['value']}")
    args.results_dir.mkdir(exist_ok=True)
    for fname, rows in (("bam_model_contrast_summary.csv", summary), ("bam_model_contrast_by_concept.csv", detail)):
        with open(args.results_dir / fname, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
