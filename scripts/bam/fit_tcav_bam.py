"""TCAV on BAM's obj and scene models, from activations and gradients stored by
`store_bam_outputs.py` (the models are TensorFlow, so captum can't hook them).

Matches `run_tcav_celeba_official_train.py`: captum's DefaultClassifier fits
each CAV (concept vs one of 6 random sets), the score is captum's sign count
(share of inputs whose class-logit gradient has a positive dot product with the
CAV), and 3 control CAVs (control vs control) give the null for a t-test.
CAVs come from --layer: block_layer4 (default), the same position as ResNet's
layer4, or block_layer3. At block_layer4 only pooling and a linear head follow,
so every input of a class has the same gradient and each CAV's sign count is
exactly 0 or 1; block_layer3 gives input-dependent gradients.

Inputs, under <bam-root>/baselines/ (<sfx> is empty for block_layer4, else _<layer>):
  tcav_sets.csv, tcav_inputs.csv          from sample_bam_concept_sets.py
  tf/<model>_tcav_sets<sfx>.npz           activations of every set image
  tf/<model>_tcav_inputs<sfx>.npz         gradients for that model's inputs
Writes results/bam_tcav_scores<sfx>.csv.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_ind
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from captum.concept._utils.classifier import DefaultClassifier
from sample_bam_concept_sets import N_CONTROL, N_RANDOM, OBJECTS, SCENES

SEED = 42
CLASSES = {"obj": OBJECTS, "scene": SCENES}


def fit_cav(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """captum's CAV for `positive` against `negative`, pointing towards `positive`."""
    x = torch.tensor(np.concatenate([positive, negative]), dtype=torch.float32)
    y = torch.tensor([1] * len(positive) + [0] * len(negative))
    classifier = DefaultClassifier()
    classifier.train_and_eval(DataLoader(TensorDataset(x, y), batch_size=len(y)))
    weights = classifier.weights().numpy()
    classes = list(classifier.classes())
    return weights[classes.index(1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--results-dir", type=Path, default=Path(os.environ.get("CARDS_RESULTS_DIR", "results")))
    parser.add_argument("--layer", default="block_layer4", choices=["block_layer3", "block_layer4"])
    args = parser.parse_args()
    sfx = "" if args.layer == "block_layer4" else f"_{args.layer}"
    base = args.bam_root / "baselines"
    torch.manual_seed(SEED)

    with open(base / "tcav_sets.csv", newline="") as handle:
        set_rows = list(csv.DictReader(handle))

    out_rows = []
    for model, classes in CLASSES.items():
        acts = np.load(base / "tf" / f"{model}_tcav_sets{sfx}.npz")
        index = {name: i for i, name in enumerate(acts["files"])}
        activations = acts["activations"]

        def members(set_name: str, activations=activations, index=index) -> np.ndarray:
            return activations[[index[r["file"]] for r in set_rows if r["set"] == set_name]]

        grads = np.load(base / "tf" / f"{model}_tcav_inputs{sfx}.npz")
        gradients, targets = grads["gradients"], grads["targets"]

        randoms = [members(f"random_{i}") for i in range(N_RANDOM)]
        controls = [members(f"control_{i}") for i in range(N_CONTROL)]
        control_cavs = [fit_cav(controls[2 * i], controls[2 * i + 1]) for i in range(N_CONTROL // 2)]

        for concept in OBJECTS + SCENES:
            exemplars = members(concept)
            cavs = [fit_cav(exemplars, random_set) for random_set in randoms]
            for target, cls in enumerate(classes):
                g = gradients[targets == target]
                signs = [float(np.mean(g @ cav > 0)) for cav in cavs]
                magnitudes = [float(np.mean(g @ cav)) for cav in cavs]
                control_signs = [float(np.mean(g @ cav > 0)) for cav in control_cavs]
                t_stat, p_value = ttest_ind(signs, control_signs)
                out_rows.append({
                    "model": model, "class": cls, "concept": concept, "n_inputs": len(g),
                    "mean_sign_count": float(np.mean(signs)), "mean_control_sign_count": float(np.mean(control_signs)),
                    "mean_magnitude": float(np.mean(magnitudes)), "t_stat": float(t_stat), "p_value": float(p_value),
                })
            print(f"{model} {concept}: done", flush=True)

    args.results_dir.mkdir(exist_ok=True)
    out_path = args.results_dir / f"bam_tcav_scores{sfx}.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"saved {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
