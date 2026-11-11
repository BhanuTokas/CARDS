"""Companion to `local_attribution_comparison_celeba_corrected_gt.py`:
scores hybrid/TCAV on CONCEPT-ABSENT images (attribute==False), instead
of assuming their score is 0 -- prompted directly ("For images where
concept is absent, can't we just assume the attribution to be zero" ->
"We should go with (a) [actually compute real scores] as that is more
principled and mirrors real-world use case").

**Ground truth IS assumed 0 for absent images** (not computed) -- this
one really is structural, not a simplifying assumption: the real
masking-based ground truth removes the concept's own CelebAMask-HQ
segmentation region, and if the concept isn't present, that region is
empty, so there's nothing to mask and the causal effect of removing
nothing is 0 by construction, not by choice.

**Hybrid and TCAV scores ARE actually computed here, not assumed** --
the more informative, and more honest, version of "what does an absent
concept look like to this method":
  - **Hybrid**: reuses `local_attribution_comparison_celeba_corrected_
    gt.py`'s own localize+threshold+mask machinery UNCHANGED, run on
    images where the concept genuinely isn't there. That script's
    existing degenerate-mask fallback (`hybrid_score=0.0` when
    `threshold_mask` finds an empty or full region) already partially
    covers this case; the interesting NEW finding is whichever absent
    images DON'T hit that fallback -- localize_concept finds SOME
    region to mask despite the concept not being present, producing a
    real (possibly large) score. That's a genuine false-positive
    signal this script can now surface, not something the original
    present-only run could ever have shown.
  - **TCAV**: needs no adaptation at all -- its local per-image score
    is a directional derivative against a fixed CAV direction, never
    dependent on whether the scored image is concept-present.

Absent-image sample: drawn from `val_hq` (CelebAMask-HQ's own held-out
val split, the SAME pool `run_celeba_full_faithfulness_attribute_
conditioned.py` draws PRESENT ground-truth candidates from -- matching
pools, not a new leakage risk), filtered to attribute==False, up to
N_ABSENT_PER_CONCEPT=100 per concept (this project's own standard
N_PER_ATTRIBUTE convention), both target tasks (Attractive, Young)
scored per sampled image, same as the present-pairs script.

TCAV's own exemplar/random/control pools stay sourced from `train_hq`
(a fitting resource, unchanged) -- only the SCORING sample moves to
absent val_hq images; the CAV-fitting apparatus itself is identical to
the present-pairs script's own, reused verbatim below (duplicated, not
imported, to avoid dragging a second heavy encoder/TCAV/model
instantiation into this process via cross-script import side effects).
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba import load_celebamask_hq_image_paths, split_celebamask_hq
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    TARGET_CLASSES,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
ALPHA = 1.0
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
N_ABSENT_PER_CONCEPT = 100
N_RANDOM, N_CONTROL, N_PER_RANDOM_SET, N_CONCEPT_EXEMPLARS = 6, 6, 25, 40
HOOK_LAYER = "layer4"
TCAV_DEVICE = "cpu"


def make_tcav_concept(concept_id: int, name: str, image_paths: list, preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    native_model_cpu = spec.load_native().to(TCAV_DEVICE).eval()

    demeaned_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                        for c in GROUNDABLE_CONCEPTS}
    orthogonalized_queries = orthogonalize_queries(demeaned_queries)

    print("Loading CelebAMask-HQ metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)
    print(f"{len(train_hq)} train_hq (TCAV exemplar pool)  {len(val_hq)} val_hq "
          f"(absent-image SCORING pool -- same held-out split the present ground truth uses)", flush=True)

    rng_py = random.Random(SEED)

    print("\n=== per-concept setup (TCAV CAVs, z-score cutoff) -- one-time cost, mirrors the present-pairs script ===", flush=True)
    all_train_paths = [image_paths_by_idx[i] for i in train_hq]
    rng_py.shuffle(all_train_paths)
    tcav_concept_id = 0
    idx = 0
    random_concepts = []
    for i in range(N_RANDOM):
        random_concepts.append(make_tcav_concept(tcav_concept_id, f"random_{i}", all_train_paths[idx:idx + N_PER_RANDOM_SET], spec.preprocess))
        idx += N_PER_RANDOM_SET
        tcav_concept_id += 1
    for i in range(N_CONTROL):
        idx += N_PER_RANDOM_SET
        tcav_concept_id += 1

    tcav = TCAV(model=native_model_cpu, layers=[HOOK_LAYER], save_path=str(RESULTS_DIR / "local_attribution_absent_tcav_cav_cache"))

    per_concept = {}
    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c_hybrid = orthogonalized_queries[concept_name].to(DEVICE)
        concept_attr_idx = attr_names.index(concept_name)

        hybrid_present_indices_paths = [
            image_paths_by_idx[i] for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]
        ]
        rng_py.shuffle(hybrid_present_indices_paths)
        sim_maps = []
        for p in hybrid_present_indices_paths[:50]:
            img = Image.open(p).convert("RGB")
            sim_maps.append(localize_concept(encoder, img, t_c_hybrid, (img.height, img.width)))
        cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

        pos_train_paths = [image_paths_by_idx[i] for i in train_hq if attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]]
        rng_py.shuffle(pos_train_paths)
        target_concept = make_tcav_concept(tcav_concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], spec.preprocess)
        tcav_concept_id += 1

        absent_val_paths = [
            image_paths_by_idx[i] for i in val_hq if not attr_labels_by_file[f"{i}.jpg"][concept_attr_idx]
        ]
        rng_py.shuffle(absent_val_paths)
        absent_sample = absent_val_paths[:N_ABSENT_PER_CONCEPT]

        per_concept[concept_name] = {
            "t_c_hybrid": t_c_hybrid, "cutoff": cutoff, "target_concept": target_concept,
            "absent_sample": absent_sample,
        }
        print(f"  [{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} cutoff={cutoff:+.4f}  "
              f"absent_val_pool={len(absent_val_paths)}  sampled={len(absent_sample)}", flush=True)

    total_pairs = sum(len(v["absent_sample"]) for v in per_concept.values())
    print(f"\n=== scoring {total_pairs} absent (image, concept) pairs x {len(TARGET_CLASSES)} tasks ===", flush=True)
    all_out_rows = []
    pair_i = 0
    for concept_name in GROUNDABLE_CONCEPTS:
        cinfo = per_concept[concept_name]
        concept_idx = GROUNDABLE_CONCEPTS.index(concept_name)

        for image_path in cinfo["absent_sample"]:
            image = Image.open(image_path).convert("RGB")
            pixels = spec.preprocess(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                orig_logits = native_model(pixels)[0]

            sim_map = localize_concept(encoder, image, cinfo["t_c_hybrid"], (image.height, image.width))
            mask = threshold_mask(sim_map, method="fixed", cutoff=cinfo["cutoff"])
            hybrid_by_task = {}
            if mask.any() and not mask.all():
                rng = np.random.default_rng(SEED + concept_idx * 10_000 + pair_i)
                candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILL_STRATEGIES]
                with torch.no_grad():
                    embeds = encoder.encode_images([image] + candidates).to(DEVICE)
                embed_orig = embeds[0]
                best_angle, best_i = None, None
                for i in range(len(FILL_STRATEGIES)):
                    diff = embed_orig - embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ cinfo["t_c_hybrid"], -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle_deg < best_angle:
                        best_angle, best_i = angle_deg, i
                masked_image = candidates[best_i]
                pixels_masked = spec.preprocess(masked_image).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    masked_logits = native_model(pixels_masked)[0]
                for task_name in TARGET_CLASSES:
                    task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                    hybrid_by_task[task_name] = (orig_logits[task_idx] - masked_logits[task_idx]).item()
            else:
                for task_name in TARGET_CLASSES:
                    hybrid_by_task[task_name] = 0.0

            tcav_by_task = {}
            for task_name in TARGET_CLASSES:
                task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                experimental_sets = [[cinfo["target_concept"], rc] for rc in random_concepts]
                scores = tcav.interpret(inputs=pixels.to(TCAV_DEVICE), experimental_sets=experimental_sets, target=task_idx)
                sign_counts = [scores[f"{cinfo['target_concept'].id}-{rc.id}"][HOOK_LAYER]["sign_count"][0].item() for rc in random_concepts]
                tcav_by_task[task_name] = float(np.mean(sign_counts))

            for task_name in TARGET_CLASSES:
                all_out_rows.append((str(image_path), concept_name, task_name, 0.0,
                                      hybrid_by_task[task_name], tcav_by_task[task_name]))

            pair_i += 1
            if pair_i % 50 == 0 or pair_i == total_pairs:
                print(f"  [{pair_i}/{total_pairs}] absent pairs scored", flush=True)

    out_path = RESULTS_DIR / "local_attribution_celeba_absent_pairs.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "concept_name", "target_task", "gt_delta_p", "hybrid_score", "tcav_score"])
        writer.writerows(all_out_rows)
    print(f"\nSaved {len(all_out_rows)} absent-image scored rows to {out_path}", flush=True)

    print("\n=== absent-image score summary (should trend near 0 for a well-behaved method; non-zero = false positive) ===", flush=True)
    for task_name in TARGET_CLASSES:
        task_rows = [r for r in all_out_rows if r[2] == task_name]
        hybrid_scores = [r[4] for r in task_rows]
        tcav_scores = [r[5] for r in task_rows]
        n_hybrid_nonzero = sum(1 for s in hybrid_scores if s != 0.0)
        print(f"  [{task_name}] n={len(task_rows)}  hybrid: mean|score|={np.mean(np.abs(hybrid_scores)):.4f} "
              f"({n_hybrid_nonzero}/{len(hybrid_scores)} nonzero, i.e. found SOME region despite absence)  "
              f"tcav: mean|score|={np.mean(np.abs(tcav_scores)):.4f}", flush=True)


if __name__ == "__main__":
    main()
