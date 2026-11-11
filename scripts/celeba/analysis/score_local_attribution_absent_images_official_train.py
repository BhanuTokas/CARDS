"""Official-train counterpart of `score_local_attribution_absent_
images.py` -- see that script's own docstring for the full shared
design (ground truth assumed 0 for absent images, structural not a
simplifying assumption; hybrid/TCAV scores ARE actually computed, not
assumed, per direct instruction "We should go with (a) [compute real
scores] as that is more principled"). Only the classifier and its
consequences change, matching `local_attribution_comparison_celeba_
official_train.py`'s own substitutions exactly:

  - Backbone: `celeba_official_train_attractive_male`, TASK_POSITIVE_
    LOGIT_INDEX = {"Attractive": 1, "Male": 3}.
  - Absent-image sample: drawn from the SAME `non_overlapping` pool
    (5,817 CelebAMask-HQ images falling in official CelebA val+test)
    the present-pairs script's own ground truth (`celeba_official_
    train_faithfulness_non_overlapping.csv`) draws its PRESENT
    candidates from -- matching pools for a fair present-vs-absent
    comparison, not a new/different pool.
  - TCAV exemplar/random/control pools: official CelebA TRAIN
    (matching the present-pairs script's own substitution).
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
from run_cards_celeba_full import CONCEPT_QUERY_TEXT

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.celeba import load_celebamask_hq_image_paths
from cards.data.celeba_attributes import (
    GROUNDABLE_CONCEPTS,
    load_attribute_labels,
    load_attribute_names,
)
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
ALPHA = 1.0
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
N_ABSENT_PER_CONCEPT = 100
N_RANDOM, N_CONTROL, N_PER_RANDOM_SET, N_CONCEPT_EXEMPLARS = 6, 6, 25, 40
HOOK_LAYER = "layer4"
TCAV_DEVICE = "cpu"
TARGET_CLASSES = ["Attractive", "Male"]
TASK_POSITIVE_LOGIT_INDEX = {"Attractive": 1, "Male": 3}


def load_official_attr_names(path: Path) -> list[str]:
    return path.read_text().splitlines()[1].split()


def load_official_attr_labels(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    result: dict[str, np.ndarray] = {}
    for line in lines[2:]:
        if not line.strip():
            continue
        parts = line.split()
        result[parts[0]] = np.array([v == "1" for v in parts[1:]], dtype=bool)
    return result


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

    spec = BACKBONES["celeba_official_train_attractive_male"]
    native_model = spec.load_native().to(DEVICE).eval()
    native_model_cpu = spec.load_native().to(TCAV_DEVICE).eval()

    demeaned_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                        for c in GROUNDABLE_CONCEPTS}
    orthogonalized_queries = orthogonalize_queries(demeaned_queries)

    print("Loading CelebAMask-HQ <-> official-CelebA mapping + official-train metadata...", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    hq_attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    hq_attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")

    hq_to_orig = {}
    with open(CELEBA_HQ_ROOT / "CelebA-HQ-to-CelebA-mapping.txt") as f:
        next(f)
        for line in f:
            parts = line.split()
            hq_to_orig[int(parts[0])] = parts[2]

    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part

    non_overlapping = [i for i in image_paths_by_idx if partition.get(hq_to_orig[i]) in ("1", "2")]
    print(f"non_overlapping absent-sampling pool: {len(non_overlapping)} HQ images "
          f"(same pool the present-pairs ground truth draws PRESENT candidates from)", flush=True)

    attr_names = load_official_attr_names(CELEBA_ROOT / "list_attr_celeba.txt")
    attr_labels = load_official_attr_labels(CELEBA_ROOT / "list_attr_celeba.txt")
    train_files = sorted(f for f, p in partition.items() if p == "0")
    print(f"{len(train_files)} official-train images (TCAV/hybrid fitting resource).", flush=True)

    rng_py = random.Random(SEED)
    all_train_paths = [CELEBA_ROOT / "img_align_celeba" / f for f in train_files]
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

    tcav = TCAV(model=native_model_cpu, layers=[HOOK_LAYER], save_path=str(RESULTS_DIR / "local_attribution_absent_official_train_tcav_cav_cache"))

    per_concept = {}
    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c_hybrid = orthogonalized_queries[concept_name].to(DEVICE)
        concept_attr_idx = attr_names.index(concept_name)
        hq_concept_attr_idx = hq_attr_names.index(concept_name)

        pos_train_paths = [
            CELEBA_ROOT / "img_align_celeba" / f for f in train_files if attr_labels[f][concept_attr_idx]
        ]
        rng_py.shuffle(pos_train_paths)
        sim_maps = []
        for p in pos_train_paths[:50]:
            img = Image.open(p).convert("RGB")
            sim_maps.append(localize_concept(encoder, img, t_c_hybrid, (img.height, img.width)))
        cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

        target_concept = make_tcav_concept(tcav_concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], spec.preprocess)
        tcav_concept_id += 1

        absent_hq_paths = [
            image_paths_by_idx[i] for i in non_overlapping if not hq_attr_labels_by_file[f"{i}.jpg"][hq_concept_attr_idx]
        ]
        rng_py.shuffle(absent_hq_paths)
        absent_sample = absent_hq_paths[:N_ABSENT_PER_CONCEPT]

        per_concept[concept_name] = {
            "t_c_hybrid": t_c_hybrid, "cutoff": cutoff, "target_concept": target_concept,
            "absent_sample": absent_sample,
        }
        print(f"  [{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} cutoff={cutoff:+.4f}  "
              f"absent_pool={len(absent_hq_paths)}  sampled={len(absent_sample)}", flush=True)

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

    out_path = RESULTS_DIR / "local_attribution_celeba_official_train_absent_pairs.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "concept_name", "target_task", "gt_delta_p", "hybrid_score", "tcav_score"])
        writer.writerows(all_out_rows)
    print(f"\nSaved {len(all_out_rows)} absent-image scored rows to {out_path}", flush=True)

    print("\n=== absent-image score summary ===", flush=True)
    for task_name in TARGET_CLASSES:
        task_rows = [r for r in all_out_rows if r[2] == task_name]
        hybrid_scores = [r[4] for r in task_rows]
        tcav_scores = [r[5] for r in task_rows]
        n_hybrid_nonzero = sum(1 for s in hybrid_scores if s != 0.0)
        print(f"  [{task_name}] n={len(task_rows)}  hybrid: mean|score|={np.mean(np.abs(hybrid_scores)):.4f} "
              f"({n_hybrid_nonzero}/{len(hybrid_scores)} nonzero)  "
              f"tcav: mean|score|={np.mean(np.abs(tcav_scores)):.4f}", flush=True)


if __name__ == "__main__":
    main()
