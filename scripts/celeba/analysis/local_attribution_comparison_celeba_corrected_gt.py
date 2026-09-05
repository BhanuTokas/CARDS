"""Re-runs the local (per-image, per-concept) attribution comparison
(`local_attribution_comparison_celeba.py`, v112) against the CORRECTED,
attribute-conditioned ground truth (`celeba_full_faithfulness_attribute_
conditioned.csv` -- both label directions pooled, candidates filtered by
the concept's own attribute value, see `run_celeba_full_faithfulness_
attribute_conditioned.py`'s own docstring), prompted directly ("Can you
run the local attribution with the corrected GT, also rank images by
magnitude, show sign in the actual figure though!").

Two changes from the original script, both requested directly:
1. **Ground truth source**: the corrected CSV instead of the original.
   Since the corrected ground truth now pools BOTH Attractive=True and
   Attractive=False images (limitation 2's fix), a meaningful fraction
   of scored pairs now have a genuinely NEGATIVE gt_delta_p/local score
   (masking a concept INCREASED the target probability) -- ranking by
   raw value alone would only ever surface the positive tail, hiding
   these.
2. **Top-10 ranking by |magnitude|, not raw value** -- surfaces the
   images each method is MOST confident about in EITHER direction, and
   **sign shown directly in the figure** (green score text = positive/
   "helps", red = negative/"hurts") since magnitude-ranking mixes both
   signs together in one row and the sign is exactly the information a
   magnitude-only ranking would otherwise hide.

Top-10 grids show `ground_truth` (real gt_delta_p) instead of
`baseline` -- checks each method against real masking ground truth
directly rather than just method-vs-method (matches the original
script's own convention; this file initially reverted to `baseline` by
oversight, caught directly and fixed: "Shouldn't the images be with gt
not baseline?" -> "Can you also correct the existing code to make sure
you don't repeat the errors again in the future?"). `baseline` is still
scored and reported in the rho table, just not shown in the grid.

Everything else (the 3 methods' own definitions, the TCAV/hybrid/
baseline apparatus, the per-concept setup) is IDENTICAL to the original
script -- see that script's own docstring for the full design rationale.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from captum.concept import TCAV, Concept
from concepts.concept_utils import ListDataset
from run_cards_celeba_full import CONCEPT_QUERY_TEXT, TASK_POSITIVE_LOGIT_INDEX

from cards.attribution.local_mode import local_score
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
from cards.data.datasets import load_celeba
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

CELEBA_HQ_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebAMask-HQ")
RESULTS_DIR = Path("results")
TOP10_DIR = RESULTS_DIR / "local_attribution_top10_corrected_gt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 50
SEED = 42
ALPHA = 1.0
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
CONCEPT_TO_IDX = {name: i for i, name in enumerate(GROUNDABLE_CONCEPTS)}

# TCAV apparatus, matching run_tcav_celeba_full.py exactly
N_RANDOM, N_CONTROL, N_PER_RANDOM_SET, N_CONCEPT_EXEMPLARS = 6, 6, 25, 40
HOOK_LAYER = "layer4"
TCAV_DEVICE = "cpu"


def load_ground_truth_rows() -> list[dict]:
    rows = []
    with open(RESULTS_DIR / "celeba_full_faithfulness_attribute_conditioned.csv", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "image": row["image"],
                "concept_name": row["concept_name"],
                "target_task": row["target_task"],
                "delta_p": float(row["delta_p"]),
            })
    return rows


def make_tcav_concept(concept_id: int, name: str, image_paths: list, preprocess, batch_size: int = 32) -> Concept:
    ds = ListDataset([str(p) for p in image_paths], preprocess=preprocess)
    from torch.utils.data import DataLoader

    return Concept(id=concept_id, name=name, data_iter=DataLoader(ds, batch_size=batch_size, shuffle=False))


def build_image_grid(rows: list[tuple[str, list[tuple[Path, float]]]], out_path: Path, thumb=128) -> None:
    """rows: [(row_label, [(image_path, score), ...]), ...] -- one row per
    method, up to 10 columns, ranked by |score| by the caller. Sign shown
    directly: green score text = positive ("helps"), red = negative
    ("hurts") -- load-bearing here since magnitude-ranking mixes both
    signs into one row, unlike the original script's raw-value ranking."""
    n_cols = max(len(images) for _, images in rows)
    n_rows = len(rows)
    pad, label_w, header_h = 4, 90, 20
    grid = Image.new("RGB", (label_w + n_cols * (thumb + pad), header_h + n_rows * (thumb + pad + header_h)), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for r, (row_label, images) in enumerate(rows):
        y0 = header_h + r * (thumb + pad + header_h)
        draw.text((2, y0), row_label, fill="black", font=font)
        for c, (img_path, score) in enumerate(images):
            x0 = label_w + c * (thumb + pad)
            try:
                thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            except OSError:
                thumb_img = Image.new("RGB", (thumb, thumb), "gray")
            grid.paste(thumb_img, (x0, y0 + header_h))
            sign_color = "lime" if score >= 0 else "red"
            draw.text((x0, y0 + header_h + thumb - 12), f"{score:+.2f}", fill=sign_color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    gt_rows = load_ground_truth_rows()
    print(f"Loaded {len(gt_rows)} raw (image, concept, task) ground-truth rows "
          f"(CORRECTED ground truth).", flush=True)

    # Group by (image, concept) -- collect which task(s) each pair needs.
    pairs: dict[tuple[str, str], dict[str, float]] = {}
    for r in gt_rows:
        key = (r["image"], r["concept_name"])
        pairs.setdefault(key, {})[r["target_task"]] = r["delta_p"]
    print(f"{len(pairs)} unique (image, concept) pairs.", flush=True)

    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
        "cache_dir": "embedding_cache",
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    cfg.dataset = {"name": "celeba", "root": str(CELEBA_HQ_ROOT)}
    cfg.pool_source = "val"
    celeba_pairs = load_celeba(CELEBA_HQ_ROOT, split="val")
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), celeba_pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()
    native_model_cpu = spec.load_native().to(TCAV_DEVICE).eval()

    demeaned_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                        for c in GROUNDABLE_CONCEPTS}
    orthogonalized_queries = orthogonalize_queries(demeaned_queries)

    print("\n=== per-concept setup (N_c, z-score cutoff, TCAV CAVs) -- one-time cost ===", flush=True)
    image_paths_by_idx = load_celebamask_hq_image_paths(CELEBA_HQ_ROOT)
    attr_names = load_attribute_names(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    attr_labels_by_file = load_attribute_labels(CELEBA_HQ_ROOT / "CelebAMask-HQ-attribute-anno.txt")
    target_indices = [attr_names.index(t) for t in TARGET_CLASSES]
    train_hq, _val_hq = split_celebamask_hq(image_paths_by_idx, attr_labels_by_file, target_indices)

    import random

    rng_py = random.Random(SEED)
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

    tcav = TCAV(model=native_model_cpu, layers=[HOOK_LAYER], save_path=str(RESULTS_DIR / "local_attribution_tcav_cav_cache"))

    per_concept = {}
    for c_i, concept_name in enumerate(GROUNDABLE_CONCEPTS):
        t_c_baseline = demeaned_queries[concept_name]
        t_c_hybrid = orthogonalized_queries[concept_name]

        present_indices, _ = retrieve_top_bottom_k(pool, t_c_baseline, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c_baseline, K)
        absent_images = torch.stack([spec.preprocess(Image.open(pool.paths[i]).convert("RGB")) for i in absent_indices]).to(DEVICE)

        hybrid_present_indices, _ = retrieve_top_bottom_k(pool, t_c_hybrid, K)
        sim_maps = []
        for i in hybrid_present_indices:
            img = Image.open(pool.paths[i]).convert("RGB")
            sim_maps.append(localize_concept(encoder, img, t_c_hybrid, (img.height, img.width)))
        cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

        pos_train_paths = [image_paths_by_idx[i] for i in train_hq if attr_labels_by_file[f"{i}.jpg"][attr_names.index(concept_name)]]
        rng_py.shuffle(pos_train_paths)
        target_concept = make_tcav_concept(tcav_concept_id, concept_name, pos_train_paths[:N_CONCEPT_EXEMPLARS], spec.preprocess)
        tcav_concept_id += 1

        per_concept[concept_name] = {
            "t_c_baseline": t_c_baseline, "t_c_hybrid": t_c_hybrid.to(DEVICE),
            "absent_images": absent_images, "cutoff": cutoff, "target_concept": target_concept,
        }
        print(f"  [{c_i + 1:>2d}/{len(GROUNDABLE_CONCEPTS)}] {concept_name:<20s} cutoff={cutoff:+.4f}", flush=True)

    print(f"\n=== scoring {len(pairs)} (image, concept) pairs ===", flush=True)
    all_out_rows = []
    top10_candidates: dict[str, dict[str, list[tuple[str, float]]]] = {
        c: {"ground_truth": [], "baseline": [], "hybrid": [], "tcav": []} for c in GROUNDABLE_CONCEPTS
    }

    for pair_i, ((image_path, concept_name), tasks) in enumerate(pairs.items()):
        cinfo = per_concept[concept_name]
        concept_idx = CONCEPT_TO_IDX[concept_name]
        image = Image.open(image_path).convert("RGB")

        pixels = spec.preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            orig_logits = native_model(pixels)[0]

        baseline_by_task = {}
        for task_name in tasks:
            task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
            def bb(batch, _idx=task_idx):
                return native_model(batch)[:, _idx]
            baseline_by_task[task_name] = local_score(bb, pixels[0], cinfo["absent_images"])

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
            for task_name in tasks:
                task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
                hybrid_by_task[task_name] = (orig_logits[task_idx] - masked_logits[task_idx]).item()
        else:
            for task_name in tasks:
                hybrid_by_task[task_name] = 0.0

        tcav_by_task = {}
        inputs = pixels
        for task_name in tasks:
            task_idx = TASK_POSITIVE_LOGIT_INDEX[task_name]
            experimental_sets = [[cinfo["target_concept"], rc] for rc in random_concepts]
            scores = tcav.interpret(inputs=inputs.to(TCAV_DEVICE), experimental_sets=experimental_sets, target=task_idx)
            sign_counts = [scores[f"{cinfo['target_concept'].id}-{rc.id}"][HOOK_LAYER]["sign_count"][0].item() for rc in random_concepts]
            tcav_by_task[task_name] = float(np.mean(sign_counts))

        for task_name, gt_delta_p in tasks.items():
            all_out_rows.append((image_path, concept_name, task_name, gt_delta_p,
                                  baseline_by_task[task_name], hybrid_by_task[task_name], tcav_by_task[task_name]))
            top10_candidates[concept_name]["ground_truth"].append((image_path, gt_delta_p))
            top10_candidates[concept_name]["baseline"].append((image_path, baseline_by_task[task_name]))
            top10_candidates[concept_name]["hybrid"].append((image_path, hybrid_by_task[task_name]))
            top10_candidates[concept_name]["tcav"].append((image_path, tcav_by_task[task_name]))

        if (pair_i + 1) % 50 == 0 or pair_i == len(pairs) - 1:
            print(f"  [{pair_i + 1}/{len(pairs)}] pairs scored", flush=True)

    with open(RESULTS_DIR / "local_attribution_celeba_pairs_corrected_gt.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "concept_name", "target_task", "gt_delta_p", "baseline_score", "hybrid_score", "tcav_score"])
        writer.writerows(all_out_rows)
    print(f"\nSaved {len(all_out_rows)} scored rows to results/local_attribution_celeba_pairs_corrected_gt.csv", flush=True)

    print("\n=== local rho (n = all scored rows, CORRECTED ground truth) ===", flush=True)
    for task_name in TARGET_CLASSES:
        task_rows = [r for r in all_out_rows if r[2] == task_name]
        gt = [r[3] for r in task_rows]
        for method_i, method_name in [(4, "baseline"), (5, "hybrid"), (6, "tcav")]:
            scores = [r[method_i] for r in task_rows]
            rho, p = spearmanr(gt, scores)
            print(f"  [{task_name}] {method_name:<10s} n={len(task_rows)} rho={rho:+.4f} p={p:.4g}", flush=True)

    # ground_truth (real gt_delta_p) shown instead of baseline -- checks
    # each method against real masking ground truth directly, matching
    # the substitution already applied to the original-ground-truth
    # grids (initially missed here, caught directly: "Shouldn't the
    # images be with gt not baseline?").
    print("\n=== building top-10-by-|magnitude| grids per concept (ground_truth/hybrid/tcav, sign shown) ===", flush=True)
    for concept_name in GROUNDABLE_CONCEPTS:
        rows_for_grid = []
        for method_name in ["ground_truth", "hybrid", "tcav"]:
            ranked = sorted(top10_candidates[concept_name][method_name], key=lambda kv: -abs(kv[1]))[:10]
            rows_for_grid.append((method_name, ranked))
        build_image_grid(rows_for_grid, TOP10_DIR / f"{concept_name}.png")
    print(f"Saved top-10 grids to {TOP10_DIR}/", flush=True)


if __name__ == "__main__":
    main()
