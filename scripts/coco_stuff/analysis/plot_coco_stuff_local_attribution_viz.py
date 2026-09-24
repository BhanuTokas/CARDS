"""Local attribution visualization for COCO-Stuff.

For 3 specific (concept, class) pairs, scores ALL class-present COCO train2017
images with Hide and Seek and TCAV, then saves top-3 / bottom-3 image grids.

Additionally, finds the TOP-5 most-common classes (by image count in
train2017 -- extended from just the single dominant class, prompted
directly: "Can we update section 2 to generate for top-5 classes?"),
scores ALL 103 balanced concepts against EACH of those 5 classes with
BOTH Hide and Seek and TCAV -- prompted directly ("can you help me add
TCAV analysis for dominant class in this code?" -> "Can you make the
optimization and give me the complete code?") -- and saves 5x103=515
grids.

VIZ_CASES (Hide and Seek + TCAV):
  (greenness, sports ball)
  (blueness,  sports ball)
  (bottle,    wine glass)

Outputs:
  results/coco_stuff/local_attribution_viz/{concept}_{class}.png             (3 files)
  results/coco_stuff/local_attribution_viz/top5_classes/{concept}_{cls}.png  (515 files,
      103 concepts x 5 classes -- flat directory, filenames already unique
      per (concept, class) pair, no per-class subdirectories needed)

Each VIZ_CASES PNG: 4 rows -- Hide and Seek top3/bottom3, TCAV top3/bottom3.
Each top5_classes PNG: 4 rows -- Hide and Seek top3/bottom3, TCAV top3/bottom3.

**Top-3/bottom-3, not top-5/bottom-5** -- matches the CelebA local-
attribution grids' own established convention, confirmed directly
("Wait, should it not be top-3 and bottom-3?" -> "Yes, switch to
top-3/bottom-3 for consistency"). `TOP_BOTTOM_N` below is the single
source of truth for both sections' slicing.

**Method renamed "CARDS"/"ConceptMask" -> "Hide and Seek"** throughout
this file's rendered grid labels (internal dict keys/variable names
still say "cards" -- only what's actually drawn on the grids changed),
prompted directly ("I changed the name to Hide and Seek, so the name
of the technique needs to be updated").

**Single-class TCAV optimization**: `_tcav_per_batch` used to backprop through
all 20 classes' logits every call, even though every caller here only ever
reads out ONE class (`cls_i`) afterward -- 19/20 of that work was pure waste.
Confirmed directly (both call sites -- VIZ_CASES' per-(concept,class) loop and
the dominant-class loop -- always pass a single fixed `cls_i`, never sweep
multiple classes per call), so `_tcav_per_batch` now takes `cls_i` directly
and backprops ONLY that class's logit sum, returning a 1D sign array instead
of the old [N, 20] array. This matters far more for the dominant-class section
(103 concepts x every dominant-class image) than it did for VIZ_CASES (3
concepts), which is why it's being made now rather than left as-is.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from collections import defaultdict

import numpy as np
import torch
from cards.data.coco_stuff import COCO_STUFF_20_CLASSES
from cards.models.coco_stuff_black_box import _coco_stuff_preprocess
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from sklearn.svm import LinearSVC

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.datasets import list_broden_concepts, load_broden, load_coco_stuff
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k
from cards.validation.broden_faithfulness import mask_region

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COCO_ROOT   = Path("/scratch/rnair21/LLMConceptEditing_old/coco")
BRODEN_ROOT = Path("/scratch/rnair21/LLMConceptEditing_old/broden_concepts")
CKPT_PATH   = Path("trained_models_new/coco_stuff/resnet18_coco_stuff.pt")
RESULTS_DIR = Path("results/coco_stuff")
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
VIZ_OUT_DIR = RESULTS_DIR / "local_attribution_viz"

BRODEN_DROPPED_CONCEPTS: set[str] = {
    "air_conditioner", "apron", "awning", "bathroom_s", "beak", "bedroom_s",
    "blotchy", "bumper", "can", "candlestick", "cap", "clock", "dining_room_s",
    "drinking_glass", "eye", "eyebrow", "faucet", "figurine", "handle",
    "inside_arm", "jar", "knob", "loudspeaker", "mouse", "mouth",
    "outside_arm", "street_s",
}

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
K           = 50
ALPHA       = 1.5
SEED        = 42
N_CAV_POS   = 40
N_CAV_NEG   = 25
TCAV_BATCH  = 32  # images per gradient-computation batch
TOP_BOTTOM_N = 3  # matches the CelebA local-attribution grids' own convention

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]

# (concept_name, class_name) pairs — scored with both CARDS and TCAV
VIZ_CASES: list[tuple[str, str]] = [
    ("greenness", "sports ball"),
    ("blueness",  "sports ball"),
    ("bottle",    "wine glass"),
]

# Display label for the masking-hybrid method's own grid rows -- renamed
# "CARDS"/"ConceptMask" -> "Hide and Seek" (prompted directly: "I changed
# the name to Hide and Seek, so the name of the technique needs to be
# updated"). `viz_candidates` still keys internally on "cards" (unchanged,
# just an internal dict key, not shown anywhere) -- only the RENDERED grid
# label goes through this map.
METHOD_DISPLAY_NAMES = {"cards": "Hide and Seek (Ours)", "tcav": "TCAV"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_balanced_concepts(counts_csv: Path, min_images: int = 100) -> set[str]:
    balanced: set[str] = set()
    with open(counts_csv, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["n_images"]) >= min_images:
                balanced.add(row["concept_name"])
    return balanced


def _build_class_to_paths(coco_root: Path, split: str) -> dict[int, list[Path]]:
    """Parse COCO annotations multi-label: class_idx → sorted list of image paths.

    Unlike load_coco_stuff, which assigns each image only its dominant (lowest-index)
    class, this captures every class present in each image.
    """
    ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
    with open(ann_path) as f:
        data = json.load(f)

    cat_name_to_id = {cat["name"]: cat["id"] for cat in data["categories"]}
    cat_id_to_class_idx: dict[int, int] = {}
    for idx, cls in enumerate(COCO_STUFF_20_CLASSES):
        coco_id = cat_name_to_id.get(cls)
        if coco_id is not None:
            cat_id_to_class_idx[coco_id] = idx

    img_id_to_class_idxs: dict[int, set[int]] = defaultdict(set)
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in cat_id_to_class_idx:
            img_id_to_class_idxs[ann["image_id"]].add(cat_id_to_class_idx[cat_id])

    img_id_to_path = {
        img["id"]: coco_root / f"{split}2017" / img["file_name"]
        for img in data["images"]
    }

    class_to_paths: dict[int, list[Path]] = {i: [] for i in range(len(COCO_STUFF_20_CLASSES))}
    for img_id, class_idxs in img_id_to_class_idxs.items():
        path = img_id_to_path.get(img_id)
        if path is None:
            continue
        for ci in class_idxs:
            class_to_paths[ci].append(path)

    for ci in class_to_paths:
        class_to_paths[ci].sort()

    return class_to_paths


# ---------------------------------------------------------------------------
# TCAV helpers
# ---------------------------------------------------------------------------

def _layer4_activations(
    images: list[Image.Image],
    model: torch.nn.Module,
    preprocess,
    device: str,
) -> np.ndarray:
    if not images:
        return np.zeros((0, 512), dtype=np.float32)
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    feats: list[torch.Tensor] = []
    hook = model.layer4.register_forward_hook(
        lambda m, i, o: feats.append(o.detach().mean(dim=[2, 3]).cpu())
    )
    with torch.no_grad():
        model(tensors)
    hook.remove()
    return feats[0].numpy()


def _fit_cav(
    pos_images: list[Image.Image],
    neg_images: list[Image.Image],
    model: torch.nn.Module,
    preprocess,
    device: str,
) -> np.ndarray | None:
    pos_feats = _layer4_activations(pos_images, model, preprocess, device)
    neg_feats = _layer4_activations(neg_images, model, preprocess, device)
    if len(pos_feats) < 2 or len(neg_feats) < 2:
        return None
    X = np.concatenate([pos_feats, neg_feats], axis=0)
    y = np.array([1] * len(pos_feats) + [0] * len(neg_feats))
    try:
        svm = LinearSVC(C=0.01, max_iter=1000, dual=True)
        svm.fit(X, y)
        cav = svm.coef_[0]
        norm = np.linalg.norm(cav)
        return (cav / norm) if norm > 1e-8 else None
    except Exception:
        return None


def _tcav_per_batch(
    test_tensors: torch.Tensor,
    model: torch.nn.Module,
    cav_weights: np.ndarray,
    device: str,
    cls_i: int,
) -> np.ndarray:
    """Returns sign array [N] in {-1, 0, +1} for ONE target class -- only
    backprops through that class's logit (see module docstring's own
    "Single-class TCAV optimization" note for why: every caller only ever
    wants one class per call, so the original all-20-classes sweep wasted
    19/20 of its backward passes)."""
    cav_t = torch.tensor(cav_weights, dtype=torch.float32).to(device)
    h_store: list[torch.Tensor] = []

    def fwd_hook(m, i, o):
        o.retain_grad()
        h_store.clear()
        h_store.append(o)

    handle = model.layer4.register_forward_hook(fwd_hook)
    with torch.enable_grad():
        logits = model(test_tensors)
    handle.remove()

    h_l = h_store[0]
    logits[:, cls_i].sum().backward()
    if h_l.grad is None:
        return np.zeros(test_tensors.shape[0], dtype=np.float32)
    grad_flat = h_l.grad.mean(dim=[2, 3])
    return torch.sign(grad_flat @ cav_t).cpu().numpy()


def _tcav_all_images(
    paths: list[str],
    model: torch.nn.Module,
    cav_weights: np.ndarray,
    preprocess,
    device: str,
    cls_i: int,
) -> list[float]:
    """TCAV sign for cls_i across all paths, batched to avoid OOM."""
    results: list[float] = []
    for start in range(0, len(paths), TCAV_BATCH):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[start:start + TCAV_BATCH]]
        batch_t    = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        signs      = _tcav_per_batch(batch_t, model, cav_weights, device, cls_i)
        results.extend(float(s) for s in signs)
    return results


# ---------------------------------------------------------------------------
# CARDS scoring (single image)
# ---------------------------------------------------------------------------

def _cards_score(
    img_path: str,
    concept_idx: int,
    img_i: int,
    encoder,
    model: torch.nn.Module,
    preprocess,
    t_c: torch.Tensor,
    t_c_dev: torch.Tensor,
    cutoff: float,
    cls_i: int,
    device: str,
) -> float:
    image   = Image.open(img_path).convert("RGB")
    sim_map = localize_concept(encoder, image, t_c, (image.height, image.width))
    mask    = threshold_mask(sim_map, method="fixed", cutoff=cutoff)

    if not mask.any() or mask.all():
        return 0.0

    img_rng    = np.random.default_rng(SEED + concept_idx * 10_000 + img_i)
    candidates = [mask_region(image, mask, strategy=s, rng=img_rng) for s in FILL_STRATEGIES]

    with torch.no_grad():
        embeds = encoder.encode_images([image] + candidates).to(device)
    embed_orig = embeds[0]

    best_angle, best_i = None, 0
    for fi in range(len(FILL_STRATEGIES)):
        diff      = embed_orig - embeds[1 + fi]
        diff_unit = diff / (diff.norm() + 1e-8)
        cos_sim   = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))
        if best_angle is None or angle_deg < best_angle:
            best_angle, best_i = angle_deg, fi

    pix_orig   = preprocess(image).unsqueeze(0)
    pix_masked = preprocess(candidates[best_i]).unsqueeze(0)
    with torch.no_grad():
        logits_pair = model(torch.cat([pix_orig, pix_masked]).to(device))
    return (logits_pair[0, cls_i] - logits_pair[1, cls_i]).item()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def build_image_grid(
    rows: list[tuple[str, list[tuple[str, float]]]],
    out_path: Path,
    thumb: int = 128,
) -> None:
    n_cols   = max(len(imgs) for _, imgs in rows)
    pad      = 4
    label_w  = 110
    header_h = 20
    w = label_w + n_cols * (thumb + pad)
    h = header_h + len(rows) * (thumb + pad + header_h)
    grid = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for r, (row_label, imgs) in enumerate(rows):
        y0 = header_h + r * (thumb + pad + header_h)
        draw.text((2, y0), row_label, fill="black", font=font)
        for c, (img_path, score) in enumerate(imgs):
            x0 = label_w + c * (thumb + pad)
            try:
                thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
            except OSError:
                thumb_img = Image.new("RGB", (thumb, thumb), "gray")
            grid.paste(thumb_img, (x0, y0 + header_h))
            color = "lime" if score >= 0 else "red"
            draw.text((x0, y0 + header_h + thumb - 12), f"{score:+.2f}", fill=color, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from torch import nn
    from torchvision.models import resnet18

    VIZ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # --- Model ---
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(COCO_STUFF_20_CLASSES))
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    model = model.to(DEVICE).eval()
    preprocess = _coco_stuff_preprocess()

    # --- Encoder + val pool ---
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {
            "name": "siglip",
            "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
            "model_name": "ViT-B-16-SigLIP",
            "pretrained": "webli",
            "device": DEVICE,
        },
        "cache_dir": "embedding_cache",
    })
    encoder     = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    cfg.dataset    = {"name": "coco_stuff", "root": str(COCO_ROOT)}
    cfg.pool_source = "val"
    val_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="val")
    val_pool  = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), val_pairs, encoder)
    print(f"Val pool: {len(val_pool.paths)} images", flush=True)

    # --- Load train pairs (for all_train_paths used by CAV neg pool) ---
    train_pairs = load_coco_stuff(COCO_ROOT, COCO_STUFF_20_CLASSES, split="train")
    print(f"Train pairs loaded: {len(train_pairs)}", flush=True)

    # --- Multi-label class → paths mapping for train split ---
    print("Building multi-label class→paths index for train2017...", flush=True)
    class_to_train_paths = _build_class_to_paths(COCO_ROOT, "train")

    # --- Collect class-present paths for VIZ_CASES ---
    class_to_concepts: dict[str, list[str]] = defaultdict(list)
    for concept, cls in VIZ_CASES:
        class_to_concepts[cls].append(concept)

    class_test_paths: dict[str, list[str]] = {}
    all_test_path_set: set[str] = set()
    for cls_name in class_to_concepts:
        cls_idx = COCO_STUFF_20_CLASSES.index(cls_name)
        present = [str(p) for p in class_to_train_paths[cls_idx]]
        class_test_paths[cls_name] = present
        all_test_path_set.update(present)
        print(f"VIZ class '{cls_name}': {len(present)} images", flush=True)

    # --- Find TOP-5 most-common classes for the dominant-class section --
    # extended from the single dominant class to the top 5 by image count,
    # prompted directly ("Can we update section 2 to generate for top-5
    # classes?"). Each entry: (count, cls_name, cls_i).
    class_image_counts = [
        (len(class_to_train_paths[ci]), cls_name, ci)
        for ci, cls_name in enumerate(COCO_STUFF_20_CLASSES)
    ]
    top5_classes = sorted(class_image_counts, reverse=True)[:5]
    top5_class_paths: dict[str, list[str]] = {
        cls_name: [str(p) for p in class_to_train_paths[cls_i]] for _count, cls_name, cls_i in top5_classes
    }
    print("Top-5 classes by train2017 image count:", flush=True)
    for count, cls_name, _cls_i in top5_classes:
        print(f"  '{cls_name}': {count} images", flush=True)

    # --- Balanced 103 concepts (same order as orthogonalization) ---
    balanced        = _load_balanced_concepts(COUNTS_CSV)
    all_concepts    = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    scored_concepts = [c for c in all_concepts if c in balanced]  # 103, deterministic order
    concept_idx_map = {c: i for i, c in enumerate(all_concepts)}  # stable index for RNG seeding
    print(f"All concepts: {len(all_concepts)}  balanced: {len(scored_concepts)}", flush=True)

    # --- CAV negative pool (VIZ_CASES only; drawn from non-test train images) ---
    all_train_paths = [str(p) for p, _ in train_pairs]
    non_test        = [p for p in all_train_paths if p not in all_test_path_set]
    perm_neg        = rng.permutation(len(non_test))
    cav_neg_pool    = [non_test[i] for i in perm_neg[: len(VIZ_CASES) * N_CAV_NEG]]

    # --- CAV negative pools for the dominant-class section, ONE PER TOP-5 CLASS
    # (separate pool/exclusion from VIZ_CASES' own and from each other -- each
    # excludes THAT class's own images specifically, sized for all 103 concepts) ---
    dom_cav_neg_pools: dict[str, list[str]] = {}
    for cls_name, cls_paths in top5_class_paths.items():
        cls_test_path_set = set(cls_paths)
        cls_non_test = [p for p in all_train_paths if p not in cls_test_path_set]
        perm_neg_cls = rng.permutation(len(cls_non_test))
        dom_cav_neg_pools[cls_name] = [cls_non_test[i] for i in perm_neg_cls[: len(scored_concepts) * N_CAV_NEG]]
        print(f"'{cls_name}' CAV negative pool: {len(dom_cav_neg_pools[cls_name])} images "
              f"({len(scored_concepts)} concepts x {N_CAV_NEG})", flush=True)

    # --- Orthogonalize over all 143 concepts ---
    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in all_concepts
    }
    queries = orthogonalize_queries(raw_queries)
    print(f"Queries orthogonalized over {len(all_concepts)} concepts", flush=True)

    # ===========================================================================
    # Section 1: VIZ_CASES — all class-present images, CARDS + TCAV
    # ===========================================================================
    viz_candidates: dict[tuple[str, str], dict[str, list[tuple[str, float]]]] = {
        case: {"cards": [], "tcav": []} for case in VIZ_CASES
    }

    cav_slot = 0  # which slot of cav_neg_pool to use next

    for cls_name, concept_list in class_to_concepts.items():
        test_paths = class_test_paths[cls_name]
        n_test     = len(test_paths)
        cls_i      = COCO_STUFF_20_CLASSES.index(cls_name)
        print(f"\n=== VIZ class='{cls_name}'  n={n_test} ===", flush=True)

        for concept_name in concept_list:
            t_c     = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)
            cidx    = concept_idx_map[concept_name]
            print(f"  concept={concept_name}", flush=True)

            # Calibrate cutoff from val present images
            present_idx, _ = retrieve_top_bottom_k(val_pool, t_c, K)
            val_sim_maps = []
            for idx in present_idx:
                img     = Image.open(val_pool.paths[idx]).convert("RGB")
                sim_map = localize_concept(encoder, img, t_c, (img.height, img.width))
                val_sim_maps.append(sim_map)
            cutoff = concept_zscore_cutoff(val_sim_maps, ALPHA)

            # Fit CAV for TCAV
            try:
                pos_paths_all, _ = load_broden(BRODEN_ROOT, concept_name)
            except ValueError:
                pos_paths_all = []
            rng_cav   = np.random.default_rng(SEED + cav_slot)
            pos_paths = sorted(pos_paths_all)
            rng_cav.shuffle(pos_paths)
            pos_paths = pos_paths[:N_CAV_POS]
            neg_slice = cav_neg_pool[cav_slot * N_CAV_NEG : (cav_slot + 1) * N_CAV_NEG]
            cav       = _fit_cav(
                [Image.open(p).convert("RGB") for p in pos_paths],
                [Image.open(p).convert("RGB") for p in neg_slice],
                model, preprocess, DEVICE,
            )
            cav_slot += 1

            # TCAV: batched over all test images
            if cav is not None:
                tcav_scores = _tcav_all_images(test_paths, model, cav, preprocess, DEVICE, cls_i)
            else:
                tcav_scores = [0.0] * n_test
                print("    CAV fit failed — TCAV scores set to 0", flush=True)

            # CARDS: per image, lazy load
            cards_entries: list[tuple[str, float]] = []
            for img_i, img_path in enumerate(test_paths):
                score = _cards_score(
                    img_path, cidx, img_i,
                    encoder, model, preprocess,
                    t_c, t_c_dev, cutoff, cls_i, DEVICE,
                )
                cards_entries.append((img_path, np.abs(score)))
                if (img_i + 1) % 100 == 0:
                    print(f"    CARDS {img_i + 1}/{n_test}", flush=True)

            viz_candidates[(concept_name, cls_name)]["cards"] = cards_entries
            viz_candidates[(concept_name, cls_name)]["tcav"]  = list(zip(test_paths, np.abs(tcav_scores)))
            print(f"    Done: {n_test} images", flush=True)

    # Build VIZ_CASES grids
    print("\n=== Building VIZ_CASES grids ===", flush=True)
    for (concept_name, cls_name), method_lists in viz_candidates.items():
        grid_rows = []
        for method, entries in method_lists.items():
            method_label = METHOD_DISPLAY_NAMES[method]
            sd = sorted(entries, key=lambda x: x[1], reverse=True)
            grid_rows.append((f"{method_label} top{TOP_BOTTOM_N}",    sd[:TOP_BOTTOM_N]))
            grid_rows.append((f"{method_label} bottom{TOP_BOTTOM_N}", sd[-TOP_BOTTOM_N:][::-1]))
        safe = f"{concept_name}_{cls_name.replace(' ', '_')}"
        out_path = VIZ_OUT_DIR / f"{safe}.png"
        build_image_grid(grid_rows, out_path)
        print(f"  Saved {out_path}", flush=True)

    # ===========================================================================
    # Section 2: TOP-5 classes × 103 balanced concepts — Hide and Seek + TCAV
    # ===========================================================================
    dom_out_dir = VIZ_OUT_DIR / "top5_classes"
    dom_out_dir.mkdir(parents=True, exist_ok=True)

    for class_rank, (count, cls_name, cls_i) in enumerate(top5_classes):
        cls_paths = top5_class_paths[cls_name]
        n_cls     = len(cls_paths)
        cls_neg_pool = dom_cav_neg_pools[cls_name]
        print(
            f"\n=== [{class_rank + 1}/5] class='{cls_name}'  n={n_cls}  concepts={len(scored_concepts)} ===",
            flush=True,
        )

        for c_i, concept_name in enumerate(scored_concepts):
            t_c     = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)
            cidx    = concept_idx_map[concept_name]

            # Calibrate cutoff from val present images
            present_idx, _ = retrieve_top_bottom_k(val_pool, t_c, K)
            val_sim_maps = []
            for idx in present_idx:
                img     = Image.open(val_pool.paths[idx]).convert("RGB")
                sim_map = localize_concept(encoder, img, t_c, (img.height, img.width))
                val_sim_maps.append(sim_map)
            cutoff = concept_zscore_cutoff(val_sim_maps, ALPHA)

            # Fit CAV for TCAV (own seed offset per (class, concept) -- distinct
            # from VIZ_CASES' own SEED + cav_slot range of 0-2, and distinct
            # per class here too, so no two (class, concept) cells draw the
            # same "random" CAV negative/positive shuffle by coincidence)
            try:
                pos_paths_all, _ = load_broden(BRODEN_ROOT, concept_name)
            except ValueError:
                pos_paths_all = []
            rng_cav   = np.random.default_rng(SEED + 1_000_000 + class_rank * 1_000_000 + c_i)
            pos_paths = sorted(pos_paths_all)
            rng_cav.shuffle(pos_paths)
            pos_paths = pos_paths[:N_CAV_POS]
            neg_slice = cls_neg_pool[c_i * N_CAV_NEG : (c_i + 1) * N_CAV_NEG]
            cav       = _fit_cav(
                [Image.open(p).convert("RGB") for p in pos_paths],
                [Image.open(p).convert("RGB") for p in neg_slice],
                model, preprocess, DEVICE,
            )

            # TCAV: batched over all class-present images (single-class optimized --
            # see module docstring)
            if cav is not None:
                tcav_scores = _tcav_all_images(cls_paths, model, cav, preprocess, DEVICE, cls_i)
            else:
                tcav_scores = [0.0] * n_cls
                print(f"    CAV fit failed for {concept_name} — TCAV scores set to 0", flush=True)

            # Hide and Seek for every class-present image
            dom_entries: list[tuple[str, float]] = []
            for img_i, img_path in enumerate(cls_paths):
                score = _cards_score(
                    img_path, cidx, img_i,
                    encoder, model, preprocess,
                    t_c, t_c_dev, cutoff, cls_i, DEVICE,
                )
                dom_entries.append((img_path, np.abs(score)))

            tcav_entries = list(zip(cls_paths, np.abs(tcav_scores)))

            sd_cards  = sorted(dom_entries, key=lambda x: x[1], reverse=True)
            sd_tcav   = sorted(tcav_entries, key=lambda x: x[1], reverse=True)
            grid_rows = [
                (f"{METHOD_DISPLAY_NAMES['cards']} top{TOP_BOTTOM_N}",    sd_cards[:TOP_BOTTOM_N]),
                (f"{METHOD_DISPLAY_NAMES['cards']} bottom{TOP_BOTTOM_N}", sd_cards[-TOP_BOTTOM_N:][::-1]),
                (f"{METHOD_DISPLAY_NAMES['tcav']} top{TOP_BOTTOM_N}",     sd_tcav[:TOP_BOTTOM_N]),
                (f"{METHOD_DISPLAY_NAMES['tcav']} bottom{TOP_BOTTOM_N}",  sd_tcav[-TOP_BOTTOM_N:][::-1]),
            ]
            safe     = f"{concept_name}_{cls_name.replace(' ', '_')}"
            out_path = dom_out_dir / f"{safe}.png"
            build_image_grid(grid_rows, out_path)
            print(
                f"  [{c_i + 1:>3d}/{len(scored_concepts)}] {concept_name:<25s} -> {out_path.name}",
                flush=True,
            )


if __name__ == "__main__":
    main()
