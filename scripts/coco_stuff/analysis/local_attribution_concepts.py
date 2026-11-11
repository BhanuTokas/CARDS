"""Local attribution visualization for ONE Broden concept against ALL 20
COCO-Stuff classes -- a concept-indexed counterpart to `plot_coco_stuff_
local_attribution_viz.py`'s Section 2 (which instead fixes the TOP-5
classes and sweeps all 103 balanced concepts). This script fixes ONE
concept (by index into the same 103 balanced concepts, same deterministic
order) and sweeps ALL 20 classes, so it can be run as one job per concept
(e.g. a SLURM array job indexed 0-102) instead of one large serial run.

For the selected concept, scores up to MAX_IMAGES_PER_CLASS class-present
train2017 images per class with Hide and Seek and TCAV, and saves one
top-3/bottom-3 grid per class (20 grids per run).

Self-contained (helpers duplicated from `plot_coco_stuff_local_attribution_
viz.py` rather than imported) -- this file and that one may not always be
checked out together in every deployment of this codebase, and every other
single-purpose script in this project follows the same no-cross-file-
import convention for exactly that reason.

Concept selection: set CONCEPT_INDEX below, or override via the
COCO_CONCEPT_INDEX environment variable (e.g. $SLURM_ARRAY_TASK_ID) for
array-job sharding -- one job per concept index, 0 to len(scored_concepts)-1.

Outputs:
  results/coco_stuff/local_attribution_viz/by_concept/{concept_name}/{class_name}.png
      (20 files per run, one per COCO-Stuff class)

Each PNG: 2 rows -- Hide and Seek top-3/bottom-3, TCAV top-3/bottom-3.
"""

from __future__ import annotations

import csv
import json
import os
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
FINAL_DIR   = Path("/data/hkerner/btokas/CARDS/results/coco_stuff")  # HPC cluster final output dir
COUNTS_CSV  = RESULTS_DIR / "coco_stuff_faithfulness_detection_counts.csv"
VIZ_OUT_DIR = FINAL_DIR / "local_attribution_viz" / "by_concept"

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
MAX_IMAGES_PER_CLASS = 500  # cap on class-present images scored per class

FILL_STRATEGIES = [
    "blur", "zero_fill", "mean_fill", "hue_shift",
    "white_fill", "zero_fill_noise", "noise_then_blur",
]

# Which of the 103 balanced concepts (0-indexed, same deterministic order
# as plot_coco_stuff_local_attribution_viz.py's Section 2) this run scores
# against all 20 classes. COCO_CONCEPT_INDEX env var overrides this for
# array-job sharding (e.g. set it to $SLURM_ARRAY_TASK_ID).
CONCEPT_INDEX = int(os.environ.get("COCO_CONCEPT_INDEX", "0"))

METHOD_DISPLAY_NAMES = {"cards": "Hide and Seek (Ours)", "tcav": "TCAV"}


# ---------------------------------------------------------------------------
# Grid renderer -- ported verbatim from `scripts/celeba/analysis/
# local_attribution_plot_utils.build_presence_grid_clean` so these grids
# match CelebA's own local-attribution grids exactly. Only the font path
# differs (Linux path for this project's HPC cluster vs. CelebA's own
# Windows dev-machine path). Verify _GRID_FONT_REGULAR/_GRID_FONT_BOLD
# actually exist on the target machine before trusting this -- `_grid_font`
# falls back to PIL's tiny default bitmap font silently on OSError if the
# path is wrong.
# ---------------------------------------------------------------------------
_GRID_FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_GRID_FONT_REGULAR = _GRID_FONT_DIR / "DejaVuSans.ttf"
_GRID_FONT_BOLD = _GRID_FONT_DIR / "DejaVuSans-Bold.ttf"

_GRID_BG = (250, 250, 250)
_GRID_PANEL_BG = (255, 255, 255)
_GRID_LABEL_COLOR = (30, 30, 30)
_GRID_DIVIDER = (0, 0, 0)
_GRID_NEUTRAL_BORDER = (60, 60, 60)


def _grid_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _paste_bordered_thumb(
    grid: Image.Image, draw: ImageDraw.ImageDraw, img_path: str,
    x0: int, y0: int, thumb: int, border: int, color: tuple[int, int, int],
) -> None:
    draw.rectangle([x0, y0, x0 + thumb + 2 * border - 1, y0 + thumb + 2 * border - 1], outline=color, width=border)
    try:
        thumb_img = Image.open(img_path).convert("RGB").resize((thumb, thumb))
    except OSError:
        thumb_img = Image.new("RGB", (thumb, thumb), "gray")
    grid.paste(thumb_img, (x0 + border, y0 + border))


def build_presence_grid_clean(
    rows: list[tuple[str, list[tuple[str, bool]], list[tuple[str, bool]]]],
    out_path: Path,
    thumb: int = 150,
    n_per_block: int = 3,
) -> None:
    """rows: [(row_label, top_n[(image, is_present)], bottom_n[(image,
    is_present)]), ...] -- one row per method, Top-N/Bottom-N blocks side
    by side separated by a vertical divider. Border color is neutral gray
    for every thumbnail; no numeric score shown."""
    n_rows = len(rows)
    left_margin, pad, border = 10, 3, 4
    right_margin = left_margin  # symmetric outer margins -- see div_x note below
    line1_h, line2_h = 40, 32
    row_h_label = line1_h + line2_h
    cell_h = thumb + 2 * border
    block_w = n_per_block * (thumb + 2 * border + pad)
    divider_w = 26
    row_block_h = row_h_label + cell_h + pad

    width = left_margin + 2 * block_w + divider_w + right_margin
    height = n_rows * row_block_h + pad

    grid = Image.new("RGB", (width, height), _GRID_BG)
    draw = ImageDraw.Draw(grid)

    row_font = _grid_font(_GRID_FONT_BOLD, 30)
    sub_font = _grid_font(_GRID_FONT_REGULAR, 28)

    bottom_x0_base = left_margin + block_w + divider_w

    for r, (row_label, top_n, bottom_n) in enumerate(rows):
        y0 = r * row_block_h
        draw.rectangle([0, y0, width, y0 + row_block_h - pad], fill=_GRID_PANEL_BG if r % 2 == 0 else _GRID_BG)
        draw.text((left_margin, y0 + 4), row_label, fill=_GRID_LABEL_COLOR, font=row_font)

        sub_y = y0 + line1_h + 2
        cell_y0 = y0 + row_h_label

        draw.text((left_margin, sub_y), "Top 3", fill=(90, 90, 90), font=sub_font)
        for c, (img_path, _is_present) in enumerate(top_n):
            x0 = left_margin + c * (thumb + 2 * border + pad)
            _paste_bordered_thumb(grid, draw, img_path, x0, cell_y0, thumb, border, _GRID_NEUTRAL_BORDER)

        # Centered in the gap between the two image blocks, minus a small
        # correction for the thick (width=6) line not drawing symmetrically
        # around the given x.
        div_x = left_margin + block_w + divider_w // 2 - 2
        draw.line([(div_x, y0), (div_x, y0 + row_block_h - pad)], fill=_GRID_DIVIDER, width=6)

        draw.text((bottom_x0_base, sub_y), "Bottom 3", fill=(90, 90, 90), font=sub_font)
        for c, (img_path, _is_present) in enumerate(bottom_n):
            x0 = bottom_x0_base + c * (thumb + 2 * border + pad)
            _paste_bordered_thumb(grid, draw, img_path, x0, cell_y0, thumb, border, _GRID_NEUTRAL_BORDER)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)


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
    """Returns the raw (signed, continuous) grad-dot-CAV array [N] for ONE
    target class -- only backprops through that class's logit. Deliberately
    NOT torch.sign(...)'d: every caller here ranks images by np.abs(...) of
    this output, and the sign-only version collapses almost every image to
    a tied value, degenerating the ranking into stable-sort tie-breaking on
    image order rather than a real per-image magnitude."""
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
    return (grad_flat @ cav_t).cpu().numpy()


def _tcav_all_images(
    paths: list[str],
    model: torch.nn.Module,
    cav_weights: np.ndarray,
    preprocess,
    device: str,
    cls_i: int,
) -> list[float]:
    """TCAV score for cls_i across all paths, batched to avoid OOM."""
    results: list[float] = []
    for start in range(0, len(paths), TCAV_BATCH):
        batch_imgs = [Image.open(p).convert("RGB") for p in paths[start:start + TCAV_BATCH]]
        batch_t    = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        scores     = _tcav_per_batch(batch_t, model, cav_weights, device, cls_i)
        results.extend(float(s) for s in scores)
    return results


# ---------------------------------------------------------------------------
# Hide and Seek scoring (single image)
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from torch import nn
    from torchvision.models import resnet18

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
    all_train_paths = [str(p) for p, _ in train_pairs]

    # --- Multi-label class → paths mapping for train split ---
    print("Building multi-label class→paths index for train2017...", flush=True)
    class_to_train_paths = _build_class_to_paths(COCO_ROOT, "train")

    # --- Balanced 103 concepts (same deterministic order as the other
    # COCO-Stuff local-attribution scripts, so CONCEPT_INDEX means the
    # same concept everywhere) ---
    balanced        = _load_balanced_concepts(COUNTS_CSV)
    all_concepts    = [c for c in list_broden_concepts(BRODEN_ROOT) if c not in BRODEN_DROPPED_CONCEPTS]
    scored_concepts = [c for c in all_concepts if c in balanced]  # 103, deterministic order
    concept_idx_map = {c: i for i, c in enumerate(all_concepts)}  # stable index for RNG seeding
    print(f"All concepts: {len(all_concepts)}  balanced: {len(scored_concepts)}", flush=True)

    if not (0 <= CONCEPT_INDEX < len(scored_concepts)):
        raise ValueError(f"COCO_CONCEPT_INDEX={CONCEPT_INDEX} out of range [0, {len(scored_concepts)})")
    concept_name = scored_concepts[CONCEPT_INDEX]
    cidx = concept_idx_map[concept_name]
    print(f"Concept [{CONCEPT_INDEX}/{len(scored_concepts)}]: '{concept_name}'", flush=True)

    # --- Orthogonalize over all 143 concepts (needed even though only ONE
    # concept is scored here -- the orthogonalized direction for that
    # concept depends on the full concept bank it's orthogonalized against,
    # matching every other script in this project) ---
    raw_queries = {
        c: demean_query(build_concept_query(c, encoder), text_center)
        for c in all_concepts
    }
    queries = orthogonalize_queries(raw_queries)
    t_c     = queries[concept_name]
    t_c_dev = t_c.to(DEVICE)
    print(f"Queries orthogonalized over {len(all_concepts)} concepts", flush=True)

    # --- Calibrate cutoff from val present images (once, for this concept) ---
    present_idx, _ = retrieve_top_bottom_k(val_pool, t_c, K)
    val_sim_maps = []
    for idx in present_idx:
        img     = Image.open(val_pool.paths[idx]).convert("RGB")
        sim_map = localize_concept(encoder, img, t_c, (img.height, img.width))
        val_sim_maps.append(sim_map)
    cutoff = concept_zscore_cutoff(val_sim_maps, ALPHA)
    print(f"Cutoff calibrated from {len(val_sim_maps)} val present images: {cutoff:.4f}", flush=True)

    # --- Broden positive images for this concept (fixed across all 20
    # classes -- only the negative pool and target class change per class) ---
    try:
        pos_paths_all, _ = load_broden(BRODEN_ROOT, concept_name)
    except ValueError:
        pos_paths_all = []
    rng_pos   = np.random.default_rng(SEED + 4_000_000 + CONCEPT_INDEX)
    pos_paths = sorted(pos_paths_all)
    rng_pos.shuffle(pos_paths)
    pos_paths = pos_paths[:N_CAV_POS]
    pos_images = [Image.open(p).convert("RGB") for p in pos_paths]
    print(f"Broden positive images for '{concept_name}': {len(pos_images)}", flush=True)

    out_dir = VIZ_OUT_DIR / concept_name.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===========================================================================
    # One concept x ALL 20 classes — Hide and Seek + TCAV
    # ===========================================================================
    for cls_i, cls_name in enumerate(COCO_STUFF_20_CLASSES):
        cls_all_paths = [str(p) for p in class_to_train_paths[cls_i]]
        n_available   = len(cls_all_paths)
        print(f"\n=== [{cls_i + 1}/20] class='{cls_name}'  available={n_available} ===", flush=True)

        if n_available == 0:
            print("    No class-present images -- skipping.", flush=True)
            continue

        # Cap at MAX_IMAGES_PER_CLASS, deterministic subsample (own seed per
        # class so the sample doesn't depend on loop order or other runs).
        if n_available > MAX_IMAGES_PER_CLASS:
            rng_sample = np.random.default_rng(SEED + 3_000_000 + cls_i)
            chosen_idx = rng_sample.choice(n_available, size=MAX_IMAGES_PER_CLASS, replace=False)
            cls_paths  = [cls_all_paths[i] for i in sorted(chosen_idx)]
        else:
            cls_paths = cls_all_paths
        n_cls = len(cls_paths)
        print(f"    Scoring {n_cls} images (capped at {MAX_IMAGES_PER_CLASS})", flush=True)

        # CAV negative pool: excludes THIS class's own images (from the full
        # class-present set, not just the capped sample, so negatives are
        # genuinely non-class images).
        cls_test_path_set = set(cls_all_paths)
        cls_non_test = [p for p in all_train_paths if p not in cls_test_path_set]
        rng_neg      = np.random.default_rng(SEED + 5_000_000 + CONCEPT_INDEX * 1_000 + cls_i)
        perm_neg     = rng_neg.permutation(len(cls_non_test))
        neg_paths    = [cls_non_test[i] for i in perm_neg[:N_CAV_NEG]]
        neg_images   = [Image.open(p).convert("RGB") for p in neg_paths]

        cav = _fit_cav(pos_images, neg_images, model, preprocess, DEVICE)

        # TCAV: batched over the sampled class-present images
        if cav is not None:
            tcav_scores = _tcav_all_images(cls_paths, model, cav, preprocess, DEVICE, cls_i)
        else:
            tcav_scores = [0.0] * n_cls
            print("    CAV fit failed — TCAV scores set to 0", flush=True)

        # Hide and Seek for every sampled class-present image
        hs_entries: list[tuple[str, float]] = []
        for img_i, img_path in enumerate(cls_paths):
            score = _cards_score(
                img_path, cidx, img_i,
                encoder, model, preprocess,
                t_c, t_c_dev, cutoff, cls_i, DEVICE,
            )
            hs_entries.append((img_path, np.abs(score)))
            if (img_i + 1) % 50 == 0:
                print(f"    Hide and Seek {img_i + 1}/{n_cls}", flush=True)

        tcav_entries = list(zip(cls_paths, np.abs(tcav_scores)))

        sd_hs   = sorted(hs_entries, key=lambda x: x[1], reverse=True)
        sd_tcav = sorted(tcav_entries, key=lambda x: x[1], reverse=True)
        grid_rows = [
            (
                METHOD_DISPLAY_NAMES["cards"],
                [(p, False) for p, _s in sd_hs[:TOP_BOTTOM_N]],
                [(p, False) for p, _s in sd_hs[-TOP_BOTTOM_N:][::-1]],
            ),
            (
                METHOD_DISPLAY_NAMES["tcav"],
                [(p, False) for p, _s in sd_tcav[:TOP_BOTTOM_N]],
                [(p, False) for p, _s in sd_tcav[-TOP_BOTTOM_N:][::-1]],
            ),
        ]
        out_path = out_dir / f"{cls_name.replace(' ', '_')}.png"
        build_presence_grid_clean(grid_rows, out_path)
        print(f"    Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
