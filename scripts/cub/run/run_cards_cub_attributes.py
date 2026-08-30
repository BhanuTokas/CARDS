"""Runs CARDS against the 87 attributes that have a faithfulness ground
truth (v38/v41), completing Part 2's 3-way comparison (PCBM and TCAV were
already scored there, v38/v40) -- the CARDS row noted as a gap since v38.

Unlike TCAV/PCBM, CARDS needs no image exemplars at all, only a text
query per concept -- so extending it from the 8-part bank
(run_cards_cub.py) to the 87-attribute bank is purely a matter of
building a natural-language phrase per attribute value (e.g.
"a bird with brown wings" for has_wing_color::brown), reusing the exact
same retrieval pool (SigLIP-embedded CUB test split, cached under the
identical cfg.dataset/pool_source as run_cards_cub.py, so this reuses
that cache instead of re-embedding 5,794 images).

Retrieval strategy: `aligned_retrieval` (cards.retrieval.aligned), not
`matched_retrieval` -- switched to the new default following v46's
retrieval-strategy ablation, which found `matched` (used in every prior
CARDS-on-CUB run) was the WORST of four strategies tested (43.5% sign
agreement, coin-flip) while `aligned` reached this investigation's first
significant CARDS result (67.4%, p=0.026) by directly optimizing
cos(mean(P_c)-mean(N_c), t_c) -- exactly the angle diagnostic v45 found
problematic under `matched`. See notes v46/v47.

K=50 and demean_query=True: the full grid (v47) found
`k=50, demean=True, baseline phrasing, SigLIP` the single best config
(71.7% sign agreement, p=0.0045) -- CARDS' strongest result anywhere in
this investigation, adopted here as the new official default per direct
instruction ("Let's update it").
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.data.cub_attributes import groundable_attributes, load_attribute_names
from cards.data.cub_parts import load_images_txt
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder
from cards.retrieval.aligned import aligned_retrieval
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool
from cards.retrieval.retrieve import retrieve_top_bottom_k

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
RESULTS_DIR = Path("results")
K = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# prefix -> template with a {value} placeholder for the attribute's own
# value (e.g. "brown", "dagger", "notched_tail"), a natural-language
# phrase, not exact grammar -- SigLIP's text encoder doesn't need
# perfect English, just semantic relevance.
PREFIX_TEMPLATES: dict[str, str] = {
    "has_bill_shape": "a bird with a {value} bill",
    "has_bill_length": "a bird with a bill {value}",
    "has_bill_color": "a bird with a {value} bill",
    "has_wing_color": "a bird with {value} wings",
    "has_wing_shape": "a bird with {value}",  # value already e.g. "rounded-wings"
    "has_wing_pattern": "a bird with a {value} wing pattern",
    "has_breast_pattern": "a bird with a {value} breast pattern",
    "has_breast_color": "a bird with a {value} breast",
    "has_back_color": "a bird with a {value} back",
    "has_back_pattern": "a bird with a {value} back pattern",
    "has_tail_shape": "a bird with a {value}",  # value already e.g. "notched_tail"
    "has_tail_pattern": "a bird with a {value} tail pattern",
    "has_upper_tail_color": "a bird with a {value} upper tail",
    "has_under_tail_color": "a bird with a {value} under tail",
    "has_throat_color": "a bird with a {value} throat",
    "has_eye_color": "a bird with {value} eyes",
    "has_forehead_color": "a bird with a {value} forehead",
    "has_nape_color": "a bird with a {value} nape",
    "has_belly_color": "a bird with a {value} belly",
    "has_belly_pattern": "a bird with a {value} belly pattern",
    "has_leg_color": "a bird with {value} legs",
    "has_crown_color": "a bird with a {value} crown",
}


def build_attribute_query_text(prefix: str, value: str) -> str:
    readable = value.replace("_", " ").replace("-", " ")
    return PREFIX_TEMPLATES[prefix].format(value=readable)


def load_classes(cub_root: Path) -> dict[int, str]:
    result = {}
    for line in (cub_root / "classes.txt").read_text().splitlines():
        class_id, raw_name = line.split(maxsplit=1)
        name = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
        result[int(class_id) - 1] = name
    return result


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "device": DEVICE,
            "encoder": {
                "name": "siglip",
                "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                "model_name": "ViT-B-16-SigLIP",
                "pretrained": "webli",
                "device": DEVICE,
            },
            "cache_dir": "embedding_cache",
        }
    )

    print("Loading SigLIP encoder + CARDS retrieval pool (CUB test split, reusing run_cards_cub.py's cache)...", flush=True)
    encoder = instantiate_encoder(cfg)
    image_paths = load_images_txt(CUB_ROOT)
    class_labels = {}
    for line in (CUB_ROOT / "image_class_labels.txt").read_text().splitlines():
        image_id, class_id = line.split()
        class_labels[image_id] = int(class_id) - 1
    test_ids = [
        line.split()[0]
        for line in (CUB_ROOT / "train_test_split.txt").read_text().splitlines()
        if line.split()[1] == "0"
    ]
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]

    cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    cfg.pool_source = "test"
    pool = load_or_build_pool(Path(cfg.cache_dir), cache_key_for(cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    print("Loading native resnet18_cub (the black box CARDS explains)...", flush=True)
    spec = BACKBONES["resnet18_cub"]
    native_model = spec.load_native().to(DEVICE).eval()
    idx_to_class = load_classes(CUB_ROOT)

    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)
    print(f"{len(groundable)}/{len(attribute_names)} official attributes have a faithfulness ground truth to score against.", flush=True)

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    rows = []
    for attr_idx, (prefix, _part_names) in groundable.items():
        attr_name = attribute_names[attr_idx]
        value = attr_name.split("::", 1)[1]
        query_text = build_attribute_query_text(prefix, value)

        t_c = build_concept_query(query_text, encoder)
        t_c = demean_query(t_c, text_center)
        present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)
        absent_indices = aligned_retrieval(pool, present_indices, t_c, K)

        present_paths = [pool.paths[i] for i in present_indices]
        absent_paths = [pool.paths[i] for i in absent_indices]
        present_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in present_paths]).to(DEVICE)
        absent_batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in absent_paths]).to(DEVICE)

        with torch.no_grad():
            present_logits = native_model(present_batch)  # (k, 200)
            absent_logits = native_model(absent_batch)

        raw_score_all_classes = (present_logits.mean(dim=0) - absent_logits.mean(dim=0)).tolist()
        for native_idx, score in enumerate(raw_score_all_classes):
            rows.append((attr_idx, attr_name, prefix, native_idx, score))

        top3 = sorted(enumerate(raw_score_all_classes), key=lambda x: -x[1])[:3]
        print(f"[{attr_idx:>3d}] {attr_name:<40s} <- {query_text!r:<45s} "
              f"top-3: {[(idx_to_class[i], round(s, 3)) for i, s in top3]}", flush=True)

    with open(RESULTS_DIR / "cards_cub_attribute_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute_index", "attribute_name", "attribute_prefix", "native_class_idx", "raw_score"])
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} (attribute, class) CARDS scores to results/cards_cub_attribute_scores.csv")


if __name__ == "__main__":
    main()
