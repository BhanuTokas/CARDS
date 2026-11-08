"""HnS (masking hybrid) on BAM, stage 1: build the masked images.

BAM's models are TensorFlow, so this stage runs the unchanged HnS pipeline
(`process_concept` + `masking_score`, called as
`score_masking_hybrid_concepts` calls it) with a stand-in black
box. The stand-in records each (original, masked) pair HnS would score, and
`store_bam_logits.py` later scores both images with each BAM model. HnS never
reads the black box while choosing masks or fills (only SigLIP), so recording
the pairs loses nothing.

Settings follow the paper's CelebA tables: ViT-B-16-SigLIP, K=50,
orthogonalized queries, z-score threshold alpha=1.0, best-of-7 fills,
template-ensembled prompts. demean is run both ways, since it has to be checked
for each new dataset.

Writes, under <bam-root>/hns/<variant>/:
  masked/<concept>__<image stem>.png  masked image, lossless
  manifest.csv                        one row per scored (concept, image)
"""

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cards.concepts.prompts import (
    GENERIC_REFERENCE_CONCEPTS,
    build_concept_query,
    compute_text_center,
    demean_query,
)
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES, masking_score
from cards.pipeline import instantiate_encoder, orthogonalize_queries, process_concept
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
K = 50
ALPHA = 1.0

# Concept name -> query text. Objects and scenes are BAM's 20 classes.
CONCEPT_QUERY_TEXT = {
    "backpack": "a backpack", "bird": "a bird", "dog": "a dog", "elephant": "an elephant",
    "kite": "a kite", "pizza": "a pizza", "stop_sign": "a stop sign", "toilet": "a toilet",
    "truck": "a truck", "zebra": "a zebra",
    "bamboo_forest": "a bamboo forest", "bedroom": "a bedroom", "bowling_alley": "a bowling alley",
    "bus_interior": "the inside of a bus", "cockpit": "an airplane cockpit", "corn_field": "a corn field",
    "laundromat": "a laundromat", "runway": "an airport runway", "ski_slope": "a ski slope",
    "track": "an outdoor running track",
}
CONCEPTS = list(CONCEPT_QUERY_TEXT)


def image_key(image: Image.Image) -> str:
    return hashlib.sha1(image.tobytes()).hexdigest()


class RecordingBlackBox:
    """Stands in for the black box: keeps each (original, masked) pair, returns zeros."""

    def __init__(self):
        self.pending: list[Image.Image] = []
        self.pairs: list[tuple[Image.Image, Image.Image]] = []

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        self.pending.append(image.copy())
        return torch.zeros(1)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        original, masked = self.pending
        self.pending = []
        self.pairs.append((original, masked))
        return torch.zeros(batch.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache")))
    parser.add_argument("--demean", choices=["true", "false"], required=True)
    args = parser.parse_args()
    demean = args.demean == "true"

    val_dir = args.bam_root / "val"
    out_dir = args.bam_root / "hns" / f"demean_{args.demean}"
    (out_dir / "masked").mkdir(parents=True, exist_ok=True)

    with open(val_dir / "manifest.csv", newline="") as handle:
        files = [row["file"] for row in csv.DictReader(handle)]
    pairs = [(val_dir / "composite" / name, 0) for name in files]

    encoder_cfg = {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                   "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE}
    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": str(args.cache_dir),
        "retrieval": {"strategy": "naive"}, "k": K,
        "masking_hybrid": {"threshold_method": "zscore", "alpha": ALPHA},
    })
    encoder = instantiate_encoder(cfg)

    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg,
                                 "dataset": {"name": "bam_val_composite"}, "pool_source": "val"})
    pool = load_or_build_pool(args.cache_dir, cache_key_for(pool_cfg), pairs, encoder)
    print(f"pool: {len(pool.paths)} images", flush=True)

    queries = {c: build_concept_query(CONCEPT_QUERY_TEXT[c], encoder) for c in CONCEPTS}
    if demean:
        center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
        queries = {c: demean_query(q, center) for c, q in queries.items()}
    queries = orthogonalize_queries(queries)

    results = [process_concept(cfg, encoder, pool, c, query=queries[c]) for c in CONCEPTS]
    print("retrieval done", flush=True)

    rows = []
    for concept_idx, (concept, result) in enumerate(zip(CONCEPTS, results)):
        present = [pool.paths[i] for i in result.present_indices]
        by_key = {image_key(Image.open(p).convert("RGB")): p for p in present}
        recorder = RecordingBlackBox()
        # Same call and seeding as score_masking_hybrid_concepts, one concept at a
        # time so each recorded pair maps to this concept.
        scored = masking_score(recorder, encoder, pool, result.present_indices, result.query,
                               fill_strategies=DEFAULT_FILL_STRATEGIES, threshold_method="zscore",
                               alpha=ALPHA, seed=SEED, concept_idx=concept_idx)
        assert len(recorder.pairs) == len(scored.delta_scores)

        for (original, masked), fill, angle in zip(recorder.pairs, scored.selected_strategies,
                                                  scored.angles_degrees):
            source = by_key[image_key(original)]
            changed = np.any(np.asarray(original) != np.asarray(masked), axis=-1)
            object_mask = np.asarray(Image.open(val_dir / "object_mask" / f"{source.stem}.png")) > 0
            masked_name = f"{concept}__{source.stem}.png"
            masked.save(out_dir / "masked" / masked_name)
            rows.append({
                "concept": concept, "file": source.name, "masked_file": masked_name,
                "fill": fill, "angle_deg": angle,
                "mask_fraction": float(changed.mean()),
                "mask_on_object": float((changed & object_mask).sum() / max(changed.sum(), 1)),
                "object_covered": float((changed & object_mask).sum() / max(object_mask.sum(), 1)),
            })
        print(f"{concept}: {len(scored.delta_scores)} scored, {scored.n_skipped_degenerate} skipped",
              flush=True)

    with open(out_dir / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} masked images in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
