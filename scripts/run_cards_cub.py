"""CUB-track analogue of run_cards_imagenet.py: runs CARDS itself against
the native resnet18_cub black box, using the 8 CUB part concepts, so
CARDS can be scored against the same CUB faithfulness ground truth
TCAV (v35) and the faithfulness extension (v36) are being scored
against.

Retrieval pool = CUB's own TEST split (5,794 images) -- same "closed
world" convention as the ImageNet track (Phase 1's plan: keep CARDS'
retrieval pool within the same domain PCBM/TCAV are scoped to), and the
same held-out population used throughout the CUB reproduction/
faithfulness checks (v34-v36).

No native-label-index restriction needed: resnet18_cub's own 200-way head
already matches CUB's 200 classes one-to-one (unlike the ImageNet track's
1000-vs-25 subset), so raw_score is computed for all 200 classes at once
per concept.

Retrieval strategy: `aligned_retrieval`, K=50, demean_query=True -- the
same config v47's grid found best on the 87-attribute bank (71.7% sign
agreement, p=0.0045), applied here too per direct instruction ("Let's
update it"). NOT independently validated on THIS 8-part bank -- v47
noted Part 1's own K=30/aligned result actually trended worse than
`matched` (sign agreement 38.5% vs. 53.8%), unlike Part 2, so whether
K=50/demean=True helps here specifically is an open question, not
confirmed. Re-run scripts/score_all_methods_against_cub_faithfulness.py
after changing this to see the actual effect on Part 1.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.models.backbones import BACKBONES  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.aligned import aligned_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
RESULTS_DIR = Path("results")
K = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no captum involved here (unlike Phase 2's TCAV script), so no device-mismatch constraint -- CUB's larger 5,794-image retrieval pool benefits from GPU embedding

TEST_CONCEPTS = ["beak", "left_eye", "right_eye", "left_wing", "right_wing", "left_leg", "right_leg", "tail"]
CONCEPT_QUERY_TEXT = {
    "beak": "a bird's beak",
    "left_eye": "a bird's eye",
    "right_eye": "a bird's eye",
    "left_wing": "a bird's wing",
    "right_wing": "a bird's wing",
    "left_leg": "a bird's leg",
    "right_leg": "a bird's leg",
    "tail": "a bird's tail",
}


def load_classes(cub_root: Path) -> dict[int, str]:
    result = {}
    for line in (cub_root / "classes.txt").read_text().splitlines():
        class_id, raw_name = line.split(maxsplit=1)
        name = raw_name.split(".", 1)[1] if "." in raw_name else raw_name
        result[int(class_id) - 1] = name  # 0-indexed, matches native model output
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

    print("Loading SigLIP encoder + building CARDS retrieval pool (CUB test split)...", flush=True)
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

    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_scores: dict[tuple[str, int], float] = {}  # (part_name, native_class_idx) -> raw_score

    for concept in TEST_CONCEPTS:
        print(f"\n=== {concept} ===", flush=True)
        t_c = build_concept_query(CONCEPT_QUERY_TEXT[concept], encoder)
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
            raw_scores[(concept, native_idx)] = score

        top3 = sorted(enumerate(raw_score_all_classes), key=lambda x: -x[1])[:3]
        print(f"top-3 classes by raw_score: {[(idx_to_class[i], round(s, 3)) for i, s in top3]}", flush=True)

    with open(RESULTS_DIR / "cards_cub_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["part", "native_class_idx", "class_name", "raw_score"])
        for (concept, native_idx), score in raw_scores.items():
            writer.writerow([concept, native_idx, idx_to_class[native_idx], score])

    print(f"\nSaved {len(raw_scores)} (part, class) CARDS scores to results/cards_cub_scores.csv")


if __name__ == "__main__":
    main()
