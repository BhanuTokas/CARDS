"""Is there a correlation between rho (or sign agreement) and the
angle(d_c, t_c) diagnostic, across CONFIGS rather than the small,
exploratory per-strategy comparisons in v45/v46? Prompted directly
("Is there any correlation between rho and angle?").

Reuses the exact 12 SigLIP configs from v47's aligned grid (K in
[15,30,50] x demean x phrasing) -- their rho/sign_agreement are already
known (hardcoded below from that run, not recomputed, to avoid paying
for the black-box scoring pass again) -- and adds ONLY the cheap part
that was never computed there: each config's own mean angle(d_c, t_c)
across all 87 concepts. Retrieval is deterministic given fixed inputs, so
re-running just the retrieval (present/absent indices), with no black-box
forward pass needed at all, exactly reproduces the same P_c/N_c split
v47 scored -- angle and rho/sign_agreement are then genuinely about the
same underlying retrieval, not two different runs.

CLIP/open_clip_h/perception excluded from this pass -- different encoders
have different embedding geometries (confirmed indirectly: v46's
matched-vs-aligned angle shrink was SigLIP-specific), so pooling angle
values across encoders into one correlation would conflate "does angle
predict rho within one encoder's own geometry" with "do different
encoders just have different baseline angles," a different question.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent / "ablate"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from run_cards_cub_attributes import PREFIX_TEMPLATES  # noqa: E402
from ablate_cards_cub_attributes import ALT_PREFIX_TEMPLATES  # noqa: E402

from cards.data.cub_attributes import groundable_attributes, load_attribute_names  # noqa: E402
from cards.data.cub_parts import load_images_txt  # noqa: E402
from cards.pipeline import instantiate_encoder  # noqa: E402
from cards.retrieval.aligned import aligned_retrieval  # noqa: E402
from cards.retrieval.embedding_cache import cache_key_for, load_or_build_pool  # noqa: E402
from cards.retrieval.retrieve import retrieve_top_bottom_k  # noqa: E402
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query  # noqa: E402

CUB_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CUB_200_2011")
ATTRIBUTE_NAMES_PATH = CUB_ROOT / "attributes" / "new_attributes.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (k, demean, phrasing) -> (rho, sign_agreement) from v47's aligned SigLIP grid, verbatim.
KNOWN_RESULTS = {
    (15, False, "baseline"): (0.1086, 0.674),
    (15, False, "ensemble"): (0.1409, 0.674),
    (15, True, "baseline"): (0.0944, 0.696),
    (15, True, "ensemble"): (0.1246, 0.630),
    (30, False, "baseline"): (0.0366, 0.652),
    (30, False, "ensemble"): (-0.0091, 0.587),
    (30, True, "baseline"): (-0.0590, 0.630),
    (30, True, "ensemble"): (-0.0141, 0.630),
    (50, False, "baseline"): (0.0223, 0.674),
    (50, False, "ensemble"): (0.0261, 0.652),
    (50, True, "baseline"): (0.0779, 0.717),
    (50, True, "ensemble"): (0.0862, 0.696),
}


def readable(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def build_query(prefix: str, value: str, encoder, phrasing: str) -> torch.Tensor:
    base_text = PREFIX_TEMPLATES[prefix].format(value=readable(value))
    if phrasing == "baseline":
        return build_concept_query(base_text, encoder)
    alt_text = ALT_PREFIX_TEMPLATES[prefix].format(value=readable(value))
    embeddings = torch.stack([build_concept_query(base_text, encoder), build_concept_query(alt_text, encoder)])
    return F.normalize(embeddings.mean(dim=0), dim=0)


def main():
    attribute_names = load_attribute_names(ATTRIBUTE_NAMES_PATH)
    groundable = groundable_attributes(attribute_names)

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

    print("Loading SigLIP + pool...", flush=True)
    encoder_cfg = OmegaConf.create({"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                                     "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE})
    encoder = instantiate_encoder(OmegaConf.create({"encoder": encoder_cfg, "device": DEVICE}))
    pool_cfg = OmegaConf.create({"seed": 0, "device": DEVICE, "encoder": encoder_cfg, "cache_dir": "embedding_cache"})
    pool_cfg.dataset = {"name": "cub", "root": str(CUB_ROOT)}
    pool_cfg.pool_source = "test"
    pairs = [(image_paths[i], class_labels[i]) for i in test_ids]
    pool = load_or_build_pool(Path(pool_cfg.cache_dir), cache_key_for(pool_cfg), pairs, encoder)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    configs = list(itertools.product([15, 30, 50], [False, True], ["baseline", "ensemble"]))
    rows = []
    for k, use_demean, phrasing in configs:
        angles = []
        for attr_idx, (prefix, _part_names) in groundable.items():
            value = attribute_names[attr_idx].split("::", 1)[1]
            t_c = build_query(prefix, value, encoder, phrasing)
            if use_demean:
                t_c = demean_query(t_c, text_center)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, k)
            absent_indices = aligned_retrieval(pool, present_indices, t_c, k)

            d_c = pool.embeddings[present_indices].mean(dim=0) - pool.embeddings[absent_indices].mean(dim=0)
            t_c_unit = F.normalize(t_c, dim=0)
            d_c_unit = F.normalize(d_c, dim=0)
            cos_sim = float(torch.clamp(t_c_unit @ d_c_unit, -1.0, 1.0))
            angles.append(float(np.degrees(np.arccos(cos_sim))))

        mean_angle = float(np.mean(angles))
        rho, sign_agreement = KNOWN_RESULTS[(k, use_demean, phrasing)]
        rows.append((k, use_demean, phrasing, mean_angle, rho, sign_agreement))
        print(f"k={k:>2d} demean={use_demean!s:>5s} {phrasing:<9s} mean_angle={mean_angle:.3f} deg  "
              f"rho={rho:+.4f}  sign={sign_agreement:.1%}", flush=True)

    angles_all = [r[3] for r in rows]
    rhos_all = [r[4] for r in rows]
    signs_all = [r[5] for r in rows]

    rho_angle_rho, rho_angle_p = spearmanr(angles_all, rhos_all)
    sign_angle_rho, sign_angle_p = spearmanr(angles_all, signs_all)

    print(f"\nAcross {len(rows)} SigLIP aligned-grid configs:")
    print(f"Spearman(mean_angle, rho)            = {rho_angle_rho:+.4f} (p={rho_angle_p:.4g})")
    print(f"Spearman(mean_angle, sign_agreement)  = {sign_angle_rho:+.4f} (p={sign_angle_p:.4g})")
    print(f"angle range: {min(angles_all):.2f} - {max(angles_all):.2f} degrees")


if __name__ == "__main__":
    main()
