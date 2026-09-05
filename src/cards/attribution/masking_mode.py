"""Step 6 -- masking hybrid (cfg.scoring_mode == "masking_hybrid"), an
alternative to global_mode.global_score's cross-image contrast.

global_score compares the black box's outputs across TWO DIFFERENT
retrieved image sets (mean(b(P_c)) - mean(b(N_c))) -- notes/
celeba_correlation_investigation.md's v79 own diagnosis is that this
lets P_c/N_c differ in whatever else correlates with the concept, not
just the concept itself. masking_score instead never touches an absent
set at all: for each PRESENT image, it localizes the concept within
that SAME image (cards.attribution.localization, no manual masks), masks
it out, and scores b(original) - b(masked) on that one image -- a
same-image counterfactual, immune to that specific confound by
construction.

Validated at full scale on CelebA (notes/celeba_correlation_
investigation.md v78-v93: the only method of CARDS/TCAV/PCBM/hybrid with
a significant, cross-encoder-replicated result there) and on CUB (notes/
cub_correlation_investigation.md v67-v72, later dropped over ground-
truth precision concerns unrelated to this mechanism itself -- see that
file's own closing note).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from cards.attribution.localization import localize_concept, threshold_mask
from cards.encoders.base import PatchLocalizableEncoder
from cards.models.base import BlackBoxModel
from cards.retrieval.pool import CandidatePool
from cards.validation.broden_faithfulness import mask_region

DEFAULT_FILL_STRATEGIES: list[str] = [
    "blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur",
]


@dataclass
class MaskingScoreResult:
    raw_score: float
    delta_scores: list[float] = field(default_factory=list)  # per present-image b(orig) - b(masked)
    selected_strategies: list[str] = field(default_factory=list)
    angles_degrees: list[float] = field(default_factory=list)
    n_skipped_degenerate: int = 0  # present images whose pseudo-mask was empty or covered everything


@torch.no_grad()
def masking_score(
    black_box: BlackBoxModel,
    encoder: PatchLocalizableEncoder,
    pool: CandidatePool,
    present_indices: list[int],
    query: torch.Tensor,
    top_pct: float = 15,
    fill_strategies: list[str] | None = None,
    threshold_method: str = "top_pct",
    seed: int = 0,
    concept_idx: int = 0,
) -> MaskingScoreResult:
    """For each present-set image: localize `query` in that image, mask
    the top `top_pct`% of pixels via whichever of `fill_strategies`
    produces the embedding shift MOST aligned with `query` (best-of-N,
    the smallest angle to `query` wins -- validated as the strongest
    single variant across notes/celeba_correlation_investigation.md
    v79-v82), then score black_box(original) - black_box(masked) on
    that one image. `raw_score` is the mean of those per-image deltas.

    `seed`/`concept_idx` reproduce the exact per-(concept, image) rng
    seeding formula validated throughout this investigation (`seed +
    concept_idx * 10_000 + int(pool_index)`), keyed on the POOL index
    (not list position) so results are reproducible across runs that
    retrieve a different-length present set for the same concept.
    """
    if fill_strategies is None:
        fill_strategies = DEFAULT_FILL_STRATEGIES

    result = MaskingScoreResult(raw_score=0.0)
    for idx in present_indices:
        image = Image.open(pool.paths[idx]).convert("RGB")
        sim_map = localize_concept(encoder, image, query, (image.height, image.width))
        mask = threshold_mask(sim_map, top_pct=top_pct, method=threshold_method)
        if not mask.any() or mask.all():
            result.n_skipped_degenerate += 1
            continue

        rng = np.random.default_rng(seed + concept_idx * 10_000 + int(idx))
        candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in fill_strategies]

        embeds = encoder.encode_images([image] + candidates)
        embed_orig = embeds[0]
        query_dev = query.to(embed_orig.device)
        best_angle, best_i = None, 0
        for i in range(len(fill_strategies)):
            diff = embed_orig - embeds[1 + i]
            diff_unit = diff / diff.norm()
            cos_sim = float(torch.clamp(diff_unit @ query_dev, -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_sim)))
            if best_angle is None or angle_deg < best_angle:
                best_angle, best_i = angle_deg, i

        masked_image = candidates[best_i]
        pixels_orig = black_box.preprocess(image).unsqueeze(0)
        pixels_masked = black_box.preprocess(masked_image).unsqueeze(0)
        batch = torch.cat([pixels_orig, pixels_masked], dim=0)
        outputs = black_box(batch)  # (2,) per-image scalar scores

        result.delta_scores.append((outputs[0] - outputs[1]).item())
        result.selected_strategies.append(fill_strategies[best_i])
        result.angles_degrees.append(best_angle)

    result.raw_score = float(np.mean(result.delta_scores)) if result.delta_scores else 0.0
    return result
