"""Phase 3 (notes/pcbm_correlation_investigation.md, CARDS vs. TCAV vs.
PCBM plan): masking-based perturbation/faithfulness metric -- the sole
ground truth this comparison is judged against, not one diagnostic among
several. A method's concept-importance claim is right or wrong exactly to
the extent that perturbing that concept in a real image (using Broden's
own segmentation mask, not a synthetic proxy) moves the model's own
prediction. Scoped to Broden's own images only: ImageNet's validation
images have no segmentation masks, Broden's ~63k do -- an accepted scope
limit, not a gap to solve (see the parent plan's Phase 3 section).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


class MultiClassModel(Protocol):
    """Deliberately distinct from `cards.models.base.BlackBoxModel`, which
    resolves to one fixed target class per instance -- this metric needs
    the model's own *per-image* top-1 prediction, which varies image to
    image, so it needs the full logit vector."""

    def preprocess(self, image: Image.Image) -> torch.Tensor: ...

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """-> (N, num_classes) full logits, not reduced to one class."""
        ...


# ImageNet normalization mean, 0-255 scale -- the "mean_fill" strategy's
# fill color, chosen to be a neutral, information-free pixel value under
# the same normalization the model itself expects.
_IMAGENET_MEAN_RGB = (124, 116, 104)  # round(0.485*255), round(0.456*255), round(0.406*255)


def mask_region(image: Image.Image, mask: np.ndarray, strategy: str = "blur", blur_sigma: float = 20.0) -> Image.Image:
    """Fills `mask` (True = region to remove) within `image`.

    "blur" (default) is the recommended strategy: zero/mean fill creates
    an out-of-distribution "hole" that can itself be salient for reasons
    unrelated to concept content (a known issue in the deletion-metric
    literature, e.g. Petsiuk et al.'s RISE explicitly prefers blur over
    constant fill for this reason) -- blur instead preserves plausible
    low-frequency scene structure while still destroying the concept's
    own fine detail.
    """
    if mask.shape != (image.height, image.width):
        raise ValueError(f"mask shape {mask.shape} doesn't match image size {(image.height, image.width)}")

    if strategy == "blur":
        filled = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    elif strategy == "mean_fill":
        filled = Image.new("RGB", image.size, _IMAGENET_MEAN_RGB)
    elif strategy == "zero_fill":
        filled = Image.new("RGB", image.size, (0, 0, 0))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return Image.composite(filled, image.convert("RGB"), mask_img)


def _area_matched_rectangle(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fallback when no valid exact-shape translation is found (relevant
    for concepts covering most of the image) -- a random square of the
    same area, not the same shape."""
    h, w = mask.shape
    area = int(mask.sum())
    side = max(1, min(int(round(np.sqrt(area))), h, w))
    y0 = int(rng.integers(0, h - side + 1))
    x0 = int(rng.integers(0, w - side + 1))
    candidate = np.zeros_like(mask)
    candidate[y0 : y0 + side, x0 : x0 + side] = True
    return candidate


def random_placements(
    mask: np.ndarray,
    rng: np.random.Generator,
    n_draws: int = 5,
    exclude_masks: list[np.ndarray] | None = None,
    max_overlap_frac: float = 0.1,
    max_attempts: int = 200,
) -> tuple[list[np.ndarray], int]:
    """Translates `mask`'s *exact* shape (not just an area-matched
    rectangle) to `n_draws` random valid offsets -- holds shape AND area
    constant between the real-concept and random conditions, isolating
    content as the only varying factor, a stronger control than area
    alone. Rejects placements that go out of bounds or overlap `mask`
    (always excluded) or any `exclude_masks` (e.g. other real concepts
    present in the same image) by more than `max_overlap_frac`. Falls
    back to `_area_matched_rectangle` after `max_attempts` failed
    attempts. Returns (placements, n_fallbacks) so callers can see how
    often the fallback triggered.
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return [], 0
    y0b, y1b = int(ys.min()), int(ys.max()) + 1
    x0b, x1b = int(xs.min()), int(xs.max()) + 1
    box_h, box_w = y1b - y0b, x1b - x0b
    local_mask = mask[y0b:y1b, x0b:x1b]

    exclude = [mask] + list(exclude_masks or [])
    placements = []
    n_fallbacks = 0

    for _ in range(n_draws):
        found = None
        if box_h <= h and box_w <= w:
            for _attempt in range(max_attempts):
                new_y0 = int(rng.integers(0, h - box_h + 1))
                new_x0 = int(rng.integers(0, w - box_w + 1))
                candidate = np.zeros_like(mask)
                candidate[new_y0 : new_y0 + box_h, new_x0 : new_x0 + box_w] = local_mask
                total = candidate.sum()
                if total == 0:
                    continue
                if all((candidate & ex).sum() / total <= max_overlap_frac for ex in exclude):
                    found = candidate
                    break
        if found is None:
            found = _area_matched_rectangle(mask, rng)
            n_fallbacks += 1
        placements.append(found)

    return placements, n_fallbacks


@dataclass
class FaithfulnessResult:
    image: str
    concept_number: int
    category: str
    predicted_class: int
    p0: float
    p_masked: float
    delta_p: float
    delta_logit: float
    random_delta_p_mean: float
    random_delta_p_std: float
    z_score: float
    n_random_fallbacks: int


@torch.no_grad()
def compute_faithfulness(
    image: Image.Image,
    image_path: str,
    concept_number: int,
    category: str,
    mask: np.ndarray,
    model: MultiClassModel,
    rng: np.random.Generator,
    n_random_draws: int = 5,
    fill_strategy: str = "blur",
    device: str = "cpu",
) -> FaithfulnessResult:
    """The core per-(image, concept) measurement. Uses the model's own
    top-1 prediction on the *unmasked* image as "the class of interest"
    -- no ImageNet-class ground truth needed, and no external
    concept->class mapping table, since Broden's images don't carry
    ImageNet labels at all."""
    x0 = model.preprocess(image).unsqueeze(0).to(device)
    logits0 = model(x0)[0]
    predicted_class = int(torch.argmax(logits0).item())
    p0 = F.softmax(logits0, dim=0)[predicted_class].item()

    masked_image = mask_region(image, mask, strategy=fill_strategy)
    x_masked = model.preprocess(masked_image).unsqueeze(0).to(device)
    logits_masked = model(x_masked)[0]
    p_masked = F.softmax(logits_masked, dim=0)[predicted_class].item()
    delta_logit = (logits0[predicted_class] - logits_masked[predicted_class]).item()

    random_masks, n_fallbacks = random_placements(mask, rng, n_draws=n_random_draws)
    random_deltas = []
    for rmask in random_masks:
        r_image = mask_region(image, rmask, strategy=fill_strategy)
        x_r = model.preprocess(r_image).unsqueeze(0).to(device)
        logits_r = model(x_r)[0]
        p_r = F.softmax(logits_r, dim=0)[predicted_class].item()
        random_deltas.append(p0 - p_r)

    delta_p = p0 - p_masked
    random_mean = float(np.mean(random_deltas)) if random_deltas else 0.0
    random_std = float(np.std(random_deltas)) if random_deltas else 0.0
    z_score = (delta_p - random_mean) / (random_std + 1e-8)

    return FaithfulnessResult(
        image=image_path,
        concept_number=concept_number,
        category=category,
        predicted_class=predicted_class,
        p0=p0,
        p_masked=p_masked,
        delta_p=delta_p,
        delta_logit=delta_logit,
        random_delta_p_mean=random_mean,
        random_delta_p_std=random_std,
        z_score=z_score,
        n_random_fallbacks=n_fallbacks,
    )


@dataclass
class AgreementResult:
    n_pairs: int
    spearman_rho: float
    spearman_p: float


def score_method_agreement(
    faithfulness_records: list[FaithfulnessResult],
    method_scores: dict[tuple[int, int], float],
    min_samples_per_pair: int = 3,
) -> AgreementResult | None:
    """Aggregates `delta_p` per unique (concept_number, predicted_class)
    pair across all Broden images sharing it (requiring >=
    min_samples_per_pair images), Spearman-correlates against any
    method's own (concept, class) -> importance score table.
    `method_scores` is the common denominator: CARDS' raw_score, TCAV's
    sign_count/magnitude, and PCBM's own weight are all reducible to
    exactly this (concept, class) -> scalar shape, so one function scores
    all three methods identically against the masking ground truth.
    Returns None if fewer than 3 pairs have both a faithfulness
    aggregate and a method score (too few for a meaningful correlation).
    """
    from collections import defaultdict

    from scipy.stats import spearmanr

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in faithfulness_records:
        grouped[(r.concept_number, r.predicted_class)].append(r.delta_p)

    aggregated = {
        pair: float(np.mean(deltas))
        for pair, deltas in grouped.items()
        if len(deltas) >= min_samples_per_pair and pair in method_scores
    }

    if len(aggregated) < 3:
        return None

    pairs = list(aggregated.keys())
    x = [aggregated[p] for p in pairs]
    y = [method_scores[p] for p in pairs]
    rho, p = spearmanr(x, y)
    return AgreementResult(n_pairs=len(pairs), spearman_rho=float(rho), spearman_p=float(p))
