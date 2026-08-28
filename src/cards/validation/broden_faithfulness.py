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


def mask_region(
    image: Image.Image, mask: np.ndarray, strategy: str = "blur",
    blur_sigma: float = 20.0, hue_shift_degrees: float = 180.0,
    noise_std: float = 20.0, rng: np.random.Generator | None = None,
) -> Image.Image:
    """Fills `mask` (True = region to remove) within `image`.

    "blur" (default) is the recommended general-purpose strategy: zero/
    mean fill creates an out-of-distribution "hole" that can itself be
    salient for reasons unrelated to concept content (a known issue in
    the deletion-metric literature, e.g. Petsiuk et al.'s RISE explicitly
    prefers blur over constant fill for this reason) -- blur instead
    preserves plausible low-frequency scene structure while still
    destroying the concept's own fine detail.

    "hue_shift" is a COLOR-SPECIFIC alternative (not a general-purpose
    replacement for blur): rotates the masked region's hue by
    `hue_shift_degrees` (180 degrees = each color's exact complementary
    opposite, the default and the maximum possible hue distance) while
    leaving saturation and value (i.e. luminance-driven edges/texture/
    shape) untouched. This targets exactly the gap blur and zero_fill
    each leave open on their own: blur is confirmed (notes/
    cub_correlation_investigation.md v61) to leave most of a masked
    region's own mean color intact (a low-pass filter, by construction),
    so it under-erases color-defined concepts specifically; zero_fill
    erases color completely but also erases the region's own spatial
    structure and creates an out-of-distribution "hole," a confound of
    its own. hue_shift erases color information (the model can no longer
    read off the ORIGINAL hue) while keeping the perturbed region
    in-distribution-shaped (still has real luminance edges/texture,
    still looks like SOME part of a bird, just the wrong color) -- a
    genuinely surgical color-only perturbation, useful for isolating
    color-attribute faithfulness specifically from pattern/shape.

    "zero_fill_noise" and "white_fill" both target a narrower confound
    zero_fill can have for specific concept VALUES that happen to
    coincide with the fill color's own identity, not attribute types in
    general (checked directly, notes/cub_correlation_investigation.md
    v61's own "black"/"solid" check found this confound doesn't actually
    suppress delta_p in practice -- these two variants exist to test that
    more rigorously, not because the plain zero_fill numbers looked
    obviously wrong).
    - "zero_fill_noise": zero_fill plus per-pixel Gaussian noise
      (`noise_std`, 0-255 scale) -- a flat, perfectly uniform black patch
      is itself unlike any real photographic region (even a genuinely
      "solid"-patterned one still has natural sensor/texture noise), so
      plain zero_fill's own "obviously edited" flatness could be an
      independent salience confound for pattern="solid" concepts
      specifically, where the ORIGINAL region is also uniform and the
      absence-of-fine-detail can't tell masked from unmasked apart on its
      own. Requires `rng` (reuses whatever np.random.Generator the caller
      already threads through compute_faithfulness's random-placement
      draws, for reproducibility -- raises if omitted rather than
      silently seeding its own).
    - "white_fill": the exact opposite constant color from zero_fill's
      black -- for color="black" concepts specifically, where the
      region's own real (dark, if imperfectly so) pixel values are
      closest in RGB space to zero_fill's own fill color, maximizing
      contrast instead.
    """
    if mask.shape != (image.height, image.width):
        raise ValueError(f"mask shape {mask.shape} doesn't match image size {(image.height, image.width)}")

    if strategy == "blur":
        filled = image.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    elif strategy == "mean_fill":
        filled = Image.new("RGB", image.size, _IMAGENET_MEAN_RGB)
    elif strategy == "zero_fill":
        filled = Image.new("RGB", image.size, (0, 0, 0))
    elif strategy == "white_fill":
        filled = Image.new("RGB", image.size, (255, 255, 255))
    elif strategy == "zero_fill_noise":
        if rng is None:
            raise ValueError("strategy='zero_fill_noise' requires an rng (np.random.Generator) for the noise draw")
        noise = rng.normal(loc=0.0, scale=noise_std, size=(image.height, image.width, 3))
        filled = Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8), mode="RGB")
    elif strategy == "hue_shift":
        h, s, v = image.convert("HSV").split()
        shift = round(hue_shift_degrees / 360.0 * 256.0)
        h_shifted = ((np.array(h, dtype=np.int16) + shift) % 256).astype(np.uint8)
        filled = Image.merge("HSV", (Image.fromarray(h_shifted, mode="L"), s, v)).convert("RGB")
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
    target_class: int | None = None,
) -> FaithfulnessResult:
    """The core per-(image, concept) measurement. By default uses the
    model's own top-1 prediction on the *unmasked* image as "the class of
    interest" -- no ImageNet-class ground truth needed, and no external
    concept->class mapping table, since Broden's images don't carry
    ImageNet labels at all. Pass `target_class` to override this with a
    real ground-truth label instead (e.g. CUB's own species labels, which
    -- unlike Broden -- genuinely exist): this measures "does masking
    concept c reduce the model's confidence in the CORRECT class" rather
    than "...in whatever the model currently guesses," a different and,
    where ground truth exists, often more useful question. The resulting
    `predicted_class` field holds whichever one was actually used --
    callers passing `target_class` are responsible for documenting that
    divergence, the field itself doesn't rename to stay compatible with
    existing readers."""
    x0 = model.preprocess(image).unsqueeze(0).to(device)
    logits0 = model(x0)[0]
    predicted_class = target_class if target_class is not None else int(torch.argmax(logits0).item())
    p0 = F.softmax(logits0, dim=0)[predicted_class].item()

    masked_image = mask_region(image, mask, strategy=fill_strategy, rng=rng)
    x_masked = model.preprocess(masked_image).unsqueeze(0).to(device)
    logits_masked = model(x_masked)[0]
    p_masked = F.softmax(logits_masked, dim=0)[predicted_class].item()
    delta_logit = (logits0[predicted_class] - logits_masked[predicted_class]).item()

    random_masks, n_fallbacks = random_placements(mask, rng, n_draws=n_random_draws)
    random_deltas = []
    for rmask in random_masks:
        r_image = mask_region(image, rmask, strategy=fill_strategy, rng=rng)
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


def _aggregate_faithfulness_pairs(
    faithfulness_records: list[FaithfulnessResult],
    method_scores: dict[tuple[int, int], float],
    min_samples_per_pair: int,
) -> dict[tuple[int, int], float]:
    """Mean `delta_p` per unique (concept_number, predicted_class) pair
    across all images sharing it, restricted to pairs with both
    >=min_samples_per_pair faithfulness samples AND a method score --
    the shared aggregation both score_method_agreement and
    score_sign_agreement build on."""
    from collections import defaultdict

    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for r in faithfulness_records:
        grouped[(r.concept_number, r.predicted_class)].append(r.delta_p)

    return {
        pair: float(np.mean(deltas))
        for pair, deltas in grouped.items()
        if len(deltas) >= min_samples_per_pair and pair in method_scores
    }


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
    from scipy.stats import spearmanr

    aggregated = _aggregate_faithfulness_pairs(faithfulness_records, method_scores, min_samples_per_pair)

    if len(aggregated) < 3:
        return None

    pairs = list(aggregated.keys())
    x = [aggregated[p] for p in pairs]
    y = [method_scores[p] for p in pairs]
    rho, p = spearmanr(x, y)
    return AgreementResult(n_pairs=len(pairs), spearman_rho=float(rho), spearman_p=float(p))


@dataclass
class SignAgreementResult:
    n_pairs: int
    n_agree: int
    agreement_frac: float
    binom_p: float  # two-sided p-value against 50% chance agreement


def score_sign_agreement(
    faithfulness_records: list[FaithfulnessResult],
    method_scores: dict[tuple[int, int], float],
    min_samples_per_pair: int = 3,
    method_threshold: float = 0.0,
) -> SignAgreementResult | None:
    """A coarser, more robust complement to score_method_agreement's exact
    Spearman ranking: for each (concept, predicted_class) pair with enough
    samples, does the method's score agree with the faithfulness ground
    truth on DIRECTION alone (mean delta_p > 0, i.e. masking the concept
    hurt the model, vs. the method's own score > method_threshold)? Exact
    rank order is fragile at the small n_pairs this investigation
    typically has (13-46); a binary "did they even point the same way"
    call is far less sensitive to a handful of noisy pairs, at the cost of
    discarding magnitude information Spearman uses.

    `method_threshold` lets a method's own "positive" boundary sit
    somewhere other than 0 -- TCAV's sign_count is a [0,1] fraction
    centered at 0.5 (not a signed quantity like CARDS' raw_score or PCBM's
    weight), so callers scoring TCAV should pass method_threshold=0.5.

    Significance is a two-sided exact binomial test against 50% (chance
    agreement), via scipy's binomtest -- appropriate for a small discrete
    count of matches/mismatches, unlike a normal-approximation z-test.
    Returns None under the same too-few-pairs condition as
    score_method_agreement.
    """
    from scipy.stats import binomtest

    aggregated = _aggregate_faithfulness_pairs(faithfulness_records, method_scores, min_samples_per_pair)

    if len(aggregated) < 3:
        return None

    n_agree = sum(
        1 for pair, gt in aggregated.items() if (gt > 0) == (method_scores[pair] > method_threshold)
    )
    n_pairs = len(aggregated)
    result = binomtest(n_agree, n_pairs, p=0.5, alternative="two-sided")
    return SignAgreementResult(
        n_pairs=n_pairs, n_agree=n_agree, agreement_frac=n_agree / n_pairs, binom_p=float(result.pvalue)
    )
