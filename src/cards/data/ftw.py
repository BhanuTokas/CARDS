"""FTW tile I/O and 4-band masking for the ConceptMask replication of the
CounterConcept paper's FTW experiment (Section 4.5).

Real tile format confirmed directly against
`Datasets/FTW/ftw/austria/s2_images/window_a/g77_00002_10.tif`: 4 bands,
uint16, band order (B04, B03, B02, B08) = (Red, Green, Blue, NIR) --
conveniently already R/G/B/NIR order, matching the checkpoint's own
expected input order. CRS/transform are real georeferencing metadata
that MUST be preserved when writing a masked tile back out, since
scoring goes through `ftw_tools.cli inference run` (the official
inference path, per direct instruction -- "We should probably use
ftw_tools.cli inference run" -- rather than an in-process model call);
it won't process the tile correctly otherwise.

Masking strategy: "hue_shift" is excluded from FTW_FILL_STRATEGIES for
the SAME reason `mask_region_nir_band` already excludes it for the NIR
band alone (broden_faithfulness.py's own docstring) -- generalized here
to all 4 bands, since these are raw physical reflectance bands, not a
display-RGB image with a meaningful joint hue. The remaining 6 strategies
each have a clean per-band numeric analog via `mask_region_nir_band`,
applied independently to all 4 raw bands with the SAME strategy/rng so
the mask is filled consistently across bands.

Two distinct representations are used deliberately:
1. `raw_bands_to_display_rgb`: an 8-bit stretch of the first 3 (R,G,B)
   bands, ONLY for feeding CARDS' own SigLIP-based localization and
   best-of-family fill SELECTION (cards.attribution.localization/
   masking_mode) -- these need a normal-looking RGB image, not raw
   digital numbers.
2. `mask_ftw_tile`: applies the STRATEGY selected via (1) directly to the
   raw uint16 4-band array (via `mask_region_nir_band` per band), never
   round-tripping through the 8-bit display image -- avoids the
   precision loss and the "what raw value does uint8=200 correspond to"
   ambiguity a round-trip would introduce for the actual PRUE-scoring
   input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image

from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES
from cards.validation.broden_faithfulness import mask_region_nir_band

# Confirmed via ftw_tools/torchgeo/datamodules.py (upstream ftw-prue repo,
# NOT the local ftw-baselines checkout -- the user flagged that one's
# preprocessing may have been altered for an RGB-only variant): raw DN / 3000,
# no per-channel mean subtraction. Reused here as the display stretch's own
# reference scale, so "looks correct in the display image" and "what PRUE
# actually sees" stay consistent.
DISPLAY_STRETCH_MAX = 3000.0

FTW_FILL_STRATEGIES: list[str] = [s for s in DEFAULT_FILL_STRATEGIES if s != "hue_shift"]


def normalize_tile_orientation(bands: np.ndarray, profile: dict) -> tuple[np.ndarray, dict]:
    """Fixes FTW's own "upside-down" tile convention -- confirmed
    universal across every sampled country (Austria, Belgium, Kenya,
    Portugal, South Africa all have transform.e > 0, i.e. row index
    increasing WITH latitude instead of against it, where a standard
    north-up GeoTIFF has transform.e < 0), NOT a one-off anomaly in a
    single tile.

    This matters because `ftw_tools.cli inference run`'s own
    patch-placement code assumes a standard-orientation transform when
    converting a patch's geographic bounds back to array indices --
    verified directly (notes/ftw_correlation_investigation.md v6) that
    feeding it an unfixed tile produces a silent all-zero output (no
    exception), while a real, sensible prediction comes back once the
    tile is normalized here. A no-op (returns `bands`/`profile`
    unchanged) if the transform is already standard-orientation, so this
    is always safe to call regardless of a given tile's own convention.
    """
    t = profile["transform"]
    if t.e < 0:
        return bands, profile
    height = bands.shape[-2]
    fixed = bands[..., ::-1, :]
    fixed_transform = Affine(t.a, t.b, t.c, t.d, -t.e, t.f + t.e * height)
    fixed_profile = profile.copy()
    fixed_profile["transform"] = fixed_transform
    return fixed, fixed_profile


def load_ftw_tile(path: Path) -> tuple[np.ndarray, dict]:
    """-> ((4, H, W) uint16 raw bands in R/G/B/NIR order, rasterio profile
    dict for writing a same-CRS/transform tile back out)."""
    with rasterio.open(path) as src:
        bands = src.read()
        profile = src.profile.copy()
    return bands, profile


def write_ftw_tile(path: Path, bands_rgbn: np.ndarray, profile: dict) -> None:
    """Writes `bands_rgbn` (N, H, W) back out with `profile`'s own
    CRS/transform/dtype, unchanged -- required for `ftw_tools.cli
    inference run` to interpret the tile correctly. Band-count-agnostic
    (works for both the 4-band single-window case and the 8-band
    stacked dual-window case -- `count` is derived from the array, not
    hardcoded)."""
    out_profile = profile.copy()
    out_profile["count"] = bands_rgbn.shape[0]
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(bands_rgbn)


def window_b_path_for(window_a_path: Path) -> Path:
    """FTW's own directory convention: window_a and window_b tiles share
    the same filename, differing only in which `s2_images/window_*`
    subdirectory they live under -- confirmed directly (matching file
    counts, same tile IDs, `.../austria/s2_images/window_a/g77_00002_10.tif`
    vs `.../austria/s2_images/window_b/g77_00002_10.tif`)."""
    parts = list(window_a_path.parts)
    idx = parts.index("window_a")  # raises ValueError if not a window_a path -- fail loudly, don't guess
    parts[idx] = "window_b"
    return Path(*parts)


def scale_for_dual_window_cli(bands: np.ndarray) -> np.ndarray:
    """Compensates for a real, confirmed normalization mismatch between
    the dual-window PRUE checkpoint (FTW_PRUE_EFNET_B7_CCBY) and the
    official `ftw_tools.cli inference run`'s hardcoded preprocessing.

    `inference.py`'s `default_preprocess` always divides by 3000 (the
    older single-window FTW baseline convention). But the dual-window
    PRUE model was trained on inputs normalized to "surface reflectance
    units (division by 10,000...)" per the PRUE global-mapping paper's
    Methods section -- confirmed EMPIRICALLY, not just by re-reading the
    paper: feeding real DNs through the CLI unscaled produces a
    confidently-wrong all-background prediction (field channel exactly
    0) on tiles the single-window model correctly finds strong field
    signal in; pre-scaling by 0.3 (so CLI's /3000 works out to /10000
    effectively) recovers a real, non-degenerate prediction matching the
    single-window model's own result on the same tile almost exactly
    (field channel mean 160.3 vs 162.8).

    Per "stick to the official code" -- this compensates on the INPUT
    side (like `normalize_tile_orientation` does for the upside-down-
    transform bug, notes v6), not by patching `default_preprocess`
    itself. Only apply this to tiles destined for the dual-window
    checkpoint -- the single-window checkpoint's own /3000 assumption is
    correct as-is (see v8/v9), do NOT apply this scaling there."""
    return np.clip(bands.astype(np.float64) * 0.3, 0, 65535).astype(bands.dtype)


def stack_ftw_windows(bands_a: np.ndarray, bands_b: np.ndarray, profile: dict) -> np.ndarray:
    """Concatenates window_a's and window_b's 4 raw bands into one 8-band
    array for the dual-window PRUE model (`in_channels=8`, confirmed
    directly from the PRUE global-mapping paper: "the encoder processes
    the 8-channel bi-temporal input (4 RGBN bands x 2 time steps)" --
    notes v10).

    Order is [window_b, window_a] (window_b's bands first) -- the
    apparent default throughout the ftw-baselines codebase (`datasets.py`'s
    "stacked" temporal option appends window_b before window_a; `cli.py`'s
    own `--swap_order` help text describes "(window_a, window_b) instead
    of the default (window_b, window_a)"), NOT explicitly confirmed by
    the paper itself. The paper DOES confirm the model was trained with
    "channel shuffling for input-order invariance" augmentation, which
    de-risks getting this detail wrong, but doesn't make it moot -- pick
    ONE consistent convention (this one) rather than mixing them.

    Both windows are assumed to already share the same shape/profile
    (same physical tile location at two dates) -- asserts rather than
    silently mismatching if they don't."""
    if bands_a.shape != bands_b.shape:
        raise ValueError(f"window_a/window_b shape mismatch: {bands_a.shape} vs {bands_b.shape}")
    return np.concatenate([bands_b, bands_a], axis=0)


def raw_bands_to_display_rgb(bands_rgbn: np.ndarray, stretch_max: float = DISPLAY_STRETCH_MAX) -> Image.Image:
    """(4, H, W) raw uint16 -> an 8-bit PIL RGB image from the first 3
    (R, G, B) bands, for localization/fill-selection ONLY -- never fed
    back to PRUE. Simple linear stretch at `stretch_max` (matching the
    model's own /3000 input normalization), clipped and rounded to uint8."""
    rgb = bands_rgbn[:3].astype(np.float64)
    rgb_u8 = np.clip(rgb / stretch_max * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np.transpose(rgb_u8, (1, 2, 0)), mode="RGB")


def mask_ftw_tile(
    bands_rgbn: np.ndarray, mask: np.ndarray, strategy: str, rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Applies `strategy` to all 4 raw bands independently via
    `mask_region_nir_band`, using the SAME mask/rng for every band so the
    fill is spatially and stochastically consistent across channels.
    `strategy` must be one of FTW_FILL_STRATEGIES (no "hue_shift" -- see
    module docstring)."""
    if strategy not in FTW_FILL_STRATEGIES:
        raise ValueError(f"strategy {strategy!r} not in FTW_FILL_STRATEGIES={FTW_FILL_STRATEGIES}")
    return np.stack([
        mask_region_nir_band(bands_rgbn[i], mask, strategy=strategy, rng=rng)
        for i in range(bands_rgbn.shape[0])
    ])
