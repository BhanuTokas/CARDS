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
    """Writes `bands_rgbn` (4, H, W) back out with `profile`'s own
    CRS/transform/dtype, unchanged -- required for `ftw_tools.cli
    inference run` to interpret the tile correctly."""
    out_profile = profile.copy()
    out_profile["count"] = bands_rgbn.shape[0]
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(bands_rgbn)


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
