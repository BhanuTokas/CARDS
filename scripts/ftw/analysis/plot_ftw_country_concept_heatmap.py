"""Country-level heatmap of ConceptMask's land-cover concept attributions
for the FTW field-segmentation model, prompted directly ("can you help me
create some form of visualization for the country level analysis").

Reads the 14 per-country SigLIP CSVs in results/ftw_conceptmask_results/
(README.md there has the full provenance) and renders a countries x
concepts grid of raw_score_mask_both, a real (not synthetic) drop in the
model's predicted field-channel score when that concept's most-associated
region is masked in both windows. Positive = masking the concept lowered
the field score (the concept supports a "field" prediction, e.g. Arable
land). Negative = masking the concept raised the field score (the concept
suppresses a "field" prediction, e.g. Urban fabric).
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RESULTS_DIR = Path("results/ftw_conceptmask_results")
OUT_PATH = RESULTS_DIR / "country_concept_heatmap.png"
CLIP_MAGNITUDE = 60 / 255.0  # symmetric color-scale bound (in /255 units); a handful of outlier cells exceed this and are clamped

SHORT_NAMES = {
    "Complex cultivation patterns": "Complex cultivation",
    "Permanent crops": "Permanent crops",
    "Arable land": "Arable land",
    "Agro-forestry areas": "Agro-forestry",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Agriculture + natural veg.",
    "Pastures": "Pastures",
    "Natural grassland and sparsely vegetated areas": "Natural grassland",
    "Beaches, dunes, sands": "Beaches/dunes",
    "Transitional woodland-shrub": "Transitional woodland",
    "Coastal wetlands": "Coastal wetlands",
    "Moors, heathland and sclerophyllous vegetation": "Moors/heathland",
    "Coniferous forest": "Coniferous forest",
    "Inland wetlands": "Inland wetlands",
    "Marine waters": "Marine waters",
    "Inland waters": "Inland waters",
    "Broad-leaved forest": "Broad-leaved forest",
    "Mixed forest": "Mixed forest",
    "Industrial or commercial units": "Industrial/commercial",
    "Urban fabric": "Urban fabric",
}


def main():
    files = sorted(glob.glob(str(RESULTS_DIR / "conceptmask_ftw_dual_window_siglip_*.csv")))
    frames = []
    for f in files:
        country = Path(f).stem.split("siglip_")[1].replace("_", " ").title()
        df = pd.read_csv(f)
        df["country"] = country
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["concept_name"] = all_df["concept_name"].map(lambda c: SHORT_NAMES.get(c.strip(), c.strip()))
    all_df["raw_score_mask_both"] = all_df["raw_score_mask_both"] / 255.0

    pivot = all_df.pivot(index="country", columns="concept_name", values="raw_score_mask_both")

    # order concepts by their across-country mean, most field-supporting -> most field-suppressing
    concept_order = pivot.mean(axis=0).sort_values(ascending=False).index
    pivot = pivot[concept_order]
    # order countries by their mean |score|, most-explained -> least-explained
    country_order = pivot.abs().mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[country_order]

    fig, ax = plt.subplots(figsize=(15, 9))
    norm = TwoSlopeNorm(vmin=-CLIP_MAGNITUDE, vcenter=0, vmax=CLIP_MAGNITUDE)
    im = ax.imshow(pivot.values, cmap="coolwarm", norm=norm, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=14)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=15)

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mean score drop from masking\n(+ supports \"field\", $-$ suppresses \"field\")", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}  ({pivot.shape[0]} countries x {pivot.shape[1]} concepts)", flush=True)


if __name__ == "__main__":
    main()
