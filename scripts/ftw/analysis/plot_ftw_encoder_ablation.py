"""Encoder-ablation bar chart for FTW ConceptMask, prompted directly
("can you also create graphs for the encoder ablation"). Compares the
project's own SigLIP default against two remote-sensing-domain CLIP
variants, RemoteCLIP and GeoRSCLIP (see results/ftw_conceptmask_results/
README.md for full provenance).

SigLIP's whole-dataset number comes from `results/conceptmask_ftw_dual_
window_full_hpc.csv` (the real pooled-across-countries SigLIP run, all
19 concepts, confirmed directly -- "SigLIP overall results should be
available separately"), NOT from averaging the 14 per-country SigLIP
CSVs in results/ftw_conceptmask_results/ (an earlier, weaker proxy this
script used before that file was pointed out). A sibling file,
`conceptmask_ftw_dual_window_full.csv`, is an earlier/incomplete run
(no header, only ~10 concepts) and is not used here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results/ftw_conceptmask_results")
OUT_PATH = RESULTS_DIR / "encoder_ablation_bars.png"
SIGLIP_FULL_CSV = Path("results/conceptmask_ftw_dual_window_full_hpc.csv")

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

ENCODER_COLORS = {"SigLIP": "#3B82F6", "RemoteCLIP": "#F59E0B", "GeoRSCLIP": "#10B981"}


def load_pooled(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.set_index("concept_name")["raw_score_mask_both"]


def main():
    siglip = load_pooled(SIGLIP_FULL_CSV)
    remoteclip = load_pooled(RESULTS_DIR / "conceptmask_ftw_dual_window_remoteclip.csv")
    georsclip = load_pooled(RESULTS_DIR / "conceptmask_ftw_dual_window_georsclip.csv")

    df = pd.DataFrame({
        "SigLIP": siglip,
        "RemoteCLIP": remoteclip,
        "GeoRSCLIP": georsclip,
    }) / 255.0
    df.index = [SHORT_NAMES.get(c.strip(), c.strip()) for c in df.index]
    df = df.sort_values("SigLIP", ascending=False)

    n_concepts = len(df)
    n_encoders = len(df.columns)
    x = np.arange(n_concepts)
    bar_w = 0.8 / n_encoders

    fig, ax = plt.subplots(figsize=(17, 8))
    for i, encoder in enumerate(df.columns):
        ax.bar(x + i * bar_w - 0.4 + bar_w / 2, df[encoder], width=bar_w,
               color=ENCODER_COLORS[encoder], edgecolor="black", linewidth=0.6, label=encoder)

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=45, ha="right", fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel("mean score drop from masking\n(+ supports \"field\", $-$ suppresses \"field\")", fontsize=14)
    ax.legend(fontsize=13, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PATH}  ({n_concepts} concepts x {n_encoders} encoders)", flush=True)


if __name__ == "__main__":
    main()
