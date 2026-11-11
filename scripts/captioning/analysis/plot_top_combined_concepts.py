"""Bar charts for the top male/female-associated concepts under the
COMBINED COCO+Broden concept set, prompted directly ("can you give me
the updated figures with both concepts?"). Only covers the 5 self-
generated modern-VLM models (vit_gpt2, blip, florence, llava,
bakllava) -- Broden captions don't exist for the other 5 externally-
sourced models yet, confirmed directly with the user before running
this ("I want to see the initial results with these models, will add
other models soon").

Male concept ("tie") is an unambiguous 4/5-model top vote-getter.
Female concept ("hair drier") was a 3-way tie at 4/5 votes with
"eyebrow" and "dog"; user picked "hair drier" directly to match the
original 10-model figure's own concept.

Matches `top_tie_per_model.png`/`top_cake_per_model.png`'s own style
(blue = masculine-leaning/positive, pink = feminine-leaning/negative,
per-bar sign, not a fixed per-model color).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path("results/captioning_bias_combined_attribution")
OUT_DIR = RESULTS_DIR
MODEL_ORDER = ["vit_gpt2", "blip", "florence", "llava", "bakllava"]
MODEL_LABELS = {"vit_gpt2": "ViT-GPT2", "blip": "BLIP", "florence": "Florence-2",
                "llava": "LLaVA", "bakllava": "BakLLaVA"}

BLUE = "#3B82F6"
PINK = "#EC1876"

CONCEPTS = [("tie", "coco", "Top male-associated concept (COCO+Broden): \"tie\""),
            ("hair drier", "coco", "Top female-associated concept (COCO+Broden): \"hair drier\"")]


def load_delta(model: str, concept_name: str, source: str) -> float | None:
    path = RESULTS_DIR / f"{model}_combined_gender_attribution.csv"
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["concept_name"] == concept_name and row["source"] == source:
                return float(row["mean_gender_delta"])
    return None


def plot_concept(concept_name: str, source: str, title: str, out_path: Path):
    labels, deltas = [], []
    for model in MODEL_ORDER:
        delta = load_delta(model, concept_name, source)
        if delta is None:
            continue
        labels.append(MODEL_LABELS[model])
        deltas.append(delta)

    colors = [BLUE if d >= 0 else PINK for d in deltas]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.bar(labels, deltas, color=colors, edgecolor="black", linewidth=1.0, width=0.7)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("mean_gender_delta\n(+ masculine-leaning, - feminine-leaning)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path} ({len(labels)} models)", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for concept_name, source, title in CONCEPTS:
        slug = concept_name.replace(" ", "_")
        plot_concept(concept_name, source, title, OUT_DIR / f"top_{slug}_combined_per_model.png")


if __name__ == "__main__":
    main()
