"""Collect the FunnyBirds result JSONs (evaluate.py) into one table (CSV + Markdown).

Run from the repo root:  python scripts/funnybirds/make_table.py [results_dir]
Reads <results_dir>/<model>_<explainer>[_tag].json (default $FUNNYBIRDS_OUT) and writes funnybirds_table.{csv,md}
there. Columns as in Hesse et al. (2023), Table 2: accuracy, CSDC, PC, DC, D, BI, SD, TS, then Com, Cor (= SD),
Con (= TS) and mX = mean(Com, Cor, Con). Also D_parts / mX_parts: distractibility with background objects left out
of every method's explanation (HnS never scores background objects, so plain D favours it).
The framework README's ResNet-50 InputXGradient row is printed for the reproduction check.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from paths import OUT

ORDER = ["InputXGradient", "IntegratedGradients", "TCAV", "TCAV_variant", "PCBM", "PCBM_CLIP_RN50", "PCBM_SigLIP", "ConceptSAE", "ConceptSHAP", "EAC", "HnS",
         "HnS_L1024_a1.5", "HnS_L1024_a2.5", "HnSTrueMask"]
LABEL = {
    "InputXGradient": "Input x Gradient",
    "IntegratedGradients": "Integrated Gradients",
    "TCAV": "TCAV (layer3, part-presence CAVs)",
    "TCAV_variant": "TCAV (layer3, variant CAVs)",
    "PCBM": "PCBM (conventional, 26 variant CAVs)",
    "PCBM_CLIP_RN50": "PCBM (CLIP-RN50 concepts)",
    "PCBM_SigLIP": "PCBM (SigLIP concepts)",
    "ConceptSAE": "ConceptSAE (26 variant-named latents)",
    "ConceptSHAP": "ConceptSHAP (26 variant-named vectors)",
    "EAC": "EAC",
    "HnS": "HnS (paper: L=196, alpha=1)",
    "HnS_L1024_a1.5": "HnS (L=1024, alpha=1.5; selected on train)",
    "HnS_L1024_a2.5": "HnS (L=1024, alpha=2.5; sensitivity)",
    "HnSTrueMask": "HnS with true part masks (oracle)",
}
COLS = ["accuracy", "CSDC", "PC", "DC", "D", "BI", "SD", "TS", "Com", "Cor", "Con", "mX", "D_parts", "mX_parts"]
README_IXG = {"accuracy": .998, "CSDC": .7353, "PC": .602, "DC": .532, "D": .54372, "BI": .99826, "SD": .54592,
              "TS": .806}


def main():
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    rows = []
    for f in sorted(d.glob("*.json")):
        r = json.loads(f.read_text())
        meta, final = r["meta"], r["final"]
        key = f.stem[len(meta["model"]) + 1:]  # explainer[_tag]; model names contain underscores (vit_b_16)
        label = LABEL.get(key, key)
        if "pcbm_fidelity" in meta:  # surrogate agreement with the black box on held-out training images
            label += f", fidelity {meta['pcbm_fidelity']:.2f}"
        if "cshap_completeness" in meta:  # Yeh et al.'s completeness on held-out training images
            label += f", completeness {meta['cshap_completeness']:.2f}"
        if "sae_fvu" in meta:  # held-out reconstruction (FVU) and prediction agreement through the SAE
            label += f", FVU {meta['sae_fvu']:.2f}, agreement {meta['sae_agreement']:.2f}"
        rows.append({"model": meta["model"], "key": key, "method": label,
                     "n_itrs": meta["nr_itrs"], "minutes": round(meta["minutes"], 1)} | final)
    rows.sort(key=lambda r: (r["model"], ORDER.index(r["key"]) if r["key"] in ORDER else 99))
    if not rows:
        sys.exit(f"no result JSONs in {d}")
    with open(d / "funnybirds_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["model", "method", *COLS, "threshold", "n_itrs", "minutes"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    lines = ["| Model | Method | " + " | ".join(COLS) + " |", "|" + "---|" * (len(COLS) + 2)]
    for r in rows:
        lines.append(f"| {r['model']} | {r['method']} | " + " | ".join(f"{r[c]:.3f}" for c in COLS) + " |")
    lines.append("| resnet50 | IxG, framework README | " + " | ".join(
        f"{README_IXG[c]:.3f}" if c in README_IXG else "" for c in COLS) + " |")
    (d / "funnybirds_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    for r in rows:
        if r["n_itrs"] < 2501:
            print(f"note: {r['key']} ran on a subset (nr_itrs={r['n_itrs']})")


if __name__ == "__main__":
    main()
