"""Same-image nulls for HnS on BAM: masked images for the corrected scores.

For each concept c and each image HnS retrieves for it, build N_NULL masks
that remove something other than c, fill them with HnS's best-of-7 rule using
c's query, and save them. The corrected score is b(original) - b(real masked)
minus the mean of b(original) - b(null masked). Same designs as the CelebA-HQ
nulls:

  other  For N_NULL other concepts picked at random, the k pixels most similar
         to that concept's query, outside c's mask (k = the size of c's mask).
  moved  c's own mask shape moved to a random spot overlapping it by at most
         10% (random_placements, strict). Draws with no valid spot are dropped.
  shape  An equal-area random smooth shape outside c's mask. This stands in
         for CelebA's background null: BAM has no pixels that belong to no
         concept, since every pixel is part of the object or the scene.

`other` and `shape` skip an image when c's mask covers more than MAX_FILL of
the area left outside it.

Two options change `other`:

  --exclude-alpha A  Keep null masks off a looser mask of c, thresholded at
                     the pooled mean + A * std (HnS itself uses ALPHA), so
                     they cannot land on the parts of c that HnS's mask
                     missed. The null keeps the size of HnS's mask.
  --on-masked        Apply the null mask to HnS's masked image instead of the
                     original, so the null's fill artifact is measured on the
                     image whose drop it corrects. The correction is then
                     b(real masked) - b(real masked + null).

--fill matched fills every null mask with the fill HnS chose for that image's
real mask (from manifest.csv) instead of re-running best-of-7, so the null
measures the artifact of the fill actually in HnS's drop. Any null.

Masks, cutoffs, fills and seeds match run_hns_bam_masks.py. As a check, the
real masked image is rebuilt and compared pixel for pixel with the saved one.

Writes <bam-root>/hns/demean_<d>/<name>_null/<concept>__<stem>__<j>.png and
<name>_null_manifest.csv (concept, file, j, other_concept, masked_file, area,
base), where <name> is the null plus "_excl" and/or "_stack" for the options,
and base says whether the null was applied to the original or the masked image.
Model outputs come from store_bam_outputs.py afterwards.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from run_hns_bam_masks import ALPHA, CONCEPTS, SEED, retrieve  # noqa: E402

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask  # noqa: E402
from cards.attribution.masking_mode import DEFAULT_FILL_STRATEGIES as FILLS  # noqa: E402
from cards.validation.broden_faithfulness import mask_region, random_placements  # noqa: E402

N_NULL = 3
MAX_FILL = 0.8  # skip when c's mask is too big to fit an equal-area mask outside it
NULL_SEEDS = {"moved": 10_000_000, "shape": 30_000_000, "other": 40_000_000}
GRID = 128


@torch.no_grad()
def best_fill(encoder, image: Image.Image, mask: np.ndarray, rng: np.random.Generator, query: torch.Tensor):
    """HnS's fill choice: the fill whose embedding shift is closest in angle to the query."""
    candidates = [mask_region(image, mask, strategy=s, rng=rng) for s in FILLS]
    embeds = encoder.encode_images([image] + candidates)
    cos = []
    for k in range(len(FILLS)):
        diff = embeds[0] - embeds[1 + k]
        cos.append(float(torch.clamp((diff / diff.norm()) @ query.to(embeds.device), -1, 1)))
    return candidates[int(np.argmax(cos))]


def random_shape(mask: np.ndarray, rng: np.random.Generator, exclude: np.ndarray | None = None) -> np.ndarray:
    """Smooth random shape with the same pixel count as `mask`, entirely outside
    it (or outside `exclude`, a superset of `mask`, when given)."""
    k = int(mask.sum())
    h, w = mask.shape
    sigma = max(1.0, 0.25 * np.sqrt(k / mask.size) * GRID)
    noise = gaussian_filter(rng.standard_normal((GRID, GRID)), sigma, mode="wrap")
    field = zoom(noise, (h / GRID, w / GRID), order=1)[:h, :w] + 1e-9 * rng.standard_normal((h, w))
    field[mask if exclude is None else exclude] = -np.inf
    return field >= np.partition(field.ravel(), -k)[-k]


def null_masks(kind, mask, rng, others, encoder, image, queries, size, exclude=None):
    """[(label, null mask)] for one image; empty when no null fits. `exclude`
    (a superset of `mask`, `other` only) is the region the null must avoid."""
    k = int(mask.sum())
    if kind == "moved":
        places, _ = random_placements(mask, rng, n_draws=N_NULL, strict=True)
        return [("", p) for p in places]
    exclude = mask if exclude is None else exclude
    if k > MAX_FILL * (~exclude).sum():
        return []
    if kind == "shape":
        return [("", random_shape(mask, rng)) for _ in range(N_NULL)]
    out = []
    for other in rng.choice(others, size=N_NULL, replace=False):
        sim = localize_concept(encoder, image, queries[other], size)
        sim[exclude] = -np.inf
        out.append((str(other), sim >= np.partition(sim.ravel(), -k)[-k]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bam-root", type=Path, default=Path(os.environ.get("BAM_ROOT", "../Datasets/BAM")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("CARDS_CACHE_DIR", "embedding_cache")))
    parser.add_argument("--demean", choices=["true", "false"], default="true")
    parser.add_argument("--null", choices=sorted(NULL_SEEDS), default="other")
    parser.add_argument("--exclude-alpha", type=float, default=None,
                        help="other only: keep nulls off c's mask at this looser z-score cutoff")
    parser.add_argument("--on-masked", action="store_true", help="other only: apply nulls to HnS's masked image")
    parser.add_argument("--fill", choices=["best", "matched"], default="best",
                        help="best: HnS's best-of-7 per null; matched: the fill HnS used on the real mask")
    args = parser.parse_args()
    if args.null != "other" and (args.exclude_alpha is not None or args.on_masked):
        parser.error("--exclude-alpha and --on-masked apply to --null other only")
    name = (args.null + ("_excl" if args.exclude_alpha is not None else "") + ("_stack" if args.on_masked else "")
            + ("_matched" if args.fill == "matched" else ""))
    hns_dir = args.bam_root / "hns" / f"demean_{args.demean}"
    out_dir = hns_dir / f"{name}_null"
    out_dir.mkdir(exist_ok=True)

    with open(hns_dir / "manifest.csv", newline="") as handle:
        real_fill = {(r["concept"], r["file"]): r["fill"] for r in csv.DictReader(handle)}
    encoder, pool, results = retrieve(args.bam_root, args.cache_dir, args.demean == "true")
    queries = {c: r.query for c, r in zip(CONCEPTS, results)}
    rows, n_checked, n_mismatch = [], 0, 0
    for ci, (concept, result) in enumerate(zip(CONCEPTS, results)):
        images = {i: Image.open(pool.paths[i]).convert("RGB") for i in result.present_indices}
        size = {i: (im.height, im.width) for i, im in images.items()}
        maps = {i: localize_concept(encoder, images[i], result.query, size[i]) for i in images}
        cutoff = concept_zscore_cutoff(list(maps.values()), ALPHA)
        if args.exclude_alpha is not None:
            loose_cutoff = concept_zscore_cutoff(list(maps.values()), args.exclude_alpha)
        others = [c for c in CONCEPTS if c != concept]
        n_images = 0
        for i, image in images.items():
            mask = threshold_mask(maps[i], method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue
            stem = Path(pool.paths[i]).stem
            real = best_fill(encoder, image, mask, np.random.default_rng(SEED + ci * 10_000 + int(i)), result.query)
            saved_image = Image.open(hns_dir / "masked" / f"{concept}__{stem}.png").convert("RGB")
            n_checked += 1
            n_mismatch += int(not np.array_equal(np.asarray(real), np.asarray(saved_image)))

            exclude = None
            if args.exclude_alpha is not None:
                exclude = mask | threshold_mask(maps[i], method="fixed", cutoff=loose_cutoff)
            rng = np.random.default_rng(NULL_SEEDS[args.null] + ci * 10_000 + int(i))
            nulls = null_masks(args.null, mask, rng, others, encoder, image, queries, size[i], exclude)
            base = saved_image if args.on_masked else image
            for j, (other, null_mask) in enumerate(nulls):
                out_name = f"{concept}__{stem}__{j}.png"
                if args.fill == "matched":
                    filled = mask_region(base, null_mask, strategy=real_fill[concept, f"{stem}.jpg"], rng=rng)
                else:
                    filled = best_fill(encoder, base, null_mask, rng, result.query)
                filled.save(out_dir / out_name)
                rows.append({"concept": concept, "file": f"{stem}.jpg", "j": j, "other_concept": other,
                             "masked_file": out_name, "area": float(null_mask.mean()),
                             "base": "masked" if args.on_masked else "original"})
            n_images += bool(nulls)
        print(f"{concept}: {n_images} images with a null; rebuild mismatches so far {n_mismatch}/{n_checked}",
              flush=True)

    with open(hns_dir / f"{name}_null_manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"done: {len(rows)} {name} null images; real-mask rebuild matched {n_checked - n_mismatch}/{n_checked}",
          flush=True)


if __name__ == "__main__":
    main()
