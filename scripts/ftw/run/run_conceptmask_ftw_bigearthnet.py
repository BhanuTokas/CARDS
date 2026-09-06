"""ConceptMask replication of the CounterConcept paper's FTW segmentation
experiment (Section 4.5, notes/ftw_correlation_investigation.md) -- a
pilot run on the currently-available LOCAL FTW tiles (5 countries:
austria/belgium/kenya/portugal/south_africa, ~9,800 tiles total), not
the full 24-country HPC dataset.

Pipeline per concept (one of the 19 BigEarthNet classes,
cards.data.bigearthnet.BIGEARTHNET_19_CLASSES):
1. Retrieval (unlike CelebA/CUB, FTW tiles have no ground-truth BigEarthNet
   label -- CARDS' own standard text-query retrieval, `retrieve_top_bottom_k`,
   picks the K tiles most similar to a text query embedding, exactly how
   CARDS is meant to be used without labeled data).
2. Per present tile: localize the concept via SigLIP patch-similarity
   (RGB-only, on an 8-bit display stretch -- `raw_bands_to_display_rgb`,
   never fed to the black box), threshold to a mask using the established
   best-known config from the CelebA track (z-score alpha=1.0, cutoff
   pooled across all K present tiles for the concept via
   `concept_zscore_cutoff` -- NOT a fixed top_pct%, which was this
   script's own first-draft default before being corrected: "Why are we
   till using TOP_PCT, shouldn't we be using the alpha based technique?"),
   with queries jointly orthogonalized (`orthogonalize_queries`, all 19
   at once, before retrieval) matching that same best-known config's
   orth=True. Generate FTW_FILL_STRATEGIES candidates on the RAW 4-band
   tile (`mask_ftw_tile`, never round-tripping through the display
   image), best-of-family select by embedding-angle-to-query (same logic
   as `masking_mode.masking_score`, reimplemented here since that
   function assumes a live in-process black-box call, not a CLI
   subprocess).
3. Normalize both the original and the selected masked tile's orientation
   (`normalize_tile_orientation` -- FTW's own tiles are universally
   "upside-down," confirmed in v6; skipped here it silently zeros the CLI
   output).
4. Score both via `ftw_tools.cli inference run` (per direct instruction,
   "We should probably use ftw_tools.cli inference run" -- one subprocess
   call per tile, no native batch support, confirmed directly against the
   CLI's own `INPUT` argument definition). Invocations run through a
   N_PARALLEL_INFERENCE-worker thread pool -- measured directly (not
   assumed) that 4 concurrent invocations finish in ~33s vs. ~44s
   sequential (~25% faster, most per-call latency is CUDA-context-init/
   model-load/import overhead rather than actual compute on a tiny
   256x256 tile), and that 8 concurrent gives the SAME per-call
   throughput as 4 (a single shared GPU's compute becomes the ceiling
   past that point) -- 4 is the measured sweet spot, not a guessed
   default.
5. delta = mean(field_channel_original) - mean(field_channel_masked),
   over the WHOLE field-channel raster (get_attr.py's own convention,
   not restricted to pixels predicted as field). raw_score = mean(delta)
   over the K present tiles.

Positive raw_score -> masking the concept REDUCES field-class confidence,
i.e. the concept HELPS predict fields (matches CARDS' sign convention
elsewhere in this project: b(original) - b(masked)).
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.bigearthnet import BIGEARTHNET_19_CLASSES
from cards.data.ftw import (
    FTW_FILL_STRATEGIES,
    load_ftw_tile,
    mask_ftw_tile,
    normalize_tile_orientation,
    raw_bands_to_display_rgb,
    write_ftw_tile,
)
from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k

FTW_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\FTW\ftw")
COUNTRIES = ["austria", "belgium", "kenya", "portugal", "south_africa"]
CHECKPOINT = Path(r"C:\Users\btokas\Projects\Datasets\FTW\checkpoints\3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt")
FTW_BASELINES_VENV_PYTHON = Path(r"C:\Users\btokas\Projects\ftw-baselines\.venv\Scripts\python.exe")
FTW_BASELINES_ROOT = Path(r"C:\Users\btokas\Projects\ftw-baselines")
RESULTS_DIR = Path("results")
FIELD_BAND = 2  # rasterio 1-indexed: band 2 = field (["background","field","boundary"][1])
K = 10
SEED = 0
ALPHA = 1.0  # the established best-known threshold config (CelebA track: z-score alpha=1.0/orth=True/SigLIP)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PARALLEL_INFERENCE = 4  # measured sweet spot -- see module docstring


def list_local_tiles() -> list[Path]:
    tiles = []
    for country in COUNTRIES:
        tiles.extend(sorted((FTW_ROOT / country / "s2_images" / "window_a").glob("*.tif")))
    return tiles


def _clean_subprocess_env() -> dict:
    """subprocess.run inherits the FULL parent environment by default --
    since the parent here is itself a `uv run` Python process (CARDS'
    own .venv, Python 3.11), that inheritance leaks VIRTUAL_ENV/
    PYTHONHOME/PYTHONPATH into the ftw-baselines venv's own Python 3.13
    interpreter, corrupting its own stdlib resolution (observed directly:
    `AssertionError: SRE module mismatch` inside `re`'s own import chain
    -- a classic cross-Python-version module mixing symptom, not an
    ftw_tools bug). Stripping these 3 vars is sufficient -- confirmed by
    the fix actually working, not guessed."""
    import os
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(var, None)
    return env


def run_ftw_inference(tile_path: Path, out_path: Path) -> None:
    # --num_workers 1: the CLI's own default (4) spawns 4 DataLoader worker
    # processes PER invocation, multiplying with our own N_PARALLEL_INFERENCE
    # concurrent invocations -- confirmed directly to exhaust the Windows
    # paging file under that combination ("OSError: The paging file is too
    # small" / "DataLoader worker exited unexpectedly"). Zero benefit anyway
    # for these tiles: patch_size=256 equals the tile size, so there is only
    # ever ONE patch to load per image.
    cmd = [
        str(FTW_BASELINES_VENV_PYTHON), "-m", "ftw_tools.cli", "inference", "run",
        str(tile_path), "--model", str(CHECKPOINT), "--out", str(out_path),
        "--gpu", "0", "--save_scores", "--patch_size", "256", "--padding", "0",
        "--num_workers", "1", "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=str(FTW_BASELINES_ROOT), capture_output=True, text=True, env=_clean_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(f"ftw_tools.cli inference run failed on {tile_path}:\n{result.stderr}")


def read_field_channel_mean(output_tif: Path) -> float:
    with rasterio.open(output_tif) as src:
        return float(src.read(FIELD_BAND).astype(np.float64).mean())


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    tile_paths = list_local_tiles()
    print(f"{len(tile_paths)} local FTW tiles across {len(COUNTRIES)} countries.", flush=True)

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    print("Embedding all local tiles (display-stretched RGB) via SigLIP...", flush=True)
    embeds = []
    batch, batch_size = [], 64
    for i, p in enumerate(tile_paths):
        bands, _ = load_ftw_tile(p)
        batch.append(raw_bands_to_display_rgb(bands))
        if len(batch) == batch_size or i == len(tile_paths) - 1:
            with torch.no_grad():
                embeds.append(encoder.encode_images(batch).cpu())
            batch = []
            if (i + 1) % 1000 == 0:
                print(f"  embedded {i + 1}/{len(tile_paths)}", flush=True)
    embeddings = torch.cat(embeds, dim=0)
    pool = CandidatePool(paths=tile_paths, embeddings=embeddings)
    print(f"pool embeddings: {pool.embeddings.shape}", flush=True)

    # Best-known config (CelebA track: z-score alpha=1.0/orth=True/SigLIP) --
    # queries are jointly orthogonalized BEFORE retrieval, all 19 at once,
    # not built independently per concept.
    raw_queries = {
        c: demean_query(build_concept_query(f"a satellite image of {c.lower()}", encoder), text_center)
        for c in BIGEARTHNET_19_CLASSES
    }
    queries = orthogonalize_queries(raw_queries)

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(max_workers=N_PARALLEL_INFERENCE) as executor:
        tmp = Path(tmpdir)
        for concept_idx, concept_name in enumerate(BIGEARTHNET_19_CLASSES):
            t_c = queries[concept_name]
            t_c_dev = t_c.to(DEVICE)

            present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

            # Phase 0 (sequential, GPU-bound but fast -- ~0.03s/tile measured):
            # localize every present tile FIRST, before any masking -- the
            # z-score cutoff needs all of this concept's own present-set
            # sim_maps pooled together (concept_zscore_cutoff), not a single
            # image's own percentile.
            cached = []  # (idx, bands, profile, display_img, sim_map)
            for idx in present_indices:
                bands, profile = load_ftw_tile(pool.paths[idx])
                display_img = raw_bands_to_display_rgb(bands)
                sim_map = localize_concept(encoder, display_img, t_c, (display_img.height, display_img.width))
                cached.append((idx, bands, profile, display_img, sim_map))
            cutoff = concept_zscore_cutoff([sm for *_, sm in cached], ALPHA)

            # Phase 1 (sequential, GPU-bound but fast): mask every present
            # tile at the shared cutoff, write orig/masked GeoTIFFs with
            # UNIQUE per-tile names (required for phase 2's concurrency).
            n_skipped_degenerate = 0
            jobs = []  # (idx, orig_tif, masked_tif, orig_out, masked_out)
            for idx, bands, profile, display_img, sim_map in cached:
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
                    n_skipped_degenerate += 1
                    continue

                rng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
                candidates_bands = [mask_ftw_tile(bands, mask, s, rng=rng) for s in FTW_FILL_STRATEGIES]
                candidates_display = [raw_bands_to_display_rgb(c) for c in candidates_bands]
                with torch.no_grad():
                    cand_embeds = encoder.encode_images([display_img] + candidates_display).to(DEVICE)
                embed_orig = cand_embeds[0]
                best_angle, best_i = None, 0
                for i in range(len(FTW_FILL_STRATEGIES)):
                    diff = embed_orig - cand_embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
                    angle_deg = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle_deg < best_angle:
                        best_angle, best_i = angle_deg, i
                masked_bands = candidates_bands[best_i]

                orig_bands, orig_profile = normalize_tile_orientation(bands, profile)
                masked_bands_n, masked_profile = normalize_tile_orientation(masked_bands, profile)

                orig_tif = tmp / f"c{concept_idx}_t{idx}_orig.tif"
                masked_tif = tmp / f"c{concept_idx}_t{idx}_masked.tif"
                orig_out = tmp / f"c{concept_idx}_t{idx}_orig_out.tif"
                masked_out = tmp / f"c{concept_idx}_t{idx}_masked_out.tif"
                write_ftw_tile(orig_tif, orig_bands, orig_profile)
                write_ftw_tile(masked_tif, masked_bands_n, masked_profile)
                jobs.append((idx, orig_tif, masked_tif, orig_out, masked_out))

            # Phase 2 (parallel, N_PARALLEL_INFERENCE workers): the CLI
            # subprocess calls, where nearly all latency lives.
            futures = {}
            for idx, orig_tif, masked_tif, orig_out, masked_out in jobs:
                futures[executor.submit(run_ftw_inference, orig_tif, orig_out)] = (idx, "orig")
                futures[executor.submit(run_ftw_inference, masked_tif, masked_out)] = (idx, "masked")
            for future in futures:
                future.result()  # raises if any invocation failed

            deltas = []
            for idx, orig_tif, masked_tif, orig_out, masked_out in jobs:
                field_orig = read_field_channel_mean(orig_out)
                field_masked = read_field_channel_mean(masked_out)
                deltas.append(field_orig - field_masked)

            raw_score = float(np.mean(deltas)) if deltas else float("nan")
            rows.append({"concept_name": concept_name, "raw_score": raw_score,
                         "n_present": len(present_indices), "n_scored": len(deltas),
                         "n_skipped_degenerate": n_skipped_degenerate})
            print(f"[{concept_idx + 1}/{len(BIGEARTHNET_19_CLASSES)}] {concept_name:<70s} "
                  f"raw_score={raw_score:+.4f} (n_scored={len(deltas)}/{len(present_indices)})", flush=True)

    out_path = RESULTS_DIR / "conceptmask_ftw_bigearthnet_pilot.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_name", "raw_score", "n_present", "n_scored", "n_skipped_degenerate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} concept scores to {out_path}", flush=True)


if __name__ == "__main__":
    main()
