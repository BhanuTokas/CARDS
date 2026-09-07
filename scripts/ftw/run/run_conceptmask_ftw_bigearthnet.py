"""ConceptMask replication of the CounterConcept paper's FTW segmentation
experiment (Section 4.5, notes/ftw_correlation_investigation.md).

Runs unchanged against either the local 5-country pilot pool or the full
24-country HPC dataset -- COUNTRIES is auto-discovered from whatever
subdirectories actually exist under FTW_ROOT (see v9 in the notes), and
every machine-specific path is env-var-overridable (defaults are this
machine's local Windows paths, so the local pilot invocation is
unaffected). No country-stratified retrieval is used deliberately --
some countries in the full dataset have too few local samples to
support it evenly (explicit user instruction, 2026-09-06).

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
   CLI's own `INPUT` argument definition), run CPU-only (`--gpu` omitted
   -> the CLI's own `gpu=None` -> `gpu=-1` -> CPU device path, confirmed
   directly in ftw_tools/inference/inference.py -- no hack needed).
   Deliberately NOT GPU: measured directly that a single 256x256-tile
   invocation shows NO GPU-vs-CPU speedup at all (~10-13s either way --
   per-call latency is dominated by torch import/model-load/CUDA-init
   overhead, not compute), so GPU only added queue contention with zero
   benefit (notes v7). Invocations run through an N_PARALLEL_INFERENCE-
   worker thread pool -- on the single local GPU this was capped at 4
   (8 concurrent gave the same per-call throughput -- the shared GPU's
   compute became the ceiling); running CPU-only removes that specific
   ceiling, so N_PARALLEL_INFERENCE defaults to the machine's own CPU
   count (`os.cpu_count()`) unless overridden, and should be re-measured
   at whatever core count an HPC allocation actually grants rather than
   assuming the local number 4 still applies.
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
import os
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

# Every machine-specific path is env-var-overridable; defaults are this
# machine's local Windows layout so the local pilot invocation is
# unaffected. On the HPC server, set these 4 vars to that filesystem's
# real locations before running (see scripts/ftw/run/submit_ftw_hpc.sh, or
# for the two-job HPC split, scripts/ftw/build/submit_ftw_build_masked_tiles.sh
# + scripts/ftw/run/submit_ftw_hpc.sh).
FTW_ROOT = Path(os.environ.get("FTW_ROOT", r"C:\Users\btokas\Projects\Datasets\FTW\ftw"))
CHECKPOINT = Path(os.environ.get(
    "FTW_CHECKPOINT", r"C:\Users\btokas\Projects\Datasets\FTW\checkpoints\3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt"))
FTW_BASELINES_VENV_PYTHON = Path(os.environ.get(
    "FTW_BASELINES_VENV_PYTHON", r"C:\Users\btokas\Projects\ftw-baselines\.venv\Scripts\python.exe"))
FTW_BASELINES_ROOT = Path(os.environ.get("FTW_BASELINES_ROOT", r"C:\Users\btokas\Projects\ftw-baselines"))
RESULTS_DIR = Path(os.environ.get("FTW_RESULTS_DIR", "results"))
FIELD_BAND = 2  # rasterio 1-indexed: band 2 = field (["background","field","boundary"][1])
K = int(os.environ.get("FTW_K", "10"))
SEED = 0
ALPHA = 1.0  # the established best-known threshold config (CelebA track: z-score alpha=1.0/orth=True/SigLIP)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # only affects the CARDS-side SigLIP embedding step
# CLI inference runs CPU-only regardless of DEVICE (see module docstring: measured
# no GPU-vs-CPU speedup for a single small-tile invocation). Not capped at 4 the
# way the single-shared-GPU pilot was -- defaults to all available CPU cores;
# override via env if that's too aggressive for the allocation's shared usage.
N_PARALLEL_INFERENCE = int(os.environ.get("FTW_N_PARALLEL_INFERENCE", str(os.cpu_count() or 4)))
# Optional: path to a cached embedding tensor (see build_pool()) -- lets the
# real-compute embedding step run once on a GPU node (scripts/ftw/build/
# embed_ftw_tiles.py) while this script's own long per-concept loop runs
# CPU-only on a separate, cheaper allocation. Unset (default) -> embeds
# in-process, on whatever DEVICE is active, no caching.
_embedding_cache_env = os.environ.get("FTW_EMBEDDING_CACHE")
EMBEDDING_CACHE = Path(_embedding_cache_env) if _embedding_cache_env else None


def list_local_tiles() -> list[Path]:
    countries = sorted(p.name for p in FTW_ROOT.iterdir() if p.is_dir() and (p / "s2_images" / "window_a").is_dir())
    tiles = []
    for country in countries:
        tiles.extend(sorted((FTW_ROOT / country / "s2_images" / "window_a").glob("*.tif")))
    print(f"{len(countries)} countries found under {FTW_ROOT}: {countries}", flush=True)
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
    # No --gpu flag: the CLI's own gpu=None -> gpu=-1 -> CPU device path
    # (confirmed in ftw_tools/inference/inference.py). CPU-only per the
    # measured no-speedup finding -- see module docstring.
    cmd = [
        str(FTW_BASELINES_VENV_PYTHON), "-m", "ftw_tools.cli", "inference", "run",
        str(tile_path), "--model", str(CHECKPOINT), "--out", str(out_path),
        "--save_scores", "--patch_size", "256", "--padding", "0",
        "--num_workers", "1", "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=str(FTW_BASELINES_ROOT), capture_output=True, text=True, env=_clean_subprocess_env())
    if result.returncode != 0:
        # Both streams, not just stderr -- a raw process abort (SIGABRT,
        # "Aborted!") often prints its real diagnostic (GDAL/glibc/
        # std::bad_alloc messages) to stdout or splits across both, and
        # stderr alone was observed to show only the bare word "Aborted!"
        # with zero other context, unhelpfully.
        raise RuntimeError(
            f"ftw_tools.cli inference run failed (exit {result.returncode}) on {tile_path}:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def read_field_channel_mean(output_tif: Path) -> float:
    with rasterio.open(output_tif) as src:
        return float(src.read(FIELD_BAND).astype(np.float64).mean())


def build_masked_tiles_for_concept(pool, encoder, concept_idx, concept_name, t_c, out_dir: Path):
    """Phase 0+1 for one concept: retrieval, localization, and best-of-
    family masking -- writes orig/masked GeoTIFF pairs into out_dir (must
    already exist) and returns (jobs, n_skipped_degenerate, n_present)
    where jobs is [(idx, orig_tif, masked_tif), ...].

    This is the ENCODER-dependent half of the pipeline: `localize_concept`
    (patch-similarity) and `encode_images` (best-of-family fill selection)
    are real batched neural-net forward passes that benefit from GPU --
    UNLIKE Phase 2's `ftw_tools.cli inference run` subprocess calls, which
    are measured to get zero benefit from GPU (see module docstring).
    Split out as its own function specifically so a GPU-only job can run
    just this, for every concept, while a separate CPU-only job runs
    Phase 2 -- direct feedback: "wouldn't concept masking also require GPU
    as we verify which type of masking works best?" -- correct; see
    scripts/ftw/build/build_ftw_masked_tiles.py and notes v10.
    """
    t_c_dev = t_c.to(DEVICE)
    present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

    # localize every present tile FIRST, before any masking -- the z-score
    # cutoff needs all of this concept's own present-set sim_maps pooled
    # together (concept_zscore_cutoff), not a single image's own percentile.
    cached = []  # (idx, bands, profile, display_img, sim_map)
    for idx in present_indices:
        bands, profile = load_ftw_tile(pool.paths[idx])
        display_img = raw_bands_to_display_rgb(bands)
        sim_map = localize_concept(encoder, display_img, t_c, (display_img.height, display_img.width))
        cached.append((idx, bands, profile, display_img, sim_map))
    cutoff = concept_zscore_cutoff([sm for *_, sm in cached], ALPHA)

    n_skipped_degenerate = 0
    jobs = []  # (idx, orig_tif, masked_tif)
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

        orig_tif = out_dir / f"c{concept_idx}_t{idx}_orig.tif"
        masked_tif = out_dir / f"c{concept_idx}_t{idx}_masked.tif"
        write_ftw_tile(orig_tif, orig_bands, orig_profile)
        write_ftw_tile(masked_tif, masked_bands_n, masked_profile)
        jobs.append((idx, orig_tif, masked_tif))

    return jobs, n_skipped_degenerate, len(present_indices)


def score_jobs(jobs, out_dir: Path, executor: ThreadPoolExecutor) -> list[float]:
    """Phase 2 for one concept's jobs: `ftw_tools.cli inference run` on
    every (idx, orig_tif, masked_tif), parallel + CPU-only, ZERO encoder/
    GPU dependency -- reusable standalone from a CPU-only job that never
    imports/instantiates an encoder at all (scripts/ftw/run/
    score_ftw_masked_tiles.py)."""
    futures = {}
    out_paths = {}
    for idx, orig_tif, masked_tif in jobs:
        orig_out = out_dir / f"{orig_tif.stem}_out.tif"
        masked_out = out_dir / f"{masked_tif.stem}_out.tif"
        out_paths[idx] = (orig_out, masked_out)
        futures[executor.submit(run_ftw_inference, orig_tif, orig_out)] = None
        futures[executor.submit(run_ftw_inference, masked_tif, masked_out)] = None
    for future in futures:
        future.result()  # raises if any invocation failed

    deltas = []
    for idx, orig_tif, masked_tif in jobs:
        orig_out, masked_out = out_paths[idx]
        field_orig = read_field_channel_mean(orig_out)
        field_masked = read_field_channel_mean(masked_out)
        deltas.append(field_orig - field_masked)
    return deltas


def build_pool(tile_paths: list[Path], encoder) -> CandidatePool:
    """Embeds every tile via SigLIP, or loads a cached embedding tensor if
    FTW_EMBEDDING_CACHE is set and already exists.

    Split out as its own function (not inlined in main()) so this one-time,
    real-batched-compute step can be precomputed by a SEPARATE short GPU job
    (scripts/ftw/build/embed_ftw_tiles.py) while the long per-concept
    masking+inference loop runs CPU-only -- avoids holding a scarce GPU idle
    for the ~95% of wall time actually spent in CPU-only
    `ftw_tools.cli inference run` subprocess calls (direct feedback: "wouldn't
    majority of time be spent on compute that doesn't use GPU? That seems
    wasteful" -- correct, see notes v9). Also works standalone with no cache
    set: computes embeddings in-process on whatever DEVICE is active.
    """
    if EMBEDDING_CACHE and EMBEDDING_CACHE.exists():
        cached = torch.load(EMBEDDING_CACHE)
        cached_paths = [Path(p) for p in cached["paths"]]
        if cached_paths != tile_paths:
            raise RuntimeError(
                f"Embedding cache {EMBEDDING_CACHE} was built from a different tile set "
                f"({len(cached_paths)} tiles vs {len(tile_paths)} now) -- delete and rebuild it "
                f"(scripts/ftw/build/embed_ftw_tiles.py) rather than silently using stale embeddings."
            )
        print(f"Loaded cached embeddings from {EMBEDDING_CACHE} ({cached['embeddings'].shape})", flush=True)
        return CandidatePool(paths=tile_paths, embeddings=cached["embeddings"])

    print("Embedding all tiles (display-stretched RGB) via SigLIP (no cache found/configured)...", flush=True)
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

    if EMBEDDING_CACHE:
        EMBEDDING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"paths": [str(p) for p in tile_paths], "embeddings": embeddings}, EMBEDDING_CACHE)
        print(f"Saved embeddings to {EMBEDDING_CACHE}", flush=True)

    return CandidatePool(paths=tile_paths, embeddings=embeddings)


def main():
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    tile_paths = list_local_tiles()
    print(f"{len(tile_paths)} FTW tiles total, K={K}, N_PARALLEL_INFERENCE={N_PARALLEL_INFERENCE}.", flush=True)

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool = build_pool(tile_paths, encoder)
    print(f"pool embeddings: {pool.embeddings.shape}", flush=True)

    # Best-known config (CelebA track: z-score alpha=1.0/orth=True/SigLIP) --
    # queries are jointly orthogonalized BEFORE retrieval, all 19 at once,
    # not built independently per concept.
    raw_queries = {
        c: demean_query(build_concept_query(f"a satellite image of {c.lower()}", encoder), text_center)
        for c in BIGEARTHNET_19_CLASSES
    }
    queries = orthogonalize_queries(raw_queries)

    # Monolithic single-job path (used locally, where GPU is available for
    # the whole run anyway -- no benefit to splitting): both phases share
    # one tempdir and one thread pool per concept, back to back. On HPC,
    # scripts/ftw/build/build_ftw_masked_tiles.py (GPU) and scripts/ftw/
    # run/score_ftw_masked_tiles.py (CPU-only) call the same two
    # functions (build_masked_tiles_for_concept, score_jobs) separately,
    # across two jobs, with a persistent manifest in between.
    rows = []
    with tempfile.TemporaryDirectory() as tmpdir, ThreadPoolExecutor(max_workers=N_PARALLEL_INFERENCE) as executor:
        tmp = Path(tmpdir)
        for concept_idx, concept_name in enumerate(BIGEARTHNET_19_CLASSES):
            t_c = queries[concept_name]
            jobs, n_skipped_degenerate, n_present = build_masked_tiles_for_concept(
                pool, encoder, concept_idx, concept_name, t_c, tmp)
            deltas = score_jobs(jobs, tmp, executor)

            raw_score = float(np.mean(deltas)) if deltas else float("nan")
            rows.append({"concept_name": concept_name, "raw_score": raw_score,
                         "n_present": n_present, "n_scored": len(deltas),
                         "n_skipped_degenerate": n_skipped_degenerate})
            print(f"[{concept_idx + 1}/{len(BIGEARTHNET_19_CLASSES)}] {concept_name:<70s} "
                  f"raw_score={raw_score:+.4f} (n_scored={len(deltas)}/{n_present})", flush=True)

    out_name = os.environ.get("FTW_OUTPUT_NAME", "conceptmask_ftw_bigearthnet_pilot.csv")
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept_name", "raw_score", "n_present", "n_scored", "n_skipped_degenerate"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} concept scores to {out_path}", flush=True)


if __name__ == "__main__":
    main()
