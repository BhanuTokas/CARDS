"""ConceptMask on the dual-window PRUE model (FTW_PRUE_EFNET_B7_CCBY) --
an explicit EXTENSION beyond the CounterConcept paper's own single-window
RGB analysis (notes v10), not a replication of it. Prompted directly:
"Let's add dual window PRUE for B7."

Differs from run_conceptmask_ftw_bigearthnet.py (the single-window
pipeline) in three real ways, each a direct design decision from notes
v10/v12:

1. RETRIEVAL (which K locations get selected) uses a COMBINED
   window_a+window_b embedding per location -- average the two
   per-window SigLIP image embeddings, then re-normalize to unit norm
   (retrieve_top_bottom_k assumes L2-normalized embeddings for its
   cosine-sim dot product; a plain average of two unit vectors is NOT
   unit-norm).
2. LOCALIZATION (where within a selected location to mask) is
   INDEPENDENT per window -- own similarity map, own pooled z-score
   cutoff, own best-of-family fill selection per window. Corrected after
   direct pushback ("Why not localize per window?") on an initial,
   weaker shared-mask proposal -- reusing one window's mask on the other
   risks masking a stale/mismatched region if the concept's appearance
   genuinely shifted between planting and harvest season, which is
   exactly the seasonal difference this experiment measures.
3. FOUR CLI calls per present location per concept, not two: orig-both
   (baseline), mask-a-only, mask-b-only, mask-both -- all through the
   SAME dual-window model (it always takes both windows as input; only
   which window(s) get masked varies) -- yielding THREE separate
   raw_scores per concept relative to the shared orig-both baseline
   ("what is impact if we only mask window_a, only mask window_b, and
   when we mask both? The actual PRUE model takes both inputs").

A real, EMPIRICALLY CONFIRMED normalization bug (notes v12, not just
reasoned about): the dual-window checkpoint needs input effectively
normalized by /10000, not the CLI's hardcoded /3000 (`default_preprocess`
in ftw_tools/inference/inference.py) -- confirmed directly: unscaled
input produces a confident, saturated ALL-BACKGROUND prediction on tiles
the single-window model finds strong field signal in; pre-scaling by 0.3
(so the CLI's /3000 works out to an effective /10000) recovers a real
prediction matching the single-window model's own result almost exactly.
Fixed via `scale_for_dual_window_cli`, applied ONLY to the 4 dual-window
variant files this script writes -- explicit user instruction: "Make
sure that the 0.3 normalization is only restricted to the dual window
version" -- the single-window script/pipeline never calls it and is
untouched by this change.

Auto-downloads the checkpoint by registry name (FTW_PRUE_EFNET_B7_CCBY)
via ftw_tools' own `torch.hub` caching -- no manual download step.
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
    scale_for_dual_window_cli,
    stack_ftw_windows,
    window_b_path_for,
    write_ftw_tile,
)
from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.retrieval.pool import CandidatePool
from cards.retrieval.retrieve import retrieve_top_bottom_k
import torch.nn.functional as F

# Every machine-specific path is env-var-overridable, same convention as
# run_conceptmask_ftw_bigearthnet.py.
FTW_ROOT = Path(os.environ.get("FTW_ROOT", r"C:\Users\btokas\Projects\Datasets\FTW\ftw"))
CHECKPOINT = os.environ.get("FTW_DUAL_CHECKPOINT", "FTW_PRUE_EFNET_B7_CCBY")  # registry name -> auto-downloads
FTW_BASELINES_VENV_PYTHON = Path(os.environ.get(
    "FTW_BASELINES_VENV_PYTHON", r"C:\Users\btokas\Projects\ftw-baselines\.venv\Scripts\python.exe"))
FTW_BASELINES_ROOT = Path(os.environ.get("FTW_BASELINES_ROOT", r"C:\Users\btokas\Projects\ftw-baselines"))
RESULTS_DIR = Path(os.environ.get("FTW_RESULTS_DIR", "results"))
FIELD_BAND = 2  # rasterio 1-indexed: band 2 = field
K = int(os.environ.get("FTW_DUAL_K", "10"))  # small pilot value first -- untested pipeline, validate before scaling
SEED = 0
ALPHA = 1.0  # established best-known threshold config (CelebA/FTW single-window track)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PARALLEL_INFERENCE = int(os.environ.get("FTW_N_PARALLEL_INFERENCE", str(os.cpu_count() or 4)))


def list_local_window_a_tiles() -> list[Path]:
    """Same auto-discovery as the single-window script, but only returns
    window_a paths WHERE a matching window_b file also exists (dual-window
    needs both)."""
    countries = sorted(p.name for p in FTW_ROOT.iterdir() if p.is_dir() and (p / "s2_images" / "window_a").is_dir())
    tiles = []
    n_missing_b = 0
    for country in countries:
        for a_path in sorted((FTW_ROOT / country / "s2_images" / "window_a").glob("*.tif")):
            if window_b_path_for(a_path).exists():
                tiles.append(a_path)
            else:
                n_missing_b += 1
    print(f"{len(countries)} countries found under {FTW_ROOT}: {countries}", flush=True)
    if n_missing_b:
        print(f"  ({n_missing_b} window_a tiles skipped -- no matching window_b file)", flush=True)
    return tiles


def _clean_subprocess_env() -> dict:
    env = os.environ.copy()
    for var in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(var, None)
    return env


def run_ftw_inference(tile_path: Path, out_path: Path) -> None:
    cmd = [
        str(FTW_BASELINES_VENV_PYTHON), "-m", "ftw_tools.cli", "inference", "run",
        str(tile_path), "--model", str(CHECKPOINT), "--out", str(out_path),
        "--save_scores", "--patch_size", "256", "--padding", "0",
        "--num_workers", "1", "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=str(FTW_BASELINES_ROOT), capture_output=True, text=True, env=_clean_subprocess_env())
    if result.returncode != 0:
        raise RuntimeError(
            f"ftw_tools.cli inference run failed (exit {result.returncode}) on {tile_path}:\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def read_field_channel_mean(output_tif: Path) -> float:
    with rasterio.open(output_tif) as src:
        return float(src.read(FIELD_BAND).astype(np.float64).mean())


def build_pool(tile_paths: list[Path], encoder) -> CandidatePool:
    """Combined window_a+window_b embedding per location: average the two
    per-window image embeddings, then re-normalize to unit norm (a plain
    average of two unit vectors is not itself unit-norm, and
    retrieve_top_bottom_k assumes L2-normalized embeddings for its
    cosine-sim dot product)."""
    print(f"Embedding {len(tile_paths)} locations (combined window_a+window_b) via SigLIP...", flush=True)
    embeds = []
    batch_a, batch_b, batch_size = [], [], 32  # halved vs single-window: 2x images per location per batch slot
    for i, a_path in enumerate(tile_paths):
        bands_a, _ = load_ftw_tile(a_path)
        bands_b, _ = load_ftw_tile(window_b_path_for(a_path))
        batch_a.append(raw_bands_to_display_rgb(bands_a))
        batch_b.append(raw_bands_to_display_rgb(bands_b))
        if len(batch_a) == batch_size or i == len(tile_paths) - 1:
            with torch.no_grad():
                embed_a = encoder.encode_images(batch_a)
                embed_b = encoder.encode_images(batch_b)
                combined = F.normalize(embed_a + embed_b, dim=-1).cpu()
            embeds.append(combined)
            batch_a, batch_b = [], []
            if (i + 1) % 1000 == 0:
                print(f"  embedded {i + 1}/{len(tile_paths)}", flush=True)
    embeddings = torch.cat(embeds, dim=0)
    return CandidatePool(paths=tile_paths, embeddings=embeddings)


def build_masked_tiles_for_concept(pool, encoder, concept_idx, concept_name, t_c, out_dir: Path):
    """Retrieval (combined embedding, already in pool) + per-window
    independent localization + 4-condition masking for one concept.
    Writes 4 files per non-degenerate present location (orig, mask_a,
    mask_b, mask_both), each already scaled via scale_for_dual_window_cli
    and orientation-normalized. Returns (jobs, n_skipped_degenerate,
    n_present) where jobs is [(idx, orig_tif, mask_a_tif, mask_b_tif,
    mask_both_tif), ...].

    A location is skipped entirely if EITHER window's localization is
    degenerate (empty/full mask) -- conservative, avoids partial-
    availability bookkeeping across the 3 masking conditions."""
    t_c_dev = t_c.to(DEVICE)
    present_indices, _ = retrieve_top_bottom_k(pool, t_c, K)

    # Phase 0: localize BOTH windows for every present location first --
    # the z-score cutoff needs each window's own present-set sim_maps
    # pooled separately (window_a's cutoff from window_a maps only, same
    # for window_b), not a single image's own percentile.
    cached = []  # (idx, bands_a, profile_a, bands_b, sim_map_a, sim_map_b)
    for idx in present_indices:
        a_path = pool.paths[idx]
        bands_a, profile_a = load_ftw_tile(a_path)
        bands_b, profile_b = load_ftw_tile(window_b_path_for(a_path))
        display_a = raw_bands_to_display_rgb(bands_a)
        display_b = raw_bands_to_display_rgb(bands_b)
        sim_map_a = localize_concept(encoder, display_a, t_c, (display_a.height, display_a.width))
        sim_map_b = localize_concept(encoder, display_b, t_c, (display_b.height, display_b.width))
        cached.append((idx, bands_a, profile_a, bands_b, sim_map_a, sim_map_b))

    cutoff_a = concept_zscore_cutoff([c[4] for c in cached], ALPHA)
    cutoff_b = concept_zscore_cutoff([c[5] for c in cached], ALPHA)

    n_skipped_degenerate = 0
    jobs = []
    for idx, bands_a, profile_a, bands_b, sim_map_a, sim_map_b in cached:
        mask_a = threshold_mask(sim_map_a, method="fixed", cutoff=cutoff_a)
        mask_b = threshold_mask(sim_map_b, method="fixed", cutoff=cutoff_b)
        if not mask_a.any() or mask_a.all() or not mask_b.any() or mask_b.all():
            n_skipped_degenerate += 1
            continue

        seed_a = SEED + concept_idx * 10_000 + int(idx) * 2
        seed_b = seed_a + 1
        rng_a = np.random.default_rng(seed_a)
        rng_b = np.random.default_rng(seed_b)

        display_a = raw_bands_to_display_rgb(bands_a)
        display_b = raw_bands_to_display_rgb(bands_b)

        cand_bands_a = [mask_ftw_tile(bands_a, mask_a, s, rng=rng_a) for s in FTW_FILL_STRATEGIES]
        cand_disp_a = [raw_bands_to_display_rgb(c) for c in cand_bands_a]
        with torch.no_grad():
            embeds_a = encoder.encode_images([display_a] + cand_disp_a).to(DEVICE)
        best_i_a = _best_fill_index(embeds_a, t_c_dev)
        masked_bands_a = cand_bands_a[best_i_a]

        cand_bands_b = [mask_ftw_tile(bands_b, mask_b, s, rng=rng_b) for s in FTW_FILL_STRATEGIES]
        cand_disp_b = [raw_bands_to_display_rgb(c) for c in cand_bands_b]
        with torch.no_grad():
            embeds_b = encoder.encode_images([display_b] + cand_disp_b).to(DEVICE)
        best_i_b = _best_fill_index(embeds_b, t_c_dev)
        masked_bands_b = cand_bands_b[best_i_b]

        variants = {
            "orig": stack_ftw_windows(bands_a, bands_b, profile_a),
            "mask_a": stack_ftw_windows(masked_bands_a, bands_b, profile_a),
            "mask_b": stack_ftw_windows(bands_a, masked_bands_b, profile_a),
            "mask_both": stack_ftw_windows(masked_bands_a, masked_bands_b, profile_a),
        }

        paths = {}
        for name, stacked in variants.items():
            stacked_n, profile_n = normalize_tile_orientation(stacked, profile_a)
            scaled = scale_for_dual_window_cli(stacked_n)
            p = out_dir / f"c{concept_idx}_t{idx}_{name}.tif"
            write_ftw_tile(p, scaled, profile_n)
            paths[name] = p

        jobs.append((idx, paths["orig"], paths["mask_a"], paths["mask_b"], paths["mask_both"]))

    return jobs, n_skipped_degenerate, len(present_indices)


def _best_fill_index(cand_embeds: torch.Tensor, t_c_dev: torch.Tensor) -> int:
    embed_orig = cand_embeds[0]
    best_angle, best_i = None, 0
    for i in range(len(FTW_FILL_STRATEGIES)):
        diff = embed_orig - cand_embeds[1 + i]
        diff_unit = diff / diff.norm()
        cos_sim = float(torch.clamp(diff_unit @ t_c_dev, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_sim)))
        if best_angle is None or angle_deg < best_angle:
            best_angle, best_i = angle_deg, i
    return best_i


def score_jobs(jobs, out_dir: Path, executor: ThreadPoolExecutor):
    """Runs the CLI on all 4 files per job, parallel. Returns
    (deltas_a, deltas_b, deltas_both, n_failed_inference) -- a location
    only contributes to a given condition's deltas if BOTH its orig AND
    that condition's masked-variant CLI call succeeded; per-tile
    failures are logged and excluded, not fatal (same resilience pattern
    as the single-window pipeline)."""
    futures = {}
    out_paths = {}
    for idx, orig_tif, mask_a_tif, mask_b_tif, mask_both_tif in jobs:
        out_paths[idx] = {}
        for name, tif in (("orig", orig_tif), ("mask_a", mask_a_tif), ("mask_b", mask_b_tif), ("mask_both", mask_both_tif)):
            out_tif = out_dir / f"{tif.stem}_out.tif"
            out_paths[idx][name] = out_tif
            futures[executor.submit(run_ftw_inference, tif, out_tif)] = (idx, name, tif)

    n_failed_inference = 0
    failed = set()  # (idx, name)
    for future in futures:
        idx, name, tif = futures[future]
        try:
            future.result()
        except Exception as e:
            print(f"  WARNING: inference failed on {tif}, skipping -- {e}", flush=True)
            failed.add((idx, name))
            n_failed_inference += 1

    deltas_a, deltas_b, deltas_both = [], [], []
    for idx, *_ in jobs:
        if (idx, "orig") in failed:
            continue
        field_orig = read_field_channel_mean(out_paths[idx]["orig"])
        if (idx, "mask_a") not in failed:
            deltas_a.append(field_orig - read_field_channel_mean(out_paths[idx]["mask_a"]))
        if (idx, "mask_b") not in failed:
            deltas_b.append(field_orig - read_field_channel_mean(out_paths[idx]["mask_b"]))
        if (idx, "mask_both") not in failed:
            deltas_both.append(field_orig - read_field_channel_mean(out_paths[idx]["mask_both"]))

    return deltas_a, deltas_b, deltas_both, n_failed_inference


def main():
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    tile_paths = list_local_window_a_tiles()
    print(f"{len(tile_paths)} dual-window-capable locations, K={K}, N_PARALLEL_INFERENCE={N_PARALLEL_INFERENCE}.", flush=True)

    cfg = OmegaConf.create({
        "seed": SEED, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)

    pool = build_pool(tile_paths, encoder)
    print(f"pool embeddings: {pool.embeddings.shape}", flush=True)

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
            jobs, n_skipped_degenerate, n_present = build_masked_tiles_for_concept(
                pool, encoder, concept_idx, concept_name, t_c, tmp)
            deltas_a, deltas_b, deltas_both, n_failed = score_jobs(jobs, tmp, executor)

            score_a = float(np.mean(deltas_a)) if deltas_a else float("nan")
            score_b = float(np.mean(deltas_b)) if deltas_b else float("nan")
            score_both = float(np.mean(deltas_both)) if deltas_both else float("nan")
            rows.append({
                "concept_name": concept_name,
                "raw_score_mask_a": score_a, "raw_score_mask_b": score_b, "raw_score_mask_both": score_both,
                "n_present": n_present, "n_masked": len(jobs), "n_skipped_degenerate": n_skipped_degenerate,
                "n_failed_inference": n_failed,
            })
            print(f"[{concept_idx + 1}/{len(BIGEARTHNET_19_CLASSES)}] {concept_name:<70s} "
                  f"mask_a={score_a:+.4f} mask_b={score_b:+.4f} mask_both={score_both:+.4f} "
                  f"(n_masked={len(jobs)}/{n_present}, n_failed={n_failed})", flush=True)

    out_name = os.environ.get("FTW_DUAL_OUTPUT_NAME", "conceptmask_ftw_dual_window_pilot.csv")
    out_path = RESULTS_DIR / out_name
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "concept_name", "raw_score_mask_a", "raw_score_mask_b", "raw_score_mask_both",
            "n_present", "n_masked", "n_skipped_degenerate", "n_failed_inference",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} concept scores to {out_path}", flush=True)


if __name__ == "__main__":
    main()
