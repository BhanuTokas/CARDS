"""Full-scale wall-clock timing benchmark, prompted directly ("Can we
compare the time for full scale run on the CelebA dataset?") -- extends
benchmark_computational_cost.py's matched small-scale benchmark
(1000-image pool, K=15, single task) to the ACTUAL production scale used
for this project's headline CelebA results:

- ConceptMask: full official-val clean retrieval pool (~16,874 images,
  HQ-train/HQ-val leakage excluded, same as every headline CelebA run),
  K=50.
- PCBM (CLIP-concepts, SigLIP): trained on the full official CelebA
  train+val partition (162,770 + 19,867 = 182,637 images), matching
  train_pcbm_clip_concepts_celeba_official_train.py's own real
  production scale exactly.
- TCAV: same real production exemplar/random/control counts as
  run_tcav_celeba_official_train.py (N_RANDOM=6, N_CONTROL=6,
  N_PER_RANDOM_SET=25, N_CONCEPT_EXEMPLARS=40, N_VAL_SAMPLES=40) -- TCAV's
  own cost is NOT pool-size-bound the way the other two are (it always
  works from small sampled exemplar sets), so "full scale" mainly means
  using the real production sample counts rather than the smaller ones
  used for the quick matched benchmark.

Both tasks (Attractive, Male) are simulated by running each method's
per-concept procedure twice, reusing the SAME black-box output index
both times -- the wall-clock cost of scoring a black box is identical
regardless of which output index is read, so this still gives an honest
total-time-for-2-tasks number without needing to wire up a second
classifier just for timing purposes. This is stated explicitly in the
output, not left implicit.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "run"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, "../post_hoc_cbm")

from run_cards_celeba_full import CONCEPT_QUERY_TEXT
from run_cards_celeba_masking_hybrid_official_val_zscore import build_clean_official_val_paths

from cards.attribution.localization import concept_zscore_cutoff, localize_concept, threshold_mask
from cards.concepts.prompts import GENERIC_REFERENCE_CONCEPTS, build_concept_query, compute_text_center, demean_query
from cards.data.celeba_attributes import GROUNDABLE_CONCEPTS
from cards.models.backbones import BACKBONES
from cards.pipeline import instantiate_encoder, orthogonalize_queries
from cards.validation.broden_faithfulness import mask_region

CELEBA_ROOT = Path(r"C:\Users\btokas\Projects\Datasets\CelebA\celeba")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
K = 50
N_TASKS = 2  # simulated, see module docstring
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
ALPHA = 1.0
N_RANDOM_TCAV = 6
N_CONTROL_TCAV = 6
N_PER_RANDOM_SET = 25
N_CONCEPT_EXEMPLARS = 40
N_VAL_SAMPLES = 40
BATCH_SIZE = 128


def encode_images_batched(encoder, paths: list[Path], label: str) -> torch.Tensor:
    chunks = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH_SIZE]]
        chunks.append(encoder.encode_images(batch))
        if (start // BATCH_SIZE) % 50 == 0:
            print(f"    [{label}] {start + len(batch)}/{len(paths)}", flush=True)
    return torch.cat(chunks, dim=0)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = []

    print(f"Device: {DEVICE}  K={K}  concepts={len(GROUNDABLE_CONCEPTS)}  simulated_tasks={N_TASKS}", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    @torch.no_grad()
    def black_box_scores(images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([spec.preprocess(im) for im in images]).to(DEVICE)
        return native_model(batch)[:, 1].detach().cpu()

    # ================= ConceptMask (full official-val pool, K=50, x2 simulated tasks) =================
    print("\n=== ConceptMask (full scale) ===", flush=True)
    t0 = time.perf_counter()
    cfg = OmegaConf.create({
        "seed": 0, "device": DEVICE,
        "encoder": {"name": "siglip", "_target_": "cards.encoders.open_clip_encoder.OpenClipEncoder",
                    "model_name": "ViT-B-16-SigLIP", "pretrained": "webli", "device": DEVICE},
    })
    encoder = instantiate_encoder(cfg)
    t_encoder_load = time.perf_counter() - t0
    print(f"  encoder load: {t_encoder_load:.2f}s", flush=True)

    t0 = time.perf_counter()
    official_val_paths = build_clean_official_val_paths()
    pool_images = [Image.open(p).convert("RGB") for p in official_val_paths]
    t_load_images = time.perf_counter() - t0
    print(f"  load {len(pool_images)} official-val images from disk: {t_load_images:.2f}s", flush=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        pool_embeds = encode_images_batched(encoder, official_val_paths, "ConceptMask pool").to(DEVICE)
    t_pool_embed = time.perf_counter() - t0
    print(f"  pool embedding ({len(pool_images)} images): {t_pool_embed:.2f}s", flush=True)

    t0 = time.perf_counter()
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                   for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)
    t_queries = time.perf_counter() - t0
    print(f"  concept query build ({len(GROUNDABLE_CONCEPTS)} concepts): {t_queries:.2f}s", flush=True)

    t0 = time.perf_counter()
    for task_rep in range(N_TASKS):
        for concept_idx, concept_name in enumerate(GROUNDABLE_CONCEPTS):
            t_c = queries[concept_name].to(DEVICE)
            sims = (pool_embeds @ t_c).detach().cpu().numpy()
            present_indices = sims.argsort()[::-1][:K]

            sim_maps, present_images = [], []
            for idx in present_indices:
                img = pool_images[idx]
                sim_map = localize_concept(encoder, img, t_c, (img.height, img.width))
                sim_maps.append(sim_map)
                present_images.append(img)
            cutoff = concept_zscore_cutoff(sim_maps, ALPHA)

            for idx, img, sim_map in zip(present_indices, present_images, sim_maps):
                mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
                if not mask.any() or mask.all():
                    continue
                mrng = np.random.default_rng(SEED + task_rep * 100_000 + concept_idx * 10_000 + int(idx))
                candidates = [mask_region(img, mask, strategy=s, rng=mrng) for s in FILL_STRATEGIES]
                with torch.no_grad():
                    cand_embeds = encoder.encode_images([img] + candidates).to(DEVICE)
                embed_orig = cand_embeds[0]
                best_angle, best_i = None, None
                for i in range(len(FILL_STRATEGIES)):
                    diff = embed_orig - cand_embeds[1 + i]
                    diff_unit = diff / diff.norm()
                    cos_sim = float(torch.clamp(diff_unit @ t_c, -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(cos_sim)))
                    if best_angle is None or angle < best_angle:
                        best_angle, best_i = angle, i
                perturbed = candidates[best_i]
                _ = black_box_scores([img, perturbed])
        print(f"  simulated task {task_rep + 1}/{N_TASKS} done, elapsed so far: {time.perf_counter() - t0:.2f}s",
              flush=True)
    t_scoring = time.perf_counter() - t0
    print(f"  retrieval+localize+mask+score, {len(GROUNDABLE_CONCEPTS)} concepts x {N_TASKS} tasks: "
          f"{t_scoring:.2f}s", flush=True)

    conceptmask_total = t_encoder_load + t_load_images + t_pool_embed + t_queries + t_scoring
    rows.append(("ConceptMask", "encoder_load", t_encoder_load))
    rows.append(("ConceptMask", "load_images_from_disk", t_load_images))
    rows.append(("ConceptMask", "pool_embed", t_pool_embed))
    rows.append(("ConceptMask", "query_build", t_queries))
    rows.append(("ConceptMask", "scoring_all_concepts_x_tasks", t_scoring))
    rows.append(("ConceptMask", "TOTAL", conceptmask_total))
    print(f"  ConceptMask TOTAL: {conceptmask_total:.2f}s", flush=True)

    # ================= TCAV (real production exemplar counts, x2 simulated tasks) =================
    print("\n=== TCAV (full scale) ===", flush=True)
    from captum.concept import TCAV, Concept
    from concepts.concept_utils import ListDataset

    t0 = time.perf_counter()
    train_pool_paths = list(official_val_paths)  # reuse the same clean pool as the exemplar/random/control source
    random.Random(SEED).shuffle(train_pool_paths)

    def make_concept(cid, name, paths, preprocess, bs=16):
        ds = ListDataset([str(p) for p in paths], preprocess=preprocess)
        return Concept(id=cid, name=name, data_iter=DataLoader(ds, batch_size=bs, shuffle=False))

    idx = 0
    concept_id = 0
    random_concepts = []
    for i in range(N_RANDOM_TCAV):
        chunk = train_pool_paths[idx: idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        random_concepts.append(make_concept(concept_id, f"random_{i}", chunk, spec.preprocess))
        concept_id += 1
    control_concepts = []
    for i in range(N_CONTROL_TCAV):
        chunk = train_pool_paths[idx: idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        control_concepts.append(make_concept(concept_id, f"control_{i}", chunk, spec.preprocess))
        concept_id += 1
    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL_TCAV // 2)]
    t_tcav_pools = time.perf_counter() - t0
    print(f"  random/control pool build: {t_tcav_pools:.2f}s", flush=True)

    tcav_model = spec.load_native().to("cpu").eval()
    val_batch = torch.stack([spec.preprocess(im) for im in pool_images[:N_VAL_SAMPLES]])

    t0 = time.perf_counter()
    tcav = TCAV(model=tcav_model, layers=["layer4"], save_path=str(RESULTS_DIR / "tcav_cost_bench_full_cache"))
    for task_rep in range(N_TASKS):
        for concept_name in GROUNDABLE_CONCEPTS:
            chunk = train_pool_paths[idx: idx + N_CONCEPT_EXEMPLARS]
            idx = (idx + N_CONCEPT_EXEMPLARS) % (len(train_pool_paths) - N_CONCEPT_EXEMPLARS)
            target_concept = make_concept(concept_id, f"{concept_name}_{task_rep}", chunk, spec.preprocess)
            concept_id += 1
            experimental_sets = [[target_concept, rc] for rc in random_concepts]
            tcav.interpret(inputs=val_batch, experimental_sets=experimental_sets, target=1)
            _ = tcav.interpret(inputs=val_batch, experimental_sets=control_sets, target=1)
        print(f"  simulated task {task_rep + 1}/{N_TASKS} done, elapsed so far: {time.perf_counter() - t0:.2f}s",
              flush=True)
    t_tcav_score = time.perf_counter() - t0
    print(f"  CAV fit + interpret, {len(GROUNDABLE_CONCEPTS)} concepts x {N_TASKS} tasks: {t_tcav_score:.2f}s",
          flush=True)

    tcav_total = t_tcav_pools + t_tcav_score
    rows.append(("TCAV", "pool_build", t_tcav_pools))
    rows.append(("TCAV", "cav_fit_and_interpret_all_concepts_x_tasks", t_tcav_score))
    rows.append(("TCAV", "TOTAL", tcav_total))
    print(f"  TCAV TOTAL: {tcav_total:.2f}s", flush=True)

    # ================= PCBM (CLIP-concepts, SigLIP) -- full official train+val =================
    print("\n=== PCBM (CLIP-concepts, SigLIP), full official train+val ===", flush=True)
    from train_pcbm import run_linear_probe

    t0 = time.perf_counter()
    partition = {}
    with open(CELEBA_ROOT / "list_eval_partition.txt") as f:
        for line in f:
            fname, part = line.split()
            partition[fname] = part
    train_files = sorted(f for f, p in partition.items() if p == "0")
    val_files = sorted(f for f, p in partition.items() if p == "1")
    train_paths = [CELEBA_ROOT / "img_align_celeba" / f for f in train_files]
    val_paths = [CELEBA_ROOT / "img_align_celeba" / f for f in val_files]
    t_pcbm_list = time.perf_counter() - t0
    print(f"  listed {len(train_paths)} train + {len(val_paths)} val images: {t_pcbm_list:.2f}s", flush=True)

    t0 = time.perf_counter()
    with torch.no_grad():
        pcbm_train_embeds = encode_images_batched(encoder, train_paths, "PCBM train").to(DEVICE)
        pcbm_val_embeds = encode_images_batched(encoder, val_paths, "PCBM val").to(DEVICE)
    t_pcbm_embed = time.perf_counter() - t0
    print(f"  full train+val embedding ({len(train_paths) + len(val_paths)} images): {t_pcbm_embed:.2f}s", flush=True)

    t0 = time.perf_counter()
    concept_vectors = encoder.encode_text([CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    train_proj = ((pcbm_train_embeds @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    val_proj = ((pcbm_val_embeds @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    t_pcbm_project = time.perf_counter() - t0
    print(f"  concept-vector build + projection: {t_pcbm_project:.2f}s", flush=True)

    t0 = time.perf_counter()

    def native_logits_for(paths: list[Path]) -> np.ndarray:
        logits = []
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            batch = torch.stack([spec.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(DEVICE)
            with torch.no_grad():
                logits.append(native_model(batch).cpu().numpy())
        return np.concatenate(logits, axis=0)

    train_surrogate = native_logits_for(train_paths)[:, :2].argmax(axis=1)
    val_surrogate = native_logits_for(val_paths)[:, :2].argmax(axis=1)
    t_pcbm_score = time.perf_counter() - t0
    print(f"  native-model scoring (train+val, for surrogate labels): {t_pcbm_score:.2f}s", flush=True)

    t0 = time.perf_counter()
    for task_rep in range(N_TASKS):

        class Args:
            seed = SEED
            lam = 0.0002
            alpha = 0.99

        run_linear_probe(Args(), (train_proj, train_surrogate), (val_proj, val_surrogate))
    t_pcbm_fit = time.perf_counter() - t0
    print(f"  elastic-net fit x {N_TASKS} simulated tasks (all {len(GROUNDABLE_CONCEPTS)} concepts jointly each): "
          f"{t_pcbm_fit:.2f}s", flush=True)

    pcbm_total = t_pcbm_list + t_pcbm_embed + t_pcbm_project + t_pcbm_score + t_pcbm_fit
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "list_files", t_pcbm_list))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "full_train_val_embed", t_pcbm_embed))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "concept_projection", t_pcbm_project))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "native_scoring", t_pcbm_score))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "elastic_net_fit_x_tasks", t_pcbm_fit))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "TOTAL", pcbm_total))
    print(f"  PCBM TOTAL: {pcbm_total:.2f}s", flush=True)

    with open(RESULTS_DIR / "computational_cost_benchmark_full_scale.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "phase", "seconds"])
        writer.writerows(rows)
    print(f"\nSaved results/computational_cost_benchmark_full_scale.csv", flush=True)


if __name__ == "__main__":
    main()
