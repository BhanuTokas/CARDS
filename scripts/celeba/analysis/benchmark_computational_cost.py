"""Real, matched-setup wall-clock timing benchmark for ConceptMask vs.
TCAV vs. PCBM (CLIP-concepts, SigLIP), prompted directly ("I want to add
a section on computational cost comparison"). No timing instrumentation
existed anywhere in this codebase before this script (confirmed via grep
for perf_counter/time.time() across scripts/celeba/); this was flagged
once before as a pending benchmark and never run.

To keep this feasible to run in one session, uses a SMALLER matched
setup than the paper's headline results (POOL_SIZE=1000 images,
K=15, single task (Attractive), reduced TCAV exemplar counts) --
applied identically across all three methods, so the comparison is
still apples-to-apples, just not at full official-train scale. Reports
this scale explicitly rather than silently, since absolute numbers
would look different (mostly just scaled up) at full scale.

Same black-box classifier (celeba_attractive_young, native ResNet18),
same encoder (SigLIP) for ConceptMask and PCBM's CLIP-concepts variant,
same 26-concept GROUNDABLE_CONCEPTS bank, same image pool (official
CelebA val, HQ-leakage-excluded) for every method that touches images.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.stats import ttest_ind
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

RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
POOL_SIZE = 1000
K = 15
FILL_STRATEGIES = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
ALPHA = 1.0
N_RANDOM_TCAV = 4
N_CONTROL_TCAV = 4
N_PER_RANDOM_SET = 20
N_CONCEPT_EXEMPLARS = 25


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rows = []

    print(f"Device: {DEVICE}  pool_size={POOL_SIZE}  K={K}  concepts={len(GROUNDABLE_CONCEPTS)}", flush=True)

    # ---------- shared setup: pool of images, black-box model ----------
    t0 = time.perf_counter()
    official_paths = build_clean_official_val_paths()
    rng = np.random.default_rng(SEED)
    pool_paths = [official_paths[i] for i in rng.choice(len(official_paths), size=POOL_SIZE, replace=False)]
    pool_images = [Image.open(p).convert("RGB") for p in pool_paths]
    t_load_images = time.perf_counter() - t0
    print(f"Loaded {POOL_SIZE} images from disk: {t_load_images:.2f}s", flush=True)

    spec = BACKBONES["celeba_attractive_young"]
    native_model = spec.load_native().to(DEVICE).eval()

    @torch.no_grad()
    def black_box_scores(images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([spec.preprocess(im) for im in images]).to(DEVICE)
        return native_model(batch)[:, 1].detach().cpu()  # Attractive logit

    # ================= ConceptMask =================
    print("\n=== ConceptMask ===", flush=True)
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
    with torch.no_grad():
        pool_embeds = encoder.encode_images(pool_images).to(DEVICE)
    t_pool_embed = time.perf_counter() - t0
    print(f"  pool embedding ({POOL_SIZE} images): {t_pool_embed:.2f}s", flush=True)

    t0 = time.perf_counter()
    text_center = compute_text_center(GENERIC_REFERENCE_CONCEPTS, encoder)
    raw_queries = {c: demean_query(build_concept_query(CONCEPT_QUERY_TEXT[c], encoder), text_center)
                   for c in GROUNDABLE_CONCEPTS}
    queries = orthogonalize_queries(raw_queries)
    t_queries = time.perf_counter() - t0
    print(f"  concept query build ({len(GROUNDABLE_CONCEPTS)} concepts): {t_queries:.2f}s", flush=True)

    t0 = time.perf_counter()
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

        deltas = []
        for j, (idx, img, sim_map) in enumerate(zip(present_indices, present_images, sim_maps)):
            mask = threshold_mask(sim_map, method="fixed", cutoff=cutoff)
            if not mask.any() or mask.all():
                continue
            mrng = np.random.default_rng(SEED + concept_idx * 10_000 + int(idx))
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
            scores = black_box_scores([img, perturbed])
            deltas.append((scores[0] - scores[1]).item())
    t_scoring = time.perf_counter() - t0
    print(f"  retrieval+localize+mask+score, all {len(GROUNDABLE_CONCEPTS)} concepts: {t_scoring:.2f}s", flush=True)

    conceptmask_total = t_encoder_load + t_pool_embed + t_queries + t_scoring
    rows.append(("ConceptMask", "encoder_load", t_encoder_load))
    rows.append(("ConceptMask", "pool_embed", t_pool_embed))
    rows.append(("ConceptMask", "query_build", t_queries))
    rows.append(("ConceptMask", "scoring_all_concepts", t_scoring))
    rows.append(("ConceptMask", "TOTAL", conceptmask_total))
    print(f"  ConceptMask TOTAL: {conceptmask_total:.2f}s "
          f"({conceptmask_total / len(GROUNDABLE_CONCEPTS):.2f}s/concept avg)", flush=True)

    # ================= TCAV =================
    print("\n=== TCAV ===", flush=True)
    from captum.concept import TCAV, Concept
    from concepts.concept_utils import ListDataset

    t0 = time.perf_counter()
    train_pool_paths = [p for p in official_paths if p not in set(pool_paths)][: (N_RANDOM_TCAV + N_CONTROL_TCAV) * N_PER_RANDOM_SET + N_CONCEPT_EXEMPLARS * len(GROUNDABLE_CONCEPTS)]
    rng_py_paths = list(train_pool_paths)
    import random
    random.Random(SEED).shuffle(rng_py_paths)

    def make_concept(cid, name, paths, preprocess, bs=16):
        ds = ListDataset([str(p) for p in paths], preprocess=preprocess)
        return Concept(id=cid, name=name, data_iter=DataLoader(ds, batch_size=bs, shuffle=False))

    idx = 0
    concept_id = 0
    random_concepts = []
    for i in range(N_RANDOM_TCAV):
        chunk = rng_py_paths[idx: idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        random_concepts.append(make_concept(concept_id, f"random_{i}", chunk, spec.preprocess))
        concept_id += 1
    control_concepts = []
    for i in range(N_CONTROL_TCAV):
        chunk = rng_py_paths[idx: idx + N_PER_RANDOM_SET]
        idx += N_PER_RANDOM_SET
        control_concepts.append(make_concept(concept_id, f"control_{i}", chunk, spec.preprocess))
        concept_id += 1
    control_sets = [[control_concepts[2 * i], control_concepts[2 * i + 1]] for i in range(N_CONTROL_TCAV // 2)]
    t_tcav_pools = time.perf_counter() - t0
    print(f"  random/control pool build: {t_tcav_pools:.2f}s", flush=True)

    # captum's TCAV needs the model and every input on the same device as its own
    # internal DataLoader-produced concept-image tensors, which it never moves off
    # CPU -- this project's standing convention is CPU-only for TCAV (matches every
    # other TCAV script in this track); a CUDA `native_model` here reproduces the
    # exact device-mismatch crash that convention exists to avoid.
    tcav_model = spec.load_native().to("cpu").eval()
    val_batch = torch.stack([spec.preprocess(im) for im in pool_images[:20]])

    t0 = time.perf_counter()
    tcav = TCAV(model=tcav_model, layers=["layer4"], save_path=str(RESULTS_DIR / "tcav_cost_bench_cache"))
    for concept_name in GROUNDABLE_CONCEPTS:
        chunk = rng_py_paths[idx: idx + N_CONCEPT_EXEMPLARS]
        idx += N_CONCEPT_EXEMPLARS
        target_concept = make_concept(concept_id, concept_name, chunk, spec.preprocess)
        concept_id += 1
        experimental_sets = [[target_concept, rc] for rc in random_concepts]
        tcav.interpret(inputs=val_batch, experimental_sets=experimental_sets, target=1)
        _ = tcav.interpret(inputs=val_batch, experimental_sets=control_sets, target=1)
    t_tcav_score = time.perf_counter() - t0
    print(f"  CAV fit + interpret, all {len(GROUNDABLE_CONCEPTS)} concepts: {t_tcav_score:.2f}s", flush=True)

    tcav_total = t_tcav_pools + t_tcav_score
    rows.append(("TCAV", "pool_build", t_tcav_pools))
    rows.append(("TCAV", "cav_fit_and_interpret_all_concepts", t_tcav_score))
    rows.append(("TCAV", "TOTAL", tcav_total))
    print(f"  TCAV TOTAL: {tcav_total:.2f}s ({tcav_total / len(GROUNDABLE_CONCEPTS):.2f}s/concept avg)", flush=True)

    # ================= PCBM (CLIP-concepts, SigLIP) =================
    print("\n=== PCBM (CLIP-concepts, SigLIP) ===", flush=True)
    from train_pcbm import run_linear_probe

    t0 = time.perf_counter()
    concept_vectors = encoder.encode_text([CONCEPT_QUERY_TEXT[c] for c in GROUNDABLE_CONCEPTS]).to(DEVICE)
    concept_norms_sq = (concept_vectors ** 2).sum(dim=1)
    train_proj = ((pool_embeds @ concept_vectors.T) / concept_norms_sq.unsqueeze(0)).detach().cpu().numpy()
    t_pcbm_project = time.perf_counter() - t0
    print(f"  concept-vector build + projection (reusing ConceptMask's own pool embeddings): {t_pcbm_project:.2f}s",
          flush=True)

    t0 = time.perf_counter()
    native_logits = []
    bs = 64
    for start in range(0, len(pool_images), bs):
        batch = torch.stack([spec.preprocess(im) for im in pool_images[start:start + bs]]).to(DEVICE)
        with torch.no_grad():
            native_logits.append(native_model(batch).cpu().numpy())
    native_logits = np.concatenate(native_logits, axis=0)
    surrogate_labels = native_logits[:, :2].argmax(axis=1)

    class Args:
        seed = SEED
        lam = 0.0002
        alpha = 0.99

    split = int(0.8 * POOL_SIZE)
    run_linear_probe(Args(), (train_proj[:split], surrogate_labels[:split]),
                      (train_proj[split:], surrogate_labels[split:]))
    t_pcbm_fit = time.perf_counter() - t0
    print(f"  native-model scoring + elastic-net fit (all {len(GROUNDABLE_CONCEPTS)} concepts jointly): "
          f"{t_pcbm_fit:.2f}s", flush=True)

    pcbm_total = t_pcbm_project + t_pcbm_fit
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "concept_projection", t_pcbm_project))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "native_scoring_and_fit", t_pcbm_fit))
    rows.append(("PCBM (CLIP-concepts, SigLIP)", "TOTAL", pcbm_total))
    print(f"  PCBM TOTAL: {pcbm_total:.2f}s "
          f"({pcbm_total / len(GROUNDABLE_CONCEPTS):.2f}s/concept avg, "
          f"but note this is a single JOINT fit across all concepts, not per-concept)", flush=True)

    with open(RESULTS_DIR / "computational_cost_benchmark.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "phase", "seconds"])
        writer.writerows(rows)
    print(f"\nSaved results/computational_cost_benchmark.csv", flush=True)


if __name__ == "__main__":
    main()
