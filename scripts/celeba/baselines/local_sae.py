"""Local (per-image) ConceptSAE on the clean CelebA ground truth (official-train black box, env CARDS_ARCH).

Uses the SAE and concept-named latents saved by concept_sae.py (OUT.pt next to its CSV). Local score of image x for
(concept, task) = z_j(x) * (w_task . d_j): the exact contribution of the concept's latent to the positive-class
logit through the linear head, i.e. the per-image term whose mean is concept_sae.py's global score.

argv[1] = concept_sae.py's .pt, argv[2] = output folder; writes sae_{SHARD}.csv. Sharded over pairs."""
import csv
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from black_boxes import ARCH, DEVICE, PREPROCESS, feats, head, load
from concept_sae import TopKSAE
from local_common import POS, gt_pairs, hq_path, shard, shard_tag


@torch.no_grad()
def main():
    ck, out_dir = torch.load(sys.argv[1]), Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    st = ck["state"]
    sae = TopKSAE(st["W_dec"].shape[0], st["W_dec"].shape[1], ck["k"], st["b_dec"]).to(DEVICE)
    sae.load_state_dict(st)
    model = load().to(DEVICE).eval()
    print("black box:", ARCH, flush=True)
    W = head(model).weight
    todo = shard(gt_pairs())
    with open(out_dir / f"sae_{shard_tag()}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["image", "concept_name", "target_task", "gt_delta_p", "sae"])
        for s in range(0, len(todo), 64):
            chunk = todo[s:s + 64]
            x = torch.stack([PREPROCESS(Image.open(hq_path(n)).convert("RGB")) for _, (n, _, _) in chunk]).to(DEVICE)
            z = sae.encode(feats(model, x)[0])
            for r, (_, (name, c, tasks)) in enumerate(chunk):
                j = ck["latents"][c]
                for t, d in tasks.items():
                    w.writerow([name, c, t, d, float(z[r, j] * (W[POS[t]] @ sae.W_dec[:, j]))])
    print("done", flush=True)


if __name__ == "__main__":
    main()
