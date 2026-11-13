"""Hide and Seek (HnS) as a FunnyBirds explainer: implements the framework's interface functions
get_part_importance() and get_important_parts() (see funnybirds-framework/explainers/explainer_wrapper.py).

Part importance PI(part, class) = logit(image)[class] - logit(image with the part's mask filled)[class], the same
local score as scripts/celeba/baselines/local_soft.py ("cards" column):
  - mask: SigLIP patch similarity to the part's text query, thresholded at the part's z-score cutoff
          (mask_source="hns"), or the true part mask from the part map, dilated 5x5 as the framework does for
          attribution maps (mask_source="true": an oracle that isolates the fill from localization errors);
  - fill: best of 7 fills, the one whose SigLIP embedding change has the smallest angle to the query;
  - an empty or full mask gives PI = 0, as in the CelebA runs.
The masked images do not depend on the class, so all 50 logits are cached per image and every class's PI is free
(target sensitivity asks for two non-predicted classes).

Important parts P at threshold t: parts with PI > t * (sum of positive PIs over the five parts). This mirrors the
framework's rule for attribution maps (PI > t * total attribution) with HnS's total effect as the total.

Background objects: HnS only scores the concepts it is queried with, so with_bg=True adds every background object
with PI = 0. This favours HnS on distractibility; evaluate.py therefore also reports a bird-parts-only
distractibility for every method.
"""
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from paths import PART_COLORS, PARTS
from PIL import Image

from cards.attribution.localization import threshold_mask
from cards.validation.broden_faithfulness import mask_region

FILLS = ["blur", "zero_fill", "mean_fill", "hue_shift", "white_fill", "zero_fill_noise", "noise_then_blur"]
N_BG = 50  # background object ids the framework checks (colour 204,204,204+i)


def to_pil(x: torch.Tensor) -> Image.Image:
    """[1,3,H,W] float in [0,1] -> RGB PIL (FunnyBirds images are 8-bit PNGs, so this is lossless)."""
    a = (x[0].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(a)


def to_tensor(im: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)


def true_masks(part_map: torch.Tensor) -> dict[str, np.ndarray]:
    """Part masks from the framework's part map ([1,3,H,W], colour values 0-255), dilated 5x5 like the framework."""
    pm = part_map[0].round().long()
    out = {}
    for part, cols in PART_COLORS.items():
        m = torch.zeros(pm.shape[1:], dtype=torch.bool, device=pm.device)
        for c in cols:
            m |= (pm[0] == c[0]) & (pm[1] == c[1]) & (pm[2] == c[2])
        m = F.max_pool2d(m[None, None].float(), 5, stride=1, padding=2)[0, 0] > 0
        out[part] = m.cpu().numpy()
    return out


class HnSExplainer:
    def __init__(self, model, encoder, queries, cutoffs, mask_source="hns", seed=42, device="cuda"):
        assert mask_source in ("hns", "true")
        self.model, self.enc, self.q, self.cut = model, encoder, queries, cutoffs
        self.mask_source, self.seed, self.device = mask_source, seed, device
        self._cache: dict[int, dict] = {}
        self.log: list[dict] = []  # one row per explained image: fills chosen, mask areas

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    @torch.no_grad()
    def _drops(self, image: torch.Tensor, part_map: torch.Tensor) -> dict[str, np.ndarray]:
        """{part: logit drop vector over all classes}, cached by image content."""
        img_np = (image[0].detach().cpu().numpy() * 255).round().astype(np.uint8)
        key = zlib.crc32(img_np.tobytes())
        if key in self._cache:
            return self._cache[key]
        pil = to_pil(image)
        if self.mask_source == "hns":
            patches = self.enc.encode_patches(pil)
            side = round(patches.shape[0] ** 0.5)
            masks = {}
            for p in PARTS:
                s = (patches @ self.q[p].to(patches.device)).float().cpu().reshape(1, 1, side, side)
                s = F.interpolate(s, size=(pil.height, pil.width), mode="bilinear", align_corners=False)[0, 0]
                masks[p] = threshold_mask(s.numpy(), method="fixed", cutoff=self.cut[p])
        else:
            masks = true_masks(part_map)
        batch, where, info = [to_tensor(pil)], {}, {"image_crc": key}
        for pi, p in enumerate(PARTS):
            m = masks[p]
            info[f"{p}_area"] = float(m.mean())
            if not m.any() or m.all():
                info[f"{p}_fill"] = ""
                continue
            rng = np.random.default_rng(self.seed + pi * 10_000 + key % 1_000_003)
            cands = [mask_region(pil, m, strategy=f, rng=rng) for f in FILLS]
            e = self.enc.encode_images([pil] + cands).to(self.device)
            q = self.q[p].to(self.device)
            angs = []
            for i in range(len(FILLS)):
                d = e[0] - e[1 + i]; d = d / d.norm()
                angs.append(float(torch.rad2deg(torch.arccos(torch.clamp(d @ q, -1.0, 1.0)))))
            j = int(np.argmin(angs))
            info[f"{p}_fill"] = FILLS[j]
            where[p] = len(batch); batch.append(to_tensor(cands[j]))
        logits = self.model(torch.stack(batch).to(self.device)).float().cpu().numpy()
        out = {p: (logits[0] - logits[where[p]]) if p in where else np.zeros(logits.shape[1]) for p in PARTS}
        self._cache[key] = out
        self.log.append(info)
        return out

    def get_part_importance(self, image, part_map, target, colors_to_part=None, with_bg=False):
        d = self._drops(image, part_map)
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        pi = {p: float(d[p][t]) for p in PARTS}
        if with_bg:
            pi.update({f"bg_{i:03d}": 0.0 for i in range(N_BG)})
        return pi

    def get_important_parts(self, image, part_map, target, colors_to_part=None, thresholds=None, with_bg=False):
        pi = self.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)
        total = sum(max(v, 0.0) for v in pi.values())
        thresholds = self.get_p_thresholds() if thresholds is None else thresholds
        return [[k for k, v in pi.items() if total > 0 and v > total * t] for t in thresholds]
