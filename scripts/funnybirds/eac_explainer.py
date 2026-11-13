"""Explain Any Concept (EAC; Sun et al., NeurIPS 2023) as a FunnyBirds attribution explainer.

Same procedure as scripts/celeba/baselines/celeba_eac.py (SAM ViT-H automatic masks, black fill in raw pixel space,
2,500 Bernoulli(.5) segment subsets, PIE surrogate = Linear(n_segments, d) + the model's frozen final layer, SGD
lr .008 / momentum .9 / batch 100 / 50 epochs, 50k Monte-Carlo coalitions per segment), with one change: the
surrogate is fit to the model's full 50-class softmax (soft-label cross-entropy) instead of a single task
probability, and Shapley values are computed for every class at once, because FunnyBirds asks for explanations of
non-predicted classes (target sensitivity). SAM runs at the model's native 256 px input (CelebA: 224 px).

Attribution map: each segment's value spread evenly over its pixels (pixel = sum over segments of value / area).
The framework's per-part sum over a part mask then equals sum_s value_s * |s & part| / |s|, the same segment-to-
concept weighting as the CelebA EAC runs, and background objects are handled exactly as for any attribution map.
Values are cached per image, so repeated calls (and every class) cost one fit.
"""
import zlib

import numpy as np
import torch

N_SAMPLES, MC, SEED = 2500, 50_000, 0
EPOCHS, BATCH, LR, MOMENTUM = 50, 100, 0.008, 0.9


def final_layer(model) -> torch.nn.Linear:
    m = model.model
    if hasattr(m, "fc"):
        return m.fc
    if hasattr(m, "classifier"):
        return m.classifier[-1]
    return m.head


class EACExplainer:
    def __init__(self, model, sam_generator, device="cuda"):
        from explainers.explainer_wrapper import AbstractAttributionExplainer

        # borrow the framework's attribution-map interface functions (part sums, thresholds, background objects)
        self._attr = AbstractAttributionExplainer.__new__(AbstractAttributionExplainer)
        self._attr.explain = self.explain
        self.model, self.gen, self.device = model, sam_generator, device
        fc = final_layer(model)
        self.w, self.b = fc.weight.detach(), fc.bias.detach()
        self._cache: dict[int, torch.Tensor] = {}
        self.log: list[dict] = []

    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)

    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg=False):
        return self._attr.get_part_importance(image, part_map, target, colors_to_part, with_bg=with_bg)

    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds, with_bg=False):
        return self._attr.get_important_parts(image, part_map, target, colors_to_part, thresholds, with_bg=with_bg)

    @torch.no_grad()
    def _model_probs(self, img, masks, subsets):
        flat = torch.from_numpy(masks.reshape(len(masks), -1)).float().to(self.device)
        out = []
        for s in range(0, len(subsets), 250):
            sub = torch.from_numpy(subsets[s:s + 250].astype(np.float32)).to(self.device)
            keep = (sub @ flat > 0).float().view(-1, 1, *masks.shape[1:])
            out.append(torch.softmax(self.model(img * keep), 1))
        return torch.cat(out)

    def _fit_values(self, subsets, probs):
        """Segment Shapley values, (n_classes, n_segments), from one PIE surrogate on the full softmax."""
        n_seg, d = subsets.shape[1], self.w.shape[1]
        torch.manual_seed(0)
        layer = torch.nn.Linear(n_seg, d)
        weight = layer.weight.detach().T.clone().to(self.device).requires_grad_()
        bias = layer.bias.detach().clone().to(self.device).requires_grad_()
        opt = torch.optim.SGD([weight, bias], lr=LR, momentum=MOMENTUM)
        x = torch.from_numpy(subsets.astype(np.float32)).to(self.device)

        def surrogate(bt):
            return torch.softmax((bt @ weight + bias) @ self.w.T + self.b, 1)

        gen = torch.Generator().manual_seed(0)
        with torch.enable_grad():
            for _ in range(EPOCHS):
                order = torch.randperm(len(x), generator=gen)
                for s in range(0, len(x), BATCH):
                    idx = order[s:s + BATCH].to(self.device)
                    loss = -(probs[idx] * torch.log(surrogate(x[idx]).clamp_min(1e-8))).sum(1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
        values = torch.zeros(self.w.shape[0], n_seg, device=self.device)
        rng = torch.Generator(device=self.device).manual_seed(0)
        with torch.no_grad():
            for i in range(n_seg):
                tot = torch.zeros(self.w.shape[0], device=self.device)
                for _ in range(0, MC, 5000):
                    c = (torch.rand(5000, n_seg, generator=rng, device=self.device) < 0.5).float()
                    c[:, i] = 1; a = surrogate(c); c[:, i] = 0
                    tot += (a - surrogate(c)).sum(0)
                values[:, i] = tot / MC
        return values

    def explain(self, input, target):
        """Attribution map [1,1,H,W] for class `target`."""
        raw = (input[0].detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
        key = zlib.crc32(raw.tobytes())
        if key not in self._cache:
            masks = np.stack([m["segmentation"] for m in self.gen.generate(raw)])
            subsets = np.random.default_rng(SEED + key % 1_000_003).random((N_SAMPLES, len(masks))) < 0.5
            img = input[:1].detach().to(self.device)
            values = self._fit_values(subsets, self._model_probs(img, masks, subsets))  # (classes, n_seg)
            m = torch.from_numpy(masks).float().to(self.device)
            per_pixel = m / m.sum((1, 2), keepdim=True).clamp_min(1)  # (n_seg, H, W), each segment sums to 1
            self._cache[key] = torch.einsum("cs,shw->chw", values, per_pixel)  # (classes, H, W)
            self.log.append({"image_crc": key, "n_segments": len(masks),
                             "uncovered": float(1 - masks.any(0).mean())})
        t = int(target.reshape(-1)[0]) if torch.is_tensor(target) else int(target)
        return self._cache[key][t][None, None].to(input.device)
