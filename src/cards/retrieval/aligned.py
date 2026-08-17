"""Step 3 alternative — alignment-optimized negative selection.

`naive`/`matched` pick N_c by ranking on cosine similarity to `t_c`
directly (each point judged independently). This module instead treats
N_c selection as a small combinatorial optimization: choose the k points
that make the *resulting* diff-of-means direction
`d_c = mean(P_c) - mean(N_c)` as well-aligned with `t_c` as possible,
which is a genuinely different (and stricter) objective than picking
individually-dissimilar points.

Why this isn't circular: `t_c` is fixed by Step 1, before any retrieval
happens, independent of which images end up in P_c or N_c -- optimizing
against it is optimizing against an external target, not a statistic
derived from the same selection being tuned. (Optimizing against `d_c`
itself, or against a direction derived from P_c after P_c was already
chosen by similarity to `t_c`, would be circular; this deliberately
isn't that.)

Bottom-k by cosine to `t_c` (what `naive` already does) is *provably*
optimal for maximizing the plain dot product `t_c . d_c` -- with
mean(P_c) fixed, that quantity is maximized by minimizing the *sum* of
included points' dot products with `t_c`, i.e. picking the k smallest,
which is exactly bottom-k. There's nothing to search for there. The
reason to do more work here is that `t_c . d_c` isn't the full picture:
maximizing the *cosine* `cos(d_c, t_c) = (t_c . d_c) / |d_c|` also
depends on `|d_c|`, which bottom-k doesn't control for -- a large
projection onto `t_c` achieved by also drifting far off-axis can still
lose to a smaller, cleaner projection. `aligned_retrieval` runs a greedy
local search directly on the cosine objective, warm-started from
bottom-k (already optimal for the linear term, a reasonable starting
point).

Caveat, now tested (see notes/pcbm_correlation_investigation.md's v11
entry): validated against the CARDS-vs-PCBM-weight correlation on
CIFAR-100. The headline number (Pearson r=0.368, nearly double the best
`matched`-retrieval result of r=0.205) is NOT trustworthy as-is --
directly maximizing `cos(d_c, t_c)` mechanically inflates `|d_c|` (raw
score magnitude) roughly uniformly across classes/concepts, and that
inflation compounds with *pre-existing, unrelated* black-box output
volatility (some classes are just noisier than others) to produce a
handful of very large, high-leverage pairs that dominate a magnitude-
sensitive statistic like Pearson r -- confirmed via a max_iters=0..5
sweep: max_iters=0 (no search at all) already matches `matched`'s
r=0.205, and ~95% of N_c gets replaced in a single greedy pass, not
gradually. But winsorizing (capping extreme scores while keeping all
pairs, the fairer full-sample comparison) still holds at r=0.29-0.32
even at a 10% cap -- clearly above the 0.205 baseline, which wouldn't
happen if the gain were pure inflation with no real signal. Net read:
this method likely does provide a real, positive effect, probably in
the 0.25-0.32 range rather than 0.368 -- but via a mechanism aggressive
enough (one-shot near-total N_c replacement) that it deserves more
scrutiny before treating it as a stable default over `matched`. If
revisited: compare on a magnitude-robust statistic (e.g. Spearman on
winsorized scores) and consider having the search penalize `|d_c|`
growth directly, which might separate the real gain from the inflation
cleanly rather than needing to infer it post-hoc.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cards.retrieval.pool import CandidatePool


def _cosine_align(mean_present: torch.Tensor, mean_absent: torch.Tensor, t_c_unit: torch.Tensor) -> float:
    d_c = mean_present - mean_absent
    return F.cosine_similarity(d_c.unsqueeze(0), t_c_unit.unsqueeze(0)).item()


def aligned_retrieval(
    pool: CandidatePool,
    present_indices: list[int],
    t_c: torch.Tensor,
    k: int,
    max_iters: int = 5,
) -> list[int]:
    """Select N_c (size `k`) to maximize cos(mean(P_c) - mean(N_c), t_c).

    Greedy local search: start from bottom-k by cosine to t_c (already
    optimal for the linear term t_c . d_c), then repeatedly try replacing
    each current member with whichever out-of-set candidate most improves
    the cosine objective, accepting only strict improvements. Stops early
    if a full pass makes no swap. `max_iters` bounds worst-case passes
    over all k current members (each pass is one vectorized
    cosine-over-all-candidates computation per member, not one-at-a-time).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = pool.embeddings.shape[0]
    present_index_tensor = torch.tensor(present_indices, dtype=torch.long)

    candidate_mask = torch.ones(n, dtype=torch.bool)
    candidate_mask[present_index_tensor] = False
    candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
    if candidate_indices.numel() < k:
        raise ValueError(f"only {candidate_indices.numel()} candidates remain, need k={k}")

    mean_present = pool.embeddings[present_index_tensor].mean(dim=0)
    t_c_unit = F.normalize(t_c, dim=0)

    candidate_similarities = pool.embeddings[candidate_indices] @ t_c_unit
    warm_start_order = torch.argsort(candidate_similarities)  # ascending: bottom-k first
    current = candidate_indices[warm_start_order[:k]].tolist()
    current_set = set(current)

    mean_absent = pool.embeddings[torch.tensor(current, dtype=torch.long)].mean(dim=0)
    current_score = _cosine_align(mean_present, mean_absent, t_c_unit)

    for _ in range(max_iters):
        improved_this_pass = False
        for member in list(current):
            out_of_set = [i for i in candidate_indices.tolist() if i not in current_set]
            if not out_of_set:
                break
            remaining = [x for x in current if x != member]
            mean_without_member = (
                pool.embeddings[torch.tensor(remaining, dtype=torch.long)].mean(dim=0)
                if remaining
                else torch.zeros_like(mean_present)
            )
            out_embeddings = pool.embeddings[torch.tensor(out_of_set, dtype=torch.long)]
            trial_means = (mean_without_member.unsqueeze(0) * (k - 1) + out_embeddings) / k
            trial_d_c = mean_present.unsqueeze(0) - trial_means
            trial_scores = F.cosine_similarity(trial_d_c, t_c_unit.unsqueeze(0).expand_as(trial_d_c), dim=1)
            best_score, best_local_idx = trial_scores.max(dim=0)

            if best_score.item() > current_score:
                replacement = out_of_set[best_local_idx.item()]
                current_set.discard(member)
                current_set.add(replacement)
                current = list(current_set)
                current_score = best_score.item()
                improved_this_pass = True

        if not improved_this_pass:
            break

    return current
