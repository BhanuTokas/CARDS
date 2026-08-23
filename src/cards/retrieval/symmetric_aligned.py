"""Step 2/3 alternative — symmetric alignment-optimized retrieval.

`aligned_retrieval` (aligned.py) optimizes only N_c, treating P_c as
fixed by naive top-k. This module extends the same idea to P_c as well:
jointly select P_c and N_c (each size k) to maximize
cos(mean(P_c) - mean(N_c), t_c), instead of fixing one side by ranking
and only searching over the other.

Warm start: top-k / bottom-k by cosine to t_c (`retrieve_top_bottom_k`)
-- provably optimal for each side's linear term in isolation (the same
argument as aligned.py: holding the other side's mean fixed, the dot
product contribution is maximized by the extreme-k picks on that side).
The greedy search then alternates swap passes over P_c and N_c members,
holding the other side fixed during each pass (coordinate ascent),
accepting only strict improvements to the joint cosine objective.

Caveat, now tested (see notes/pcbm_correlation_investigation.md's v12
entry): validated against the same CARDS-vs-PCBM correlation benchmark
used throughout this project, CIFAR-100 + SigLIP + de-meaned + k=30.
Result: symmetric (raw r=0.258, winsorized=0.227, sign agree=53.1%)
underperforms one-sided `aligned_retrieval` (raw r=0.368,
winsorized=0.291, sign agree=55.8%) on every metric, and even trails
plain `matched` (sign agree=55.5%) on sign agreement specifically.
Giving the search freedom on both sides did not compound the gain --
it partially erased it. Mechanism confirmed (not just inferred) via
retrieval-level instrumentation, see the "Diagnostics on v12" addendum
in notes/pcbm_correlation_investigation.md: the search replaces ~85%
of P_c's members (mean 25.4/30), and the swapped-in members have mean
individual cosine-to-t_c of just 0.020 (vs. 0.148 for the swapped-out
members it discards) -- it wins its own objective (achieved
cos(d_c, t_c) 0.749 vs. 0.659 for one-sided `aligned`) by trading
genuinely on-concept P_c images for essentially concept-irrelevant ones
that merely happen to improve the aggregate direction-of-difference
fit. `aligned_retrieval`'s N_c also churns heavily (~96%) without this
problem, because N_c's role ("concept absent") only requires *not*
looking like the concept -- a loose constraint aggressive re-picking
doesn't violate. P_c's role requires the opposite: individual,
verifiable relevance, which is exactly what unrestricted search erodes.
Net: prefer one-sided `aligned` over this strategy; kept available as a
documented experimental option, not a recommended default.

`present_candidate_pool_size` (added after the above diagnosis)
implements the suggested fix directly: it restricts P_c swap-in
candidates to the top-N pool images by *individual* cosine similarity
to t_c (N = this argument, default None = unrestricted, the behavior
validated above), rather than letting the search pull from the entire
pool. N_c's candidate pool is left unrestricted either way, since the
diagnosis found N_c's churn isn't the problem.

Now also validated (see notes/pcbm_correlation_investigation.md's v13
entry), with a mixed verdict: the fix works in the sense that any
constrained multiplier clears unconstrained symmetric's r=0.258 by a
wide margin, confirming individual P_c relevance was indeed what
mattered. But swept across present_candidate_pool_size = {1,2,3,5} * k,
every multiplier > 1 (i.e. any actual P_c search room) *underperforms*
multiplier=1 (r=0.371) -- and multiplier=1 is mathematically just
one-sided `aligned_retrieval` with P_c fully pinned, re-implemented via
this module's coordinate-ascent code path. No tested amount of "some
but constrained" P_c freedom beat full pinning. Practical takeaway:
there is no known value of this parameter that improves on plain
`aligned_retrieval` -- use `aligned_retrieval` directly instead of this
module with a large present_candidate_pool_size. Kept here, documented
as such, for anyone who wants to push the relevance-vs-flexibility
tradeoff further (e.g. a soft-penalty version instead of a hard cutoff).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cards.retrieval.pool import CandidatePool


def _cosine_align(mean_present: torch.Tensor, mean_absent: torch.Tensor, t_c_unit: torch.Tensor) -> float:
    d_c = mean_present - mean_absent
    return F.cosine_similarity(d_c.unsqueeze(0), t_c_unit.unsqueeze(0)).item()


def _swap_pass(
    embeddings: torch.Tensor,
    members: list[int],
    member_set: set[int],
    other_mean: torch.Tensor,
    t_c_unit: torch.Tensor,
    k: int,
    used_mask: torch.Tensor,
    role: str,
    current_score: float,
    eligible_mask: torch.Tensor | None = None,
) -> tuple[list[int], set[int], torch.Tensor, float, bool]:
    """One swap pass over `members` (P_c if role == 'present', N_c if
    role == 'absent'), holding `other_mean` (the fixed side's mean)
    constant. `eligible_mask`, if given, further restricts swap-in
    candidates (e.g. to individually concept-relevant images) beyond
    just excluding already-used indices. Mutates `used_mask` and
    `member_set` in place as swaps are accepted. Returns (members,
    member_set, this_side_mean, current_score, improved_this_pass)."""
    improved = False
    this_mean = embeddings[torch.tensor(members, dtype=torch.long)].mean(dim=0)

    for member in list(members):
        candidate_mask = ~used_mask
        if eligible_mask is not None:
            candidate_mask = candidate_mask & eligible_mask
        candidate_indices = candidate_mask.nonzero(as_tuple=True)[0]
        if candidate_indices.numel() == 0:
            break
        remaining = [x for x in members if x != member]
        mean_without_member = (
            embeddings[torch.tensor(remaining, dtype=torch.long)].mean(dim=0)
            if remaining
            else torch.zeros_like(this_mean)
        )
        candidate_embeddings = embeddings[candidate_indices]
        trial_means = (mean_without_member.unsqueeze(0) * (k - 1) + candidate_embeddings) / k
        if role == "present":
            trial_d_c = trial_means - other_mean.unsqueeze(0)
        else:
            trial_d_c = other_mean.unsqueeze(0) - trial_means
        trial_scores = F.cosine_similarity(trial_d_c, t_c_unit.unsqueeze(0).expand_as(trial_d_c), dim=1)
        best_score, best_local_idx = trial_scores.max(dim=0)

        if best_score.item() > current_score:
            replacement = candidate_indices[best_local_idx].item()
            member_set.discard(member)
            member_set.add(replacement)
            members = list(member_set)
            used_mask[member] = False
            used_mask[replacement] = True
            this_mean = trial_means[best_local_idx]
            current_score = best_score.item()
            improved = True

    return members, member_set, this_mean, current_score, improved


def symmetric_aligned_retrieval(
    pool: CandidatePool,
    t_c: torch.Tensor,
    k: int,
    max_iters: int = 5,
    present_candidate_pool_size: int | None = None,
) -> tuple[list[int], list[int]]:
    """Jointly select P_c and N_c (each size k) to maximize
    cos(mean(P_c) - mean(N_c), t_c).

    Greedy local search warm-started from naive top-k/bottom-k (optimal
    for each side's linear term in isolation, same argument as
    aligned_retrieval). Each outer iteration does one swap pass over P_c
    (holding N_c's mean fixed), then one swap pass over N_c (holding
    P_c's now-updated mean fixed) -- coordinate ascent on the joint
    objective. Stops early if a full iteration makes no swap on either
    side. `max_iters` bounds worst-case outer iterations.

    `present_candidate_pool_size`, if given, restricts P_c swap-in
    candidates to the top-N pool images by individual cosine similarity
    to t_c (see the module docstring for why this only applies to P_c,
    not N_c). Must be >= k. None (default) leaves P_c unrestricted, the
    behavior validated (and found wanting) in v12.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    n = pool.embeddings.shape[0]
    if 2 * k > n:
        raise ValueError(f"2*k ({2 * k}) exceeds pool size ({n}); shrink k or grow the pool")
    if present_candidate_pool_size is not None and present_candidate_pool_size < k:
        raise ValueError(
            f"present_candidate_pool_size ({present_candidate_pool_size}) must be >= k ({k})"
        )

    t_c_unit = F.normalize(t_c, dim=0)
    similarities = pool.embeddings @ t_c_unit
    order = torch.argsort(similarities, descending=True)
    present = order[:k].tolist()
    absent = order[-k:].tolist()
    present_set, absent_set = set(present), set(absent)

    present_eligible_mask = None
    if present_candidate_pool_size is not None:
        present_eligible_mask = torch.zeros(n, dtype=torch.bool)
        present_eligible_mask[order[:present_candidate_pool_size]] = True

    used_mask = torch.zeros(n, dtype=torch.bool)
    used_mask[torch.tensor(present, dtype=torch.long)] = True
    used_mask[torch.tensor(absent, dtype=torch.long)] = True

    mean_present = pool.embeddings[torch.tensor(present, dtype=torch.long)].mean(dim=0)
    mean_absent = pool.embeddings[torch.tensor(absent, dtype=torch.long)].mean(dim=0)
    current_score = _cosine_align(mean_present, mean_absent, t_c_unit)

    for _ in range(max_iters):
        present, present_set, mean_present, current_score, improved_present = _swap_pass(
            pool.embeddings, present, present_set, mean_absent, t_c_unit, k, used_mask, "present", current_score,
            eligible_mask=present_eligible_mask,
        )
        absent, absent_set, mean_absent, current_score, improved_absent = _swap_pass(
            pool.embeddings, absent, absent_set, mean_present, t_c_unit, k, used_mask, "absent", current_score
        )
        if not (improved_present or improved_absent):
            break

    return present, absent
