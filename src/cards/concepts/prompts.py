"""Step 1 — Concept prompt construction.

Ensemble multiple template phrasings per concept (reusing the k-vs-1
prompt-ensembling construction from CounterConcept), average the resulting
text embeddings to get a single query vector `t_c` per concept.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cards.encoders.base import TextEncoder

DEFAULT_TEMPLATES: list[str] = [
    "a photo of {concept}",
    "a photo containing {concept}",
    "an image showing {concept}",
]

# Fallback reference set for compute_text_center when a run's own concepts
# are too few (< 10) to give a stable center and no explicit
# demean_reference_concepts was supplied -- see cards.pipeline.run, which
# logs a warning every time this fallback is actually used, since it's a
# generic stand-in, not a substitute for a real concept-bank-specific
# reference set. Deliberately broad/diverse (objects, animals, materials,
# scenes) rather than tied to any one dataset's vocabulary.
GENERIC_REFERENCE_CONCEPTS: list[str] = [
    "dog", "cat", "bird", "fish", "person", "car", "bicycle", "boat",
    "tree", "flower", "grass", "mountain", "cloud", "sky", "water",
    "house", "building", "road", "window", "door",
    "chair", "table", "book", "cup", "bottle", "clock", "lamp", "phone",
    "shoe", "hat",
]


def build_concept_query(
    concept: str,
    encoder: TextEncoder,
    templates: list[str] = DEFAULT_TEMPLATES,
) -> torch.Tensor:
    """Embed `concept` under each template and average to a single query vector `t_c`.

    Each per-template embedding is already unit-norm (TextEncoder contract),
    but their mean isn't, so the result is renormalized after averaging.
    """
    prompts = [template.format(concept=concept) for template in templates]
    embeddings = encoder.encode_text(prompts)  # (len(templates), dim)
    return F.normalize(embeddings.mean(dim=0), dim=0)


def build_concept_bank(
    concepts: list[str],
    encoder: TextEncoder,
    templates: list[str] = DEFAULT_TEMPLATES,
) -> dict[str, torch.Tensor]:
    """Build `{concept: t_c}` for a full concept bank."""
    return {concept: build_concept_query(concept, encoder, templates) for concept in concepts}


def compute_text_center(
    concepts: list[str],
    encoder: TextEncoder,
    templates: list[str] = DEFAULT_TEMPLATES,
) -> torch.Tensor:
    """Mean t_c over `concepts` -- the text-modality "center" used to
    de-mean queries against the CLIP modality gap (see demean_query).

    Not renormalized: this is a de-meaning offset, not a direction, and
    subtracting an offset from t_c should not be treated as a query
    vector on its own. `concepts` should be a broad, stable reference
    set (e.g. a full concept bank), not just the handful of concepts
    being scored in one particular run -- too small a reference set
    gives an unstable center (and demeaning a single concept against a
    "center" computed only from itself is degenerate: the result would
    be the zero vector).
    """
    if len(concepts) < 10:
        raise ValueError(
            f"compute_text_center needs a broad reference set to give a stable "
            f"estimate, got only {len(concepts)} concepts -- pass a larger "
            f"concept bank, not just the concepts being scored in this run"
        )
    embeddings = torch.stack([build_concept_query(c, encoder, templates) for c in concepts])
    return embeddings.mean(dim=0)


def demean_query(t_c: torch.Tensor, text_center: torch.Tensor) -> torch.Tensor:
    """Subtract the text-modality center from t_c to counteract the CLIP
    modality gap before retrieval.

    Deliberately NOT renormalized to unit length: retrieve_top_bottom_k's
    ranking is invariant to the query's norm (a positive scalar just
    scales every similarity equally), and matched_retrieval's
    _project_out renormalizes its direction argument internally -- so an
    unnormalized vector is safe everywhere t_c is actually consumed
    downstream (confirmed against cards.retrieval.confound/retrieve).
    Do NOT add the *other* modality's center back in (e.g. t_c + (mean_
    image - mean_text)): empirically this makes things worse, not
    better -- the image-center term is large and constant across every
    concept, and dominates/washes out each concept's own small
    distinguishing signal. See notes/pcbm_correlation_investigation.md's
    v8 entry for the measurements behind this.
    """
    return t_c - text_center
