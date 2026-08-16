# Broden concept dataset: label corruption finding

Initial validation of CARDS' Steps 1-2 (design doc Section 2, item 5 — retrieval
purity against Broden ground truth) using `scripts/broden_purity_check.py` and
`scripts/broden_label_flags.py` turned up something stronger than "CLIP's
retrieval is imperfect on some concepts": **unambiguous, severe ground-truth
label corruption**, concentrated in specific concepts, not spread evenly as
retrieval difficulty would be.

## Dataset provenance

Broden Dataset: [`post_hoc_cbm`'s
README](https://github.com/mertyg/post-hoc-cbm/blob/main/README.md) links to
it as a Google Drive download, crediting Amirata Ghorbani and Abubakar Abid,
and describing it as "mostly inherited from the Broden Dataset." Abubakar Abid
is also a co-author (with Mert Yuksekgonul and James Zou) of ["Meaningfully
Debugging Model Mistakes using Conceptual Counterfactual
Explanations"](https://arxiv.org/abs/2106.12723) (CCE, ICML 2022) — the same
CCE baseline the CARDS design doc compares against throughout — making it very
likely this concept dataset originates from that paper's own experimental
setup, then got reused and redistributed via `post_hoc_cbm`.

## What was found

Visually inspecting the flagged images (`results/broden_label_flags.csv`) for
every concept with notably low average precision:

| Concept | AP | A "ground-truth positive" image actually shows |
|---|---|---|
| `air_conditioner` | 0.65 | An airport control tower |
| `bathroom_s` | 0.29 | An office building exterior; a street with traffic signs |
| `street_s` | 0.31 | A dam / hydroelectric spillway |
| `knob` | — | A windmill |
| `blotchy` | 0.12 | A stone aqueduct/arch bridge |
| `apron` | 0.25 | Red British telephone booths |
| `outside_arm` | 0.26 | A wall of guitars in a music shop |
| `mouse` | 0.35 | An operating room; a lab workshop |
| `handle` | 0.33 | A postcard collage; a meeting table with laptops |
| `dining_room_s` | 0.44 | Dirt-track car racing; a muddy forest trail |

9/9 checked low-AP concepts showed this pattern. As a control, the same check
on two high-AP concepts (`dog` 0.97, `bus` 1.00) found nothing wrong — their
"worst" images were still genuinely on-concept, just harder examples (an
oddly-angled dog, a small bus in a wide landscape).

An earlier pass through this investigation wrongly attributed the `_s` (scene)
concepts' low scores to "genuine scene-retrieval difficulty." That conclusion
doesn't hold up once the actual flagged images are looked at, and is retracted
here.

## Root cause, traced as far as possible

Cross-referencing two mislabeled images against the raw, original NetDissect
Broden release (`../NetDissect/dataset/broden1_224/index.csv` + `label.csv`,
decoding the actual segmentation/scene labels) confirms the raw ground truth
is correct:

- A `knob`-labeled image in `broden_concepts/` is genuinely labeled `blade`
  (label 355 — a windmill's rotor blades) in the raw Broden index.
- A `street_s`-labeled image in `broden_concepts/` is genuinely labeled
  `dam-s` (label 931) in the raw Broden index.

Both are real, valid Broden labels — just filed under the wrong concept folder
somewhere in whatever process built the downloadable `broden_concepts`
package. That process isn't in this repo or in `post_hoc_cbm` (which only
*reads* `broden_concepts/`, via `broden_concept_loaders` in
`data/concept_loaders.py`), so the exact mechanism is unrecovered. Checked and
ruled out:

- **A simple row-index shift** in `c_part.csv` / `c_scene.csv` — `knob` (row
  28) vs. `blade` (row 92), `street-s` (row 1) vs. `dam-s` (row 271): no
  consistent offset.
- **"Rare concepts get padded with random images"** — plausible for `knob`
  (99 real images) but `street_s` has 2241 genuine images and isn't rare, yet
  shows the same corruption.

Net effect: this specific concept-dataset package, reused across at least
`post_hoc_cbm` and possibly other concept-explanation work building on the
same Ghorbani/Abid/CCE asset, has real image-to-concept mislabeling
independent of any code in this repo.

## Takeaway

Low AP against this dataset is a strong signal to check for mislabeling
before trusting it as a retrieval-quality or difficulty measurement — it is
not safe to assume a low score means CARDS' retrieval (or any retrieval
method) is failing.

## Update: corruption is pervasive, not confined to low-AP concepts

Follow-up spot-check (prompted by scoping a CIFAR-100 PCBM concept bank,
which would source from this dataset) sampled the single worst-ranked
flagged positive across a spread of concepts from AP 0.44 up to 0.87 --
well above the ≤0.436 range of the originally-checked concepts:

| concept | AP | worst flagged positive | verdict |
|---|---|---|---|
| bumper | 0.440 | shopping carts | corrupted |
| candlestick | 0.462 | baby cribs | corrupted |
| faucet | 0.483 | speaker/projector | corrupted |
| pedestal | 0.486 | bakery display case | corrupted |
| clock | 0.608 | stacked warehouse goods | corrupted |
| can | 0.647 | pianos | corrupted |
| handle_bar | 0.700 | folded fabric | corrupted |
| glass | 0.785 | TV/entertainment setup | likely corrupted |
| bucket | 0.748 | shearing scene, bucket visible in background | plausible |
| ottoman | 0.853 | padded round stool | plausible |
| **armchair** | **0.872** | **dog in grass wearing a hat** | **corrupted** |
| dog (control) | 0.972 | genuine dog | clean |
| bus (control) | 0.997 | distant landscape, ambiguous | plausible |

`armchair` at AP 0.872 -- top quartile of all 170 concepts -- still has a
blatantly mislabeled worst-ranked positive. This overturns the earlier
framing ("corruption is concentrated in low-AP concepts"): it looks
pervasive across most/all of the `broden_concepts` package at some rate,
with AP tracking *severity* (what fraction of a concept's positives are
affected) rather than presence/absence of corruption. A single AP
threshold cannot cleanly separate corrupted from clean concepts --
`n_flags` / flag-rate from `broden_label_flags.csv` doesn't help either,
since it flags a near-constant ~15% of each concept's pool regardless of
AP (an artifact of how the flagging script picks candidates, not a
severity signal).

Practical implication: any future use of this dataset as ground truth
(e.g. training a PCBM concept bank) should not rely on an AP-threshold
filter alone. Rebuilding the relevant concept(s) directly from the raw
NetDissect release (`../NetDissect/dataset/broden1_224/`, already
confirmed correct where cross-referenced -- see Root cause section
above) is the more defensible path. Decision on how to proceed
deferred -- see `notes/` for the live discussion when it resumes.

## Open follow-ups

- Decide the concrete path forward for a corruption-free concept source
  (rebuild from raw NetDissect vs. strict-threshold-plus-residual-risk
  vs. sidestep Broden entirely with CLIP-prompted concepts) -- paused,
  to be revisited.
- Finding or reconstructing the actual script that built `broden_concepts/`
  from the raw Broden release, to pin down the exact mechanism.

## Reproducing this

```
uv run python scripts/broden_purity_check.py     # results/broden_purity_summary.csv
uv run python scripts/broden_label_flags.py       # results/broden_label_flags.csv
```

`results/broden_concept_summary.csv` and `results/broden_label_flags.csv` are
the full 170-concept outputs from the run this finding is based on.
