"""Import smoke test: fails fast if the package/env isn't wired up correctly."""

import cards
import cards.attribution.global_mode
import cards.attribution.local_mode
import cards.attribution.normalization
import cards.concepts.prompts
import cards.directions.estimate
import cards.directions.orthogonalize
import cards.encoders.base
import cards.retrieval.confound
import cards.retrieval.pool
import cards.retrieval.retrieve


def test_package_imports():
    assert cards is not None
