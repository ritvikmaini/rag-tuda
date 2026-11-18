"""Answer shapes: the backend, the evals checks and the v2 page read one fixture. The evals runner keeps its own copy of the
checks because evals must not import the assistant; this fixture proves they agree."""

import json
import os

import pytest
from evals import checks
from rag_core.domain import answer_shapes

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
with open(os.path.join(ROOT, "tests", "contracts", "fixtures", "answer_shapes.json"), encoding="utf-8") as fh:
    FIXTURE = json.load(fh)


BACKEND = answer_shapes


@pytest.mark.parametrize("implementation", [BACKEND, checks], ids=["backend", "evals"])
def test_every_sample_has_its_shape_and_its_cited_share(implementation):
    for sample in FIXTURE["samples"]:
        assert implementation.detect_shape(sample["text"]) == sample["shape"], sample["text"]
        for shape, share in sample["cited"].items():
            assert list(implementation.cited_share(sample["text"], shape)) == share, (sample["text"], shape)


def test_the_fixture_names_the_backends_shapes_and_variants():
    assert FIXTURE["shapes"] == list(BACKEND.SHAPES) == list(checks.SHAPES)
    assert FIXTURE["variants"] == list(BACKEND.VARIANTS)
