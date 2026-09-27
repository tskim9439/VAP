"""extract_json — zero-padded indices from long English streams ('[[049, "dogs"]]') are accepted on a second, lenient pass."""
import pytest
from vapasr.data.semcommit_llm import extract_json


def test_leading_zero_indices():
    assert extract_json('{"semantic_boundaries": [[049, "dogs"]], "fillers": [[007, 008]]}') == {"semantic_boundaries": [[49, "dogs"]], "fillers": [[7, 8]]}


def test_strict_first_and_values_kept():
    assert extract_json('<think>x</think>{"a": [0, 10, 0.5], "w": "0 007"}') == {"a": [0, 10, 0.5], "w": "0 007"}   # valid JSON untouched
    with pytest.raises(ValueError):
        extract_json("no json here")
