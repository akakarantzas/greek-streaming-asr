from __future__ import annotations

import pytest

from src.eval.wer import compute_wer


def test_wer_exact_match() -> None:
    result = compute_wer(["hello world"], ["hello world"])

    assert result["wer"] == 0.0
    assert result["total_words"] == 2


def test_wer_insertion() -> None:
    result = compute_wer(["hello"], ["hello world"])

    assert result["insertions"] == 1
    assert result["wer"] == 1.0


def test_wer_deletion() -> None:
    result = compute_wer(["hello world"], ["hello"])

    assert result["deletions"] == 1
    assert result["wer"] == 0.5


def test_wer_substitution() -> None:
    result = compute_wer(["hello world"], ["hello there"])

    assert result["substitutions"] == 1
    assert result["wer"] == 0.5


def test_wer_empty_hypothesis() -> None:
    result = compute_wer(["hello world"], [""])

    assert result["deletions"] == 2
    assert result["wer"] == 1.0


def test_wer_greek_punctuation_normalization() -> None:
    result = compute_wer(["Γεια σου, κόσμε!"], ["γεια σου κόσμε"])

    assert result["wer"] == 0.0
    assert result["total_words"] == 3


def test_wer_result_includes_normalization_description() -> None:
    result = compute_wer(["hello"], ["hello"])

    assert "normalization" in result
    assert "lowercase" in result["normalization"]
    assert "punctuation" in result["normalization"]


def test_wer_mismatched_list_lengths_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_wer(["hello"], ["hello", "world"])


def test_wer_greek_final_sigma_and_case_do_not_count_as_errors() -> None:
    result = compute_wer(["ΚΑΛΗΜΈΡΑ ΚΌΣΜΟΣ"], ["καλημέρα κόσμοσ"])

    assert result["wer"] == 0.0


def test_wer_greek_accents_counted_by_default() -> None:
    result = compute_wer(["πότε έρχεσαι"], ["ποτε έρχεσαι"])

    assert result["substitutions"] == 1
    assert result["wer"] == 0.5


def test_wer_greek_accent_insensitive_mode() -> None:
    result = compute_wer(["πότε έρχεσαι"], ["ποτε ερχεσαι"], strip_accents=True)

    assert result["wer"] == 0.0
    assert "strip Greek accents" in result["normalization"]


def test_wer_greek_punctuation_is_ignored() -> None:
    result = compute_wer(["«Καλημέρα», είπε· πότε;"], ["καλημέρα είπε πότε"])

    assert result["wer"] == 0.0
