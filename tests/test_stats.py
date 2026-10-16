from __future__ import annotations

from src.profiling.stats import mean, percentile


def test_mean_of_values() -> None:
    assert mean([100.0, 200.0]) == 150.0


def test_mean_handles_empty_values() -> None:
    assert mean([]) is None


def test_percentile_interpolates_between_values() -> None:
    assert percentile([100.0, 200.0], 95) == 195.0


def test_percentile_handles_empty_and_single_values() -> None:
    assert percentile([], 50) is None
    assert percentile([42.0], 95) == 42.0
