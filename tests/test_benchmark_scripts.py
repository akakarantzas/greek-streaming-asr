from __future__ import annotations

import json

import pytest

from scripts.make_benchmark_plots import (
    has_concurrency_latency_data,
    has_latency_distribution_data,
    load_benchmark_artifacts,
)
from scripts.profile_inference import parse_stream_counts


def test_parse_stream_counts() -> None:
    assert parse_stream_counts("1,2,4,8") == [1, 2, 4, 8]


def test_parse_stream_counts_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_stream_counts("1,0")


def test_load_benchmark_artifacts(tmp_path) -> None:
    (tmp_path / "profile_summary.json").write_text(
        json.dumps({"per_input_latencies_ms": [1.0]}),
        encoding="utf-8",
    )

    artifacts = load_benchmark_artifacts(str(tmp_path))

    assert artifacts == {"profile_summary": {"per_input_latencies_ms": [1.0]}}


def test_plot_data_presence_helpers() -> None:
    artifacts = {
        "profile_summary": {
            "per_input_latencies_ms": [1.0, 2.0],
        },
        "concurrency_summary": [{"stream_count": 1, "p95_latency_ms": 100.0}],
    }

    assert has_latency_distribution_data(artifacts)
    assert has_concurrency_latency_data(artifacts)


def test_missing_plot_data_helpers_return_false() -> None:
    assert not has_latency_distribution_data({})
    assert not has_concurrency_latency_data({})


def test_cpu_profile_reports_no_gpu_numbers(tmp_path, monkeypatch) -> None:
    import scripts.profile_inference as profile_script

    (tmp_path / "a.wav").write_bytes(b"")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": "a.wav", "text": "γεια"}), encoding="utf-8")
    monkeypatch.setattr(profile_script, "load_model", lambda **_kwargs: object())
    monkeypatch.setattr(
        profile_script,
        "profile_inference",
        lambda **_kwargs: {"max_memory_allocated_mb": 0.0, "avg_gpu_util_pct": 37.0},
    )
    monkeypatch.setattr(
        profile_script,
        "concurrency_test",
        lambda **_kwargs: [{"stream_count": 1, "max_memory_allocated_mb": 0.0}],
    )
    out = tmp_path / "out"

    assert profile_script.main(
        ["--manifest", str(manifest), "--output-dir", str(out), "--device", "cpu"]
    ) == 0

    profile = json.loads((out / "profile_summary.json").read_text(encoding="utf-8"))
    concurrency = json.loads((out / "concurrency_summary.json").read_text(encoding="utf-8"))
    assert profile["max_memory_allocated_mb"] is None
    assert profile["avg_gpu_util_pct"] is None
    assert concurrency[0]["max_memory_allocated_mb"] is None
