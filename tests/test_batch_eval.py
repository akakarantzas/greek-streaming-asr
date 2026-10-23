from __future__ import annotations

import csv
import json

import pytest

from scripts.run_batch_eval import (
    _attach_metric_to_rows,
    build_summary,
    language_readiness_for,
    percentile,
    run_batch_eval,
)


def test_crash_mid_run_keeps_finished_transcripts(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"audio_filepath": f"{name}.wav", "text": "γεια σου"}) + "\n"
            for name in ("a", "b", "c")
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results.csv"

    def inference_fn(row: dict) -> dict:
        if row["audio_filepath"].endswith("b.wav"):
            raise RuntimeError("CUDA out of memory")
        return {"transcript": "γεια σου", "latency_ms": 10.0}

    with pytest.raises(RuntimeError, match="out of memory"):
        run_batch_eval(
            manifest_path=str(manifest),
            model_id_or_path="model",
            language="el",
            target_lang="el-GR",
            output=str(output),
            summary_output=str(tmp_path / "summary.json"),
            device="cpu",
            validate_files=False,
            inference_fn=inference_fn,
        )

    with output.open(encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["hypothesis_text"] for row in rows] == ["γεια σου"]
    assert rows[0]["audio_filepath"].endswith("a.wav")
    assert not (tmp_path / "summary.json").exists()


def test_build_summary_computes_greek_wer_and_latency_stats() -> None:
    rows = [
        {
            "reference_text": "hello world",
            "hypothesis_text": "hello world",
            "latency_ms": 100.0,
        },
        {
            "reference_text": "good morning",
            "hypothesis_text": "good evening",
            "latency_ms": 200.0,
        },
    ]

    summary = build_summary(
        per_file_rows=rows,
        language="el",
        model_id_or_path="model",
        target_lang="el-GR",
        language_readiness="adaptation_ready",
        manifest_path="manifest.jsonl",
        warnings=[],
    )

    assert summary["metric_name"] == "wer"
    assert summary["metric_value"] == 0.25
    assert summary["metric_value_no_accents"] == 0.25
    assert summary["avg_latency_ms"] == 150.0
    assert summary["p50_latency_ms"] == 150.0
    assert summary["p95_latency_ms"] == 195.0
    assert summary["substitutions"] == 1


def test_build_summary_labels_greek_as_adaptation_ready() -> None:
    readiness, warnings = language_readiness_for("el")

    assert readiness == "adaptation_ready"
    assert "exploratory" in warnings[0]


def test_attach_metric_to_rows_uses_per_file_metric() -> None:
    rows = [
        {
            "reference_text": "hello world",
            "hypothesis_text": "hello mars",
            "wer": None,
            "cer": None,
        },
        {
            "reference_text": "good morning",
            "hypothesis_text": "good morning",
            "wer": None,
            "cer": None,
        },
    ]
    summary = {"metric_name": "wer", "metric_value": 0.25}

    _attach_metric_to_rows(rows, summary)

    assert rows[0]["wer"] == 0.5
    assert rows[1]["wer"] == 0.0


def test_percentile_handles_empty_values() -> None:
    assert percentile([], 50) is None


def test_percentile_handles_single_value() -> None:
    assert percentile([42.0], 95) == 42.0


def test_build_summary_reports_strict_and_accent_insensitive_wer() -> None:
    rows = [
        {
            "reference_text": "πότε έρχεσαι",
            "hypothesis_text": "ποτε ερχεσαι",
            "latency_ms": 100.0,
        },
    ]

    summary = build_summary(
        per_file_rows=rows,
        language="el",
        model_id_or_path="model",
        target_lang="el-GR",
        language_readiness="adaptation_ready",
        manifest_path="manifest.jsonl",
        warnings=[],
    )

    assert summary["metric_value"] == 1.0
    assert summary["metric_value_no_accents"] == 0.0


def test_attach_metric_to_rows_adds_accent_insensitive_wer() -> None:
    rows = [
        {
            "reference_text": "πότε έρχεσαι",
            "hypothesis_text": "ποτε ερχεσαι",
            "wer": None,
            "wer_no_accents": None,
        },
    ]

    _attach_metric_to_rows(rows, {"metric_name": "wer", "metric_value": 1.0})

    assert rows[0]["wer"] == 1.0
    assert rows[0]["wer_no_accents"] == 0.0
