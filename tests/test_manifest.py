from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.manifest import load_jsonl_manifest


def test_valid_manifest(tmp_path) -> None:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"placeholder")
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_file,
        [{"audio_filepath": str(audio_file), "text": "ground truth", "duration": 3.2}],
    )

    rows = load_jsonl_manifest(str(manifest_file))

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": 3.2,
        }
    ]


def test_relative_audio_path_resolves_from_manifest_directory(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"
    sample_dir = tmp_path / "samples"
    manifest_dir.mkdir()
    sample_dir.mkdir()
    audio_file = sample_dir / "sample.wav"
    audio_file.write_bytes(b"placeholder")
    manifest_file = manifest_dir / "manifest.jsonl"
    _write_jsonl(
        manifest_file,
        [{"audio_filepath": "../samples/sample.wav", "text": "ground truth"}],
    )

    rows = load_jsonl_manifest(str(manifest_file))

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": None,
        }
    ]


def test_missing_audio_filepath(tmp_path) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_file, [{"text": "ground truth"}])

    with pytest.raises(ValueError, match="line 1.*audio_filepath"):
        load_jsonl_manifest(str(manifest_file), validate_files=False)


def test_missing_text(tmp_path) -> None:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"placeholder")
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest_file, [{"audio_filepath": str(audio_file)}])

    with pytest.raises(ValueError, match="line 1.*text"):
        load_jsonl_manifest(str(manifest_file))


def test_invalid_duration(tmp_path) -> None:
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"placeholder")
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_file,
        [{"audio_filepath": str(audio_file), "text": "ground truth", "duration": "3.2"}],
    )

    with pytest.raises(ValueError, match="line 1.*duration"):
        load_jsonl_manifest(str(manifest_file))


def test_invalid_json_line(tmp_path) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    manifest_file.write_text('{"audio_filepath": "sample.wav", "text": ', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_jsonl_manifest(str(manifest_file), validate_files=False)


def test_manifest_with_only_blank_lines_is_rejected(tmp_path) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    manifest_file.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="no rows"):
        load_jsonl_manifest(str(manifest_file), validate_files=False)


def test_missing_audio_file_when_validate_files_true(tmp_path) -> None:
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_file,
        [{"audio_filepath": str(tmp_path / "missing.wav"), "text": "ground truth"}],
    )

    with pytest.raises(ValueError, match="line 1.*does not exist"):
        load_jsonl_manifest(str(manifest_file), validate_files=True)


def test_allows_missing_audio_file_when_validate_files_false(tmp_path) -> None:
    audio_file = tmp_path / "missing.wav"
    manifest_file = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest_file,
        [{"audio_filepath": str(audio_file), "text": "ground truth"}],
    )

    rows = load_jsonl_manifest(str(manifest_file), validate_files=False)

    assert rows == [
        {
            "audio_filepath": str(audio_file),
            "text": "ground truth",
            "duration": None,
        }
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
