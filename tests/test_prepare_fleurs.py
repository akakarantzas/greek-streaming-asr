from __future__ import annotations

import io
import tarfile

import pytest

from scripts.prepare_fleurs import extract_wavs, parse_tsv, write_manifest
from src.eval.manifest import load_jsonl_manifest


def _tsv_line(file_name: str, text: str, num_samples: int) -> str:
    return "\t".join(["1", file_name, text, text.lower(), "x", str(num_samples), "FEMALE"])


def _tar_gz(files: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_fleurs_split_becomes_a_loadable_manifest(tmp_path) -> None:
    rows = parse_tsv(
        _tsv_line("a.wav", "Καλημέρα, κόσμε.", 32000)
        + "\n"
        + _tsv_line("b.wav", "Γεια σου.", 16000)
        + "\n"
    )
    audio_dir = tmp_path / "test"
    audio_dir.mkdir()
    archive = _tar_gz({"test/a.wav": b"aaa", "test/b.wav": b"bbb", "test/other.wav": b"ccc"})

    found = extract_wavs(archive, {"a.wav", "b.wav"}, audio_dir)
    write_manifest(rows, tmp_path / "el_gr_test.jsonl", audio_subdir="test")

    assert found == {"a.wav", "b.wav"}
    assert sorted(path.name for path in audio_dir.iterdir()) == ["a.wav", "b.wav"]
    manifest = load_jsonl_manifest(str(tmp_path / "el_gr_test.jsonl"), validate_files=True)
    assert [row["text"] for row in manifest] == ["Καλημέρα, κόσμε.", "Γεια σου."]
    assert [row["duration"] for row in manifest] == [2.0, 1.0]


def test_tsv_rejects_file_names_with_a_path() -> None:
    with pytest.raises(ValueError, match="unexpected file name"):
        parse_tsv(_tsv_line("../evil.wav", "κείμενο", 16000))
