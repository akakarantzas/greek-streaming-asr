from __future__ import annotations

import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.asr.model_loader import (
    DEFAULT_MODEL_ID,
    _looks_like_local_nemo_path,
    _validate_local_nemo_path,
    format_bytes_as_gb,
    get_gpu_memory_allocated_mb,
    get_gpu_name,
    resolve_device,
)


def test_default_model_id_is_treated_as_hugging_face_id() -> None:
    assert not _looks_like_local_nemo_path(DEFAULT_MODEL_ID)


def test_nemo_suffix_is_treated_as_local_path() -> None:
    assert _looks_like_local_nemo_path("model.nemo")


def test_windows_path_is_treated_as_local_path() -> None:
    assert _looks_like_local_nemo_path(r"checkpoints\model.nemo")


def test_validate_local_nemo_path_accepts_existing_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.nemo"
        checkpoint.write_text("placeholder")

        assert _validate_local_nemo_path(str(checkpoint)) == checkpoint


def test_validate_local_nemo_path_rejects_missing_checkpoint() -> None:
    with _workspace_temp_dir() as temp_dir:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _validate_local_nemo_path(str(temp_dir / "missing.nemo"))


def test_validate_local_nemo_path_rejects_non_nemo_file() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.ckpt"
        checkpoint.write_text("placeholder")

        with pytest.raises(ValueError, match=".nemo suffix"):
            _validate_local_nemo_path(str(checkpoint))


def test_validate_local_nemo_path_rejects_directory() -> None:
    with _workspace_temp_dir() as temp_dir:
        checkpoint = temp_dir / "model.nemo"
        checkpoint.mkdir()

        with pytest.raises(ValueError, match="must be a file"):
            _validate_local_nemo_path(str(checkpoint))


def test_format_bytes_as_gb() -> None:
    assert format_bytes_as_gb(0) == "0.00 GB"
    assert format_bytes_as_gb(1073741824) == "1.00 GB"


@contextmanager
def _workspace_temp_dir():
    with tempfile.TemporaryDirectory(
        prefix="model_loader_",
        dir=Path.cwd() / "tests",
    ) as temp_dir:
        yield Path(temp_dir)


def test_gpu_memory_is_none_when_torch_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    assert get_gpu_memory_allocated_mb() is None


def test_gpu_memory_is_none_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert get_gpu_memory_allocated_mb() is None


def test_gpu_memory_is_reported_in_mb_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        memory_allocated=lambda: 5 * 1024**2,
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    assert get_gpu_memory_allocated_mb() == 5.0


def test_resolve_device_passes_explicit_values_through() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_uses_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert resolve_device("auto") == "cpu"

    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device("auto") == "cpu"


def test_get_gpu_name_is_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert get_gpu_name() is None
