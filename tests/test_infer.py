from __future__ import annotations

import pytest

from src.asr.infer import (
    extract_transcript_text,
    language_readiness_for,
    transcribe_file,
)


class TranscriptObject:
    def __init__(self, text: str) -> None:
        self.text = text


def test_extract_transcript_from_string() -> None:
    assert extract_transcript_text("hello world") == "hello world"


def test_extract_transcript_from_object_with_text() -> None:
    assert extract_transcript_text(TranscriptObject("object text")) == "object text"


def test_extract_transcript_from_list_of_strings() -> None:
    assert extract_transcript_text(["first", "second"]) == "first\nsecond"


def test_extract_transcript_from_list_of_objects_with_text() -> None:
    result = [TranscriptObject("first"), TranscriptObject("second")]

    assert extract_transcript_text(result) == "first\nsecond"


def test_extract_transcript_rejects_unknown_result_type() -> None:
    with pytest.raises(RuntimeError, match="Could not extract transcript text"):
        extract_transcript_text(object())


def test_language_readiness_warns_for_greek_exploratory_use() -> None:
    readiness, warnings = language_readiness_for("el-GR")

    assert readiness == "adaptation_ready"
    assert "exploratory" in warnings[0]
    assert language_readiness_for("auto") == ("auto", [])


def test_target_lang_support_requires_a_named_parameter(tmp_path) -> None:
    class KwargsModel:
        def transcribe(self, audio, **config_kwargs):
            self.config_kwargs = config_kwargs
            return ["text"]

    class NamedModel:
        def transcribe(self, audio, target_lang=None):
            return ["text"]

    kwargs_model = KwargsModel()
    via_kwargs = transcribe_file(kwargs_model, tmp_path / "a.wav", "el-GR")
    named = transcribe_file(NamedModel(), tmp_path / "a.wav", "el-GR")

    assert kwargs_model.config_kwargs == {"target_lang": "el-GR"}
    assert via_kwargs["target_lang_support"] is False
    assert "unverified" in via_kwargs["warnings"][0]
    assert named["target_lang_support"] is True
    assert named["warnings"] == []
