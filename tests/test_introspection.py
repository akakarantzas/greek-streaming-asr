from __future__ import annotations

from src.asr.introspection import call_accepts_parameter, call_has_named_parameter


def _named(audio, target_lang=None):
    return audio, target_lang


def _kwargs_only(audio, **kwargs):
    return audio, kwargs


def test_named_parameter_is_found_by_both_helpers() -> None:
    assert call_has_named_parameter(_named, "target_lang")
    assert call_accepts_parameter(_named, "target_lang")


def test_var_keyword_counts_as_accepting_but_not_as_named() -> None:
    assert call_accepts_parameter(_kwargs_only, "target_lang")
    assert not call_has_named_parameter(_kwargs_only, "target_lang")


def test_missing_parameter_is_not_accepted() -> None:
    assert not call_accepts_parameter(_named, "revision")
    assert not call_has_named_parameter(_named, "revision")


def test_uninspectable_callable_returns_false() -> None:
    assert not call_accepts_parameter(object(), "target_lang")
    assert not call_has_named_parameter(object(), "target_lang")
