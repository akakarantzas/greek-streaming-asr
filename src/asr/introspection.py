from __future__ import annotations

import inspect
from typing import Any


def call_has_named_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """True only if the signature names the parameter explicitly."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return parameter_name in signature.parameters


def call_accepts_parameter(callable_obj: Any, parameter_name: str) -> bool:
    """True if the parameter is named or the callable takes **kwargs."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False

    if parameter_name in signature.parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
