from __future__ import annotations

import math
from dataclasses import replace

import pytest

from app.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("decode_timeout_seconds", 0.0),
        ("queue_timeout_seconds", -1.0),
        ("request_idle_timeout_seconds", math.nan),
        ("ws_idle_timeout_seconds", math.inf),
        ("ws_max_session_seconds", 0.0),
    ],
)
def test_invalid_timeout_configuration_fails_startup(field, value) -> None:
    settings = replace(Settings(), **{field: value})

    with pytest.raises(ValueError):
        settings.validate()
