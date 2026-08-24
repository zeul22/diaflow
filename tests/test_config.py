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


@pytest.mark.parametrize(
    "origin",
    ["ws://voice.example", "https://voice.example/path", "voice.example"],
)
def test_invalid_websocket_origin_configuration_fails_startup(origin) -> None:
    settings = replace(Settings(), ws_allowed_origins=(origin,))

    with pytest.raises(ValueError, match="WS_ALLOWED_ORIGINS"):
        settings.validate()


def test_websocket_origins_are_normalized_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "WS_ALLOWED_ORIGINS",
        " HTTPS://VOICE.EXAMPLE/ , http://localhost:3000 ",
    )

    settings = Settings.from_env()

    assert settings.ws_allowed_origins == (
        "https://voice.example",
        "http://localhost:3000",
    )
