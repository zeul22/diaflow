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


def test_known_wavlm_backend_is_accepted() -> None:
    replace(Settings(), model_backend="wavlm_onnx").validate()


def test_unknown_model_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="MODEL_BACKEND"):
        replace(Settings(), model_backend="mystery").validate()


def test_degraded_audio_cannot_be_easier_to_pass_than_good_audio() -> None:
    with pytest.raises(ValueError, match="GENDER_CONFIDENCE_THRESHOLD_DEGRADED"):
        replace(
            Settings(),
            gender_confidence_threshold=0.80,
            gender_confidence_threshold_degraded=0.60,
        ).validate()
    with pytest.raises(ValueError, match="AGE_CONFIDENCE_THRESHOLD_DEGRADED"):
        replace(
            Settings(),
            age_confidence_threshold=0.50,
            age_confidence_threshold_degraded=0.30,
        ).validate()


def test_age_reliable_range_must_be_ordered_and_adult() -> None:
    with pytest.raises(ValueError, match="AGE_RELIABLE_MIN_YEARS"):
        replace(Settings(), age_reliable_min_years=12.0).validate()
    with pytest.raises(ValueError, match="AGE_RELIABLE_MIN_YEARS"):
        replace(
            Settings(), age_reliable_min_years=80.0, age_reliable_max_years=70.0
        ).validate()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ws_emit_backoff", 0.5),
        ("ws_max_emit_interval_seconds", 0.5),
        ("ws_analysis_window_seconds", 1.0),
        ("ws_analysis_window_seconds", 60.0),
    ],
)
def test_invalid_progressive_streaming_configuration_fails_startup(
    field, value
) -> None:
    with pytest.raises(ValueError):
        replace(Settings(), **{field: value}).validate()


def test_replica_count_is_bounded() -> None:
    with pytest.raises(ValueError, match="INFERENCE_CONCURRENCY"):
        replace(Settings(), inference_concurrency=64).validate()
