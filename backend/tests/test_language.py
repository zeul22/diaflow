from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import _carry_language, create_app
from app.models.language import (
    LanguageEstimate,
    decide_language,
    parse_language_code,
)
from app.models.pool import EstimatorPool
from app.schemas import (
    AgeBracketPrediction,
    AnalysisResponse,
    AudioQuality,
    GenderPrediction,
    LanguagePrediction,
)
from tests.conftest import FakeEstimator, speechlike_pcm, wav_bytes


class FakeIdentifier:
    name = "fake-language-id"

    def __init__(self, code: str = "en", confidence: float = 0.91) -> None:
        self.calls = 0
        self.code = code
        self.confidence = confidence
        self.warmed = False

    def identify(self, samples) -> LanguageEstimate:
        assert samples.ndim == 1
        self.calls += 1
        return LanguageEstimate(code=self.code, confidence=self.confidence)

    def warmup(self) -> None:
        self.warmed = True


def _response(language: LanguagePrediction | None) -> AnalysisResponse:
    from uuid import uuid4

    return AnalysisResponse(
        contact_id=uuid4(),
        gender=GenderPrediction(prediction="male", confidence=0.9),
        age_bracket=AgeBracketPrediction(prediction="31-45", confidence=0.5),
        processing_ms=10,
        audio_quality=AudioQuality.GOOD,
        language=language,
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("en: English", "en"),
        ("hi: Hindi", "hi"),
        ("ceb: Cebuano", "ceb"),
        ("  ru: Russian ", "ru"),
        ("English", "unknown"),
        ("", "unknown"),
    ],
)
def test_upstream_labels_reduce_to_language_tags(label, expected) -> None:
    assert parse_language_code(label) == expected


LABELS = ("en: English", "nl: Dutch", "la: Latin", "th: Thai")


def _decide(values, threshold=0.35, margin_ratio=2.0) -> LanguageEstimate:
    return decide_language(
        np.asarray(values, dtype=np.float64),
        LABELS,
        threshold=threshold,
        margin_ratio=margin_ratio,
    )


def test_a_clear_winner_is_accepted_below_a_naive_half_probability() -> None:
    """The real posteriors measured for an English clip on a 5s window.

    Mass spreads over 107 related languages, so English led the runner-up 3.7x
    while scoring 0.468. A bare 0.5 floor rejected a correct, unambiguous answer.
    """

    estimate = _decide([0.4679, 0.1269, 0.1155, 0.0001])

    assert estimate.code == "en"
    assert estimate.confidence == pytest.approx(0.4679)


def test_a_genuinely_ambiguous_result_still_abstains() -> None:
    assert _decide([0.38, 0.31, 0.05, 0.01]).code == "unknown"


def test_a_confident_result_is_accepted() -> None:
    estimate = _decide([0.0005, 0.0003, 0.0003, 0.9989])

    assert estimate.code == "th"
    assert estimate.confidence == pytest.approx(0.9989)


def test_a_low_floor_is_still_enforced() -> None:
    # Huge margin, but too little absolute mass to name a language.
    assert _decide([0.20, 0.01, 0.01, 0.01]).code == "unknown"


def test_malformed_posteriors_abstain() -> None:
    assert _decide([float("nan"), 0.1, 0.1, 0.1]).code == "unknown"
    assert (
        decide_language(
            np.asarray([1.0]), ("en: English",), threshold=0.35, margin_ratio=2.0
        ).code
        == "unknown"
    )
    assert (
        decide_language(
            np.asarray([0.9, 0.1]), LABELS, threshold=0.35, margin_ratio=2.0
        ).code
        == "unknown"
    )


def test_language_field_is_absent_when_the_deployment_disables_it(client) -> None:
    response = client.post(
        "/analyze",
        content=wav_bytes(speechlike_pcm()),
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 200, response.text
    assert "language" not in response.json()


def test_language_is_reported_when_enabled(settings) -> None:
    identifier = FakeIdentifier(code="hi", confidence=0.88)
    app = create_app(
        settings=settings,
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([identifier]),
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            content=wav_bytes(speechlike_pcm()),
            headers={"Content-Type": "audio/wav"},
        )

    assert response.status_code == 200, response.text
    assert response.json()["language"] == {"prediction": "hi", "confidence": 0.88}
    assert identifier.calls == 1
    assert identifier.warmed


def test_low_confidence_language_is_withheld_rather_than_guessed(settings) -> None:
    app = create_app(
        settings=settings,
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([FakeIdentifier(code="unknown", confidence=0.0)]),
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            content=wav_bytes(speechlike_pcm()),
            headers={"Content-Type": "audio/wav"},
        )

    assert response.json()["language"] == {"prediction": "unknown", "confidence": 0.0}


def test_a_new_confident_language_replaces_the_previous_one() -> None:
    """A caller who switches language must not be reported as the old one."""

    english = LanguagePrediction(prediction="en", confidence=0.9)
    hindi = LanguagePrediction(prediction="hi", confidence=0.72)

    result, latest = _carry_language(_response(hindi), english)

    assert latest == hindi
    assert result.language == hindi


def test_an_unknown_window_keeps_the_last_confident_language() -> None:
    """One bad window is not evidence that the caller stopped speaking a language."""

    english = LanguagePrediction(prediction="en", confidence=0.9)
    blank = LanguagePrediction(prediction="unknown", confidence=0.0)

    result, latest = _carry_language(_response(blank), english)

    assert latest == english
    assert result.language == english
    # With nothing to carry forward, unknown is reported honestly.
    result, latest = _carry_language(_response(blank), None)
    assert latest is None
    assert result.language == blank
    # A disabled backend stays disabled.
    disabled = _response(None)
    assert _carry_language(disabled, None) == (disabled, None)


def _stream_languages(app, chunks: int = 8) -> list[dict]:
    raw_pcm = (speechlike_pcm(float(chunks)) * 32767.0).astype("<i2").tobytes()
    one_second = 16_000 * 2
    predictions = []
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/analyze") as websocket,
    ):
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        for index in range(chunks):
            websocket.send_bytes(raw_pcm[index * one_second : (index + 1) * one_second])
        websocket.send_json({"type": "end"})
        while True:
            message = websocket.receive_json()
            if message["type"] != "prediction":
                continue
            predictions.append(message)
            if message["is_final"]:
                break
    return predictions


def test_a_mid_session_language_switch_is_reflected_live(settings) -> None:
    class SwitchingIdentifier(FakeIdentifier):
        def identify(self, samples):
            estimate = super().identify(samples)
            # Speaks English, then switches part-way through the call.
            return (
                estimate
                if self.calls < 2
                else LanguageEstimate(code="hi", confidence=0.77)
            )

    identifier = SwitchingIdentifier()
    app = create_app(
        settings=replace(settings, language_refresh_seconds=0.0),
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([identifier]),
    )

    predictions = _stream_languages(app)

    languages = [item["language"]["prediction"] for item in predictions]
    assert languages[0] == "en"
    assert languages[-1] == "hi"
    assert predictions[-1]["language"]["confidence"] == 0.77


def test_language_rechecks_are_rate_limited_between_updates(settings) -> None:
    identifier = FakeIdentifier()
    app = create_app(
        # A refresh window wider than the session means one check plus the final.
        settings=replace(settings, language_refresh_seconds=60.0),
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([identifier]),
    )

    predictions = _stream_languages(app)

    assert len(predictions) > 3
    # Not once per update: the cached answer is reused between rechecks.
    assert identifier.calls < len(predictions)
    assert predictions[-1]["language"] == {"prediction": "en", "confidence": 0.91}


def test_language_backend_configuration_is_validated() -> None:
    replace(Settings(), language_backend="voxlingua_ecapa").validate()
    with pytest.raises(ValueError, match="LANGUAGE_BACKEND"):
        replace(Settings(), language_backend="guess").validate()
    with pytest.raises(ValueError, match="LANGUAGE_CONFIDENCE_THRESHOLD"):
        replace(Settings(), language_confidence_threshold=1.4).validate()
