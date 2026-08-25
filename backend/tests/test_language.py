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
    # Six seconds: above LANGUAGE_MIN_SECONDS, so the classifier is consulted.
    identifier = FakeIdentifier(code="hi", confidence=0.88)
    app = create_app(
        settings=settings,
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([identifier]),
    )

    with TestClient(app) as client:
        response = client.post(
            "/analyze",
            content=wav_bytes(speechlike_pcm(6.0)),
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
            content=wav_bytes(speechlike_pcm(6.0)),
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

    predictions = _stream_languages(app, chunks=12)

    # Early updates report `unknown`: under LANGUAGE_MIN_SECONDS the classifier
    # is not consulted, so nothing is determined rather than guessed. Only the
    # determined answers are compared here.
    languages = [
        item["language"]["prediction"]
        for item in predictions
        if item.get("language") and item["language"]["prediction"] != "unknown"
    ]
    assert languages, "no update ever accumulated enough audio for a language"
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


def test_short_audio_never_reaches_the_language_model(settings) -> None:
    """Below five seconds this model is confidently wrong, not just uncertain.

    Measured on real English speech: two seconds returned Pashto at 0.429 and
    three seconds returned Latin at 0.929. A confidence threshold cannot filter a
    0.929, so short audio must not be shown to the classifier at all.
    """

    identifier = FakeIdentifier()
    app = create_app(
        settings=settings,
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([identifier]),
    )

    with TestClient(app) as client:
        short = client.post(
            "/v1/analyze",
            content=wav_bytes(speechlike_pcm(3.0)),
            headers={"Content-Type": "audio/wav"},
        ).json()
        long = client.post(
            "/v1/analyze",
            content=wav_bytes(speechlike_pcm(6.0)),
            headers={"Content-Type": "audio/wav"},
        ).json()
        metrics = client.get("/metrics").text

    # `unknown`, not absent: the deployment does language identification, it
    # just could not determine one. Absent is reserved for "not configured".
    assert short["language"] == {"prediction": "unknown", "confidence": 0.0}
    assert long["language"]["prediction"] == "en"
    # The classifier ran once, for the long clip only.
    assert identifier.calls == 1
    assert 'voice_attribute_language_skipped_total{reason="too_short"} 1.0' in metrics


def test_legacy_iso_codes_are_normalised() -> None:
    """VoxLingua107 ships superseded tags; "IW" in a UI is not a language."""

    assert parse_language_code("iw: Hebrew") == "he"
    assert parse_language_code("jw: Javanese") == "jv"
    assert parse_language_code("in: Indonesian") == "id"
    assert parse_language_code("en: English") == "en"


# Real posteriors measured on a 3-second window of genuine English speech.
# Latin wins outright across all 107 classes; English is present but tiny.
_THREE_SECOND_ENGLISH = {"la": 0.929, "en": 0.033, "nn": 0.006, "hi": 0.001}


def _measured(mapping, allowed=(), threshold=0.50, margin_ratio=2.0):
    labels = tuple(f"{code}: X" for code in mapping)
    values = np.asarray(list(mapping.values()), dtype=np.float64)
    return decide_language(
        values,
        labels,
        threshold=threshold,
        margin_ratio=margin_ratio,
        allowed=allowed,
    )


def test_allowlist_removes_languages_a_deployment_will_never_hear() -> None:
    """Latin beat English 0.929 to 0.033 on real English speech.

    No logistics caller speaks Latin, but the class still competes and wins.
    Restricting contention to the deployment's languages is what fixes that.
    """

    # Unrestricted, the model names Latin.
    assert _measured(_THREE_SECOND_ENGLISH).code == "la"
    # Restricted, Latin cannot win -- but the evidence is still too weak to
    # name anything, so the honest answer is abstention rather than English.
    assert (
        _measured(_THREE_SECOND_ENGLISH, allowed=("en", "hi", "ta")).code == "unknown"
    )


def test_the_floor_applies_to_the_raw_posterior_not_a_renormalized_one() -> None:
    """Renormalizing over a subset inflates confidence without new evidence.

    That 3-second window renormalizes to English at 0.945 from a raw 0.033.
    Reporting 0.945 would be a fabricated number.
    """

    estimate = _measured(_THREE_SECOND_ENGLISH, allowed=("en", "hi"), threshold=0.02)

    assert estimate.code == "en"
    # The raw posterior, not the 0.945 the subset would renormalize to.
    assert estimate.confidence == pytest.approx(0.033)


def test_allowlist_keeps_a_confident_answer() -> None:
    measured = {"en": 0.641, "nl": 0.075, "la": 0.060, "hi": 0.001}

    assert _measured(measured, allowed=("en", "hi", "ta")).code == "en"
    assert _measured(measured).code == "en"


def test_a_single_entry_allowlist_is_rejected() -> None:
    """One permitted language means the answer is predetermined, not measured."""

    assert _measured(_THREE_SECOND_ENGLISH, allowed=("en",)).code == "unknown"
    with pytest.raises(ValueError, match="LANGUAGE_ALLOWLIST"):
        replace(Settings(), language_allowlist=("en",)).validate()
    with pytest.raises(ValueError, match="LANGUAGE_ALLOWLIST"):
        replace(Settings(), language_allowlist=("en", "!!")).validate()


def test_allowlist_floor_measures_mass_in_the_served_languages() -> None:
    """The floor should ask whether this is a language you serve.

    Judging one class against the full 107-way distribution is too strict once
    96 of those classes are impossible: English at 0.45 is 48x chance level yet
    a flat 0.50 floor rejects it. With an allowlist the floor is applied to the
    permitted mass instead -- which is not renormalization, because the reported
    confidence is still the raw posterior.
    """

    # English clearly leads, but no single class reaches 0.50.
    spread = {"en": 0.45, "hi": 0.05, "ta": 0.02, "la": 0.30, "cy": 0.18}

    # Without an allowlist the flat floor rejects a perfectly good answer.
    assert _measured(spread).code == "unknown"

    # With one, the permitted mass is 0.52 and English wins it by 9x.
    estimate = _measured(spread, allowed=("en", "hi", "ta"))
    assert estimate.code == "en"
    assert estimate.confidence == pytest.approx(0.45)


def test_mass_outside_the_served_languages_still_abstains() -> None:
    """If the model thinks it is hearing something you do not serve, say so."""

    # Almost all the mass sits in excluded classes.
    elsewhere = {"en": 0.04, "hi": 0.01, "la": 0.60, "cy": 0.35}

    assert _measured(elsewhere, allowed=("en", "hi")).code == "unknown"


def test_ambiguity_between_served_languages_abstains() -> None:
    """A near-tie between two languages you serve is exactly when to abstain."""

    tie = {"en": 0.40, "hi": 0.38, "ta": 0.02, "la": 0.20}

    # Plenty of permitted mass, but the margin rule refuses the coin flip.
    assert _measured(tie, allowed=("en", "hi", "ta")).code == "unknown"
