from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from app.config import Settings
from app.inference.postprocess import (
    QualitySummary,
    age_bracket_confidence,
    age_sigma_years,
    age_to_bracket,
    build_response,
)
from app.models.base import RawAttributes
from app.schemas import AudioQuality


def _quality(label: AudioQuality = AudioQuality.GOOD) -> QualitySummary:
    return QualitySummary(
        label=label,
        duration_seconds=5.0,
        voiced_seconds=4.0,
        speech_ratio=0.8,
        snr_db=20.0,
        clipping_ratio=0.0,
        rms_dbfs=-18.0,
        spectral_flatness=0.1,
        low_frequency_ratio=0.1,
    )


def _response(raw: RawAttributes, quality: QualitySummary, settings: Settings):
    return build_response(
        contact_id=uuid4(),
        raw=raw,
        quality=quality,
        processing_ms=20,
        settings=settings,
    )


def test_age_bracket_boundaries_are_unambiguous() -> None:
    assert age_to_bracket(17.99) == "unknown"
    assert age_to_bracket(18.0) == "18-30"
    assert age_to_bracket(30.99) == "18-30"
    assert age_to_bracket(31.0) == "31-45"
    assert age_to_bracket(46.0) == "46-60"
    assert age_to_bracket(59.99) == "46-60"
    assert age_to_bracket(60.0) == "60+"
    assert age_to_bracket(120.01) == "unknown"


def test_direct_ordinal_age_probabilities_bypass_regression_heuristic() -> None:
    raw = RawAttributes(
        gender_probabilities={"female": 0.2, "male": 0.8},
        age_bracket_probabilities={
            "18-30": 0.1,
            "31-45": 0.2,
            "46-60": 0.6,
            "60+": 0.1,
        },
    )

    result = _response(raw, _quality(), Settings())

    assert result.age_bracket.prediction == "46-60"
    assert result.age_bracket.confidence == 0.6


def test_ensemble_disagreement_widens_age_uncertainty() -> None:
    settings = Settings()

    confident = age_sigma_years(40.0, 0.0, settings)
    uncertain = age_sigma_years(40.0, 12.0, settings)

    assert confident == pytest.approx(settings.age_residual_sigma_years)
    assert uncertain > confident


def test_age_abstention_is_reachable_for_an_in_range_estimate() -> None:
    """The old fixed-sigma confidence could never fall below its threshold.

    Every adult estimate scored at least 0.42, so AGE_CONFIDENCE_THRESHOLD was
    unreachable and age only ever abstained on out-of-range regressions. Per
    sample spread has to make the threshold live.
    """

    settings = Settings()
    agreeing = RawAttributes(
        gender_probabilities={"female": 0.1, "male": 0.9},
        age_years=47.0,
        age_spread_years=0.5,
        speaker_homogeneity=0.9,
        ensemble_windows=3,
    )
    disagreeing = replace(agreeing, age_spread_years=22.0)

    assert _response(agreeing, _quality(), settings).age_bracket.prediction == "46-60"
    withheld = _response(disagreeing, _quality(), settings).age_bracket
    assert withheld.prediction == "unknown"
    assert withheld.confidence == 0.0


def test_degraded_audio_raises_the_bar_instead_of_scaling_confidence() -> None:
    settings = Settings()
    raw = RawAttributes(
        gender_probabilities={"female": 0.32, "male": 0.68},
        age_years=38.0,
        age_spread_years=1.0,
    )

    good = _response(raw, _quality(AudioQuality.GOOD), settings)
    degraded = _response(raw, _quality(AudioQuality.DEGRADED), settings)

    # The emitted probability is the model's own, never multiplied by a quality
    # factor: 0.68 stays 0.68 rather than becoming 0.68 * 0.75.
    assert good.gender.prediction == "male"
    assert good.gender.confidence == pytest.approx(0.68)
    # The same probability does not clear the stricter degraded threshold.
    assert degraded.gender.prediction == "unknown"
    assert degraded.gender.confidence == 0.0


def test_confident_gender_is_reported_unscaled_on_degraded_audio() -> None:
    raw = RawAttributes(
        gender_probabilities={"female": 0.06, "male": 0.94},
        age_years=38.0,
        age_spread_years=1.0,
    )

    degraded = _response(raw, _quality(AudioQuality.DEGRADED), Settings())

    assert degraded.gender.prediction == "male"
    assert degraded.gender.confidence == pytest.approx(0.94)


def test_open_top_bracket_is_not_reported_as_near_certain() -> None:
    """A 90-year estimate used to score 0.999 -- the least reliable region of an
    SVR trained on VoxCeleb2 produced the highest confidence in the API."""

    settings = Settings()

    def confidence(age: float) -> float:
        return age_bracket_confidence(age, "60+", age_sigma_years(age, 1.0, settings))

    at_75 = confidence(75.0)
    at_90 = confidence(90.0)
    at_110 = confidence(110.0)

    assert at_90 < 0.85
    # Confidence falls as the estimate moves further outside the range where the
    # upstream head has training support.
    assert at_110 < at_90 < at_75


def test_segment_with_more_than_one_speaker_is_downgraded() -> None:
    settings = Settings()
    raw = RawAttributes(
        gender_probabilities={"female": 0.34, "male": 0.66},
        age_years=38.0,
        age_spread_years=1.0,
        speaker_homogeneity=0.05,
        ensemble_windows=3,
    )

    result = _response(raw, _quality(AudioQuality.GOOD), settings)

    # Clean signal, but the sub-window embeddings describe different speakers.
    assert result.audio_quality is AudioQuality.DEGRADED
    assert result.gender.prediction == "unknown"


def test_raw_age_estimate_is_hidden_unless_explicitly_enabled() -> None:
    raw = RawAttributes(
        gender_probabilities={"female": 0.1, "male": 0.9},
        age_years=38.4,
    )

    assert _response(raw, _quality(), Settings()).debug_age_years is None

    exposed = _response(
        raw, _quality(), replace(Settings(), expose_debug_age_years=True)
    )
    assert exposed.debug_age_years == pytest.approx(38.4)


def test_the_debug_estimate_keeps_values_the_bracket_mapping_abstains_on() -> None:
    """The residual statistics it exists to measure need the tails.

    A 14-year estimate is `unknown` in the contract, but silently dropping it
    from the diagnostic would bias the measured spread toward zero.
    """

    settings = replace(Settings(), expose_debug_age_years=True)
    raw = RawAttributes(
        gender_probabilities={"female": 0.1, "male": 0.9},
        age_years=14.0,
    )

    result = _response(raw, _quality(), settings)

    assert result.age_bracket.prediction == "unknown"
    assert result.debug_age_years == pytest.approx(14.0)


def test_single_view_backend_keeps_the_residual_only_behaviour() -> None:
    raw = RawAttributes(
        gender_probabilities={"female": 0.1, "male": 0.9},
        age_years=38.0,
    )

    result = _response(raw, _quality(), Settings())

    assert result.audio_quality is AudioQuality.GOOD
    assert result.age_bracket.prediction == "31-45"
