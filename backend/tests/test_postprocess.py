from __future__ import annotations

from uuid import uuid4

from app.config import Settings
from app.inference.postprocess import QualitySummary, age_to_bracket, build_response
from app.models.base import RawAttributes
from app.schemas import AudioQuality


def test_age_bracket_boundaries_are_unambiguous() -> None:
    assert age_to_bracket(17.99) == "unknown"
    assert age_to_bracket(18.0) == "18-30"
    assert age_to_bracket(30.99) == "18-30"
    assert age_to_bracket(31.0) == "31-45"
    assert age_to_bracket(46.0) == "46-60"
    assert age_to_bracket(59.99) == "46-60"
    assert age_to_bracket(60.0) == "60+"
    assert age_to_bracket(120.0) == "60+"
    assert age_to_bracket(120.01) == "unknown"


def test_direct_ordinal_age_probabilities_bypass_regression_heuristic() -> None:
    quality = QualitySummary(
        label=AudioQuality.GOOD,
        duration_seconds=5.0,
        voiced_seconds=4.0,
        speech_ratio=0.8,
        snr_db=20.0,
        clipping_ratio=0.0,
        rms_dbfs=-18.0,
        spectral_flatness=0.1,
        low_frequency_ratio=0.1,
    )
    raw = RawAttributes(
        gender_probabilities={"female": 0.2, "male": 0.8},
        age_bracket_probabilities={
            "18-30": 0.1,
            "31-45": 0.2,
            "46-60": 0.6,
            "60+": 0.1,
        },
    )

    result = build_response(
        contact_id=uuid4(),
        raw=raw,
        quality=quality,
        processing_ms=20,
        settings=Settings(),
    )

    assert result.age_bracket.prediction == "46-60"
    assert result.age_bracket.confidence == 0.6
