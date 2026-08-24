from __future__ import annotations

import math
from dataclasses import dataclass
from uuid import UUID

from app.config import Settings
from app.models.base import RawAttributes
from app.schemas import (
    AgeBracketPrediction,
    AnalysisResponse,
    AudioQuality,
    GenderPrediction,
)


@dataclass(frozen=True, slots=True)
class QualitySummary:
    label: AudioQuality
    duration_seconds: float
    voiced_seconds: float
    speech_ratio: float
    snr_db: float
    clipping_ratio: float
    rms_dbfs: float
    spectral_flatness: float
    low_frequency_ratio: float


AGE_RANGES: tuple[tuple[str, float, float], ...] = (
    ("18-30", 18.0, 31.0),
    ("31-45", 31.0, 46.0),
    ("46-60", 46.0, 60.0),
    ("60+", 60.0, math.inf),
)
MAX_PLAUSIBLE_AGE_YEARS = 120.0


def age_to_bracket(age_years: float) -> str:
    if (
        not math.isfinite(age_years)
        or age_years < 18.0
        or age_years > MAX_PLAUSIBLE_AGE_YEARS
    ):
        return "unknown"
    if age_years < 31.0:
        return "18-30"
    if age_years < 46.0:
        return "31-45"
    if age_years < 60.0:
        return "46-60"
    return "60+"


def _normal_cdf(value: float, mean: float, sigma: float) -> float:
    if value == math.inf:
        return 1.0
    return 0.5 * (1.0 + math.erf((value - mean) / (sigma * math.sqrt(2.0))))


def _age_bracket_confidence(age_years: float, bracket: str, sigma: float) -> float:
    adult_mass = max(1e-9, 1.0 - _normal_cdf(18.0, age_years, sigma))
    for candidate, low, high in AGE_RANGES:
        if candidate == bracket:
            probability = _normal_cdf(high, age_years, sigma) - _normal_cdf(
                low, age_years, sigma
            )
            return min(1.0, max(0.0, probability / adult_mass))
    return 0.0


def _quality_factor(quality: AudioQuality, settings: Settings) -> float:
    if quality is AudioQuality.GOOD:
        return 1.0
    if quality is AudioQuality.DEGRADED:
        return settings.degraded_confidence_factor
    return 0.0


def _valid_probability(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric


def _direct_age_prediction(
    probabilities: dict[str, float] | object,
    *,
    factor: float,
    threshold: float,
) -> AgeBracketPrediction:
    if not hasattr(probabilities, "get"):
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    normalized: dict[str, float] = {
        label: _valid_probability(probabilities.get(label, 0.0))
        for label, _, _ in AGE_RANGES
    }
    total = sum(normalized.values())
    if total <= 0.0:
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    normalized = {label: value / total for label, value in normalized.items()}
    label = max(normalized, key=normalized.get)
    confidence = normalized[label] * factor
    if confidence < threshold:
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    return AgeBracketPrediction(
        prediction=label,
        confidence=round(min(1.0, confidence), 4),
    )


def unknown_response(
    contact_id: UUID,
    processing_ms: int,
    quality: AudioQuality = AudioQuality.INSUFFICIENT,
) -> AnalysisResponse:
    return AnalysisResponse(
        contact_id=contact_id,
        gender=GenderPrediction(prediction="unknown", confidence=0.0),
        age_bracket=AgeBracketPrediction(prediction="unknown", confidence=0.0),
        processing_ms=processing_ms,
        audio_quality=quality,
    )


def build_response(
    *,
    contact_id: UUID,
    raw: RawAttributes,
    quality: QualitySummary,
    processing_ms: int,
    settings: Settings,
) -> AnalysisResponse:
    if quality.label is AudioQuality.INSUFFICIENT:
        return unknown_response(contact_id, processing_ms, quality.label)

    factor = _quality_factor(quality.label, settings)
    female = _valid_probability(raw.gender_probabilities.get("female", 0.0))
    male = _valid_probability(raw.gender_probabilities.get("male", 0.0))
    total = female + male
    if total > 0.0:
        female, male = female / total, male / total
    gender_label = "female" if female >= male else "male"
    gender_confidence = max(female, male) * factor
    if gender_confidence < settings.gender_confidence_threshold:
        gender = GenderPrediction(prediction="unknown", confidence=0.0)
    else:
        gender = GenderPrediction(
            prediction=gender_label,
            confidence=round(min(1.0, gender_confidence), 4),
        )

    if raw.age_bracket_probabilities is not None:
        age = _direct_age_prediction(
            raw.age_bracket_probabilities,
            factor=factor,
            threshold=settings.age_confidence_threshold,
        )
    else:
        age_label = (
            age_to_bracket(float(raw.age_years))
            if raw.age_years is not None
            else "unknown"
        )
        age_confidence = 0.0
        if age_label != "unknown" and raw.age_years is not None:
            age_confidence = (
                _age_bracket_confidence(
                    float(raw.age_years), age_label, settings.age_residual_sigma_years
                )
                * factor
            )
        if age_label == "unknown" or age_confidence < settings.age_confidence_threshold:
            age = AgeBracketPrediction(prediction="unknown", confidence=0.0)
        else:
            age = AgeBracketPrediction(
                prediction=age_label,
                confidence=round(min(1.0, age_confidence), 4),
            )

    return AnalysisResponse(
        contact_id=contact_id,
        gender=gender,
        age_bracket=age,
        processing_ms=max(0, processing_ms),
        audio_quality=quality.label,
    )
