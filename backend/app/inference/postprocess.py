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
    LanguagePrediction,
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
MIN_ADULT_AGE_YEARS = 18.0
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


def age_sigma_years(
    age_years: float, spread_years: float | None, settings: Settings
) -> float:
    """Total age uncertainty for one sample.

    Three independent terms:

    * the head's population residual (``AGE_RESIDUAL_SIGMA_YEARS``), which must
      be measured on a domain holdout rather than assumed;
    * this sample's disagreement across the sub-window ensemble, which is what
      stops confidence from being a fixed function of the bracket geometry;
    * an extrapolation term outside the range where the upstream regressor has
      training support, so an estimate of 90 is not reported as near-certain
      merely because the top bracket is open-ended.
    """

    sigma = settings.age_residual_sigma_years
    if spread_years is not None and math.isfinite(spread_years) and spread_years > 0.0:
        sigma = math.sqrt(sigma * sigma + spread_years * spread_years)
    distance = max(
        0.0,
        settings.age_reliable_min_years - age_years,
        age_years - settings.age_reliable_max_years,
    )
    if distance > 0.0:
        sigma *= 1.0 + settings.age_extrapolation_sigma_per_year * distance
    return sigma


def age_bracket_confidence(age_years: float, bracket: str, sigma: float) -> float:
    """Probability mass inside the selected bracket, given an adult speaker.

    The top bracket is integrated to ``MAX_PLAUSIBLE_AGE_YEARS`` rather than to
    infinity, and the conditioning mass uses the same bounded support, so ``60+``
    no longer collects every unit of tail mass an unbounded integral would give
    it.
    """

    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    support = _normal_cdf(MAX_PLAUSIBLE_AGE_YEARS, age_years, sigma) - _normal_cdf(
        MIN_ADULT_AGE_YEARS, age_years, sigma
    )
    if support <= 1e-9:
        return 0.0
    for candidate, low, high in AGE_RANGES:
        if candidate == bracket:
            bounded_high = min(high, MAX_PLAUSIBLE_AGE_YEARS)
            probability = _normal_cdf(bounded_high, age_years, sigma) - _normal_cdf(
                low, age_years, sigma
            )
            return min(1.0, max(0.0, probability / support))
    return 0.0


def speaker_evidence_is_weak(raw: RawAttributes, settings: Settings) -> bool:
    """True when the sub-window embeddings disagree about who is speaking.

    A caller-plus-agent or caller-plus-bystander segment produces a low pairwise
    similarity. The API exposes only three quality states, so this is surfaced as
    ``degraded`` -- a stricter abstention threshold -- rather than as a silent
    prediction about whichever speaker dominated the window.
    """

    homogeneity = raw.speaker_homogeneity
    if homogeneity is None or not math.isfinite(homogeneity):
        return False
    return homogeneity < settings.min_speaker_homogeneity


def effective_quality(
    label: AudioQuality, raw: RawAttributes, settings: Settings
) -> AudioQuality:
    if label is AudioQuality.GOOD and speaker_evidence_is_weak(raw, settings):
        return AudioQuality.DEGRADED
    return label


def _thresholds(quality: AudioQuality, settings: Settings) -> tuple[float, float]:
    """Abstention thresholds for (gender, age) at this audio quality.

    Confidence itself is never rescaled by quality. Poor audio raises the bar a
    result must clear instead, which keeps the emitted number interpretable as
    the model's own probability.
    """

    if quality is AudioQuality.DEGRADED:
        return (
            settings.gender_confidence_threshold_degraded,
            settings.age_confidence_threshold_degraded,
        )
    return settings.gender_confidence_threshold, settings.age_confidence_threshold


def _valid_probability(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        return 0.0
    return numeric


def _direct_age_prediction(
    probabilities: dict[str, float] | object,
    *,
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
    confidence = normalized[label]
    if confidence < threshold:
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    return AgeBracketPrediction(
        prediction=label,
        confidence=round(min(1.0, confidence), 4),
    )


def _regression_age_prediction(
    raw: RawAttributes,
    *,
    threshold: float,
    settings: Settings,
) -> AgeBracketPrediction:
    if raw.age_years is None:
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    age_years = float(raw.age_years)
    label = age_to_bracket(age_years)
    if label == "unknown":
        return AgeBracketPrediction(prediction="unknown", confidence=0.0)
    sigma = age_sigma_years(age_years, raw.age_spread_years, settings)
    confidence = age_bracket_confidence(age_years, label, sigma)
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
    language: LanguagePrediction | None = None,
) -> AnalysisResponse:
    if quality.label is AudioQuality.INSUFFICIENT:
        return unknown_response(contact_id, processing_ms, quality.label)

    reported_quality = effective_quality(quality.label, raw, settings)
    gender_threshold, age_threshold = _thresholds(reported_quality, settings)

    female = _valid_probability(raw.gender_probabilities.get("female", 0.0))
    male = _valid_probability(raw.gender_probabilities.get("male", 0.0))
    total = female + male
    if total > 0.0:
        female, male = female / total, male / total
    gender_label = "female" if female >= male else "male"
    gender_confidence = max(female, male)
    if gender_confidence < gender_threshold:
        gender = GenderPrediction(prediction="unknown", confidence=0.0)
    else:
        gender = GenderPrediction(
            prediction=gender_label,
            confidence=round(min(1.0, gender_confidence), 4),
        )

    if raw.age_bracket_probabilities is not None:
        age = _direct_age_prediction(
            raw.age_bracket_probabilities,
            threshold=age_threshold,
        )
    else:
        age = _regression_age_prediction(
            raw,
            threshold=age_threshold,
            settings=settings,
        )

    debug_age_years = None
    if (
        settings.expose_debug_age_years
        and raw.age_years is not None
        and math.isfinite(float(raw.age_years))
    ):
        # Deliberately the unclamped regressor output, including values outside
        # 18-120 that the bracket mapping abstains on: the residual statistics
        # this exists to measure are wrong if the tails are silently discarded.
        debug_age_years = round(float(raw.age_years), 3)

    return AnalysisResponse(
        contact_id=contact_id,
        gender=gender,
        age_bracket=age,
        processing_ms=max(0, processing_ms),
        audio_quality=reported_quality,
        language=language,
        debug_age_years=debug_age_years,
    )
