from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AudioQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class PersistenceMode(StrEnum):
    NONE = "none"
    RESULT = "result"
    RESULT_AND_AUDIO = "result_and_audio"


class PersistenceStatus(StrEnum):
    PENDING = "pending"
    STORED = "stored"
    PARTIAL = "partial"
    DELETION_PENDING = "deletion_pending"
    DELETED = "deleted"
    FAILED = "failed"


class PersistenceReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: PersistenceMode
    status: PersistenceStatus
    chunks_received: int = Field(default=0, ge=0)
    chunks_stored: int = Field(default=0, ge=0)
    segments_stored: int = Field(default=0, ge=0)
    bytes_stored: int = Field(default=0, ge=0)
    audio_expires_at: datetime | None = None
    result_expires_at: datetime | None = None


class GenderPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["male", "female", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class AgeBracketPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: Literal["18-30", "31-45", "46-60", "60+", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class LanguagePrediction(BaseModel):
    """Best-effort spoken-language identity.

    ``prediction`` is an ISO-639 style language tag or `unknown`. It names a
    language only: it is not a locale, accent, dialect, region, or nationality,
    and it says nothing about where a caller is from.
    """

    model_config = ConfigDict(extra="forbid")

    prediction: str = Field(pattern=r"^(unknown|[a-z]{2,8})$")
    confidence: float = Field(ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_id: UUID
    gender: GenderPrediction
    age_bracket: AgeBracketPrediction
    processing_ms: int = Field(ge=0)
    audio_quality: AudioQuality
    # Present only when the deployment enables language identification.
    language: LanguagePrediction | None = None
    # Evaluation-only. The regressor's raw age estimate in years, exposed so the
    # Common Voice harness can measure MAE and the residual spread that
    # AGE_RESIDUAL_SIGMA_YEARS is supposed to encode -- neither of which is
    # computable from bracket labels alone. Off by default, never persisted, and
    # not part of the API contract. It is a finer-grained personal inference than
    # the bracket the contract promises, so it stays off outside evaluation.
    debug_age_years: float | None = None
    analysis_id: UUID | None = None
    persistence: PersistenceReceipt | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class WebSocketStart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["start"]
    contact_id: UUID | None = None
    encoding: Literal["pcm_s16le", "pcm_s16be", "pcm_f32le", "mulaw", "alaw"]
    sample_rate: int = Field(ge=8_000, le=96_000)
    channels: int = Field(default=1, ge=1, le=2)
    # `seq32` prefixes every audio frame with a 4-byte big-endian sequence
    # number so loss and reordering introduced upstream can be repaired and,
    # more importantly, counted. `raw` keeps the original unlabelled framing.
    framing: Literal["raw", "seq32"] = "raw"
    persistence_mode: PersistenceMode = PersistenceMode.NONE
    consent_reference: str | None = Field(default=None, min_length=1, max_length=256)
