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


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_id: UUID
    gender: GenderPrediction
    age_bracket: AgeBracketPrediction
    processing_ms: int = Field(ge=0)
    audio_quality: AudioQuality
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
    persistence_mode: PersistenceMode = PersistenceMode.NONE
    consent_reference: str | None = Field(default=None, min_length=1, max_length=256)
