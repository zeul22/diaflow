from __future__ import annotations

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class AudioQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


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
