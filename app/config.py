from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "voice-attribute-service"
    service_version: str = "0.1.0"
    model_backend: str = "ecapa"
    model_root: Path = Path("/opt/models")
    model_device: str = "cpu"
    torch_threads: int = 2
    warmup_model: bool = True

    target_sample_rate: int = 16_000
    max_upload_bytes: int = 12 * 1024 * 1024
    multipart_overhead_bytes: int = 256 * 1024
    max_audio_seconds: float = 30.0
    inference_window_seconds: float = 5.0
    decode_timeout_seconds: float = 8.0

    min_audio_seconds: float = 1.25
    min_voiced_seconds: float = 0.65
    gender_confidence_threshold: float = 0.60
    age_confidence_threshold: float = 0.28
    degraded_confidence_factor: float = 0.75
    age_residual_sigma_years: float = 10.0

    inference_concurrency: int = 1
    queue_timeout_seconds: float = 1.0
    ws_emit_interval_seconds: float = 1.0
    ws_start_timeout_seconds: float = 10.0
    request_idle_timeout_seconds: float = 10.0
    ws_idle_timeout_seconds: float = 30.0
    ws_max_session_seconds: float = 120.0
    ffmpeg_binary: str = "ffmpeg"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        defaults = cls()
        settings = cls(
            service_name=os.getenv("SERVICE_NAME", defaults.service_name),
            service_version=os.getenv("SERVICE_VERSION", defaults.service_version),
            model_backend=os.getenv("MODEL_BACKEND", defaults.model_backend).lower(),
            model_root=Path(os.getenv("MODEL_ROOT", str(defaults.model_root))),
            model_device=os.getenv("MODEL_DEVICE", defaults.model_device),
            torch_threads=int(os.getenv("TORCH_THREADS", defaults.torch_threads)),
            warmup_model=_env_bool("WARMUP_MODEL", defaults.warmup_model),
            target_sample_rate=int(
                os.getenv("TARGET_SAMPLE_RATE", defaults.target_sample_rate)
            ),
            max_upload_bytes=int(
                os.getenv("MAX_UPLOAD_BYTES", defaults.max_upload_bytes)
            ),
            multipart_overhead_bytes=int(
                os.getenv("MULTIPART_OVERHEAD_BYTES", defaults.multipart_overhead_bytes)
            ),
            max_audio_seconds=float(
                os.getenv("MAX_AUDIO_SECONDS", defaults.max_audio_seconds)
            ),
            inference_window_seconds=float(
                os.getenv("INFERENCE_WINDOW_SECONDS", defaults.inference_window_seconds)
            ),
            decode_timeout_seconds=float(
                os.getenv("DECODE_TIMEOUT_SECONDS", defaults.decode_timeout_seconds)
            ),
            min_audio_seconds=float(
                os.getenv("MIN_AUDIO_SECONDS", defaults.min_audio_seconds)
            ),
            min_voiced_seconds=float(
                os.getenv("MIN_VOICED_SECONDS", defaults.min_voiced_seconds)
            ),
            gender_confidence_threshold=float(
                os.getenv(
                    "GENDER_CONFIDENCE_THRESHOLD",
                    defaults.gender_confidence_threshold,
                )
            ),
            age_confidence_threshold=float(
                os.getenv("AGE_CONFIDENCE_THRESHOLD", defaults.age_confidence_threshold)
            ),
            degraded_confidence_factor=float(
                os.getenv(
                    "DEGRADED_CONFIDENCE_FACTOR",
                    defaults.degraded_confidence_factor,
                )
            ),
            age_residual_sigma_years=float(
                os.getenv(
                    "AGE_RESIDUAL_SIGMA_YEARS",
                    defaults.age_residual_sigma_years,
                )
            ),
            inference_concurrency=int(
                os.getenv("INFERENCE_CONCURRENCY", defaults.inference_concurrency)
            ),
            queue_timeout_seconds=float(
                os.getenv("QUEUE_TIMEOUT_SECONDS", defaults.queue_timeout_seconds)
            ),
            ws_emit_interval_seconds=float(
                os.getenv("WS_EMIT_INTERVAL_SECONDS", defaults.ws_emit_interval_seconds)
            ),
            ws_start_timeout_seconds=float(
                os.getenv("WS_START_TIMEOUT_SECONDS", defaults.ws_start_timeout_seconds)
            ),
            request_idle_timeout_seconds=float(
                os.getenv(
                    "REQUEST_IDLE_TIMEOUT_SECONDS",
                    defaults.request_idle_timeout_seconds,
                )
            ),
            ws_idle_timeout_seconds=float(
                os.getenv("WS_IDLE_TIMEOUT_SECONDS", defaults.ws_idle_timeout_seconds)
            ),
            ws_max_session_seconds=float(
                os.getenv("WS_MAX_SESSION_SECONDS", defaults.ws_max_session_seconds)
            ),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", defaults.ffmpeg_binary),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.model_backend != "ecapa":
            raise ValueError("MODEL_BACKEND must be 'ecapa'")
        if self.target_sample_rate != 16_000:
            raise ValueError("TARGET_SAMPLE_RATE must be 16000 for the ECAPA model")
        if self.max_upload_bytes <= 0 or self.multipart_overhead_bytes < 0:
            raise ValueError("upload byte limits must be positive")
        if not math.isfinite(self.max_audio_seconds) or not (
            1.0 <= self.max_audio_seconds <= 300.0
        ):
            raise ValueError("MAX_AUDIO_SECONDS must be between 1 and 300")
        if not math.isfinite(self.inference_window_seconds) or not (
            1.0 <= self.inference_window_seconds <= self.max_audio_seconds
        ):
            raise ValueError(
                "INFERENCE_WINDOW_SECONDS must be between 1 and MAX_AUDIO_SECONDS"
            )
        if not math.isfinite(self.min_audio_seconds) or not (
            0.1 <= self.min_audio_seconds <= self.max_audio_seconds
        ):
            raise ValueError("MIN_AUDIO_SECONDS is outside the accepted range")
        if not math.isfinite(self.min_voiced_seconds) or not (
            0.1 <= self.min_voiced_seconds <= self.max_audio_seconds
        ):
            raise ValueError("MIN_VOICED_SECONDS is outside the accepted range")
        for field_name, value in (
            ("GENDER_CONFIDENCE_THRESHOLD", self.gender_confidence_threshold),
            ("AGE_CONFIDENCE_THRESHOLD", self.age_confidence_threshold),
            ("DEGRADED_CONFIDENCE_FACTOR", self.degraded_confidence_factor),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if (
            not math.isfinite(self.age_residual_sigma_years)
            or self.age_residual_sigma_years <= 0
        ):
            raise ValueError("AGE_RESIDUAL_SIGMA_YEARS must be positive")
        if self.inference_concurrency <= 0 or self.torch_threads <= 0:
            raise ValueError("inference and torch thread counts must be positive")
        for field_name, value in (
            ("DECODE_TIMEOUT_SECONDS", self.decode_timeout_seconds),
            ("QUEUE_TIMEOUT_SECONDS", self.queue_timeout_seconds),
            ("WS_EMIT_INTERVAL_SECONDS", self.ws_emit_interval_seconds),
            ("WS_START_TIMEOUT_SECONDS", self.ws_start_timeout_seconds),
            ("REQUEST_IDLE_TIMEOUT_SECONDS", self.request_idle_timeout_seconds),
            ("WS_IDLE_TIMEOUT_SECONDS", self.ws_idle_timeout_seconds),
            ("WS_MAX_SESSION_SECONDS", self.ws_max_session_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
