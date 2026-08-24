from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit


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


def _env_optional(name: str, default: str | None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    return normalized or None


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "voice-attribute-service"
    service_version: str = "0.1.0"
    model_backend: str = "ecapa"
    model_root: Path = Path("/opt/models")
    model_device: str = "cpu"
    torch_threads: int = 2
    warmup_model: bool = True
    wavlm_model_path: Path = Path("/opt/models/wavlm/model.onnx")
    wavlm_model_revision: str = "unconfigured"
    onnx_intra_threads: int = 2

    target_sample_rate: int = 16_000
    max_upload_bytes: int = 12 * 1024 * 1024
    multipart_overhead_bytes: int = 256 * 1024
    max_audio_seconds: float = 30.0
    inference_window_seconds: float = 5.0
    decode_timeout_seconds: float = 8.0

    min_audio_seconds: float = 1.25
    min_voiced_seconds: float = 0.65

    # Level normalization is applied to the inference window only, after the
    # quality gate has judged the untouched signal. Denoising is off by default
    # because suppression helps listeners and ASR but attenuates the low-energy
    # spectral detail paralinguistic models depend on; see docs/AUDIO_PIPELINE.md.
    agc_enabled: bool = True
    agc_target_dbfs: float = -20.0
    agc_max_gain_db: float = 20.0
    agc_min_level_dbfs: float = -45.0
    denoise_backend: str = "none"
    denoise_over_subtraction: float = 1.5
    denoise_floor_db: float = -18.0
    denoise_noise_percentile: float = 10.0

    # Sequenced framing lets a gateway's packet loss and reordering be repaired
    # and, more importantly, counted rather than silently analyzed as speech.
    ws_reorder_window_frames: int = 8
    ws_max_loss_ratio: float = 0.15

    # Emitted confidence is the model's own probability and is never scaled by a
    # quality fudge factor: multiplying a probability by 0.75 leaves a number that
    # is neither calibrated nor a quality signal. Degraded audio instead has to
    # clear a stricter abstention threshold. Both pairs must be replaced from a
    # coverage-versus-error curve on consented domain data before production.
    gender_confidence_threshold: float = 0.60
    gender_confidence_threshold_degraded: float = 0.75
    age_confidence_threshold: float = 0.35
    age_confidence_threshold_degraded: float = 0.45

    # Age uncertainty has three components: the head's population residual, the
    # per-sample disagreement across the encoder's sub-window ensemble, and an
    # extrapolation term for estimates outside the range where the upstream head
    # has meaningful training support.
    age_residual_sigma_years: float = 10.0
    age_reliable_min_years: float = 20.0
    age_reliable_max_years: float = 70.0
    age_extrapolation_sigma_per_year: float = 0.15

    # Sub-window ensembling supplies the per-sample age spread, reduces window
    # selection variance, and detects more than one speaker in the segment.
    ensemble_windows: int = 3
    ensemble_min_window_seconds: float = 2.0
    min_speaker_homogeneity: float = 0.30
    require_calibrated_gender: bool = True

    # Evaluation only: return the regressor's raw age estimate so the harness can
    # measure MAE and the residual spread. Never persisted. See AUDIO_PIPELINE.md.
    expose_debug_age_years: bool = False

    # Language identification needs its own encoder, so it is deployment-scoped
    # and costs a second forward pass when enabled.
    language_backend: str = "none"
    # Posteriors spread across 107 related languages, so acceptance needs a
    # floor and a margin over the runner-up rather than a high floor alone.
    language_confidence_threshold: float = 0.35
    language_margin_ratio: float = 2.0
    # New audio required before a streaming session re-checks the language. Zero
    # re-checks on every progressive update, at one extra encoder pass each.
    language_refresh_seconds: float = 3.0

    inference_concurrency: int = 1
    queue_timeout_seconds: float = 1.0
    ws_emit_interval_seconds: float = 1.0
    ws_max_emit_interval_seconds: float = 4.0
    ws_emit_backoff: float = 1.5
    ws_analysis_window_seconds: float = 10.0
    ws_start_timeout_seconds: float = 10.0
    request_idle_timeout_seconds: float = 10.0
    ws_idle_timeout_seconds: float = 30.0
    ws_max_session_seconds: float = 120.0
    ws_allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    ffmpeg_binary: str = "ffmpeg"
    log_level: str = "INFO"

    # Defence in depth only: the primary rate limit belongs at the ingress, which
    # can drop traffic before it costs a connection. This bounds expensive work
    # per client if that is absent or misconfigured. The limit is per container,
    # so N containers permit N times the rate.
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: float = 60.0
    rate_limit_burst: int = 10
    rate_limit_max_tracked_clients: int = 10_000
    # How many proxies append to X-Forwarded-For. Zero ignores the header, which
    # is right when reached directly or when uvicorn --proxy-headers already
    # rewrote the peer address. Getting this wrong either collapses every caller
    # into one bucket or lets a caller forge its identity.
    trusted_proxy_hops: int = 0
    # Emit HSTS on HTTPS responses. Only meaningful once TLS terminates in front
    # of the service; it is a promise to browsers, not a control this app applies.
    hsts_max_age_seconds: int = 0

    # Retention is explicitly opt-in at request level even when the deployment
    # enables these services. Raw voice bytes live in S3, never PostgreSQL.
    persistence_enabled: bool = False
    database_url: str = (
        "postgresql://voice_attributes:local-only@postgres:5432/voice_attributes"
    )
    s3_endpoint_url: str = "http://object-storage:9000"
    s3_region: str = "us-east-1"
    s3_bucket: str = "voice-audio"
    s3_access_key: str = "voice-local"
    s3_secret_key: str = "voice-local-development-only"
    s3_create_bucket: bool = True
    s3_server_side_encryption: str | None = None
    audio_retention_hours: int = 24
    result_retention_days: int = 30
    storage_segment_seconds: float = 1.0
    storage_operation_timeout_seconds: float = 5.0
    storage_worker_threads: int = 4

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
            wavlm_model_path=Path(
                os.getenv("WAVLM_MODEL_PATH", str(defaults.wavlm_model_path))
            ),
            wavlm_model_revision=os.getenv(
                "WAVLM_MODEL_REVISION", defaults.wavlm_model_revision
            ),
            onnx_intra_threads=int(
                os.getenv("ONNX_INTRA_THREADS", defaults.onnx_intra_threads)
            ),
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
            agc_enabled=_env_bool("AGC_ENABLED", defaults.agc_enabled),
            agc_target_dbfs=float(
                os.getenv("AGC_TARGET_DBFS", defaults.agc_target_dbfs)
            ),
            agc_max_gain_db=float(
                os.getenv("AGC_MAX_GAIN_DB", defaults.agc_max_gain_db)
            ),
            agc_min_level_dbfs=float(
                os.getenv("AGC_MIN_LEVEL_DBFS", defaults.agc_min_level_dbfs)
            ),
            denoise_backend=os.getenv(
                "DENOISE_BACKEND", defaults.denoise_backend
            ).lower(),
            denoise_over_subtraction=float(
                os.getenv("DENOISE_OVER_SUBTRACTION", defaults.denoise_over_subtraction)
            ),
            denoise_floor_db=float(
                os.getenv("DENOISE_FLOOR_DB", defaults.denoise_floor_db)
            ),
            denoise_noise_percentile=float(
                os.getenv("DENOISE_NOISE_PERCENTILE", defaults.denoise_noise_percentile)
            ),
            ws_reorder_window_frames=int(
                os.getenv("WS_REORDER_WINDOW_FRAMES", defaults.ws_reorder_window_frames)
            ),
            ws_max_loss_ratio=float(
                os.getenv("WS_MAX_LOSS_RATIO", defaults.ws_max_loss_ratio)
            ),
            gender_confidence_threshold=float(
                os.getenv(
                    "GENDER_CONFIDENCE_THRESHOLD",
                    defaults.gender_confidence_threshold,
                )
            ),
            gender_confidence_threshold_degraded=float(
                os.getenv(
                    "GENDER_CONFIDENCE_THRESHOLD_DEGRADED",
                    defaults.gender_confidence_threshold_degraded,
                )
            ),
            age_confidence_threshold=float(
                os.getenv("AGE_CONFIDENCE_THRESHOLD", defaults.age_confidence_threshold)
            ),
            age_confidence_threshold_degraded=float(
                os.getenv(
                    "AGE_CONFIDENCE_THRESHOLD_DEGRADED",
                    defaults.age_confidence_threshold_degraded,
                )
            ),
            age_residual_sigma_years=float(
                os.getenv(
                    "AGE_RESIDUAL_SIGMA_YEARS",
                    defaults.age_residual_sigma_years,
                )
            ),
            age_reliable_min_years=float(
                os.getenv("AGE_RELIABLE_MIN_YEARS", defaults.age_reliable_min_years)
            ),
            age_reliable_max_years=float(
                os.getenv("AGE_RELIABLE_MAX_YEARS", defaults.age_reliable_max_years)
            ),
            age_extrapolation_sigma_per_year=float(
                os.getenv(
                    "AGE_EXTRAPOLATION_SIGMA_PER_YEAR",
                    defaults.age_extrapolation_sigma_per_year,
                )
            ),
            ensemble_windows=int(
                os.getenv("ENSEMBLE_WINDOWS", defaults.ensemble_windows)
            ),
            ensemble_min_window_seconds=float(
                os.getenv(
                    "ENSEMBLE_MIN_WINDOW_SECONDS",
                    defaults.ensemble_min_window_seconds,
                )
            ),
            min_speaker_homogeneity=float(
                os.getenv("MIN_SPEAKER_HOMOGENEITY", defaults.min_speaker_homogeneity)
            ),
            require_calibrated_gender=_env_bool(
                "REQUIRE_CALIBRATED_GENDER", defaults.require_calibrated_gender
            ),
            expose_debug_age_years=_env_bool(
                "EXPOSE_DEBUG_AGE_YEARS", defaults.expose_debug_age_years
            ),
            language_backend=os.getenv(
                "LANGUAGE_BACKEND", defaults.language_backend
            ).lower(),
            language_confidence_threshold=float(
                os.getenv(
                    "LANGUAGE_CONFIDENCE_THRESHOLD",
                    defaults.language_confidence_threshold,
                )
            ),
            language_margin_ratio=float(
                os.getenv("LANGUAGE_MARGIN_RATIO", defaults.language_margin_ratio)
            ),
            language_refresh_seconds=float(
                os.getenv("LANGUAGE_REFRESH_SECONDS", defaults.language_refresh_seconds)
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
            ws_max_emit_interval_seconds=float(
                os.getenv(
                    "WS_MAX_EMIT_INTERVAL_SECONDS",
                    defaults.ws_max_emit_interval_seconds,
                )
            ),
            ws_emit_backoff=float(
                os.getenv("WS_EMIT_BACKOFF", defaults.ws_emit_backoff)
            ),
            ws_analysis_window_seconds=float(
                os.getenv(
                    "WS_ANALYSIS_WINDOW_SECONDS",
                    defaults.ws_analysis_window_seconds,
                )
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
            ws_allowed_origins=tuple(
                origin.strip().lower().rstrip("/")
                for origin in os.getenv(
                    "WS_ALLOWED_ORIGINS", ",".join(defaults.ws_allowed_origins)
                ).split(",")
                if origin.strip()
            ),
            ffmpeg_binary=os.getenv("FFMPEG_BINARY", defaults.ffmpeg_binary),
            rate_limit_enabled=_env_bool(
                "RATE_LIMIT_ENABLED", defaults.rate_limit_enabled
            ),
            rate_limit_requests_per_minute=float(
                os.getenv(
                    "RATE_LIMIT_REQUESTS_PER_MINUTE",
                    defaults.rate_limit_requests_per_minute,
                )
            ),
            rate_limit_burst=int(
                os.getenv("RATE_LIMIT_BURST", defaults.rate_limit_burst)
            ),
            rate_limit_max_tracked_clients=int(
                os.getenv(
                    "RATE_LIMIT_MAX_TRACKED_CLIENTS",
                    defaults.rate_limit_max_tracked_clients,
                )
            ),
            trusted_proxy_hops=int(
                os.getenv("TRUSTED_PROXY_HOPS", defaults.trusted_proxy_hops)
            ),
            hsts_max_age_seconds=int(
                os.getenv("HSTS_MAX_AGE_SECONDS", defaults.hsts_max_age_seconds)
            ),
            log_level=os.getenv("LOG_LEVEL", defaults.log_level).upper(),
            persistence_enabled=_env_bool(
                "PERSISTENCE_ENABLED", defaults.persistence_enabled
            ),
            database_url=os.getenv("DATABASE_URL", defaults.database_url),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", defaults.s3_endpoint_url),
            s3_region=os.getenv("S3_REGION", defaults.s3_region),
            s3_bucket=os.getenv("S3_BUCKET", defaults.s3_bucket),
            s3_access_key=os.getenv("S3_ACCESS_KEY", defaults.s3_access_key),
            s3_secret_key=os.getenv("S3_SECRET_KEY", defaults.s3_secret_key),
            s3_create_bucket=_env_bool("S3_CREATE_BUCKET", defaults.s3_create_bucket),
            s3_server_side_encryption=_env_optional(
                "S3_SERVER_SIDE_ENCRYPTION",
                defaults.s3_server_side_encryption,
            ),
            audio_retention_hours=int(
                os.getenv("AUDIO_RETENTION_HOURS", defaults.audio_retention_hours)
            ),
            result_retention_days=int(
                os.getenv("RESULT_RETENTION_DAYS", defaults.result_retention_days)
            ),
            storage_segment_seconds=float(
                os.getenv("STORAGE_SEGMENT_SECONDS", defaults.storage_segment_seconds)
            ),
            storage_operation_timeout_seconds=float(
                os.getenv(
                    "STORAGE_OPERATION_TIMEOUT_SECONDS",
                    defaults.storage_operation_timeout_seconds,
                )
            ),
            storage_worker_threads=int(
                os.getenv("STORAGE_WORKER_THREADS", defaults.storage_worker_threads)
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.model_backend not in {"ecapa", "wavlm_onnx"}:
            raise ValueError("MODEL_BACKEND must be 'ecapa' or 'wavlm_onnx'")
        if self.target_sample_rate != 16_000:
            raise ValueError("TARGET_SAMPLE_RATE must be 16000 for supported models")
        if self.onnx_intra_threads <= 0:
            raise ValueError("ONNX_INTRA_THREADS must be positive")
        if not self.wavlm_model_revision or len(self.wavlm_model_revision) > 80:
            raise ValueError("WAVLM_MODEL_REVISION must contain 1 to 80 characters")
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
            (
                "GENDER_CONFIDENCE_THRESHOLD_DEGRADED",
                self.gender_confidence_threshold_degraded,
            ),
            ("AGE_CONFIDENCE_THRESHOLD", self.age_confidence_threshold),
            (
                "AGE_CONFIDENCE_THRESHOLD_DEGRADED",
                self.age_confidence_threshold_degraded,
            ),
            ("MIN_SPEAKER_HOMOGENEITY", self.min_speaker_homogeneity),
            ("LANGUAGE_CONFIDENCE_THRESHOLD", self.language_confidence_threshold),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1")
        if self.language_backend not in {"none", "voxlingua_ecapa"}:
            raise ValueError("LANGUAGE_BACKEND must be 'none' or 'voxlingua_ecapa'")
        if self.denoise_backend not in {"none", "spectral_gate"}:
            raise ValueError("DENOISE_BACKEND must be 'none' or 'spectral_gate'")
        if not -60.0 <= self.agc_target_dbfs <= 0.0:
            raise ValueError("AGC_TARGET_DBFS must be between -60 and 0")
        if not 0.0 <= self.agc_max_gain_db <= 60.0:
            raise ValueError("AGC_MAX_GAIN_DB must be between 0 and 60")
        if not -90.0 <= self.agc_min_level_dbfs <= 0.0:
            raise ValueError("AGC_MIN_LEVEL_DBFS must be between -90 and 0")
        if not 1.0 <= self.denoise_over_subtraction <= 4.0:
            raise ValueError("DENOISE_OVER_SUBTRACTION must be between 1 and 4")
        if not -60.0 <= self.denoise_floor_db <= 0.0:
            raise ValueError("DENOISE_FLOOR_DB must be between -60 and 0")
        if not 1.0 <= self.denoise_noise_percentile <= 50.0:
            raise ValueError("DENOISE_NOISE_PERCENTILE must be between 1 and 50")
        if not 1 <= self.ws_reorder_window_frames <= 256:
            raise ValueError("WS_REORDER_WINDOW_FRAMES must be between 1 and 256")
        if not 0.0 <= self.ws_max_loss_ratio <= 1.0:
            raise ValueError("WS_MAX_LOSS_RATIO must be between 0 and 1")
        if not math.isfinite(self.language_margin_ratio) or (
            self.language_margin_ratio < 1.0
        ):
            raise ValueError("LANGUAGE_MARGIN_RATIO must be at least 1.0")
        if not math.isfinite(self.language_refresh_seconds) or (
            self.language_refresh_seconds < 0.0
        ):
            raise ValueError("LANGUAGE_REFRESH_SECONDS cannot be negative")
        # Degraded audio must never be easier to pass than good audio. Allowing it
        # would silently reintroduce the quality-scaling bug in reverse.
        if self.gender_confidence_threshold_degraded < self.gender_confidence_threshold:
            raise ValueError(
                "GENDER_CONFIDENCE_THRESHOLD_DEGRADED cannot be below "
                "GENDER_CONFIDENCE_THRESHOLD"
            )
        if self.age_confidence_threshold_degraded < self.age_confidence_threshold:
            raise ValueError(
                "AGE_CONFIDENCE_THRESHOLD_DEGRADED cannot be below "
                "AGE_CONFIDENCE_THRESHOLD"
            )
        if (
            not math.isfinite(self.age_residual_sigma_years)
            or self.age_residual_sigma_years <= 0
        ):
            raise ValueError("AGE_RESIDUAL_SIGMA_YEARS must be positive")
        if not 18.0 <= self.age_reliable_min_years < self.age_reliable_max_years:
            raise ValueError(
                "AGE_RELIABLE_MIN_YEARS must be at least 18 and below "
                "AGE_RELIABLE_MAX_YEARS"
            )
        if not math.isfinite(self.age_reliable_max_years) or (
            self.age_reliable_max_years > 120.0
        ):
            raise ValueError("AGE_RELIABLE_MAX_YEARS must be finite and at most 120")
        if (
            not math.isfinite(self.age_extrapolation_sigma_per_year)
            or self.age_extrapolation_sigma_per_year < 0.0
        ):
            raise ValueError("AGE_EXTRAPOLATION_SIGMA_PER_YEAR cannot be negative")
        if not 1 <= self.ensemble_windows <= 8:
            raise ValueError("ENSEMBLE_WINDOWS must be between 1 and 8")
        if not math.isfinite(self.ensemble_min_window_seconds) or not (
            0.5 <= self.ensemble_min_window_seconds <= self.inference_window_seconds
        ):
            raise ValueError(
                "ENSEMBLE_MIN_WINDOW_SECONDS must be between 0.5 and "
                "INFERENCE_WINDOW_SECONDS"
            )
        if self.inference_concurrency <= 0 or self.torch_threads <= 0:
            raise ValueError("inference and torch thread counts must be positive")
        if self.inference_concurrency > 16:
            raise ValueError(
                "INFERENCE_CONCURRENCY is the model replica count; above 16 the "
                "memory cost is almost certainly unintended"
            )
        if self.ws_emit_backoff < 1.0 or not math.isfinite(self.ws_emit_backoff):
            raise ValueError("WS_EMIT_BACKOFF must be at least 1.0")
        if self.ws_max_emit_interval_seconds < self.ws_emit_interval_seconds:
            raise ValueError(
                "WS_MAX_EMIT_INTERVAL_SECONDS cannot be below WS_EMIT_INTERVAL_SECONDS"
            )
        if not math.isfinite(self.ws_analysis_window_seconds) or not (
            self.inference_window_seconds
            <= self.ws_analysis_window_seconds
            <= self.max_audio_seconds
        ):
            raise ValueError(
                "WS_ANALYSIS_WINDOW_SECONDS must be between INFERENCE_WINDOW_SECONDS "
                "and MAX_AUDIO_SECONDS"
            )
        for field_name, value in (
            ("DECODE_TIMEOUT_SECONDS", self.decode_timeout_seconds),
            ("QUEUE_TIMEOUT_SECONDS", self.queue_timeout_seconds),
            ("WS_EMIT_INTERVAL_SECONDS", self.ws_emit_interval_seconds),
            ("WS_MAX_EMIT_INTERVAL_SECONDS", self.ws_max_emit_interval_seconds),
            ("WS_START_TIMEOUT_SECONDS", self.ws_start_timeout_seconds),
            ("REQUEST_IDLE_TIMEOUT_SECONDS", self.request_idle_timeout_seconds),
            ("WS_IDLE_TIMEOUT_SECONDS", self.ws_idle_timeout_seconds),
            ("WS_MAX_SESSION_SECONDS", self.ws_max_session_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        for origin in self.ws_allowed_origins:
            parsed = urlsplit(origin)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "WS_ALLOWED_ORIGINS must contain comma-separated HTTP(S) origins"
                )
        if self.rate_limit_requests_per_minute <= 0.0 or not math.isfinite(
            self.rate_limit_requests_per_minute
        ):
            raise ValueError("RATE_LIMIT_REQUESTS_PER_MINUTE must be positive")
        if not 1 <= self.rate_limit_burst <= 10_000:
            raise ValueError("RATE_LIMIT_BURST must be between 1 and 10000")
        if not 1 <= self.rate_limit_max_tracked_clients <= 1_000_000:
            raise ValueError(
                "RATE_LIMIT_MAX_TRACKED_CLIENTS must be between 1 and 1000000"
            )
        if not 0 <= self.trusted_proxy_hops <= 8:
            raise ValueError("TRUSTED_PROXY_HOPS must be between 0 and 8")
        if not 0 <= self.hsts_max_age_seconds <= 63_072_000:
            raise ValueError("HSTS_MAX_AGE_SECONDS must be between 0 and 63072000")
        if self.audio_retention_hours <= 0 or self.result_retention_days <= 0:
            raise ValueError("retention periods must be positive")
        if not 1 <= self.storage_worker_threads <= 64:
            raise ValueError("STORAGE_WORKER_THREADS must be between 1 and 64")
        for field_name, value in (
            ("STORAGE_SEGMENT_SECONDS", self.storage_segment_seconds),
            (
                "STORAGE_OPERATION_TIMEOUT_SECONDS",
                self.storage_operation_timeout_seconds,
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive")
        if self.persistence_enabled:
            database = urlsplit(self.database_url)
            if database.scheme not in {"postgres", "postgresql"} or not database.netloc:
                raise ValueError("DATABASE_URL must be a PostgreSQL URL")
            endpoint = urlsplit(self.s3_endpoint_url)
            if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
                raise ValueError("S3_ENDPOINT_URL must be an HTTP(S) URL")
            for field_name, value in (
                ("S3_REGION", self.s3_region),
                ("S3_BUCKET", self.s3_bucket),
                ("S3_ACCESS_KEY", self.s3_access_key),
                ("S3_SECRET_KEY", self.s3_secret_key),
            ):
                if not value:
                    raise ValueError(f"{field_name} cannot be empty")
