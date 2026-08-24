from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class PersistenceMode(StrEnum):
    """The data a caller explicitly permits the service to retain."""

    NONE = "none"
    RESULT = "result"
    RESULT_AND_AUDIO = "result_and_audio"

    @property
    def rank(self) -> int:
        return {
            PersistenceMode.NONE: 0,
            PersistenceMode.RESULT: 1,
            PersistenceMode.RESULT_AND_AUDIO: 2,
        }[self]

    @classmethod
    def parse(cls, value: PersistenceMode | str) -> PersistenceMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(value.strip().lower())
        except (AttributeError, ValueError) as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"persistence mode must be one of: {choices}") from exc


class SessionTransport(StrEnum):
    REST = "rest"
    WEBSOCKET = "websocket"


class SessionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PersistenceSettings:
    """Persistence-specific configuration, disabled unless explicitly enabled.

    ``mode`` is the maximum mode the deployment permits. Each request must still
    pass its desired mode to ``start_rest`` or ``start_ws``; omission means
    ``none`` even when the deployment supports persistence.
    """

    mode: PersistenceMode = PersistenceMode.NONE
    database_url: str | None = None
    retention_hours: float = 24.0
    result_retention_hours: float | None = None
    cleanup_interval_seconds: float = 300.0
    cleanup_batch_size: int = 100
    ws_segment_seconds: float = 1.0
    db_pool_min_size: int = 1
    db_pool_max_size: int = 10
    db_command_timeout_seconds: float = 10.0
    s3_bucket: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_session_token: str | None = None
    s3_create_bucket: bool = False
    s3_server_side_encryption: str | None = "AES256"
    s3_operation_timeout_seconds: float = 5.0
    s3_max_workers: int = 4
    s3_orphan_lifecycle_days: int | None = None
    object_key_prefix: str = "voice-attributes"

    @classmethod
    def from_env(cls) -> PersistenceSettings:
        defaults = cls()
        return cls(
            mode=PersistenceMode.parse(
                os.getenv("PERSISTENCE_MODE", defaults.mode.value)
            ),
            database_url=os.getenv("PERSISTENCE_DATABASE_URL"),
            retention_hours=float(
                os.getenv("PERSISTENCE_RETENTION_HOURS", defaults.retention_hours)
            ),
            result_retention_hours=_env_optional_float(
                "PERSISTENCE_RESULT_RETENTION_HOURS",
                defaults.result_retention_hours,
            ),
            cleanup_interval_seconds=float(
                os.getenv(
                    "PERSISTENCE_CLEANUP_INTERVAL_SECONDS",
                    defaults.cleanup_interval_seconds,
                )
            ),
            cleanup_batch_size=int(
                os.getenv("PERSISTENCE_CLEANUP_BATCH_SIZE", defaults.cleanup_batch_size)
            ),
            ws_segment_seconds=float(
                os.getenv("PERSISTENCE_WS_SEGMENT_SECONDS", defaults.ws_segment_seconds)
            ),
            db_pool_min_size=int(
                os.getenv("PERSISTENCE_DB_POOL_MIN_SIZE", defaults.db_pool_min_size)
            ),
            db_pool_max_size=int(
                os.getenv("PERSISTENCE_DB_POOL_MAX_SIZE", defaults.db_pool_max_size)
            ),
            db_command_timeout_seconds=float(
                os.getenv(
                    "PERSISTENCE_DB_COMMAND_TIMEOUT_SECONDS",
                    defaults.db_command_timeout_seconds,
                )
            ),
            s3_bucket=os.getenv("PERSISTENCE_S3_BUCKET"),
            s3_endpoint_url=os.getenv("PERSISTENCE_S3_ENDPOINT_URL"),
            s3_region=os.getenv("PERSISTENCE_S3_REGION", defaults.s3_region),
            s3_access_key_id=os.getenv("PERSISTENCE_S3_ACCESS_KEY_ID"),
            s3_secret_access_key=os.getenv("PERSISTENCE_S3_SECRET_ACCESS_KEY"),
            s3_session_token=os.getenv("PERSISTENCE_S3_SESSION_TOKEN"),
            s3_create_bucket=_env_bool(
                "PERSISTENCE_S3_CREATE_BUCKET", defaults.s3_create_bucket
            ),
            s3_server_side_encryption=_env_optional(
                "PERSISTENCE_S3_SERVER_SIDE_ENCRYPTION",
                defaults.s3_server_side_encryption,
            ),
            s3_operation_timeout_seconds=float(
                os.getenv(
                    "PERSISTENCE_S3_OPERATION_TIMEOUT_SECONDS",
                    defaults.s3_operation_timeout_seconds,
                )
            ),
            s3_max_workers=int(
                os.getenv("PERSISTENCE_S3_MAX_WORKERS", defaults.s3_max_workers)
            ),
            s3_orphan_lifecycle_days=_env_optional_int(
                "PERSISTENCE_S3_ORPHAN_LIFECYCLE_DAYS",
                defaults.s3_orphan_lifecycle_days,
            ),
            object_key_prefix=os.getenv(
                "PERSISTENCE_OBJECT_KEY_PREFIX", defaults.object_key_prefix
            ),
        )

    def validate(
        self,
        *,
        require_database_url: bool = True,
        require_s3_bucket: bool = True,
    ) -> None:
        mode = PersistenceMode.parse(self.mode)
        if (
            mode is not PersistenceMode.NONE
            and require_database_url
            and not self.database_url
        ):
            raise ValueError(
                "PERSISTENCE_DATABASE_URL is required when persistence is enabled"
            )
        if (
            mode is PersistenceMode.RESULT_AND_AUDIO
            and require_s3_bucket
            and not self.s3_bucket
        ):
            raise ValueError(
                "PERSISTENCE_S3_BUCKET is required for result_and_audio mode"
            )
        if not 0.016 <= self.retention_hours <= 24 * 365:
            raise ValueError("retention_hours must be between one minute and one year")
        if self.result_retention_hours is not None and not (
            0.016 <= self.result_retention_hours <= 24 * 365
        ):
            raise ValueError(
                "result_retention_hours must be between one minute and one year"
            )
        if not 1.0 <= self.cleanup_interval_seconds <= 86_400:
            raise ValueError("cleanup_interval_seconds must be between 1 and 86400")
        if not 1 <= self.cleanup_batch_size <= 10_000:
            raise ValueError("cleanup_batch_size must be between 1 and 10000")
        if not 0.1 <= self.ws_segment_seconds <= 10.0:
            raise ValueError("ws_segment_seconds must be between 0.1 and 10")
        if not 1 <= self.db_pool_min_size <= self.db_pool_max_size:
            raise ValueError("database pool sizes are invalid")
        if self.db_pool_max_size > 100:
            raise ValueError("db_pool_max_size must not exceed 100")
        if not 0.1 <= self.db_command_timeout_seconds <= 300:
            raise ValueError("db_command_timeout_seconds must be between 0.1 and 300")
        if not 0.1 <= self.s3_operation_timeout_seconds <= 300:
            raise ValueError("s3_operation_timeout_seconds must be between 0.1 and 300")
        if not 1 <= self.s3_max_workers <= 64:
            raise ValueError("s3_max_workers must be between 1 and 64")
        if self.s3_orphan_lifecycle_days is not None and not (
            1 <= self.s3_orphan_lifecycle_days <= 3_650
        ):
            raise ValueError("s3_orphan_lifecycle_days must be between 1 and 3650")
        prefix = self.object_key_prefix.strip("/")
        if not prefix or any(part in {".", ".."} for part in prefix.split("/")):
            raise ValueError("object_key_prefix is invalid")


@dataclass(frozen=True, slots=True)
class PersistenceHandle:
    session_id: UUID | None
    contact_id: UUID
    mode: PersistenceMode
    transport: SessionTransport
    created_at: datetime
    expires_at: datetime | None
    content_type: str | None = None
    encoding: str | None = None
    sample_rate: int | None = None
    channels: int | None = None

    @property
    def enabled(self) -> bool:
        return self.mode is not PersistenceMode.NONE and self.session_id is not None


@dataclass(frozen=True, slots=True)
class LogicalChunkSlice:
    chunk_index: int
    source_byte_start: int
    source_byte_end: int
    segment_byte_start: int
    segment_byte_end: int

    def as_dict(self) -> dict[str, int]:
        return {
            "chunk_index": self.chunk_index,
            "source_byte_start": self.source_byte_start,
            "source_byte_end": self.source_byte_end,
            "segment_byte_start": self.segment_byte_start,
            "segment_byte_end": self.segment_byte_end,
        }


@dataclass(frozen=True, slots=True)
class StoredSegment:
    segment_id: UUID
    session_id: UUID
    sequence: int
    bucket: str
    object_key: str
    byte_start: int
    byte_end: int
    byte_size: int
    sha256: str
    content_type: str
    logical_chunks: tuple[LogicalChunkSlice, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class SessionManifest:
    session_id: UUID
    contact_id: UUID
    mode: PersistenceMode
    transport: SessionTransport
    status: SessionStatus
    request_id: str | None
    content_type: str | None
    encoding: str | None
    sample_rate: int | None
    channels: int | None
    model_name: str | None
    result: Mapping[str, Any] | None
    error_code: str | None
    error_message: str | None
    segment_count: int
    audio_bytes: int
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    expires_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)
    segments: tuple[StoredSegment, ...] = ()


def utc_now() -> datetime:
    return datetime.now(UTC)


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


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    return float(normalized) if normalized else None


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    return int(normalized) if normalized else None
