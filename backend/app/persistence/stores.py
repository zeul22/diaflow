from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from importlib.resources import files
from typing import Any, Protocol
from uuid import UUID

from app.persistence.models import (
    LogicalChunkSlice,
    PersistenceHandle,
    PersistenceMode,
    PersistenceSettings,
    SessionManifest,
    SessionStatus,
    SessionTransport,
    StoredSegment,
)


class MetadataStore(Protocol):
    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def create_session(
        self,
        handle: PersistenceHandle,
        *,
        request_id: str | None,
        metadata: Mapping[str, Any],
    ) -> None: ...

    async def add_segment(self, segment: StoredSegment) -> None: ...

    async def complete_session(
        self,
        session_id: UUID,
        *,
        result: Mapping[str, Any],
        model_name: str | None,
        completed_at: datetime,
    ) -> bool: ...

    async def fail_session(
        self,
        session_id: UUID,
        *,
        error_code: str,
        error_message: str,
        completed_at: datetime,
    ) -> bool: ...

    async def get_session(self, session_id: UUID) -> SessionManifest | None: ...

    async def list_sessions(
        self,
        *,
        contact_id: UUID | None,
        limit: int,
        before: datetime | None,
    ) -> Sequence[SessionManifest]: ...

    async def list_expired(self, *, before: datetime, limit: int) -> Sequence[UUID]: ...

    async def delete_session(self, session_id: UUID) -> bool: ...


class ObjectStore(Protocol):
    @property
    def bucket(self) -> str: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def put_bytes(
        self,
        *,
        object_key: str,
        payload: bytes,
        content_type: str,
        metadata: Mapping[str, str],
    ) -> None: ...

    async def delete_objects(self, object_keys: Sequence[str]) -> None: ...


class AsyncpgMetadataStore:
    """PostgreSQL metadata implementation with a lazily imported driver."""

    _migration_lock_id = 8_604_240_517

    def __init__(self, settings: PersistenceSettings) -> None:
        if not settings.database_url:
            raise ValueError("database_url is required")
        self._settings = settings
        self._pool: Any | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        try:
            import asyncpg
        except ImportError as exc:
            raise RuntimeError(
                "asyncpg is required when PostgreSQL persistence is enabled"
            ) from exc
        self._pool = await asyncpg.create_pool(
            dsn=self._settings.database_url,
            min_size=self._settings.db_pool_min_size,
            max_size=self._settings.db_pool_max_size,
            command_timeout=self._settings.db_command_timeout_seconds,
        )
        try:
            await self._initialize_schema()
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    async def _initialize_schema(self) -> None:
        pool = self._require_pool()
        migration = (
            files("app.persistence")
            .joinpath("migrations", "001_initial.sql")
            .read_text(encoding="utf-8")
        )
        async with pool.acquire() as connection:
            await connection.execute(
                "SELECT pg_advisory_lock($1)", self._migration_lock_id
            )
            try:
                await connection.execute(migration)
            finally:
                await connection.execute(
                    "SELECT pg_advisory_unlock($1)", self._migration_lock_id
                )

    async def create_session(
        self,
        handle: PersistenceHandle,
        *,
        request_id: str | None,
        metadata: Mapping[str, Any],
    ) -> None:
        if handle.session_id is None or handle.expires_at is None:
            raise ValueError("an enabled persistence handle is required")
        await self._require_pool().execute(
            """
            INSERT INTO persistence_sessions (
                session_id, contact_id, mode, transport, request_id, content_type,
                encoding, sample_rate, channels, metadata, created_at, updated_at,
                expires_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $11, $12
            )
            """,
            handle.session_id,
            handle.contact_id,
            handle.mode.value,
            handle.transport.value,
            request_id,
            handle.content_type,
            handle.encoding,
            handle.sample_rate,
            handle.channels,
            _json_dump(metadata),
            handle.created_at,
            handle.expires_at,
        )

    async def add_segment(self, segment: StoredSegment) -> None:
        pool = self._require_pool()
        async with pool.acquire() as connection, connection.transaction():
            await connection.execute(
                """
                INSERT INTO persistence_audio_segments (
                    segment_id, session_id, sequence, bucket, object_key, byte_start,
                    byte_end, byte_size, sha256, content_type, logical_chunks,
                    created_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12
                )
                """,
                segment.segment_id,
                segment.session_id,
                segment.sequence,
                segment.bucket,
                segment.object_key,
                segment.byte_start,
                segment.byte_end,
                segment.byte_size,
                segment.sha256,
                segment.content_type,
                _json_dump([item.as_dict() for item in segment.logical_chunks]),
                segment.created_at,
            )
            updated = await connection.execute(
                """
                UPDATE persistence_sessions
                SET segment_count = segment_count + 1,
                    audio_bytes = audio_bytes + $2,
                    updated_at = $3
                WHERE session_id = $1 AND status = 'pending'
                """,
                segment.session_id,
                segment.byte_size,
                segment.created_at,
            )
            if updated != "UPDATE 1":
                raise RuntimeError(
                    "persistence session is missing or no longer pending"
                )

    async def complete_session(
        self,
        session_id: UUID,
        *,
        result: Mapping[str, Any],
        model_name: str | None,
        completed_at: datetime,
    ) -> bool:
        updated = await self._require_pool().execute(
            """
            UPDATE persistence_sessions
            SET status = 'completed', result = $2::jsonb, model_name = $3,
                completed_at = $4, updated_at = $4
            WHERE session_id = $1 AND status = 'pending'
            """,
            session_id,
            _json_dump(result),
            model_name,
            completed_at,
        )
        return updated == "UPDATE 1"

    async def fail_session(
        self,
        session_id: UUID,
        *,
        error_code: str,
        error_message: str,
        completed_at: datetime,
    ) -> bool:
        updated = await self._require_pool().execute(
            """
            UPDATE persistence_sessions
            SET status = 'failed', error_code = $2, error_message = $3,
                completed_at = $4, updated_at = $4
            WHERE session_id = $1 AND status = 'pending'
            """,
            session_id,
            error_code,
            error_message,
            completed_at,
        )
        return updated == "UPDATE 1"

    async def get_session(self, session_id: UUID) -> SessionManifest | None:
        pool = self._require_pool()
        row = await pool.fetchrow(
            "SELECT * FROM persistence_sessions WHERE session_id = $1", session_id
        )
        if row is None:
            return None
        segment_rows = await pool.fetch(
            """
            SELECT * FROM persistence_audio_segments
            WHERE session_id = $1 ORDER BY sequence
            """,
            session_id,
        )
        return _manifest_from_row(
            row, [_segment_from_row(item) for item in segment_rows]
        )

    async def list_sessions(
        self,
        *,
        contact_id: UUID | None,
        limit: int,
        before: datetime | None,
    ) -> Sequence[SessionManifest]:
        pool = self._require_pool()
        rows = await pool.fetch(
            """
            SELECT * FROM persistence_sessions
            WHERE ($1::uuid IS NULL OR contact_id = $1)
              AND ($2::timestamptz IS NULL OR created_at < $2)
            ORDER BY created_at DESC, session_id DESC
            LIMIT $3
            """,
            contact_id,
            before,
            limit,
        )
        if not rows:
            return ()
        # Collection views use the aggregate counters on the session row. Exact
        # segment offsets are intentionally fetched only by get_session(), which
        # avoids an unbounded metadata fan-out on history pages.
        return tuple(_manifest_from_row(row, ()) for row in rows)

    async def list_expired(self, *, before: datetime, limit: int) -> Sequence[UUID]:
        rows = await self._require_pool().fetch(
            """
            SELECT session_id FROM persistence_sessions
            WHERE expires_at <= $1 ORDER BY expires_at, session_id LIMIT $2
            """,
            before,
            limit,
        )
        return tuple(row["session_id"] for row in rows)

    async def delete_session(self, session_id: UUID) -> bool:
        deleted = await self._require_pool().execute(
            "DELETE FROM persistence_sessions WHERE session_id = $1", session_id
        )
        return deleted == "DELETE 1"

    def _require_pool(self) -> Any:
        if self._pool is None:
            raise RuntimeError("PostgreSQL metadata store is not connected")
        return self._pool


class Boto3ObjectStore:
    """S3-compatible object storage with blocking SDK calls off the event loop."""

    def __init__(self, settings: PersistenceSettings) -> None:
        if not settings.s3_bucket:
            raise ValueError("s3_bucket is required")
        self._settings = settings
        self._client: Any | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=settings.s3_max_workers,
            thread_name_prefix="voice-s3",
        )
        self._executor_closed = False

    @property
    def bucket(self) -> str:
        if self._settings.s3_bucket is None:
            raise RuntimeError("S3 bucket is not configured")
        return self._settings.s3_bucket

    async def connect(self) -> None:
        if self._client is not None:
            return
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required when audio persistence is enabled"
            ) from exc
        client_kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self._settings.s3_region,
            "config": Config(
                connect_timeout=self._settings.s3_operation_timeout_seconds,
                read_timeout=self._settings.s3_operation_timeout_seconds,
                max_pool_connections=self._settings.s3_max_workers,
                retries={"mode": "standard", "total_max_attempts": 2},
            ),
        }
        optional = {
            "endpoint_url": self._settings.s3_endpoint_url,
            "aws_access_key_id": self._settings.s3_access_key_id,
            "aws_secret_access_key": self._settings.s3_secret_access_key,
            "aws_session_token": self._settings.s3_session_token,
        }
        client_kwargs.update({key: value for key, value in optional.items() if value})
        self._client = boto3.client(**client_kwargs)
        try:
            await self._run(self._client.head_bucket, Bucket=self.bucket)
        except Exception:
            if not self._settings.s3_create_bucket:
                self._client = None
                raise
            await self._create_bucket()
        if self._settings.s3_orphan_lifecycle_days is not None:
            await self._configure_orphan_lifecycle()

    async def _create_bucket(self) -> None:
        client = self._require_client()
        kwargs: dict[str, Any] = {"Bucket": self.bucket}
        if (
            not self._settings.s3_endpoint_url
            and self._settings.s3_region != "us-east-1"
        ):
            kwargs["CreateBucketConfiguration"] = {
                "LocationConstraint": self._settings.s3_region
            }
        await self._run(client.create_bucket, **kwargs)

    async def _configure_orphan_lifecycle(self) -> None:
        prefix = self._settings.object_key_prefix.strip("/") + "/"
        await self._run(
            self._require_client().put_bucket_lifecycle_configuration,
            Bucket=self.bucket,
            LifecycleConfiguration={
                "Rules": [
                    {
                        "ID": "voice-attributes-orphan-safety",
                        "Status": "Enabled",
                        "Filter": {"Prefix": prefix},
                        "Expiration": {"Days": self._settings.s3_orphan_lifecycle_days},
                    }
                ]
            },
        )

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None and callable(getattr(client, "close", None)):
            await self._run(client.close)
        if not self._executor_closed:
            self._executor_closed = True
            executor, self._executor = self._executor, None
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=False,
            )

    async def put_bytes(
        self,
        *,
        object_key: str,
        payload: bytes,
        content_type: str,
        metadata: Mapping[str, str],
    ) -> None:
        kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": object_key,
            "Body": payload,
            "ContentType": content_type,
            "ContentLength": len(payload),
            "Metadata": dict(metadata),
        }
        if self._settings.s3_server_side_encryption:
            kwargs["ServerSideEncryption"] = self._settings.s3_server_side_encryption
        await self._run(self._require_client().put_object, **kwargs)

    async def delete_objects(self, object_keys: Sequence[str]) -> None:
        client = self._require_client()
        for offset in range(0, len(object_keys), 1_000):
            batch = object_keys[offset : offset + 1_000]
            if not batch:
                continue
            response = await self._run(
                client.delete_objects,
                Bucket=self.bucket,
                Delete={
                    "Objects": [{"Key": object_key} for object_key in batch],
                    "Quiet": True,
                },
            )
            errors = response.get("Errors", [])
            if errors:
                failed = ", ".join(item.get("Key", "unknown") for item in errors)
                raise RuntimeError(f"S3 failed to delete objects: {failed}")

    async def _run(self, function: Any, /, **kwargs: Any) -> Any:
        executor = self._executor
        if executor is None or self._executor_closed:
            raise RuntimeError("S3 object-store executor is closed")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, partial(function, **kwargs))

    def _require_client(self) -> Any:
        if self._client is None:
            raise RuntimeError("S3 object store is not connected")
        return self._client


def _json_dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _json_load(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _segment_from_row(row: Mapping[str, Any]) -> StoredSegment:
    slices = tuple(
        LogicalChunkSlice(**item) for item in (_json_load(row["logical_chunks"]) or [])
    )
    return StoredSegment(
        segment_id=row["segment_id"],
        session_id=row["session_id"],
        sequence=row["sequence"],
        bucket=row["bucket"],
        object_key=row["object_key"],
        byte_start=row["byte_start"],
        byte_end=row["byte_end"],
        byte_size=row["byte_size"],
        sha256=row["sha256"],
        content_type=row["content_type"],
        logical_chunks=slices,
        created_at=row["created_at"],
    )


def _manifest_from_row(
    row: Mapping[str, Any], segments: Sequence[StoredSegment]
) -> SessionManifest:
    return SessionManifest(
        session_id=row["session_id"],
        contact_id=row["contact_id"],
        mode=PersistenceMode(row["mode"]),
        transport=SessionTransport(row["transport"]),
        status=SessionStatus(row["status"]),
        request_id=row["request_id"],
        content_type=row["content_type"],
        encoding=row["encoding"],
        sample_rate=row["sample_rate"],
        channels=row["channels"],
        model_name=row["model_name"],
        result=_json_load(row["result"]),
        error_code=row["error_code"],
        error_message=row["error_message"],
        segment_count=row["segment_count"],
        audio_bytes=row["audio_bytes"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        expires_at=row["expires_at"],
        metadata=_json_load(row["metadata"]) or {},
        segments=tuple(segments),
    )
