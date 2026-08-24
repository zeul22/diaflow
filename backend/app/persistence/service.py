from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from weakref import WeakValueDictionary

from app.persistence.models import (
    LogicalChunkSlice,
    PersistenceHandle,
    PersistenceMode,
    PersistenceSettings,
    SessionManifest,
    SessionTransport,
    StoredSegment,
    utc_now,
)
from app.persistence.stores import (
    AsyncpgMetadataStore,
    Boto3ObjectStore,
    MetadataStore,
    ObjectStore,
)

logger = logging.getLogger(__name__)
_CONTENT_TYPE_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]*$"
)
_BYTES_PER_SAMPLE = {
    "pcm_s16le": 2,
    "pcm_s16be": 2,
    "pcm_f32le": 4,
    "mulaw": 1,
    "alaw": 1,
}


class PersistenceError(RuntimeError):
    pass


class PersistenceConfigurationError(PersistenceError):
    pass


class PersistenceUnavailableError(PersistenceError):
    pass


class PersistenceModeNotAllowedError(PersistenceError):
    pass


class PersistenceSessionError(PersistenceError):
    pass


@dataclass(frozen=True, slots=True)
class _PendingSegment:
    sequence: int
    payload: bytes
    byte_start: int
    byte_end: int
    logical_chunks: tuple[LogicalChunkSlice, ...]


@dataclass(slots=True)
class _StreamAccumulator:
    target_bytes: int
    buffer: bytearray = field(default_factory=bytearray)
    slices: list[LogicalChunkSlice] = field(default_factory=list)
    source_cursor: int = 0
    segment_start: int = 0
    next_chunk_index: int = 0
    next_sequence: int = 0

    def append(
        self, payload: bytes | bytearray | memoryview, chunk_index: int | None
    ) -> tuple[_PendingSegment, ...]:
        view = memoryview(payload).cast("B")
        if not view:
            return ()
        index = self.next_chunk_index if chunk_index is None else chunk_index
        if index != self.next_chunk_index:
            raise PersistenceSessionError(
                f"expected logical chunk {self.next_chunk_index}, received {index}"
            )
        chunk_source_start = self.source_cursor
        chunk_offset = 0
        pending: list[_PendingSegment] = []
        while chunk_offset < len(view):
            room = self.target_bytes - len(self.buffer)
            take = min(room, len(view) - chunk_offset)
            segment_offset = len(self.buffer)
            self.buffer.extend(view[chunk_offset : chunk_offset + take])
            self.slices.append(
                LogicalChunkSlice(
                    chunk_index=index,
                    source_byte_start=chunk_source_start + chunk_offset,
                    source_byte_end=chunk_source_start + chunk_offset + take,
                    segment_byte_start=segment_offset,
                    segment_byte_end=segment_offset + take,
                )
            )
            chunk_offset += take
            if len(self.buffer) == self.target_bytes:
                pending.append(self._take_segment())
        self.source_cursor += len(view)
        self.next_chunk_index += 1
        return tuple(pending)

    def flush(self) -> _PendingSegment | None:
        if not self.buffer:
            return None
        return self._take_segment()

    def wipe(self) -> None:
        if self.buffer:
            self.buffer[:] = b"\x00" * len(self.buffer)
            self.buffer.clear()
        self.slices.clear()

    def _take_segment(self) -> _PendingSegment:
        payload = bytes(self.buffer)
        segment = _PendingSegment(
            sequence=self.next_sequence,
            payload=payload,
            byte_start=self.segment_start,
            byte_end=self.segment_start + len(payload),
            logical_chunks=tuple(self.slices),
        )
        self.buffer[:] = b"\x00" * len(self.buffer)
        self.buffer.clear()
        self.slices.clear()
        self.segment_start = segment.byte_end
        self.next_sequence += 1
        return segment


@dataclass(slots=True)
class _SessionRuntime:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    stream: _StreamAccumulator | None = None
    rest_audio_written: bool = False
    closed: bool = False


class PersistenceService:
    """Coordinates opt-in PostgreSQL metadata and S3 audio persistence.

    The service intentionally has no implicit persistence. A deployment defines
    the maximum permitted mode in ``PersistenceSettings.mode`` and every request
    selects an equal or less permissive mode explicitly.
    """

    def __init__(
        self,
        settings: PersistenceSettings | None = None,
        *,
        metadata_store: MetadataStore | None = None,
        object_store: ObjectStore | None = None,
        clock: Any = utc_now,
    ) -> None:
        self.settings = settings or PersistenceSettings()
        self._maximum_mode = PersistenceMode.parse(self.settings.mode)
        try:
            self.settings.validate(
                require_database_url=metadata_store is None,
                require_s3_bucket=object_store is None,
            )
        except ValueError as exc:
            raise PersistenceConfigurationError(str(exc)) from exc
        self._metadata = metadata_store
        self._objects = object_store
        self._clock = clock
        self._ready = False
        self._connect_lock = asyncio.Lock()
        self._runtimes: dict[UUID, _SessionRuntime] = {}
        # Completed records can live for days, so lock bookkeeping must not
        # retain one object per historical request. Active runtimes and waiting
        # operations keep their locks alive while this weak index coordinates
        # concurrent deletion calls.
        self._operation_locks: WeakValueDictionary[UUID, asyncio.Lock] = (
            WeakValueDictionary()
        )
        self._cleanup_stop = asyncio.Event()
        self._cleanup_task: asyncio.Task[None] | None = None

    @property
    def maximum_mode(self) -> PersistenceMode:
        return self._maximum_mode

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def enabled(self) -> bool:
        return self._maximum_mode is not PersistenceMode.NONE

    async def connect(self) -> None:
        async with self._connect_lock:
            if self._ready:
                return
            if not self.enabled:
                self._ready = True
                return
            metadata = self._metadata or AsyncpgMetadataStore(self.settings)
            objects = self._objects
            try:
                await metadata.connect()
                if self._maximum_mode is PersistenceMode.RESULT_AND_AUDIO:
                    objects = objects or Boto3ObjectStore(self.settings)
                    await objects.connect()
            except Exception as exc:
                await _close_quietly(objects)
                await _close_quietly(metadata)
                raise PersistenceUnavailableError(
                    "persistence backends could not be initialized"
                ) from exc
            self._metadata = metadata
            self._objects = objects
            self._ready = True
            self._cleanup_stop.clear()
            self._cleanup_task = asyncio.create_task(
                self._cleanup_loop(), name="persistence-retention-cleanup"
            )
            logger.info(
                "persistence_ready",
                extra={"event_data": {"maximum_mode": self._maximum_mode.value}},
            )

    async def close(self) -> None:
        async with self._connect_lock:
            self._ready = False
            self._cleanup_stop.set()
            task, self._cleanup_task = self._cleanup_task, None
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            for runtime in self._runtimes.values():
                if runtime.stream is not None:
                    runtime.stream.wipe()
                runtime.closed = True
            self._runtimes.clear()
            self._operation_locks.clear()
            await _close_quietly(self._objects)
            await _close_quietly(self._metadata)

    async def start_rest(
        self,
        *,
        contact_id: UUID,
        mode: PersistenceMode | str = PersistenceMode.NONE,
        request_id: str | None = None,
        content_type: str | None = None,
        encoding: str | None = None,
        sample_rate: int | None = None,
        channels: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PersistenceHandle:
        return await self._start(
            contact_id=contact_id,
            requested_mode=mode,
            transport=SessionTransport.REST,
            request_id=request_id,
            content_type=content_type,
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            metadata=metadata,
        )

    async def start_ws(
        self,
        *,
        contact_id: UUID,
        mode: PersistenceMode | str = PersistenceMode.NONE,
        request_id: str | None = None,
        content_type: str | None = None,
        encoding: str,
        sample_rate: int,
        channels: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> PersistenceHandle:
        handle = await self._start(
            contact_id=contact_id,
            requested_mode=mode,
            transport=SessionTransport.WEBSOCKET,
            request_id=request_id,
            content_type=content_type,
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
            metadata=metadata,
        )
        if handle.enabled and handle.mode is PersistenceMode.RESULT_AND_AUDIO:
            runtime = self._runtime(handle)
            runtime.stream = _StreamAccumulator(
                target_bytes=_segment_target_bytes(
                    encoding=encoding,
                    sample_rate=sample_rate,
                    channels=channels,
                    seconds=self.settings.ws_segment_seconds,
                )
            )
        return handle

    async def _start(
        self,
        *,
        contact_id: UUID,
        requested_mode: PersistenceMode | str,
        transport: SessionTransport,
        request_id: str | None,
        content_type: str | None,
        encoding: str | None,
        sample_rate: int | None,
        channels: int | None,
        metadata: Mapping[str, Any] | None,
    ) -> PersistenceHandle:
        mode = self._authorize_mode(requested_mode)
        created_at = self._clock()
        if mode is PersistenceMode.NONE:
            return PersistenceHandle(
                session_id=None,
                contact_id=contact_id,
                mode=mode,
                transport=transport,
                created_at=created_at,
                expires_at=None,
                content_type=_safe_content_type(content_type),
                encoding=encoding,
                sample_rate=sample_rate,
                channels=channels,
            )
        self._ensure_ready()
        retention_hours = self.settings.retention_hours
        if (
            mode is PersistenceMode.RESULT
            and self.settings.result_retention_hours is not None
        ):
            retention_hours = self.settings.result_retention_hours
        expires_at = created_at + timedelta(hours=retention_hours)
        handle = PersistenceHandle(
            session_id=uuid4(),
            contact_id=contact_id,
            mode=mode,
            transport=transport,
            created_at=created_at,
            expires_at=expires_at,
            content_type=_safe_content_type(content_type),
            encoding=encoding,
            sample_rate=sample_rate,
            channels=channels,
        )
        runtime = _SessionRuntime()
        if handle.session_id is None:
            raise AssertionError("enabled handle must have a session ID")
        self._runtimes[handle.session_id] = runtime
        self._operation_locks[handle.session_id] = runtime.lock
        try:
            await self._require_metadata().create_session(
                handle,
                request_id=_limited_text(request_id, 80),
                metadata=_json_mapping(metadata or {}),
            )
        except Exception as exc:
            self._runtimes.pop(handle.session_id, None)
            self._operation_locks.pop(handle.session_id, None)
            raise PersistenceUnavailableError(
                "the persistence session could not be created"
            ) from exc
        logger.info(
            "persistence_session_started",
            extra={
                "event_data": {
                    "session_id": str(handle.session_id),
                    "mode": mode.value,
                    "transport": transport.value,
                }
            },
        )
        return handle

    async def store_rest(
        self,
        handle: PersistenceHandle,
        payload: bytes | bytearray | memoryview,
        *,
        content_type: str | None = None,
    ) -> StoredSegment | None:
        if not handle.enabled or handle.mode is not PersistenceMode.RESULT_AND_AUDIO:
            return None
        if not payload:
            raise PersistenceSessionError("cannot persist an empty REST audio payload")
        runtime = self._runtime(handle)
        async with runtime.lock:
            self._ensure_open(runtime)
            if runtime.rest_audio_written:
                raise PersistenceSessionError("REST audio has already been persisted")
            length = len(payload)
            pending = _PendingSegment(
                sequence=0,
                payload=bytes(payload),
                byte_start=0,
                byte_end=length,
                logical_chunks=(
                    LogicalChunkSlice(
                        chunk_index=0,
                        source_byte_start=0,
                        source_byte_end=length,
                        segment_byte_start=0,
                        segment_byte_end=length,
                    ),
                ),
            )
            segment = await self._persist_segment(
                handle, pending, content_type=content_type
            )
            runtime.rest_audio_written = True
            return segment

    async def store_chunk(
        self,
        handle: PersistenceHandle,
        payload: bytes | bytearray | memoryview,
        *,
        chunk_index: int | None = None,
    ) -> tuple[StoredSegment, ...]:
        if not handle.enabled or handle.mode is not PersistenceMode.RESULT_AND_AUDIO:
            return ()
        runtime = self._runtime(handle)
        if runtime.stream is None:
            raise PersistenceSessionError("the handle is not a WebSocket audio session")
        async with runtime.lock:
            self._ensure_open(runtime)
            pending = runtime.stream.append(payload, chunk_index)
            stored = [
                await self._persist_segment(handle, item, content_type=None)
                for item in pending
            ]
            return tuple(stored)

    async def flush(self, handle: PersistenceHandle) -> StoredSegment | None:
        if not handle.enabled or handle.mode is not PersistenceMode.RESULT_AND_AUDIO:
            return None
        runtime = self._runtime(handle)
        async with runtime.lock:
            self._ensure_open(runtime)
            return await self._flush_locked(handle, runtime)

    async def complete(
        self,
        handle: PersistenceHandle,
        *,
        result: Mapping[str, Any] | Any,
        model_name: str | None = None,
    ) -> SessionManifest | None:
        if not handle.enabled:
            return None
        runtime = self._runtime(handle)
        async with runtime.lock:
            self._ensure_open(runtime)
            await self._flush_locked(handle, runtime)
            completed_at = self._clock()
            session_id = _session_id(handle)
            try:
                updated = await self._require_metadata().complete_session(
                    session_id,
                    result=_json_mapping(_model_payload(result)),
                    model_name=_limited_text(model_name, 200),
                    completed_at=completed_at,
                )
            except Exception as exc:
                raise PersistenceUnavailableError(
                    "the persistence result could not be committed"
                ) from exc
            if not updated:
                raise PersistenceSessionError("session is missing or already finalized")
            runtime.closed = True
            if runtime.stream is not None:
                runtime.stream.wipe()
            self._runtimes.pop(session_id, None)
        manifest = await self.get(session_id)
        logger.info(
            "persistence_session_completed",
            extra={"event_data": {"session_id": str(session_id)}},
        )
        return manifest

    async def fail(
        self,
        handle: PersistenceHandle,
        *,
        error_code: str,
        error_message: str,
        flush_audio: bool = True,
    ) -> SessionManifest | None:
        if not handle.enabled:
            return None
        runtime = self._runtime(handle)
        flush_error: Exception | None = None
        async with runtime.lock:
            self._ensure_open(runtime)
            if flush_audio:
                try:
                    await self._flush_locked(handle, runtime)
                except Exception as exc:
                    flush_error = exc
            completed_at = self._clock()
            session_id = _session_id(handle)
            message = _limited_text(error_message, 1_000) or "request failed"
            if flush_error is not None:
                message = f"{message}; trailing audio segment was not persisted"
            try:
                updated = await self._require_metadata().fail_session(
                    session_id,
                    error_code=_limited_text(error_code, 100) or "REQUEST_FAILED",
                    error_message=message,
                    completed_at=completed_at,
                )
            except Exception as exc:
                raise PersistenceUnavailableError(
                    "the failed persistence session could not be finalized"
                ) from exc
            if not updated:
                raise PersistenceSessionError("session is missing or already finalized")
            runtime.closed = True
            if runtime.stream is not None:
                runtime.stream.wipe()
            self._runtimes.pop(session_id, None)
        manifest = await self.get(session_id)
        if flush_error is not None:
            raise PersistenceUnavailableError(
                "the trailing audio segment could not be persisted"
            ) from flush_error
        return manifest

    async def get(self, session_id: UUID) -> SessionManifest | None:
        if not self.enabled:
            return None
        self._ensure_ready()
        try:
            return await self._require_metadata().get_session(session_id)
        except Exception as exc:
            raise PersistenceUnavailableError(
                "the persistence manifest could not be read"
            ) from exc

    async def list(
        self,
        *,
        contact_id: UUID | None = None,
        limit: int = 50,
        before: datetime | None = None,
    ) -> Sequence[SessionManifest]:
        if not self.enabled:
            return ()
        self._ensure_ready()
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        try:
            return await self._require_metadata().list_sessions(
                contact_id=contact_id, limit=limit, before=before
            )
        except Exception as exc:
            raise PersistenceUnavailableError(
                "persistence manifests could not be listed"
            ) from exc

    async def delete(self, session_id: UUID) -> bool:
        if not self.enabled:
            return False
        self._ensure_ready()
        lock = self._operation_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            manifest = await self.get(session_id)
            if manifest is None:
                return False
            runtime = self._runtimes.get(session_id)
            if runtime is not None:
                runtime.closed = True
            object_keys = [segment.object_key for segment in manifest.segments]
            if object_keys:
                try:
                    await self._require_objects().delete_objects(object_keys)
                except Exception as exc:
                    raise PersistenceUnavailableError(
                        "audio objects could not be deleted; metadata was retained"
                    ) from exc
            try:
                deleted = await self._require_metadata().delete_session(session_id)
            except Exception as exc:
                raise PersistenceUnavailableError(
                    "session metadata could not be deleted"
                ) from exc
            if deleted:
                if runtime is not None and runtime.stream is not None:
                    runtime.stream.wipe()
                self._runtimes.pop(session_id, None)
                self._operation_locks.pop(session_id, None)
                logger.info(
                    "persistence_session_deleted",
                    extra={"event_data": {"session_id": str(session_id)}},
                )
            return deleted

    async def purge_expired(self, *, before: datetime | None = None) -> int:
        if not self.enabled:
            return 0
        self._ensure_ready()
        cutoff = before or self._clock()
        try:
            session_ids = await self._require_metadata().list_expired(
                before=cutoff, limit=self.settings.cleanup_batch_size
            )
        except Exception as exc:
            raise PersistenceUnavailableError(
                "expired sessions could not be listed"
            ) from exc
        deleted = 0
        for session_id in session_ids:
            try:
                deleted += int(await self.delete(session_id))
            except PersistenceError:
                logger.exception(
                    "persistence_expired_session_delete_failed",
                    extra={"event_data": {"session_id": str(session_id)}},
                )
        return deleted

    async def _cleanup_loop(self) -> None:
        while not self._cleanup_stop.is_set():
            try:
                await asyncio.wait_for(
                    self._cleanup_stop.wait(),
                    timeout=self.settings.cleanup_interval_seconds,
                )
            except TimeoutError:
                try:
                    deleted = await self.purge_expired()
                    if deleted:
                        logger.info(
                            "persistence_retention_cleanup",
                            extra={"event_data": {"deleted_sessions": deleted}},
                        )
                except Exception:
                    logger.exception("persistence_retention_cleanup_failed")

    async def _flush_locked(
        self, handle: PersistenceHandle, runtime: _SessionRuntime
    ) -> StoredSegment | None:
        if runtime.stream is None:
            return None
        pending = runtime.stream.flush()
        if pending is None:
            return None
        return await self._persist_segment(handle, pending, content_type=None)

    async def _persist_segment(
        self,
        handle: PersistenceHandle,
        pending: _PendingSegment,
        *,
        content_type: str | None,
    ) -> StoredSegment:
        session_id = _session_id(handle)
        objects = self._require_objects()
        segment_id = uuid4()
        checksum = hashlib.sha256(pending.payload).hexdigest()
        created_at = self._clock()
        object_key = _object_key(
            prefix=self.settings.object_key_prefix,
            created_at=created_at,
            session_id=session_id,
            sequence=pending.sequence,
            segment_id=segment_id,
            encoding=handle.encoding,
            content_type=content_type or handle.content_type,
        )
        segment = StoredSegment(
            segment_id=segment_id,
            session_id=session_id,
            sequence=pending.sequence,
            bucket=objects.bucket,
            object_key=object_key,
            byte_start=pending.byte_start,
            byte_end=pending.byte_end,
            byte_size=len(pending.payload),
            sha256=checksum,
            content_type=_safe_content_type(content_type or handle.content_type),
            logical_chunks=pending.logical_chunks,
            created_at=created_at,
        )
        put_task = asyncio.create_task(
            objects.put_bytes(
                object_key=object_key,
                payload=pending.payload,
                content_type=segment.content_type,
                metadata={
                    "session-id": str(session_id),
                    "segment-id": str(segment_id),
                    "sequence": str(segment.sequence),
                    "sha256": checksum,
                },
            )
        )
        try:
            await asyncio.shield(put_task)
        except asyncio.CancelledError:
            uploaded = False
            try:
                await _finish_task(put_task)
                uploaded = True
            except Exception:
                pass
            if uploaded:
                await _delete_object_quietly(objects, object_key)
            raise
        except Exception as exc:
            raise PersistenceUnavailableError(
                "audio segment could not be written to object storage"
            ) from exc
        metadata_task = asyncio.create_task(
            self._require_metadata().add_segment(segment)
        )
        try:
            await asyncio.shield(metadata_task)
        except asyncio.CancelledError:
            committed = False
            try:
                await _finish_task(metadata_task)
                committed = True
            except Exception:
                pass
            if not committed:
                await _delete_object_quietly(objects, object_key)
            raise
        except Exception as exc:
            await _delete_object_quietly(objects, object_key)
            raise PersistenceUnavailableError(
                "audio segment metadata could not be committed"
            ) from exc
        return segment

    def _authorize_mode(self, requested: PersistenceMode | str) -> PersistenceMode:
        try:
            mode = PersistenceMode.parse(requested)
        except ValueError as exc:
            raise PersistenceModeNotAllowedError(str(exc)) from exc
        if mode.rank > self._maximum_mode.rank:
            raise PersistenceModeNotAllowedError(
                f"requested mode '{mode.value}' exceeds deployment maximum "
                f"'{self._maximum_mode.value}'"
            )
        return mode

    def _runtime(self, handle: PersistenceHandle) -> _SessionRuntime:
        session_id = _session_id(handle)
        runtime = self._runtimes.get(session_id)
        if runtime is None:
            raise PersistenceSessionError("session is not active in this process")
        return runtime

    @staticmethod
    def _ensure_open(runtime: _SessionRuntime) -> None:
        if runtime.closed:
            raise PersistenceSessionError("session is already finalized")

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise PersistenceUnavailableError("persistence service is not ready")

    def _require_metadata(self) -> MetadataStore:
        self._ensure_ready()
        if self._metadata is None:
            raise PersistenceUnavailableError("metadata persistence is unavailable")
        return self._metadata

    def _require_objects(self) -> ObjectStore:
        self._ensure_ready()
        if self._objects is None:
            raise PersistenceUnavailableError("audio object storage is unavailable")
        return self._objects


def _session_id(handle: PersistenceHandle) -> UUID:
    if handle.session_id is None:
        raise PersistenceSessionError("persistence is disabled for this request")
    return handle.session_id


def _segment_target_bytes(
    *, encoding: str, sample_rate: int, channels: int, seconds: float
) -> int:
    if sample_rate <= 0 or channels <= 0:
        raise PersistenceConfigurationError(
            "sample_rate and channels must be positive for WebSocket persistence"
        )
    bytes_per_sample = _BYTES_PER_SAMPLE.get(encoding)
    if bytes_per_sample is None:
        raise PersistenceConfigurationError(
            f"unsupported WebSocket persistence encoding '{encoding}'"
        )
    frame_bytes = channels * bytes_per_sample
    target = int(round(sample_rate * frame_bytes * seconds))
    target -= target % frame_bytes
    return max(frame_bytes, target)


def _safe_content_type(value: str | None) -> str:
    normalized = (value or "").split(";", 1)[0].strip().lower()
    if len(normalized) <= 128 and _CONTENT_TYPE_PATTERN.fullmatch(normalized):
        return normalized
    return "application/octet-stream"


def _object_key(
    *,
    prefix: str,
    created_at: datetime,
    session_id: UUID,
    sequence: int,
    segment_id: UUID,
    encoding: str | None,
    content_type: str | None,
) -> str:
    extensions = {
        "wav": "wav",
        "pcm_s16le": "pcm",
        "pcm_s16be": "pcm",
        "pcm_f32le": "f32",
        "mulaw": "ulaw",
        "alaw": "alaw",
    }
    content_extensions = {
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/webm": "webm",
        "audio/wav": "wav",
    }
    extension = extensions.get(encoding or "") or content_extensions.get(
        _safe_content_type(content_type), "bin"
    )
    date_path = created_at.strftime("%Y/%m/%d")
    return (
        f"{prefix.strip('/')}/{date_path}/{session_id}/segments/"
        f"{sequence:08d}-{segment_id.hex}.{extension}"
    )


def _model_payload(value: Any) -> Mapping[str, Any]:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json", exclude_none=True)
    if not isinstance(value, Mapping):
        raise ValueError("persisted result must be an object")
    if "debug_age_years" in value:
        # Evaluation-only diagnostics never enter retained records: a stored
        # point estimate of a caller's age is more sensitive than the bracket
        # the API contract promises, and has no retention basis.
        value = {key: item for key, item in value.items() if key != "debug_age_years"}
    return value


def _json_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(value, default=_json_default))
    if not isinstance(normalized, dict):
        raise ValueError("persisted metadata must be an object")
    return normalized


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _limited_text(value: str | None, maximum: int) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized[:maximum] or None


async def _close_quietly(store: Any | None) -> None:
    if store is None:
        return
    try:
        await store.close()
    except Exception:
        logger.exception("persistence_backend_close_failed")


async def _finish_task(task: asyncio.Task[Any]) -> Any:
    """Join a shielded storage operation even if cancellation repeats."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


async def _delete_object_quietly(objects: ObjectStore, object_key: str) -> None:
    task = asyncio.create_task(objects.delete_objects([object_key]))
    try:
        await _finish_task(task)
    except Exception:
        logger.critical(
            "persistence_orphan_object_cleanup_failed",
            extra={"event_data": {"object_key": object_key}},
            exc_info=True,
        )
