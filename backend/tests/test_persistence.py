from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.persistence import (
    PersistenceMode,
    PersistenceModeNotAllowedError,
    PersistenceService,
    PersistenceSettings,
    PersistenceUnavailableError,
    SessionManifest,
    SessionStatus,
    StoredSegment,
)


class FakeMetadataStore:
    def __init__(self) -> None:
        self.connected = False
        self.sessions: dict[UUID, SessionManifest] = {}
        self.fail_add_segment = False

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def create_session(
        self, handle, *, request_id: str | None, metadata: dict[str, Any]
    ) -> None:
        assert handle.session_id is not None
        assert handle.expires_at is not None
        self.sessions[handle.session_id] = SessionManifest(
            session_id=handle.session_id,
            contact_id=handle.contact_id,
            mode=handle.mode,
            transport=handle.transport,
            status=SessionStatus.PENDING,
            request_id=request_id,
            content_type=handle.content_type,
            encoding=handle.encoding,
            sample_rate=handle.sample_rate,
            channels=handle.channels,
            model_name=None,
            result=None,
            error_code=None,
            error_message=None,
            segment_count=0,
            audio_bytes=0,
            created_at=handle.created_at,
            updated_at=handle.created_at,
            completed_at=None,
            expires_at=handle.expires_at,
            metadata=metadata,
        )

    async def add_segment(self, segment: StoredSegment) -> None:
        if self.fail_add_segment:
            raise RuntimeError("database rejected segment")
        manifest = self.sessions[segment.session_id]
        self.sessions[segment.session_id] = replace(
            manifest,
            segment_count=manifest.segment_count + 1,
            audio_bytes=manifest.audio_bytes + segment.byte_size,
            updated_at=segment.created_at,
            segments=(*manifest.segments, segment),
        )

    async def complete_session(
        self,
        session_id: UUID,
        *,
        result,
        model_name: str | None,
        completed_at: datetime,
    ) -> bool:
        manifest = self.sessions.get(session_id)
        if manifest is None or manifest.status is not SessionStatus.PENDING:
            return False
        self.sessions[session_id] = replace(
            manifest,
            status=SessionStatus.COMPLETED,
            result=result,
            model_name=model_name,
            completed_at=completed_at,
            updated_at=completed_at,
        )
        return True

    async def fail_session(
        self,
        session_id: UUID,
        *,
        error_code: str,
        error_message: str,
        completed_at: datetime,
    ) -> bool:
        manifest = self.sessions.get(session_id)
        if manifest is None or manifest.status is not SessionStatus.PENDING:
            return False
        self.sessions[session_id] = replace(
            manifest,
            status=SessionStatus.FAILED,
            error_code=error_code,
            error_message=error_message,
            completed_at=completed_at,
            updated_at=completed_at,
        )
        return True

    async def get_session(self, session_id: UUID) -> SessionManifest | None:
        return self.sessions.get(session_id)

    async def list_sessions(
        self,
        *,
        contact_id: UUID | None,
        limit: int,
        before: datetime | None,
    ) -> tuple[SessionManifest, ...]:
        manifests = [
            item
            for item in self.sessions.values()
            if (contact_id is None or item.contact_id == contact_id)
            and (before is None or item.created_at < before)
        ]
        manifests.sort(key=lambda item: item.created_at, reverse=True)
        return tuple(manifests[:limit])

    async def list_expired(self, *, before: datetime, limit: int) -> tuple[UUID, ...]:
        return tuple(
            item.session_id
            for item in sorted(
                self.sessions.values(), key=lambda manifest: manifest.expires_at
            )
            if item.expires_at <= before
        )[:limit]

    async def delete_session(self, session_id: UUID) -> bool:
        return self.sessions.pop(session_id, None) is not None


class FakeObjectStore:
    bucket = "test-audio"

    def __init__(self) -> None:
        self.connected = False
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.deleted: list[str] = []

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def put_bytes(
        self,
        *,
        object_key: str,
        payload: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        del content_type
        self.objects[object_key] = payload
        self.metadata[object_key] = metadata

    async def delete_objects(self, object_keys) -> None:
        for object_key in object_keys:
            self.objects.pop(object_key, None)
            self.metadata.pop(object_key, None)
            self.deleted.append(object_key)


class MutableClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def audio_settings(**changes) -> PersistenceSettings:
    values = {
        "mode": PersistenceMode.RESULT_AND_AUDIO,
        "retention_hours": 1.0,
        "cleanup_interval_seconds": 3_600,
        **changes,
    }
    return replace(PersistenceSettings(), **values)


def test_default_service_is_a_true_noop() -> None:
    async def scenario() -> None:
        service = PersistenceService()
        await service.connect()
        handle = await service.start_rest(contact_id=uuid4())

        assert service.ready
        assert not service.enabled
        assert not handle.enabled
        assert handle.mode is PersistenceMode.NONE
        assert await service.store_rest(handle, b"caller audio") is None
        assert await service.complete(handle, result={"prediction": "unknown"}) is None
        await service.close()

    asyncio.run(scenario())


def test_result_mode_stores_metadata_but_never_audio() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = FakeObjectStore()
        service = PersistenceService(
            audio_settings(), metadata_store=metadata, object_store=objects
        )
        await service.connect()
        contact_id = uuid4()
        handle = await service.start_rest(
            contact_id=contact_id,
            mode=PersistenceMode.RESULT,
            request_id="request-1",
            metadata={"source": "upload"},
        )

        assert await service.store_rest(handle, b"must-not-be-stored") is None
        manifest = await service.complete(
            handle,
            result={"gender": {"prediction": "unknown", "confidence": 0.0}},
            model_name="production-model",
        )

        assert manifest is not None
        assert manifest.status is SessionStatus.COMPLETED
        assert manifest.contact_id == contact_id
        assert manifest.segment_count == 0
        assert manifest.audio_bytes == 0
        assert manifest.model_name == "production-model"
        assert objects.objects == {}
        await service.close()

    asyncio.run(scenario())


def test_rest_audio_uses_random_key_hash_and_exact_offsets() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = FakeObjectStore()
        clock = MutableClock()
        service = PersistenceService(
            audio_settings(),
            metadata_store=metadata,
            object_store=objects,
            clock=clock,
        )
        await service.connect()
        contact_id = uuid4()
        payload = b"m4a-payload-contents"
        handle = await service.start_rest(
            contact_id=contact_id,
            mode="result_and_audio",
            content_type="audio/mp4",
            encoding="auto",
            sample_rate=16_000,
            channels=1,
        )
        segment = await service.store_rest(handle, payload)

        assert segment is not None
        assert objects.objects[segment.object_key] == payload
        assert segment.object_key.endswith(".m4a")
        assert str(contact_id) not in segment.object_key
        assert segment.sha256 == hashlib.sha256(payload).hexdigest()
        assert (segment.byte_start, segment.byte_end) == (0, len(payload))
        assert segment.logical_chunks[0].as_dict() == {
            "chunk_index": 0,
            "source_byte_start": 0,
            "source_byte_end": len(payload),
            "segment_byte_start": 0,
            "segment_byte_end": len(payload),
        }
        await service.complete(handle, result={"ok": True})
        await service.close()

    asyncio.run(scenario())


def test_websocket_chunks_coalesce_to_one_second_segments() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = FakeObjectStore()
        service = PersistenceService(
            audio_settings(ws_segment_seconds=1.0),
            metadata_store=metadata,
            object_store=objects,
        )
        await service.connect()
        handle = await service.start_ws(
            contact_id=uuid4(),
            mode="result_and_audio",
            encoding="pcm_s16le",
            sample_rate=16_000,
            channels=1,
        )
        chunks = (b"a" * 12_800, b"b" * 22_400, b"c" * 30_000)

        assert await service.store_chunk(handle, chunks[0], chunk_index=0) == ()
        first = await service.store_chunk(handle, chunks[1], chunk_index=1)
        second = await service.store_chunk(handle, chunks[2], chunk_index=2)
        trailing = await service.flush(handle)

        assert len(first) == 1
        assert len(second) == 1
        assert trailing is not None
        segments = (*first, *second, trailing)
        assert [item.byte_size for item in segments] == [32_000, 32_000, 1_200]
        assert [item.byte_start for item in segments] == [0, 32_000, 64_000]
        restored = b"".join(objects.objects[item.object_key] for item in segments)
        assert restored == b"".join(chunks)
        assert segments[0].logical_chunks[0].chunk_index == 0
        assert segments[0].logical_chunks[1].chunk_index == 1
        assert segments[1].logical_chunks[0].source_byte_start == 32_000

        manifest = await service.complete(handle, result={"final": True})
        assert manifest is not None
        assert manifest.segment_count == 3
        assert manifest.audio_bytes == sum(map(len, chunks))
        await service.close()

    asyncio.run(scenario())


def test_object_is_compensated_when_segment_metadata_fails() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        metadata.fail_add_segment = True
        objects = FakeObjectStore()
        service = PersistenceService(
            audio_settings(), metadata_store=metadata, object_store=objects
        )
        await service.connect()
        handle = await service.start_rest(
            contact_id=uuid4(), mode=PersistenceMode.RESULT_AND_AUDIO
        )

        with pytest.raises(PersistenceUnavailableError):
            await service.store_rest(handle, b"temporary-audio")

        assert objects.objects == {}
        assert len(objects.deleted) == 1
        await service.close()

    asyncio.run(scenario())


def test_cancelled_object_put_finishes_then_deletes_the_object() -> None:
    class SlowObjectStore(FakeObjectStore):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def put_bytes(self, **kwargs) -> None:
            self.started.set()
            await self.release.wait()
            await super().put_bytes(**kwargs)

    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = SlowObjectStore()
        service = PersistenceService(
            audio_settings(), metadata_store=metadata, object_store=objects
        )
        await service.connect()
        handle = await service.start_rest(
            contact_id=uuid4(), mode=PersistenceMode.RESULT_AND_AUDIO
        )
        operation = asyncio.create_task(service.store_rest(handle, b"caller-audio"))
        await objects.started.wait()

        operation.cancel()
        objects.release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation

        assert objects.objects == {}
        assert len(objects.deleted) == 1
        assert next(iter(metadata.sessions.values())).segment_count == 0
        await service.close()

    asyncio.run(scenario())


def test_deployment_maximum_prevents_mode_escalation() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        service = PersistenceService(
            replace(PersistenceSettings(), mode=PersistenceMode.RESULT),
            metadata_store=metadata,
        )
        await service.connect()

        with pytest.raises(PersistenceModeNotAllowedError):
            await service.start_rest(
                contact_id=uuid4(), mode=PersistenceMode.RESULT_AND_AUDIO
            )
        await service.close()

    asyncio.run(scenario())


def test_delete_removes_objects_before_metadata() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = FakeObjectStore()
        service = PersistenceService(
            audio_settings(), metadata_store=metadata, object_store=objects
        )
        await service.connect()
        handle = await service.start_rest(
            contact_id=uuid4(), mode=PersistenceMode.RESULT_AND_AUDIO
        )
        segment = await service.store_rest(handle, b"delete-me")
        assert segment is not None
        await service.complete(handle, result={"ok": True})

        assert handle.session_id is not None
        assert await service.delete(handle.session_id)
        assert segment.object_key in objects.deleted
        assert metadata.sessions == {}
        assert not await service.delete(handle.session_id)
        await service.close()

    asyncio.run(scenario())


def test_retention_cleanup_deletes_expired_session() -> None:
    async def scenario() -> None:
        metadata = FakeMetadataStore()
        objects = FakeObjectStore()
        clock = MutableClock()
        service = PersistenceService(
            audio_settings(retention_hours=0.02),
            metadata_store=metadata,
            object_store=objects,
            clock=clock,
        )
        await service.connect()
        handle = await service.start_rest(
            contact_id=uuid4(), mode=PersistenceMode.RESULT_AND_AUDIO
        )
        await service.store_rest(handle, b"short-lived")
        await service.complete(handle, result={"ok": True})
        clock.now += timedelta(minutes=2)

        assert await service.purge_expired() == 1
        assert metadata.sessions == {}
        assert objects.objects == {}
        await service.close()

    asyncio.run(scenario())
