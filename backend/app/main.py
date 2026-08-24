from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST
from pydantic import ValidationError

from app.audio.ingest import (
    ingest_request,
    request_audio_metadata,
    resolve_contact_id,
    wipe_buffer,
)
from app.audio.types import SourceSpec, source_spec_from_values
from app.config import Settings
from app.errors import InputTimeoutError, InvalidRequestError, ServiceError
from app.inference.service import AnalysisService
from app.models.base import AttributeEstimator
from app.models.factory import create_estimator
from app.observability.logging import configure_logging
from app.observability.metrics import Metrics
from app.persistence import (
    PersistenceError,
    PersistenceHandle,
    PersistenceModeNotAllowedError,
    PersistenceService,
    PersistenceSettings,
    PersistenceUnavailableError,
    SessionManifest,
)
from app.persistence import (
    PersistenceMode as StorageMode,
)
from app.schemas import (
    AnalysisResponse,
    ErrorResponse,
    PersistenceMode,
    PersistenceReceipt,
    PersistenceStatus,
    WebSocketStart,
)

logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
KNOWN_PATHS = {
    "/analyze",
    "/analyses",
    "/healthz",
    "/persistence/capabilities",
    "/readyz",
    "/metrics",
}


def _request_id(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if REQUEST_ID_PATTERN.fullmatch(supplied):
        return supplied
    return str(uuid4())


def _error_payload(error: ServiceError, request_id: str | None) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "request_id": request_id,
        }
    }


def _build_persistence(settings: Settings) -> PersistenceService:
    maximum_mode = (
        StorageMode.RESULT_AND_AUDIO
        if settings.persistence_enabled
        else StorageMode.NONE
    )
    return PersistenceService(
        PersistenceSettings(
            mode=maximum_mode,
            database_url=settings.database_url,
            retention_hours=float(settings.audio_retention_hours),
            result_retention_hours=float(settings.result_retention_days * 24),
            ws_segment_seconds=settings.storage_segment_seconds,
            db_command_timeout_seconds=settings.storage_operation_timeout_seconds,
            s3_bucket=settings.s3_bucket,
            s3_endpoint_url=settings.s3_endpoint_url,
            s3_region=settings.s3_region,
            s3_access_key_id=settings.s3_access_key,
            s3_secret_access_key=settings.s3_secret_key,
            s3_create_bucket=settings.s3_create_bucket,
            s3_server_side_encryption=settings.s3_server_side_encryption,
            s3_operation_timeout_seconds=settings.storage_operation_timeout_seconds,
            s3_max_workers=settings.storage_worker_threads,
            s3_orphan_lifecycle_days=(
                (settings.audio_retention_hours + 23) // 24 + 1
                if settings.s3_create_bucket
                else None
            ),
            object_key_prefix="voice-attributes/v1",
        )
    )


def _persistence_error(error: PersistenceError) -> ServiceError:
    if isinstance(error, PersistenceModeNotAllowedError):
        return ServiceError(
            "PERSISTENCE_NOT_AVAILABLE",
            "The requested retention mode is not enabled for this deployment",
            503,
        )
    if isinstance(error, PersistenceUnavailableError):
        return ServiceError(
            "STORAGE_UNAVAILABLE",
            "Opted-in storage is temporarily unavailable; nothing is reported as stored",
            503,
        )
    return ServiceError(
        "STORAGE_ERROR",
        "The opted-in storage operation could not be completed",
        503,
    )


def _retention_policy(
    mode: PersistenceMode | str | None,
    consent_reference: str | None,
) -> tuple[StorageMode, dict[str, str]]:
    try:
        requested = PersistenceMode(mode or PersistenceMode.NONE)
    except ValueError as exc:
        raise InvalidRequestError(
            "INVALID_PERSISTENCE_MODE",
            "Persistence mode must be none, result, or result_and_audio",
        ) from exc
    reference = (consent_reference or "").strip()
    if requested is PersistenceMode.RESULT_AND_AUDIO and not reference:
        raise InvalidRequestError(
            "CONSENT_REFERENCE_REQUIRED",
            "Audio retention requires an explicit consent reference",
        )
    if len(reference) > 256:
        raise InvalidRequestError(
            "INVALID_CONSENT_REFERENCE",
            "Consent reference must not exceed 256 characters",
        )
    metadata: dict[str, str] = {}
    if reference:
        metadata["consent_reference_sha256"] = hashlib.sha256(
            reference.encode("utf-8")
        ).hexdigest()
    return StorageMode(requested.value), metadata


def _request_retention_policy(
    request: Request, fields: dict[str, str]
) -> tuple[StorageMode, dict[str, str]]:
    header_mode = request.headers.get("x-persistence-mode")
    field_mode = fields.get("persistence_mode")
    if header_mode and field_mode and header_mode.strip().lower() != field_mode.lower():
        raise InvalidRequestError(
            "CONFLICTING_PERSISTENCE_MODE",
            "Persistence mode sources disagree",
        )
    header_consent = request.headers.get("x-consent-reference")
    field_consent = fields.get("consent_reference")
    if header_consent and field_consent and header_consent.strip() != field_consent:
        raise InvalidRequestError(
            "CONFLICTING_CONSENT_REFERENCE",
            "Consent reference sources disagree",
        )
    return _retention_policy(
        header_mode or field_mode,
        header_consent or field_consent,
    )


def _stored_chunk_count(manifest: SessionManifest) -> int:
    return len(
        {
            item.chunk_index
            for segment in manifest.segments
            for item in segment.logical_chunks
        }
    )


def _receipt_from_manifest(manifest: SessionManifest) -> PersistenceReceipt:
    status = {
        "pending": PersistenceStatus.PENDING,
        "completed": PersistenceStatus.STORED,
        "failed": PersistenceStatus.FAILED,
    }[manifest.status.value]
    audio_retained = manifest.mode is StorageMode.RESULT_AND_AUDIO
    chunks = _stored_chunk_count(manifest)
    return PersistenceReceipt(
        mode=PersistenceMode(manifest.mode.value),
        status=status,
        chunks_received=chunks,
        chunks_stored=chunks,
        segments_stored=manifest.segment_count,
        bytes_stored=manifest.audio_bytes,
        audio_expires_at=manifest.expires_at if audio_retained else None,
        result_expires_at=manifest.expires_at,
    )


def _pending_receipt(
    handle: PersistenceHandle,
    *,
    chunks_received: int = 0,
    chunks_stored: int = 0,
    segments_stored: int = 0,
    bytes_stored: int = 0,
) -> PersistenceReceipt:
    return PersistenceReceipt(
        mode=PersistenceMode(handle.mode.value),
        status=PersistenceStatus.PENDING,
        chunks_received=chunks_received,
        chunks_stored=chunks_stored,
        segments_stored=segments_stored,
        bytes_stored=bytes_stored,
        audio_expires_at=(
            handle.expires_at if handle.mode is StorageMode.RESULT_AND_AUDIO else None
        ),
        result_expires_at=handle.expires_at,
    )


def _with_persistence(
    result: AnalysisResponse,
    handle: PersistenceHandle,
    receipt: PersistenceReceipt,
) -> AnalysisResponse:
    return result.model_copy(
        update={"analysis_id": handle.session_id, "persistence": receipt}
    )


async def _delete_incomplete_session(
    storage: PersistenceService,
    handle: PersistenceHandle,
) -> None:
    """Finish deletion before propagating request cancellation or failure."""

    if handle.session_id is None:
        return
    task = asyncio.create_task(storage.delete(handle.session_id))
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        task.result()
        raise


def _manifest_payload(manifest: SessionManifest, *, detail: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "analysis_id": str(manifest.session_id),
        "session_id": str(manifest.session_id),
        "contact_id": str(manifest.contact_id),
        "mode": manifest.mode.value,
        "transport": manifest.transport.value,
        "status": manifest.status.value,
        "model_name": manifest.model_name,
        "result": manifest.result,
        "segment_count": manifest.segment_count,
        "audio_bytes": manifest.audio_bytes,
        "created_at": manifest.created_at.isoformat(),
        "completed_at": (
            manifest.completed_at.isoformat() if manifest.completed_at else None
        ),
        "expires_at": manifest.expires_at.isoformat(),
        "error_code": manifest.error_code,
    }
    if detail:
        payload["segments"] = [
            {
                "sequence": segment.sequence,
                "object_key": segment.object_key,
                "byte_start": segment.byte_start,
                "byte_end": segment.byte_end,
                "byte_size": segment.byte_size,
                "sha256": segment.sha256,
                "content_type": segment.content_type,
                "logical_chunks": [item.as_dict() for item in segment.logical_chunks],
                "created_at": segment.created_at.isoformat(),
            }
            for segment in manifest.segments
        ]
    return payload


def create_app(
    *,
    settings: Settings | None = None,
    estimator: AttributeEstimator | None = None,
    persistence: PersistenceService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_settings.validate()
    configure_logging(resolved_settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        storage = persistence or _build_persistence(resolved_settings)
        model = estimator or await asyncio.to_thread(
            create_estimator, resolved_settings
        )
        if resolved_settings.warmup_model:
            await asyncio.to_thread(model.warmup)
        application.state.analysis_service = AnalysisService(
            settings=resolved_settings,
            estimator=model,
            metrics=metrics,
        )
        await storage.connect()
        application.state.persistence_service = storage
        logger.info(
            "service_ready",
            extra={
                "event_data": {
                    "service": resolved_settings.service_name,
                    "version": resolved_settings.service_version,
                    "model": model.name,
                    "persistence_maximum": storage.maximum_mode.value,
                }
            },
        )
        try:
            yield
        finally:
            application.state.analysis_service.ready = False
            await storage.close()

    app = FastAPI(
        title="Voice Contact Attribute Service",
        version=resolved_settings.service_version,
        description=(
            "Low-latency perceived voice-gender and age-bracket estimates with "
            "explicit audio quality gating. Retention is disabled per request by "
            "default and available only through an explicit opt-in policy."
        ),
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.metrics = metrics

    @app.middleware("http")
    async def request_observability(request: Request, call_next):
        request_id = _request_id(request)
        request.state.request_id = request_id
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            if request.url.path == "/analyses" or request.url.path.startswith(
                "/analyses/"
            ):
                response.headers["Cache-Control"] = "no-store"
            return response
        finally:
            elapsed = time.perf_counter() - started
            metric_path = request.url.path
            if metric_path.startswith("/analyses/"):
                metric_path = "/analyses/{id}"
            elif metric_path not in KNOWN_PATHS:
                metric_path = "__other__"
            metrics.http_requests.labels(
                path=metric_path, status=str(status_code)
            ).inc()
            metrics.http_duration.labels(path=metric_path).observe(elapsed)
            logger.info(
                "http_request_completed",
                extra={
                    "event_data": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": metric_path,
                        "status": status_code,
                        "duration_ms": round(elapsed * 1_000),
                    }
                },
            )

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, exc: ServiceError):
        request_id = getattr(request.state, "request_id", None)
        metrics.errors.labels(code=exc.code).inc()
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc, request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        del exc
        error = InvalidRequestError(
            "INVALID_REQUEST", "Request parameters failed validation"
        )
        metrics.errors.labels(code=error.code).inc()
        return JSONResponse(
            status_code=error.status_code,
            content=_error_payload(error, getattr(request.state, "request_id", None)),
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", None)
        metrics.errors.labels(code="INTERNAL_ERROR").inc()
        logger.exception(
            "unhandled_request_error",
            extra={"event_data": {"request_id": request_id}},
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected internal error occurred",
                    "request_id": request_id,
                }
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        service = getattr(request.app.state, "analysis_service", None)
        storage = getattr(request.app.state, "persistence_service", None)
        ready = bool(service and service.ready and storage and storage.ready)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "not_ready"},
        )

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    @app.post(
        "/analyze",
        response_model=AnalysisResponse,
        response_model_exclude_none=True,
        responses={
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["audio"],
                            "properties": {
                                "audio": {"type": "string", "format": "binary"},
                                "contact_id": {"type": "string", "format": "uuid"},
                                "persistence_mode": {
                                    "type": "string",
                                    "enum": ["none", "result", "result_and_audio"],
                                    "default": "none",
                                },
                                "consent_reference": {"type": "string"},
                            },
                        }
                    },
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                    "application/octet-stream": {
                        "schema": {"type": "string", "format": "binary"}
                    },
                },
            }
        },
    )
    async def analyze(request: Request) -> AnalysisResponse:
        ingested = await ingest_request(request, resolved_settings)
        handle: PersistenceHandle | None = None
        storage: PersistenceService | None = None
        retention_completed = False
        try:
            contact_id = resolve_contact_id(request, ingested)
            encoding, sample_rate, channels = request_audio_metadata(request, ingested)
            source = source_spec_from_values(
                encoding=encoding,
                sample_rate=sample_rate,
                channels=channels,
                content_type=ingested.content_type,
            )
            requested_mode, retention_metadata = _request_retention_policy(
                request, ingested.fields
            )
            storage = request.app.state.persistence_service
            try:
                handle = await storage.start_rest(
                    contact_id=contact_id,
                    mode=requested_mode,
                    request_id=request.state.request_id,
                    content_type=ingested.content_type,
                    encoding=source.encoding,
                    sample_rate=source.sample_rate,
                    channels=source.channels,
                    metadata=retention_metadata,
                )
                await storage.store_rest(
                    handle,
                    ingested.payload,
                    content_type=ingested.content_type,
                )
            except PersistenceError as exc:
                raise _persistence_error(exc) from exc
            service: AnalysisService = request.app.state.analysis_service
            result = await service.analyze(
                payload=ingested.payload,
                source=source,
                contact_id=contact_id,
            )
            try:
                manifest = await storage.complete(
                    handle,
                    result=result,
                    model_name=service.estimator.name,
                )
            except PersistenceError as exc:
                raise _persistence_error(exc) from exc
            if manifest is None:
                return result
            response = _with_persistence(
                result,
                handle,
                _receipt_from_manifest(manifest),
            )
            retention_completed = True
            return response
        finally:
            if (
                storage is not None
                and handle is not None
                and handle.enabled
                and not retention_completed
            ):
                try:
                    await _delete_incomplete_session(storage, handle)
                except asyncio.CancelledError:
                    raise
                except PersistenceError:
                    logger.exception("persistence_rest_cleanup_failed")
            wipe_buffer(ingested.payload)

    @app.get("/persistence/capabilities")
    async def persistence_capabilities(request: Request) -> dict[str, Any]:
        storage: PersistenceService = request.app.state.persistence_service
        return {
            "enabled": storage.enabled,
            "maximum_mode": storage.maximum_mode.value,
            "default_mode": "none",
            "audio_retention_hours": resolved_settings.audio_retention_hours,
            "result_retention_days": resolved_settings.result_retention_days,
            "audio_requires_consent_reference": True,
            "audio_storage": "s3-compatible object storage",
            "metadata_storage": "postgresql",
        }

    @app.get("/analyses")
    async def list_analyses(
        request: Request,
        contact_id: UUID | None = None,
        limit: int = 20,
        before: datetime | None = None,
    ) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise InvalidRequestError(
                "INVALID_LIMIT", "limit must be between 1 and 100"
            )
        storage: PersistenceService = request.app.state.persistence_service
        try:
            manifests = await storage.list(
                contact_id=contact_id,
                limit=limit,
                before=before,
            )
        except PersistenceError as exc:
            raise _persistence_error(exc) from exc
        return {"items": [_manifest_payload(item, detail=False) for item in manifests]}

    @app.get("/analyses/{analysis_id}")
    async def get_analysis(request: Request, analysis_id: UUID) -> dict[str, Any]:
        storage: PersistenceService = request.app.state.persistence_service
        try:
            manifest = await storage.get(analysis_id)
        except PersistenceError as exc:
            raise _persistence_error(exc) from exc
        if manifest is None:
            raise ServiceError(
                "ANALYSIS_NOT_FOUND", "Stored analysis was not found", 404
            )
        return _manifest_payload(manifest, detail=True)

    @app.delete("/analyses/{analysis_id}")
    async def delete_analysis(request: Request, analysis_id: UUID) -> dict[str, str]:
        storage: PersistenceService = request.app.state.persistence_service
        try:
            deleted = await storage.delete(analysis_id)
        except PersistenceError as exc:
            raise _persistence_error(exc) from exc
        if not deleted:
            raise ServiceError(
                "ANALYSIS_NOT_FOUND", "Stored analysis was not found", 404
            )
        return {"analysis_id": str(analysis_id), "status": "deleted"}

    @app.websocket("/ws/analyze")
    async def websocket_analyze(websocket: WebSocket) -> None:
        await _handle_websocket(websocket, resolved_settings)

    return app


async def _receive_start(
    websocket: WebSocket, timeout_seconds: float
) -> WebSocketStart:
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout=timeout_seconds)
    except TimeoutError as exc:
        raise InvalidRequestError(
            "WS_START_TIMEOUT", "A start message was not received in time"
        ) from exc
    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    text = message.get("text")
    if text is None:
        raise InvalidRequestError(
            "WS_PROTOCOL_ERROR", "The first WebSocket message must be start JSON"
        )
    try:
        return WebSocketStart.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InvalidRequestError(
            "WS_PROTOCOL_ERROR", "Invalid WebSocket start message"
        ) from exc


def _stream_duration_seconds(buffer: bytearray, start: WebSocketStart) -> float:
    return _stream_duration_for_bytes(len(buffer), start)


def _stream_duration_for_bytes(byte_count: int, start: WebSocketStart) -> float:
    bytes_per_sample = {
        "pcm_s16le": 2,
        "pcm_s16be": 2,
        "pcm_f32le": 4,
        "mulaw": 1,
        "alaw": 1,
    }[start.encoding]
    return byte_count / (bytes_per_sample * start.channels * start.sample_rate)


async def _send_prediction(
    websocket: WebSocket,
    result: AnalysisResponse,
    *,
    sequence: int,
    is_final: bool,
) -> None:
    payload = result.model_dump(mode="json", exclude_none=True)
    payload.update({"type": "prediction", "sequence": sequence, "is_final": is_final})
    await websocket.send_json(payload)


async def _send_storage_progress(
    websocket: WebSocket,
    handle: PersistenceHandle,
    receipt: PersistenceReceipt,
) -> None:
    await websocket.send_json(
        {
            "type": "storage",
            "analysis_id": str(handle.session_id),
            "persistence": receipt.model_dump(mode="json", exclude_none=True),
        }
    )


async def _send_ws_error(
    websocket: WebSocket, error: ServiceError, request_id: str
) -> None:
    try:
        await websocket.send_json(
            {"type": "error", **_error_payload(error, request_id)}
        )
        close_code = (
            1009
            if error.status_code == 413
            else 1013
            if error.status_code == 503
            else 1008
        )
        await websocket.close(code=close_code, reason=error.code)
    except RuntimeError:
        pass


async def _handle_websocket(websocket: WebSocket, settings: Settings) -> None:
    request_id = str(uuid4())
    buffer = bytearray()
    accepted = False
    completed = False
    handle: PersistenceHandle | None = None
    started_at = time.perf_counter()
    try:
        origin = websocket.headers.get("origin")
        if origin and origin.lower().rstrip("/") not in settings.ws_allowed_origins:
            logger.warning(
                "websocket_origin_rejected",
                extra={"event_data": {"request_id": request_id}},
            )
            await websocket.close(code=1008, reason="WS_ORIGIN_FORBIDDEN")
            return
        await websocket.accept(headers=[(b"x-request-id", request_id.encode("ascii"))])
        accepted = True
        start = await _receive_start(websocket, settings.ws_start_timeout_seconds)
        contact_id = start.contact_id or uuid4()
        source = SourceSpec(
            encoding=start.encoding,
            sample_rate=start.sample_rate,
            channels=start.channels,
            content_type="application/octet-stream",
        )
        service: AnalysisService = websocket.app.state.analysis_service
        storage: PersistenceService = websocket.app.state.persistence_service
        requested_mode, retention_metadata = _retention_policy(
            start.persistence_mode,
            start.consent_reference,
        )
        handle = await storage.start_ws(
            contact_id=contact_id,
            mode=requested_mode,
            request_id=request_id,
            content_type=source.content_type,
            encoding=start.encoding,
            sample_rate=start.sample_rate,
            channels=start.channels,
            metadata=retention_metadata,
        )
        sequence = 0
        last_emitted_at = 0.0
        chunks_received = 0
        chunk_sizes: dict[int, int] = {}
        stored_chunk_bytes: dict[int, int] = {}
        segments_stored = 0
        bytes_stored = 0
        if handle.enabled:
            await websocket.send_json(
                {
                    "type": "started",
                    "contact_id": str(contact_id),
                    "analysis_id": str(handle.session_id),
                    "persistence": _pending_receipt(handle).model_dump(
                        mode="json", exclude_none=True
                    ),
                }
            )

        while True:
            session_remaining = settings.ws_max_session_seconds - (
                time.perf_counter() - started_at
            )
            if session_remaining <= 0:
                raise InputTimeoutError(
                    "WS_SESSION_TIMEOUT",
                    "WebSocket session exceeded its maximum duration",
                )
            receive_timeout = min(settings.ws_idle_timeout_seconds, session_remaining)
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=receive_timeout
                )
            except TimeoutError as exc:
                if session_remaining <= settings.ws_idle_timeout_seconds:
                    raise InputTimeoutError(
                        "WS_SESSION_TIMEOUT",
                        "WebSocket session exceeded its maximum duration",
                    ) from exc
                raise InputTimeoutError(
                    "WS_IDLE_TIMEOUT",
                    "Timed out waiting for a WebSocket audio or control frame",
                ) from exc
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break
            binary = message.get("bytes")
            text = message.get("text")
            if binary is not None:
                projected_bytes = len(buffer) + len(binary)
                if projected_bytes > settings.max_upload_bytes:
                    from app.errors import RequestTooLargeError

                    raise RequestTooLargeError()
                duration = _stream_duration_for_bytes(projected_bytes, start)
                if duration > settings.max_audio_seconds:
                    raise InvalidRequestError(
                        "AUDIO_TOO_LONG",
                        f"Stream exceeds the {settings.max_audio_seconds:g}-second limit",
                    )
                if handle.mode is StorageMode.RESULT_AND_AUDIO:
                    chunk_sizes[chunks_received] = len(binary)
                    stored = await storage.store_chunk(
                        handle,
                        binary,
                        chunk_index=chunks_received,
                    )
                    chunks_received += 1
                    for segment in stored:
                        segments_stored += 1
                        bytes_stored += segment.byte_size
                        for item in segment.logical_chunks:
                            stored_chunk_bytes[item.chunk_index] = (
                                stored_chunk_bytes.get(item.chunk_index, 0)
                                + item.source_byte_end
                                - item.source_byte_start
                            )
                    chunks_stored = sum(
                        stored_chunk_bytes.get(index, 0) >= size
                        for index, size in chunk_sizes.items()
                    )
                    await _send_storage_progress(
                        websocket,
                        handle,
                        _pending_receipt(
                            handle,
                            chunks_received=chunks_received,
                            chunks_stored=chunks_stored,
                            segments_stored=segments_stored,
                            bytes_stored=bytes_stored,
                        ),
                    )
                buffer.extend(binary)
                if (
                    duration >= settings.min_audio_seconds
                    and duration - last_emitted_at >= settings.ws_emit_interval_seconds
                ):
                    result = await service.analyze(
                        payload=buffer,
                        source=source,
                        contact_id=contact_id,
                    )
                    if handle.enabled:
                        result = _with_persistence(
                            result,
                            handle,
                            _pending_receipt(
                                handle,
                                chunks_received=chunks_received,
                                chunks_stored=sum(
                                    stored_chunk_bytes.get(index, 0) >= size
                                    for index, size in chunk_sizes.items()
                                ),
                                segments_stored=segments_stored,
                                bytes_stored=bytes_stored,
                            ),
                        )
                    sequence += 1
                    await _send_prediction(
                        websocket, result, sequence=sequence, is_final=False
                    )
                    last_emitted_at = duration
                continue

            if text is None:
                raise InvalidRequestError(
                    "WS_PROTOCOL_ERROR",
                    "Expected binary audio or a JSON control message",
                )
            try:
                control = json.loads(text)
            except json.JSONDecodeError as exc:
                raise InvalidRequestError(
                    "WS_PROTOCOL_ERROR", "WebSocket control message is not valid JSON"
                ) from exc
            if not isinstance(control, dict):
                raise InvalidRequestError(
                    "WS_PROTOCOL_ERROR",
                    "WebSocket control messages must be JSON objects",
                )
            if control == {"type": "ping"}:
                await websocket.send_json({"type": "pong"})
                continue
            if control.get("type") != "end" or set(control) != {"type"}:
                raise InvalidRequestError(
                    "WS_PROTOCOL_ERROR", "Expected an end or ping control message"
                )
            if not buffer:
                raise InvalidRequestError(
                    "MISSING_AUDIO", "No audio chunks were received"
                )
            result = await service.analyze(
                payload=buffer,
                source=source,
                contact_id=contact_id,
            )
            manifest = await storage.complete(
                handle,
                result=result,
                model_name=service.estimator.name,
            )
            if manifest is not None:
                result = _with_persistence(
                    result,
                    handle,
                    _receipt_from_manifest(manifest),
                )
            sequence += 1
            completed = True
            await _send_prediction(websocket, result, sequence=sequence, is_final=True)
            await websocket.close(code=1000)
            break
    except WebSocketDisconnect:
        pass
    except PersistenceError as exc:
        failure = _persistence_error(exc)
        websocket.app.state.metrics.errors.labels(code=failure.code).inc()
        if accepted:
            await _send_ws_error(websocket, failure, request_id)
    except ServiceError as exc:
        failure = exc
        websocket.app.state.metrics.errors.labels(code=exc.code).inc()
        if accepted:
            await _send_ws_error(websocket, exc, request_id)
    except Exception:
        failure = ServiceError(
            "INTERNAL_ERROR", "An unexpected internal error occurred", 500
        )
        logger.exception(
            "websocket_unhandled_error",
            extra={"event_data": {"request_id": request_id}},
        )
        if accepted:
            await _send_ws_error(websocket, failure, request_id)
    finally:
        if handle is not None and handle.enabled and not completed:
            storage = getattr(websocket.app.state, "persistence_service", None)
            if storage is not None:
                try:
                    await _delete_incomplete_session(storage, handle)
                except asyncio.CancelledError:
                    raise
                except PersistenceError:
                    logger.exception("persistence_websocket_cleanup_failed")
        wipe_buffer(buffer)
        logger.info(
            "websocket_session_completed",
            extra={
                "event_data": {
                    "request_id": request_id,
                    "completed": completed,
                    "duration_ms": round((time.perf_counter() - started_at) * 1_000),
                }
            },
        )


app = create_app()
