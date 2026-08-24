from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

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
from app.models.ecapa import EcapaAttributeEstimator
from app.observability.logging import configure_logging
from app.observability.metrics import Metrics
from app.schemas import AnalysisResponse, ErrorResponse, WebSocketStart

logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
KNOWN_PATHS = {"/analyze", "/healthz", "/readyz", "/metrics"}


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


def create_app(
    *,
    settings: Settings | None = None,
    estimator: AttributeEstimator | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_settings.validate()
    configure_logging(resolved_settings.log_level)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        model = estimator or await asyncio.to_thread(
            EcapaAttributeEstimator, resolved_settings
        )
        if resolved_settings.warmup_model:
            await asyncio.to_thread(model.warmup)
        application.state.analysis_service = AnalysisService(
            settings=resolved_settings,
            estimator=model,
            metrics=metrics,
        )
        logger.info(
            "service_ready",
            extra={
                "event_data": {
                    "service": resolved_settings.service_name,
                    "version": resolved_settings.service_version,
                    "model": model.name,
                }
            },
        )
        try:
            yield
        finally:
            application.state.analysis_service.ready = False

    app = FastAPI(
        title="Voice Contact Attribute Service",
        version=resolved_settings.service_version,
        description=(
            "Low-latency perceived voice-gender and age-bracket estimates with "
            "explicit audio quality gating. Audio is request-scoped and is not stored."
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
            return response
        finally:
            elapsed = time.perf_counter() - started
            metric_path = (
                request.url.path if request.url.path in KNOWN_PATHS else "__other__"
            )
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
        ready = bool(service and service.ready)
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
        try:
            contact_id = resolve_contact_id(request, ingested)
            encoding, sample_rate, channels = request_audio_metadata(request, ingested)
            source = source_spec_from_values(
                encoding=encoding,
                sample_rate=sample_rate,
                channels=channels,
                content_type=ingested.content_type,
            )
            service: AnalysisService = request.app.state.analysis_service
            return await service.analyze(
                payload=ingested.payload,
                source=source,
                contact_id=contact_id,
            )
        finally:
            wipe_buffer(ingested.payload)

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
    bytes_per_sample = {
        "pcm_s16le": 2,
        "pcm_s16be": 2,
        "pcm_f32le": 4,
        "mulaw": 1,
        "alaw": 1,
    }[start.encoding]
    return len(buffer) / (bytes_per_sample * start.channels * start.sample_rate)


async def _send_prediction(
    websocket: WebSocket,
    result: AnalysisResponse,
    *,
    sequence: int,
    is_final: bool,
) -> None:
    payload = result.model_dump(mode="json")
    payload.update({"type": "prediction", "sequence": sequence, "is_final": is_final})
    await websocket.send_json(payload)


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
        sequence = 0
        last_emitted_at = 0.0

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
                if len(buffer) + len(binary) > settings.max_upload_bytes:
                    from app.errors import RequestTooLargeError

                    raise RequestTooLargeError()
                buffer.extend(binary)
                duration = _stream_duration_seconds(buffer, start)
                if duration > settings.max_audio_seconds:
                    raise InvalidRequestError(
                        "AUDIO_TOO_LONG",
                        f"Stream exceeds the {settings.max_audio_seconds:g}-second limit",
                    )
                if (
                    duration >= settings.min_audio_seconds
                    and duration - last_emitted_at >= settings.ws_emit_interval_seconds
                ):
                    result = await service.analyze(
                        payload=buffer,
                        source=source,
                        contact_id=contact_id,
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
            sequence += 1
            await _send_prediction(websocket, result, sequence=sequence, is_final=True)
            completed = True
            await websocket.close(code=1000)
            break
    except WebSocketDisconnect:
        pass
    except ServiceError as exc:
        websocket.app.state.metrics.errors.labels(code=exc.code).inc()
        if accepted:
            await _send_ws_error(websocket, exc, request_id)
    except Exception:
        logger.exception(
            "websocket_unhandled_error",
            extra={"event_data": {"request_id": request_id}},
        )
        if accepted:
            internal = ServiceError(
                "INTERNAL_ERROR", "An unexpected internal error occurred", 500
            )
            await _send_ws_error(websocket, internal, request_id)
    finally:
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
