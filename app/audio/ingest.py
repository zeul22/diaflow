from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from fastapi import Request
from python_multipart.multipart import MultipartParser, parse_options_header

from app.config import Settings
from app.errors import (
    InputTimeoutError,
    InvalidRequestError,
    RequestTooLargeError,
    UnsupportedMediaError,
)


@dataclass(slots=True)
class IngestedAudio:
    payload: bytearray
    content_type: str | None
    fields: dict[str, str] = field(default_factory=dict)


def wipe_buffer(buffer: bytearray) -> None:
    if not buffer:
        return
    view = memoryview(buffer)
    view[:] = b"\x00" * len(buffer)
    view.release()
    buffer.clear()


def _validate_content_length(request: Request, maximum: int) -> None:
    value = request.headers.get("content-length")
    if value is None:
        return
    try:
        length = int(value)
    except ValueError as exc:
        raise InvalidRequestError(
            "INVALID_CONTENT_LENGTH", "Content-Length must be an integer"
        ) from exc
    if length < 0:
        raise InvalidRequestError(
            "INVALID_CONTENT_LENGTH", "Content-Length cannot be negative"
        )
    if length > maximum:
        raise RequestTooLargeError()


async def _request_chunks(request: Request, settings: Settings) -> AsyncIterator[bytes]:
    iterator = request.stream().__aiter__()
    while True:
        try:
            chunk = await asyncio.wait_for(
                anext(iterator), timeout=settings.request_idle_timeout_seconds
            )
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise InputTimeoutError(
                "REQUEST_BODY_TIMEOUT",
                "Timed out while waiting for the next request-body chunk",
            ) from exc
        yield chunk


async def _read_raw(request: Request, settings: Settings) -> bytearray:
    _validate_content_length(request, settings.max_upload_bytes)
    output = bytearray()
    try:
        async for chunk in _request_chunks(request, settings):
            if len(output) + len(chunk) > settings.max_upload_bytes:
                raise RequestTooLargeError()
            output.extend(chunk)
    except Exception:
        wipe_buffer(output)
        raise
    if not output:
        raise InvalidRequestError("MISSING_AUDIO", "The request body is empty")
    return output


class _MultipartSink:
    _AUDIO_FIELDS = {"audio", "file"}
    _TEXT_FIELDS = {"contact_id", "encoding", "sample_rate", "channels"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio = bytearray()
        self.fields: dict[str, str] = {}
        self.audio_content_type: str | None = None
        self.current_headers: dict[bytes, bytes] = {}
        self.current_header_name = bytearray()
        self.current_header_value = bytearray()
        self.current_field: str | None = None
        self.current_text = bytearray()
        self.audio_parts = 0
        self.part_count = 0

    def callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
        }

    def on_part_begin(self) -> None:
        self.current_headers = {}
        self.current_header_name.clear()
        self.current_header_value.clear()
        self.current_field = None
        self.current_text.clear()
        self.part_count += 1
        if self.part_count > 8:
            raise InvalidRequestError(
                "TOO_MANY_PARTS", "Multipart requests may contain at most 8 parts"
            )

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.current_header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.current_header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        name = bytes(self.current_header_name).strip().lower()
        value = bytes(self.current_header_value).strip()
        if len(name) > 128 or len(value) > 2_048:
            raise InvalidRequestError(
                "INVALID_MULTIPART_HEADER", "Multipart header is too large"
            )
        self.current_headers[name] = value
        self.current_header_name.clear()
        self.current_header_value.clear()

    def on_headers_finished(self) -> None:
        disposition = self.current_headers.get(b"content-disposition")
        if disposition is None:
            raise InvalidRequestError(
                "INVALID_MULTIPART", "Each multipart part needs Content-Disposition"
            )
        kind, options = parse_options_header(disposition)
        if kind.lower() != b"form-data" or b"name" not in options:
            raise InvalidRequestError(
                "INVALID_MULTIPART", "Invalid multipart Content-Disposition"
            )
        try:
            self.current_field = options[b"name"].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidRequestError(
                "INVALID_MULTIPART", "Multipart field names must be UTF-8"
            ) from exc
        if self.current_field in self._AUDIO_FIELDS:
            self.audio_parts += 1
            if self.audio_parts > 1:
                raise InvalidRequestError(
                    "MULTIPLE_AUDIO_PARTS", "Send exactly one audio file"
                )
            content_type = self.current_headers.get(b"content-type")
            if content_type:
                self.audio_content_type = content_type.decode(
                    "latin-1", errors="replace"
                )

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self.current_field in self._AUDIO_FIELDS:
            if len(self.audio) + len(chunk) > self.settings.max_upload_bytes:
                raise RequestTooLargeError()
            self.audio.extend(chunk)
        elif self.current_field in self._TEXT_FIELDS:
            if len(self.current_text) + len(chunk) > 256:
                raise InvalidRequestError(
                    "INVALID_FORM_FIELD", "Multipart text field is too large"
                )
            self.current_text.extend(chunk)

    def on_part_end(self) -> None:
        if self.current_field in self._TEXT_FIELDS:
            if self.current_field in self.fields:
                raise InvalidRequestError(
                    "DUPLICATE_FORM_FIELD",
                    f"Multipart field '{self.current_field}' was provided more than once",
                )
            try:
                value = self.current_text.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise InvalidRequestError(
                    "INVALID_FORM_FIELD", "Multipart text fields must be UTF-8"
                ) from exc
            if value:
                self.fields[self.current_field] = value


async def _read_multipart(
    request: Request, content_type: str, settings: Settings
) -> IngestedAudio:
    total_limit = settings.max_upload_bytes + settings.multipart_overhead_bytes
    _validate_content_length(request, total_limit)
    _, options = parse_options_header(content_type.encode("latin-1"))
    boundary = options.get(b"boundary")
    if not boundary or len(boundary) > 200:
        raise InvalidRequestError(
            "INVALID_MULTIPART", "Multipart boundary is missing or invalid"
        )

    sink = _MultipartSink(settings)
    parser = MultipartParser(boundary, sink.callbacks(), max_size=total_limit)
    received = 0
    try:
        async for chunk in _request_chunks(request, settings):
            received += len(chunk)
            if received > total_limit:
                raise RequestTooLargeError()
            parser.write(chunk)
        parser.finalize()
    except Exception:
        wipe_buffer(sink.audio)
        raise
    if not sink.audio:
        raise InvalidRequestError(
            "MISSING_AUDIO", "Multipart request must include a non-empty 'audio' field"
        )
    return IngestedAudio(
        payload=sink.audio,
        content_type=sink.audio_content_type,
        fields=sink.fields,
    )


async def ingest_request(request: Request, settings: Settings) -> IngestedAudio:
    content_type = request.headers.get("content-type", "application/octet-stream")
    media_type = content_type.split(";", 1)[0].strip().lower()
    content_encoding = request.headers.get("content-encoding", "identity").lower()
    if content_encoding not in {"", "identity"}:
        raise UnsupportedMediaError("HTTP Content-Encoding is not supported")
    if media_type == "multipart/form-data":
        return await _read_multipart(request, content_type, settings)
    if not (
        media_type.startswith("audio/") or media_type == "application/octet-stream"
    ):
        raise UnsupportedMediaError()
    payload = await _read_raw(request, settings)
    return IngestedAudio(payload=payload, content_type=content_type)


def resolve_contact_id(request: Request, ingested: IngestedAudio) -> UUID:
    candidates = [
        request.query_params.get("contact_id"),
        request.headers.get("x-contact-id"),
        ingested.fields.get("contact_id"),
    ]
    values = [
        candidate.strip() for candidate in candidates if candidate and candidate.strip()
    ]
    if not values:
        return uuid4()
    if any(value != values[0] for value in values[1:]):
        raise InvalidRequestError(
            "CONFLICTING_CONTACT_ID", "contact_id values do not match"
        )
    try:
        return UUID(values[0])
    except ValueError as exc:
        raise InvalidRequestError(
            "INVALID_CONTACT_ID", "contact_id must be a UUID"
        ) from exc


def request_audio_metadata(
    request: Request, ingested: IngestedAudio
) -> tuple[str | None, str | int | None, str | int | None]:
    encoding = (
        request.query_params.get("encoding")
        or request.headers.get("x-audio-encoding")
        or ingested.fields.get("encoding")
    )
    sample_rate = (
        request.query_params.get("sample_rate")
        or request.headers.get("x-audio-sample-rate")
        or ingested.fields.get("sample_rate")
    )
    channels = (
        request.query_params.get("channels")
        or request.headers.get("x-audio-channels")
        or ingested.fields.get("channels")
    )
    return encoding, sample_rate, channels
