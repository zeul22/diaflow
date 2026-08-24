from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

from app.audio.decoder import AudioDecoder
from app.audio.quality import analyze_quality, prepare_inference_window
from app.audio.types import SourceSpec
from app.config import Settings
from app.errors import ModelUnavailableError, ServiceBusyError
from app.inference.postprocess import build_response, unknown_response
from app.models.base import AttributeEstimator
from app.observability.metrics import Metrics
from app.schemas import AnalysisResponse, AudioQuality

logger = logging.getLogger(__name__)
T = TypeVar("T")


async def _thread_call(
    function: Callable[..., T],
    *args: Any,
    on_cancel: Callable[[T], None] | None = None,
) -> T:
    """Let a worker finish before request-owned memory can be cleared.

    Cancelling ``asyncio.to_thread`` only cancels the awaiter; its OS thread keeps
    running. Shielding and joining it prevents cleanup from mutating audio while
    decode or inference still reads that memory.
    """

    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            result = await asyncio.shield(task)
        except Exception:
            logger.exception("cancelled_worker_failed")
        else:
            if on_cancel is not None:
                on_cancel(result)
        raise


class AnalysisService:
    def __init__(
        self,
        *,
        settings: Settings,
        estimator: AttributeEstimator,
        metrics: Metrics,
    ) -> None:
        self.settings = settings
        self.estimator = estimator
        self.metrics = metrics
        self.decoder = AudioDecoder(settings)
        self._semaphore = asyncio.Semaphore(settings.inference_concurrency)
        self.ready = True

    async def analyze(
        self,
        *,
        payload: bytes | bytearray,
        source: SourceSpec,
        contact_id: UUID,
    ) -> AnalysisResponse:
        if not self.ready:
            raise ModelUnavailableError()
        started = time.perf_counter()
        decoded = await _thread_call(
            self.decoder.decode,
            payload,
            source,
            on_cancel=lambda value: value.samples.fill(0.0),
        )
        samples = decoded.samples
        inference_window = None
        try:
            quality_source = SourceSpec(
                encoding=decoded.source_encoding,
                sample_rate=decoded.source_sample_rate,
                channels=source.channels,
                content_type=source.content_type,
                # FFmpeg is only the decoder. Using it says nothing about the
                # signal quality. Remain conservative only when the container's
                # source stream metadata could not be recovered.
                force_degraded=not decoded.source_metadata_known,
            )
            quality = await _thread_call(
                analyze_quality, samples, quality_source, self.settings
            )
            self.metrics.audio_quality.labels(quality=quality.label.value).inc()
            if quality.label is AudioQuality.INSUFFICIENT:
                elapsed_ms = round((time.perf_counter() - started) * 1_000)
                return unknown_response(contact_id, elapsed_ms, quality.label)

            inference_window = prepare_inference_window(samples, self.settings)
            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self.settings.queue_timeout_seconds,
                )
            except TimeoutError as exc:
                raise ServiceBusyError() from exc
            inference_started = time.perf_counter()
            self.metrics.inference_inflight.inc()
            try:
                raw = await _thread_call(self.estimator.predict, inference_window)
            finally:
                self.metrics.inference_inflight.dec()
                self._semaphore.release()
            inference_seconds = time.perf_counter() - inference_started
            self.metrics.inference_duration.labels(backend=self.estimator.name).observe(
                inference_seconds
            )
            elapsed_ms = round((time.perf_counter() - started) * 1_000)
            logger.info(
                "analysis_completed",
                extra={
                    "event_data": {
                        "processing_ms": elapsed_ms,
                        "inference_ms": round(inference_seconds * 1_000),
                        "audio_seconds": quality.duration_seconds,
                        "audio_quality": quality.label.value,
                        "model": self.estimator.name,
                    }
                },
            )
            return build_response(
                contact_id=contact_id,
                raw=raw,
                quality=quality,
                processing_ms=elapsed_ms,
                settings=self.settings,
            )
        finally:
            samples.fill(0.0)
            if inference_window is not None and inference_window is not samples:
                inference_window.fill(0.0)
