from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

from app.audio.decoder import AudioDecoder
from app.audio.enhance import enhance_window
from app.audio.quality import analyze_quality, prepare_inference_window
from app.audio.types import SourceSpec
from app.config import Settings
from app.errors import ModelUnavailableError, ServiceBusyError
from app.inference.postprocess import (
    build_response,
    speaker_evidence_is_weak,
    unknown_response,
)
from app.models.base import AttributeEstimator
from app.models.pool import EstimatorPool, as_pool
from app.observability.metrics import Metrics
from app.schemas import AnalysisResponse, AudioQuality, LanguagePrediction

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
        estimator: AttributeEstimator | EstimatorPool,
        metrics: Metrics,
        language: EstimatorPool | None = None,
    ) -> None:
        self.settings = settings
        self.estimator = estimator
        self.metrics = metrics
        self.decoder = AudioDecoder(settings)
        self._language_pool = language
        # Concurrency is the number of loaded replicas, not a semaphore over one
        # locked instance. A single injected estimator therefore serves one
        # inference at a time no matter what the setting says.
        self._pool = as_pool(estimator)
        self.ready = True

    @property
    def language_enabled(self) -> bool:
        return self._language_pool is not None

    async def _identify_language(
        self, window: Any
    ) -> tuple[LanguagePrediction | None, float]:
        """Best-effort language identification for one window.

        Language is explicitly best-effort, so a busy language pool degrades to
        no language field rather than failing the whole analysis.
        """

        pool = self._language_pool
        if pool is None:
            return None, 0.0
        try:
            identifier = await pool.acquire(self.settings.queue_timeout_seconds)
        except TimeoutError:
            self.metrics.language_skipped.labels(reason="busy").inc()
            logger.warning("language_identification_skipped")
            return None, 0.0
        started = time.perf_counter()
        try:
            estimate = await _thread_call(identifier.identify, window)
        finally:
            pool.release(identifier)
        elapsed = time.perf_counter() - started
        self.metrics.language_duration.labels(backend=pool.name).observe(elapsed)
        self.metrics.language_results.labels(
            outcome="unknown" if estimate.code == "unknown" else "determined"
        ).inc()
        return (
            LanguagePrediction(
                prediction=estimate.code,
                confidence=round(estimate.confidence, 4),
            ),
            elapsed,
        )

    async def analyze(
        self,
        *,
        payload: bytes | bytearray,
        source: SourceSpec,
        contact_id: UUID,
        settled_language: LanguagePrediction | None = None,
        transport_degraded: bool = False,
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
        enhanced = None
        try:
            quality_source = SourceSpec(
                encoding=decoded.source_encoding,
                sample_rate=decoded.source_sample_rate,
                channels=source.channels,
                content_type=source.content_type,
                # FFmpeg is only the decoder. Using it says nothing about the
                # signal quality. Remain conservative only when the container's
                # source stream metadata could not be recovered.
                # Concealed packet loss is synthetic audio; past a threshold the
                # caller must be told the signal was partly reconstructed.
                force_degraded=not decoded.source_metadata_known or transport_degraded,
            )
            quality = await _thread_call(
                analyze_quality, samples, quality_source, self.settings
            )
            self.metrics.audio_quality.labels(quality=quality.label.value).inc()
            if quality.label is AudioQuality.INSUFFICIENT:
                elapsed_ms = round((time.perf_counter() - started) * 1_000)
                return unknown_response(contact_id, elapsed_ms, quality.label)

            inference_window = prepare_inference_window(samples, self.settings)
            # Enhancement runs after the quality gate, never before: the gate has
            # to judge the signal as it arrived, or a near-silent recording gets
            # amplified into looking usable.
            enhanced, enhancement = await _thread_call(
                enhance_window, inference_window, self.settings
            )
            queue_started = time.perf_counter()
            try:
                replica = await self._pool.acquire(self.settings.queue_timeout_seconds)
            except TimeoutError as exc:
                self.metrics.queue_wait.observe(time.perf_counter() - queue_started)
                self.metrics.queue_rejections.inc()
                raise ServiceBusyError() from exc
            self.metrics.queue_wait.observe(time.perf_counter() - queue_started)
            inference_started = time.perf_counter()
            self.metrics.inference_inflight.inc()
            try:
                raw = await _thread_call(replica.predict, enhanced)
            finally:
                self.metrics.inference_inflight.dec()
                self._pool.release(replica)
            inference_seconds = time.perf_counter() - inference_started
            self.metrics.inference_duration.labels(backend=self.estimator.name).observe(
                inference_seconds
            )
            multi_speaker = speaker_evidence_is_weak(raw, self.settings)
            if multi_speaker:
                self.metrics.speaker_homogeneity_downgrades.inc()
            language = settled_language
            language_seconds = 0.0
            if language is None:
                language, language_seconds = await self._identify_language(enhanced)
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
                        # Deliberately no predicted values here. The identified
                        # language is an inference about the caller, and
                        # docs/PRIVACY.md promises logs carry no predictions;
                        # coverage is tracked in an aggregate metric instead.
                        "language_ms": round(language_seconds * 1_000),
                        "agc_gain_db": enhancement.gain_db,
                        "denoised": enhancement.denoised,
                        "ensemble_windows": raw.ensemble_windows,
                        # Signal statistics, not attributes of the speaker: model
                        # disagreement in years and sub-window similarity.
                        "age_spread_years": raw.age_spread_years,
                        "speaker_homogeneity": raw.speaker_homogeneity,
                        "multi_speaker_suspected": multi_speaker,
                    }
                },
            )
            return build_response(
                contact_id=contact_id,
                raw=raw,
                quality=quality,
                processing_ms=elapsed_ms,
                settings=self.settings,
                language=language,
            )
        finally:
            samples.fill(0.0)
            for buffer in (inference_window, enhanced):
                if buffer is not None and buffer is not samples:
                    buffer.fill(0.0)
