from __future__ import annotations

import logging

from app.config import Settings
from app.models.base import AttributeEstimator
from app.models.pool import EstimatorPool, warn_on_thread_oversubscription

logger = logging.getLogger(__name__)


def create_estimator(settings: Settings) -> AttributeEstimator:
    """Build the one model selected for this deployment.

    Model selection is deliberately not request-controlled: separate replicas
    can be canaried and capacity-planned without letting callers load arbitrary
    artifacts or create unbounded metric labels.
    """

    if settings.model_backend == "ecapa":
        from app.models.ecapa import EcapaAttributeEstimator

        return EcapaAttributeEstimator(settings)
    if settings.model_backend == "wavlm_onnx":
        from app.models.wavlm import WavlmAttributeEstimator

        return WavlmAttributeEstimator(settings)
    raise ValueError(f"Unsupported model backend: {settings.model_backend}")


def create_estimator_pool(settings: Settings) -> EstimatorPool:
    """Load ``INFERENCE_CONCURRENCY`` replicas of the selected model.

    Each replica holds its own weights, so this multiplies resident memory. It
    is the honest cost of genuinely concurrent inference on this architecture;
    the alternative for high concurrency is the GPU micro-batching path in
    ADR-002, not a larger semaphore over one instance.
    """

    replicas = settings.inference_concurrency
    warn_on_thread_oversubscription(replicas, settings.torch_threads)
    pool = EstimatorPool([create_estimator(settings) for _ in range(replicas)])
    logger.info(
        "model_pool_loaded",
        extra={"event_data": {"backend": pool.name, "replicas": pool.size}},
    )
    return pool


def create_language_pool(settings: Settings) -> EstimatorPool | None:
    """Load language-identification replicas, or nothing when disabled.

    Language identification is a separate encoder rather than another head on
    the speaker embedding, so it is off unless a deployment asks for it and
    accepts the extra forward pass.
    """

    if settings.language_backend == "none":
        return None
    if settings.language_backend != "voxlingua_ecapa":
        raise ValueError(f"Unsupported language backend: {settings.language_backend}")

    from app.models.language import VoxLinguaLanguageIdentifier

    pool = EstimatorPool(
        [
            VoxLinguaLanguageIdentifier(settings)
            for _ in range(settings.inference_concurrency)
        ]
    )
    logger.info(
        "language_pool_loaded",
        extra={"event_data": {"backend": pool.name, "replicas": pool.size}},
    )
    return pool
