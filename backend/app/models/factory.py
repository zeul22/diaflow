from __future__ import annotations

from app.config import Settings
from app.models.base import AttributeEstimator


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
