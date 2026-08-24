from __future__ import annotations

import logging
import math
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from app.config import Settings
from app.models.base import RawAttributes

logger = logging.getLogger(__name__)

GENDER_LABELS = ("female", "male")
AGE_LABELS = ("18-30", "31-45", "46-60", "60+")
EXPECTED_OUTPUTS = {"gender_logits", "age_logits"}


def _softmax(values: npt.ArrayLike, expected_size: int) -> npt.NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != expected_size or not np.all(np.isfinite(array)):
        raise RuntimeError("The WavLM artifact returned invalid logits")
    shifted = array - float(np.max(array))
    exponentials = np.exp(shifted)
    total = float(exponentials.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise RuntimeError("The WavLM artifact returned invalid probabilities")
    return exponentials / total


class WavlmAttributeEstimator:
    """Runtime for an owned, calibrated WavLM Base+ ONNX artifact.

    The ONNX graph is intentionally an application-owned artifact rather than a
    public demographic checkpoint. It must contain the WavLM encoder, the two
    logistics-trained heads, and their held-out calibration. Its stable contract
    is one float32 ``audio`` input and ``gender_logits``/``age_logits`` outputs.
    """

    def __init__(self, settings: Settings, *, session: Any | None = None) -> None:
        self._lock = threading.Lock()
        self._sample_rate = settings.target_sample_rate
        self._revision = settings.wavlm_model_revision
        model_path = Path(settings.wavlm_model_path)

        if session is None:
            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Missing owned WavLM ONNX artifact at {model_path}"
                )
            try:
                # Official ONNX Runtime builds enable telemetry by default.
                # Set this before importing ORT so no uploader, event, or
                # persistent device identifier is initialized.
                os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - deployment-only path
                raise RuntimeError(
                    "The wavlm_onnx backend requires the project's 'wavlm' extra"
                ) from exc
            options = ort.SessionOptions()
            options.intra_op_num_threads = settings.onnx_intra_threads
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if settings.model_device.startswith("cuda")
                else ["CPUExecutionProvider"]
            )
            session = ort.InferenceSession(
                str(model_path), sess_options=options, providers=providers
            )

        inputs = session.get_inputs()
        outputs = {item.name for item in session.get_outputs()}
        if len(inputs) != 1 or inputs[0].name != "audio":
            raise RuntimeError("WavLM ONNX input must be named 'audio'")
        if not EXPECTED_OUTPUTS.issubset(outputs):
            raise RuntimeError(
                "WavLM ONNX outputs must include gender_logits and age_logits"
            )
        self._session = session

    @property
    def name(self) -> str:
        return f"wavlm-base-plus-domain-{self._revision}"

    def predict(self, samples: npt.NDArray[np.float32]) -> RawAttributes:
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Expected a non-empty mono waveform")
        if not np.all(np.isfinite(waveform)):
            raise ValueError("Waveform contains non-finite samples")
        with self._lock:
            gender_logits, age_logits = self._session.run(
                ["gender_logits", "age_logits"], {"audio": waveform[None, :]}
            )
        gender = _softmax(gender_logits, len(GENDER_LABELS))
        age = _softmax(age_logits, len(AGE_LABELS))
        return RawAttributes(
            gender_probabilities=dict(zip(GENDER_LABELS, gender, strict=True)),
            age_bracket_probabilities=dict(zip(AGE_LABELS, age, strict=True)),
        )

    def warmup(self) -> None:
        logger.info("model_warmup_started")
        signal = np.zeros(self._sample_rate, dtype=np.float32)
        self.predict(signal)
        signal.fill(0.0)
        logger.info("model_warmup_completed")
