from __future__ import annotations

import logging
import math
import threading
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.config import Settings
from app.models.base import RawAttributes
from app.models.kernel_heads import KernelHeadBundle

logger = logging.getLogger(__name__)


def sub_window_bounds(
    size: int, count: int, minimum_samples: int
) -> tuple[tuple[int, int], ...]:
    """Split one inference window into equal, overlapping sub-windows.

    Equal lengths let every sub-window be encoded in a single batched forward
    pass. When the segment is too short to carve, the caller falls back to a
    single view and reports no dispersion rather than an invented one.
    """

    if size <= 0:
        raise ValueError("Cannot split an empty window")
    if count <= 1:
        return ((0, size),)
    length = max(minimum_samples, (size * 2) // (count + 1))
    if length >= size:
        return ((0, size),)
    hop = (size - length) // (count - 1)
    if hop <= 0:
        return ((0, size),)
    return tuple((index * hop, index * hop + length) for index in range(count))


def min_pairwise_cosine(embeddings: npt.NDArray[np.float32]) -> float | None:
    """Lowest pairwise cosine similarity between sub-window embeddings.

    ECAPA is trained to separate speakers, so a segment holding the caller plus
    an agent, dispatcher, or bystander shows a low similarity between windows.
    This is a coarse single-speaker check, not diarization.
    """

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return None
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        return None
    unit = embeddings / norms[:, None]
    similarity = unit @ unit.T
    lower = np.tril_indices(embeddings.shape[0], k=-1)
    return float(np.min(similarity[lower]))


def dispersion_years(values: list[float | None]) -> float | None:
    """Population standard deviation of the per-window age estimates."""

    finite = [value for value in values if value is not None and math.isfinite(value)]
    if len(finite) < 2:
        return None
    return float(np.std(np.asarray(finite, dtype=np.float64)))


class EcapaAttributeEstimator:
    def __init__(self, settings: Settings) -> None:
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - exercised by container startup
            raise RuntimeError(
                "The ECAPA backend requires the project's 'model' dependencies"
            ) from exc

        self._torch = torch
        self._lock = threading.Lock()
        self._device = settings.model_device
        self._sample_rate = settings.target_sample_rate
        self._ensemble_windows = settings.ensemble_windows
        self._minimum_window_samples = max(
            1,
            int(settings.ensemble_min_window_seconds * settings.target_sample_rate),
        )
        model_root = Path(settings.model_root)
        ecapa_root = model_root / "ecapa"
        head_path = model_root / "attribute_heads.npz"
        if not (ecapa_root / "hyperparams.yaml").is_file():
            raise FileNotFoundError(f"Missing ECAPA model under {ecapa_root}")

        torch.set_num_threads(settings.torch_threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        self._heads = KernelHeadBundle(
            head_path,
            require_calibrated_gender=settings.require_calibrated_gender,
        )
        cache_dir = Path("/tmp/speechbrain-ecapa")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = EncoderClassifier.from_hparams(
            source=str(ecapa_root),
            savedir=str(cache_dir),
            run_opts={"device": self._device},
            # The upstream YAML names its Hugging Face source. Override it so
            # runtime startup remains strictly offline and uses pinned files.
            overrides={"pretrained_path": str(ecapa_root)},
        )

    @property
    def name(self) -> str:
        return "speechbrain-ecapa-svm-svr"

    def _embed(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        tensor = self._torch.from_numpy(batch).to(self._device)
        lengths = self._torch.ones(batch.shape[0], device=self._device)
        embedding = self._encoder.encode_batch(tensor, lengths)
        vectors = (
            embedding.reshape(batch.shape[0], -1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        del tensor, lengths, embedding
        return vectors

    def predict(self, samples: npt.NDArray[np.float32]) -> RawAttributes:
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Expected a non-empty mono waveform")
        bounds = sub_window_bounds(
            waveform.size, self._ensemble_windows, self._minimum_window_samples
        )
        batch = np.stack([waveform[start:end] for start, end in bounds])
        try:
            with self._lock, self._torch.inference_mode():
                embeddings = self._embed(batch)
            # The heads were fitted on utterance-level embeddings, so the mean
            # embedding stays closest to their training distribution and gives
            # the point estimate. The per-window predictions only supply spread.
            attributes = self._heads.predict(embeddings.mean(axis=0))
            if embeddings.shape[0] < 2:
                return replace(attributes, ensemble_windows=1)
            per_window = [
                self._heads.predict(embeddings[index]).age_years
                for index in range(embeddings.shape[0])
            ]
            return replace(
                attributes,
                age_spread_years=dispersion_years(per_window),
                speaker_homogeneity=min_pairwise_cosine(embeddings),
                ensemble_windows=int(embeddings.shape[0]),
            )
        finally:
            batch.fill(0.0)

    def warmup(self) -> None:
        logger.info("model_warmup_started")
        # Warm the batched ensemble path, not just a single forward pass, so the
        # first real request does not pay for allocating the wider batch.
        warmup_samples = max(self._sample_rate, self._minimum_window_samples * 2)
        warmup_signal = np.zeros(warmup_samples, dtype=np.float32)
        self.predict(warmup_signal)
        warmup_signal.fill(0.0)
        logger.info("model_warmup_completed")
