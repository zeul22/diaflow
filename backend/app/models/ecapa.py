from __future__ import annotations

import logging
import threading
from contextlib import suppress
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.config import Settings
from app.models.base import RawAttributes
from app.models.kernel_heads import KernelHeadBundle

logger = logging.getLogger(__name__)


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
        model_root = Path(settings.model_root)
        ecapa_root = model_root / "ecapa"
        head_path = model_root / "attribute_heads.npz"
        if not (ecapa_root / "hyperparams.yaml").is_file():
            raise FileNotFoundError(f"Missing ECAPA model under {ecapa_root}")

        torch.set_num_threads(settings.torch_threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(1)
        self._heads = KernelHeadBundle(head_path)
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

    def predict(self, samples: npt.NDArray[np.float32]) -> RawAttributes:
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Expected a non-empty mono waveform")
        with self._lock, self._torch.inference_mode():
            tensor = self._torch.from_numpy(waveform).unsqueeze(0).to(self._device)
            lengths = self._torch.ones(1, device=self._device)
            embedding = self._encoder.encode_batch(tensor, lengths)
            vector = embedding.squeeze().detach().cpu().numpy().astype(np.float32)
            del tensor, lengths, embedding
            return self._heads.predict(vector)

    def warmup(self) -> None:
        logger.info("model_warmup_started")
        warmup_signal = np.zeros(self._sample_rate, dtype=np.float32)
        self.predict(warmup_signal)
        warmup_signal.fill(0.0)
        logger.info("model_warmup_completed")
