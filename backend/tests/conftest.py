from __future__ import annotations

import io
import wave
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.base import RawAttributes


class FakeEstimator:
    name = "fake-estimator"

    def __init__(self) -> None:
        self.calls = 0
        self.warmed = False

    def predict(self, samples: np.ndarray) -> RawAttributes:
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        self.calls += 1
        return RawAttributes(
            gender_probabilities={"female": 0.08, "male": 0.92},
            age_years=40.0,
        )

    def warmup(self) -> None:
        self.warmed = True


@pytest.fixture
def settings() -> Settings:
    return replace(
        Settings(),
        warmup_model=True,
        max_upload_bytes=2 * 1024 * 1024,
        multipart_overhead_bytes=64 * 1024,
        queue_timeout_seconds=0.5,
    )


@pytest.fixture
def fake_estimator() -> FakeEstimator:
    return FakeEstimator()


@pytest.fixture
def client(settings: Settings, fake_estimator: FakeEstimator):
    app = create_app(settings=settings, estimator=fake_estimator)
    with TestClient(app) as test_client:
        yield test_client


def speechlike_pcm(duration: float = 3.0, sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * (125.0 * time + 2.0 * np.sin(2.0 * np.pi * 0.8 * time))
    envelope = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(2.0 * np.pi * 2.3 * time)) ** 2
    waveform = envelope * (
        0.28 * np.sin(phase)
        + 0.13 * np.sin(2.0 * phase)
        + 0.08 * np.sin(3.0 * phase)
        + 0.04 * np.sin(4.0 * phase)
    )
    return np.clip(waveform, -0.95, 0.95).astype(np.float32)


def wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(output, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(pcm.tobytes())
    return output.getvalue()
