from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.audio.decoder import AudioDecoder
from app.audio.quality import analyze_quality
from app.audio.types import SourceSpec
from app.schemas import AudioQuality
from tests.conftest import speechlike_pcm, wav_bytes


def test_native_wav_decoder_resamples(settings) -> None:
    original = speechlike_pcm(duration=2.0, sample_rate=8_000)
    decoded = AudioDecoder(settings).decode(
        wav_bytes(original, sample_rate=8_000),
        SourceSpec(encoding="wav", sample_rate=8_000),
    )

    assert decoded.sample_rate == 16_000
    assert decoded.source_sample_rate == 8_000
    assert abs(decoded.duration_seconds - 2.0) < 0.01
    assert decoded.samples.dtype == np.float32


def test_quality_gate_marks_narrowband_as_degraded(settings) -> None:
    samples = speechlike_pcm()
    quality = analyze_quality(
        samples,
        SourceSpec(encoding="mulaw", sample_rate=8_000),
        settings,
    )

    assert quality.label in {AudioQuality.DEGRADED, AudioQuality.INSUFFICIENT}


def test_raw_pcm_alignment_is_validated(settings) -> None:
    decoder = AudioDecoder(replace(settings, max_audio_seconds=5.0))
    try:
        decoder.decode(
            b"\x00",
            SourceSpec(encoding="pcm_s16le", sample_rate=16_000, channels=1),
        )
    except Exception as exc:
        assert getattr(exc, "code", None) == "INVALID_AUDIO"
    else:  # pragma: no cover
        raise AssertionError("misaligned PCM was accepted")


def test_wav_sample_rate_is_bounded_before_resampling(settings) -> None:
    decoder = AudioDecoder(settings)
    payload = wav_bytes(np.zeros(2, dtype=np.float32), sample_rate=1)

    with pytest.raises(Exception) as caught:
        decoder.decode(payload, SourceSpec(encoding="wav"))

    assert getattr(caught.value, "code", None) == "INVALID_AUDIO"


def test_wav_duration_is_bounded_before_resampling(settings, monkeypatch) -> None:
    limited = replace(settings, max_audio_seconds=2.0)
    decoder = AudioDecoder(limited)
    payload = wav_bytes(np.zeros(24_000, dtype=np.float32), sample_rate=8_000)

    def fail_if_called(*_args) -> np.ndarray:
        raise AssertionError("resampler ran before the source-duration check")

    monkeypatch.setattr(decoder, "_resample_linear", fail_if_called)
    with pytest.raises(Exception) as caught:
        decoder.decode(payload, SourceSpec(encoding="wav"))

    assert getattr(caught.value, "code", None) == "INVALID_AUDIO"
