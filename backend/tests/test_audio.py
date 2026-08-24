from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.audio.decoder import AudioDecoder, DecodedAudio
from app.audio.quality import analyze_quality, prepare_inference_window
from app.audio.types import SourceSpec
from app.inference.postprocess import QualitySummary
from app.inference.service import AnalysisService
from app.observability.metrics import Metrics
from app.schemas import AudioQuality
from tests.conftest import FakeEstimator, speechlike_pcm, wav_bytes


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


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        # Verified bit-exact against FFmpeg's decoders across all 256 codes.
        # These hand-written decoders carry the telephony path, which is the
        # realistic logistics input, so the companding tables are pinned here.
        (
            "mulaw",
            {
                0: -0.9803467,
                1: -0.9490967,
                127: 0.0,
                128: 0.9803467,
                200: 0.041870117,
                254: 0.00024414062,
            },
        ),
        (
            "alaw",
            {
                0: -0.16796875,
                1: -0.16015625,
                127: -0.025878906,
                128: 0.16796875,
                200: 0.014404297,
                254: 0.026855469,
            },
        ),
    ],
)
def test_g711_companding_matches_the_reference_decoders(encoding, expected) -> None:
    decode = {
        "mulaw": AudioDecoder._decode_mulaw,
        "alaw": AudioDecoder._decode_alaw,
    }[encoding]

    decoded = decode(bytes(range(256)))

    assert decoded.dtype == np.float32
    assert decoded.size == 256
    for code, value in expected.items():
        assert decoded[code] == pytest.approx(value, abs=1e-6), f"code {code}"
    # Companding is symmetric about the sign bit and stays in range.
    assert np.abs(decoded).max() <= 1.0
    assert decoded[:128].min() == pytest.approx(-decoded[128:].max(), abs=1e-6)


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

    monkeypatch.setattr("app.audio.decoder.resample", fail_if_called)
    with pytest.raises(Exception) as caught:
        decoder.decode(payload, SourceSpec(encoding="wav"))

    assert getattr(caught.value, "code", None) == "INVALID_AUDIO"


def test_ffmpeg_decode_carries_input_codec_and_sample_rate(
    settings, monkeypatch
) -> None:
    samples = speechlike_pcm(duration=1.5)

    def fake_run(command, **kwargs):
        assert command[command.index("-loglevel") + 1] == "info"
        assert kwargs["timeout"] == settings.decode_timeout_seconds
        return SimpleNamespace(
            returncode=0,
            stdout=samples.astype("<f4").tobytes(),
            stderr=(
                b"Input #0, mov,mp4,m4a, from 'pipe:0':\n"
                b"  Stream #0:0: Audio: aac (LC), 48000 Hz, mono, fltp\n"
                b"Output #0, f32le, to 'pipe:1':\n"
                b"  Stream #0:0: Audio: pcm_f32le, 16000 Hz, mono, flt\n"
            ),
        )

    monkeypatch.setattr("app.audio.decoder.subprocess.run", fake_run)
    decoded = AudioDecoder(settings).decode(b"mock-m4a", SourceSpec(encoding="auto"))

    assert decoded.used_ffmpeg is True
    assert decoded.source_encoding == "aac"
    assert decoded.source_sample_rate == 48_000
    assert decoded.source_metadata_known is True


@pytest.mark.parametrize(
    ("metadata_known", "expected_quality"),
    [
        (True, AudioQuality.GOOD),
        (False, AudioQuality.DEGRADED),
    ],
)
def test_service_does_not_treat_ffmpeg_as_a_quality_failure(
    settings, monkeypatch, metadata_known, expected_quality
) -> None:
    service = AnalysisService(
        settings=settings,
        estimator=FakeEstimator(),
        metrics=Metrics(),
    )
    decoded = DecodedAudio(
        samples=speechlike_pcm(),
        sample_rate=16_000,
        source_sample_rate=48_000 if metadata_known else 16_000,
        source_encoding="aac" if metadata_known else "auto",
        source_metadata_known=metadata_known,
        used_ffmpeg=True,
    )
    monkeypatch.setattr(service.decoder, "decode", lambda *_args: decoded)

    def fake_quality(_samples, source, _settings):
        label = (
            AudioQuality.DEGRADED if source.is_quality_limited else AudioQuality.GOOD
        )
        return QualitySummary(
            label=label,
            duration_seconds=3.0,
            voiced_seconds=2.5,
            speech_ratio=0.8,
            snr_db=20.0,
            clipping_ratio=0.0,
            rms_dbfs=-20.0,
            spectral_flatness=0.1,
            low_frequency_ratio=0.2,
        )

    monkeypatch.setattr("app.inference.service.analyze_quality", fake_quality)
    response = asyncio.run(
        service.analyze(
            payload=b"mock-m4a",
            source=SourceSpec(encoding="auto", content_type="audio/mp4"),
            contact_id=uuid4(),
        )
    )

    assert response.audio_quality is expected_quality


def test_inference_window_prefers_speech_over_louder_stationary_noise(settings) -> None:
    rng = np.random.default_rng(42)
    stationary_noise = rng.normal(0.0, 0.42, 5 * 16_000).astype(np.float32)
    speech = (0.55 * speechlike_pcm(duration=5.0)).astype(np.float32)
    recording = np.concatenate((stationary_noise, speech))

    selected = prepare_inference_window(recording, settings)
    selected_quality = analyze_quality(
        selected,
        SourceSpec(encoding="pcm_f32le", sample_rate=16_000),
        settings,
    )

    assert selected.size == 5 * 16_000
    assert selected_quality.spectral_flatness < 0.30
