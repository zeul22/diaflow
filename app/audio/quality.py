from __future__ import annotations

import numpy as np
import numpy.typing as npt

from app.audio.types import SourceSpec
from app.config import Settings
from app.inference.postprocess import QualitySummary
from app.schemas import AudioQuality


def _frames(
    samples: npt.NDArray[np.float32], frame_size: int, hop_size: int
) -> npt.NDArray[np.float32]:
    if samples.size < frame_size:
        padded = np.pad(samples, (0, frame_size - samples.size))
        return padded.reshape(1, frame_size)
    windows = np.lib.stride_tricks.sliding_window_view(samples, frame_size)
    return np.asarray(windows[::hop_size], dtype=np.float32)


def analyze_quality(
    samples: npt.NDArray[np.float32],
    source: SourceSpec,
    settings: Settings,
) -> QualitySummary:
    sample_rate = settings.target_sample_rate
    duration = samples.size / sample_rate
    centered = samples - float(np.mean(samples))
    overall_rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
    rms_dbfs = 20.0 * np.log10(max(overall_rms, 1e-8))
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.995))

    frame_size = int(0.025 * sample_rate)
    hop_size = int(0.010 * sample_rate)
    framed = _frames(centered, frame_size, hop_size)
    frame_rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-8))
    zero_crossing = np.mean(
        np.signbit(framed[:, 1:]) != np.signbit(framed[:, :-1]), axis=1
    )

    spectrum = np.abs(np.fft.rfft(framed * np.hanning(frame_size), axis=1)) ** 2
    arithmetic = np.mean(spectrum + 1e-12, axis=1)
    geometric = np.exp(np.mean(np.log(spectrum + 1e-12), axis=1))
    flatness = geometric / arithmetic
    median_flatness = float(np.median(flatness))

    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    low_mask = frequencies < 250.0
    total_power = float(np.sum(spectrum) + 1e-12)
    low_frequency_ratio = float(np.sum(spectrum[:, low_mask]) / total_power)

    p20 = float(np.percentile(frame_db, 20))
    p90 = float(np.percentile(frame_db, 90))
    dynamic_range = p90 - p20
    energy_threshold = -44.0 if dynamic_range < 6.0 else max(-48.0, p20 + 5.0)
    voiced = (
        (frame_db > energy_threshold)
        & (flatness < 0.50)
        & (zero_crossing > 0.003)
        & (zero_crossing < 0.35)
    )
    if not np.any(voiced) and rms_dbfs > -36.0 and median_flatness < 0.30:
        voiced = frame_db > -44.0

    speech_ratio = float(np.mean(voiced))
    voiced_seconds = min(duration, float(np.sum(voiced) * hop_size / sample_rate))
    if np.any(voiced) and np.any(~voiced):
        speech_level = float(np.median(frame_rms[voiced]))
        noise_level = float(np.median(frame_rms[~voiced]))
        snr_db = float(20.0 * np.log10((speech_level + 1e-8) / (noise_level + 1e-8)))
    elif speech_ratio >= 0.75 and median_flatness < 0.35:
        snr_db = 25.0
    else:
        snr_db = 0.0
    snr_db = float(np.clip(snr_db, -10.0, 40.0))

    insufficient = (
        duration < settings.min_audio_seconds
        or voiced_seconds < settings.min_voiced_seconds
        or speech_ratio < 0.08
        or rms_dbfs < -48.0
    )
    degraded = (
        duration < 2.0
        or speech_ratio < 0.22
        or snr_db < 8.0
        or clipping_ratio > 0.02
        or low_frequency_ratio > 0.55
        or source.is_quality_limited
    )
    if insufficient:
        label = AudioQuality.INSUFFICIENT
    elif degraded:
        label = AudioQuality.DEGRADED
    else:
        label = AudioQuality.GOOD

    return QualitySummary(
        label=label,
        duration_seconds=round(duration, 4),
        voiced_seconds=round(voiced_seconds, 4),
        speech_ratio=round(speech_ratio, 4),
        snr_db=round(snr_db, 2),
        clipping_ratio=round(clipping_ratio, 6),
        rms_dbfs=round(float(rms_dbfs), 2),
        spectral_flatness=round(median_flatness, 4),
        low_frequency_ratio=round(low_frequency_ratio, 4),
    )


def prepare_inference_window(
    samples: npt.NDArray[np.float32], settings: Settings
) -> npt.NDArray[np.float32]:
    samples = np.asarray(samples, dtype=np.float32)
    window_samples = int(
        settings.inference_window_seconds * settings.target_sample_rate
    )
    centered = samples - float(np.mean(samples))
    if centered.size <= window_samples:
        return np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)

    block_size = max(1, settings.target_sample_rate // 10)
    usable = centered[: centered.size - (centered.size % block_size)]
    blocks = usable.reshape(-1, block_size)
    block_energy = np.mean(blocks * blocks, axis=1)
    block_count = max(1, int(settings.inference_window_seconds * 10))

    # Rank windows by speech-like evidence instead of raw loudness. Logistics
    # recordings commonly contain stationary engine or road noise that can be
    # louder than the caller; an energy-only selector systematically chose it.
    block_rms = np.sqrt(block_energy + 1e-12)
    block_db = 20.0 * np.log10(np.maximum(block_rms, 1e-8))
    zero_crossing = np.mean(
        np.signbit(blocks[:, 1:]) != np.signbit(blocks[:, :-1]), axis=1
    )
    window = np.hanning(block_size)
    power = np.abs(np.fft.rfft(blocks * window, axis=1)) ** 2
    arithmetic = np.mean(power + 1e-12, axis=1)
    geometric = np.exp(np.mean(np.log(power + 1e-12), axis=1))
    flatness = geometric / arithmetic

    activity_floor = max(-50.0, float(np.percentile(block_db, 20)) + 3.0)
    active = block_db > activity_floor
    plausible_crossing = (zero_crossing > 0.003) & (zero_crossing < 0.35)
    tonal_voice_shape = np.clip((0.55 - flatness) / 0.45, 0.0, 1.0)

    # Natural speech changes in level and spectral shape. Give that movement a
    # bounded bonus so a loud, steady tone cannot dominate solely through RMS.
    level_change = np.zeros_like(block_db)
    level_change[1:] = np.abs(np.diff(block_db))
    normalized_power = power / np.maximum(np.sum(power, axis=1, keepdims=True), 1e-12)
    spectral_change = np.zeros_like(block_db)
    spectral_change[1:] = 0.5 * np.sum(
        np.abs(np.diff(normalized_power, axis=0)), axis=1
    )
    movement = np.clip(level_change / 6.0 + spectral_change / 0.20, 0.0, 1.0)
    speech_evidence = (
        active.astype(np.float32)
        * plausible_crossing.astype(np.float32)
        * tonal_voice_shape.astype(np.float32)
        * (0.65 + 0.35 * movement.astype(np.float32))
    )
    rolling = np.convolve(speech_evidence, np.ones(block_count), mode="valid")
    if float(np.max(rolling)) <= 1e-6:
        # Conservative fallback for unusual but usable speech spectra.
        rolling = np.convolve(block_energy, np.ones(block_count), mode="valid")
    best_block = int(np.argmax(rolling))
    start = best_block * block_size
    selected = centered[start : start + window_samples]
    return np.clip(selected, -1.0, 1.0).astype(np.float32, copy=False)
