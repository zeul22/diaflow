"""Level normalization and optional noise reduction for the inference window.

Three requests usually arrive together -- denoising, automatic gain control, and
echo cancellation -- and they have very different answers here.

**AGC (on by default).** Callers arrive at wildly different levels: a headset in
an office, a speakerphone on a dashboard, a handset in a warehouse. The heads
were fitted on utterance-level embeddings from reasonably levelled corpora, so
presenting them a signal 25 dB quieter is avoidable variance. Normalizing the
inference window to a target level is cheap, reversible, and does not change the
spectral envelope the model reads.

**Denoising (off by default, and that is deliberate).** Noise suppression helps a
human listener and helps ASR. It measurably *hurts* speaker and paralinguistic
models, because it attenuates exactly the low-energy spectral detail that
carries voice identity, and it leaves behind artifacts a model has never seen in
training. The honest fix for noisy logistics audio is training on noisy logistics
audio, which is what ADR-002 targets. Spectral gating is provided for evaluation
and for deployments that measure a gain on their own data -- never silently on.

**Echo cancellation (not possible here).** AEC subtracts a *known* far-end
reference from the near-end mixture. This service receives one already-mixed
channel and never sees what the agent played, so there is no reference to
subtract; any "echo cancellation" applied here would be guesswork. It has to
happen where the reference exists: in the browser (``echoCancellation`` is
requested on capture) or in the telephony gateway.

Ordering matters. The quality gate runs on the *unmodified* signal so that a
near-silent recording still reports ``insufficient`` instead of being amplified
into apparent usability. Enhancement applies only to the window handed to the
model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from app.config import Settings

logger = logging.getLogger(__name__)

_FRAME = 512
_HOP = 128
_EPSILON = 1e-10


@dataclass(frozen=True, slots=True)
class EnhancementReport:
    gain_db: float = 0.0
    denoised: bool = False
    noise_floor_dbfs: float = 0.0


def _speech_rms(samples: npt.NDArray[np.float32]) -> float:
    """RMS of the louder frames, so silence between words does not skew gain."""

    frame = 400  # 25 ms at 16 kHz
    usable = samples.size - (samples.size % frame)
    if usable < frame:
        return float(np.sqrt(np.mean(samples * samples) + _EPSILON))
    frames = samples[:usable].reshape(-1, frame)
    energies = np.mean(frames * frames, axis=1)
    # The top third approximates speech-active frames without a second VAD.
    threshold = np.percentile(energies, 67.0)
    active = energies[energies >= threshold]
    return float(np.sqrt(active.mean() + _EPSILON))


def normalize_loudness(
    samples: npt.NDArray[np.float32], settings: Settings
) -> tuple[npt.NDArray[np.float32], float]:
    """Scale toward a target level, with a capped gain and no noise pumping."""

    level = _speech_rms(samples)
    level_dbfs = 20.0 * np.log10(max(level, 1e-8))
    if level_dbfs < settings.agc_min_level_dbfs:
        # Too quiet to be speech; amplifying it would only raise the noise floor
        # and hand the model a louder version of nothing.
        return samples, 0.0
    gain_db = float(
        np.clip(
            settings.agc_target_dbfs - level_dbfs,
            -settings.agc_max_gain_db,
            settings.agc_max_gain_db,
        )
    )
    if abs(gain_db) < 0.5:
        return samples, 0.0
    scaled = samples * float(10.0 ** (gain_db / 20.0))
    # Never clip: back the gain off if the peak would exceed full scale.
    peak = float(np.abs(scaled).max())
    if peak > 0.999:
        scaled *= 0.999 / peak
        gain_db += 20.0 * float(np.log10(0.999 / peak))
    return scaled.astype(np.float32, copy=False), gain_db


def _window() -> npt.NDArray[np.float64]:
    return np.hanning(_FRAME + 1)[:-1]


def spectral_gate(
    samples: npt.NDArray[np.float32], settings: Settings
) -> tuple[npt.NDArray[np.float32], float]:
    """Stationary-noise suppression by spectral subtraction with a floor.

    Suited to the steady sources in this domain -- engine, road, HVAC, fan noise
    -- and not to transients such as door slams or air brakes. The spectral floor
    limits the classic "musical noise" artifact by never removing more than a
    fixed proportion of any bin.
    """

    if samples.size < _FRAME * 2:
        return samples, 0.0
    window = _window()
    padded = np.pad(samples.astype(np.float64), (_FRAME, _FRAME))
    count = 1 + (padded.size - _FRAME) // _HOP
    indices = np.arange(_FRAME)[None, :] + _HOP * np.arange(count)[:, None]
    frames = padded[indices] * window
    spectrum = np.fft.rfft(frames, axis=1)
    magnitude = np.abs(spectrum)

    # Per-bin noise estimate from the quietest frames: robust for stationary
    # noise and does not need a separate noise-only recording.
    noise = np.percentile(magnitude, settings.denoise_noise_percentile, axis=0)
    floor_gain = float(10.0 ** (settings.denoise_floor_db / 20.0))
    reduced = magnitude - settings.denoise_over_subtraction * noise[None, :]
    reduced = np.maximum(reduced, floor_gain * magnitude)
    gains = reduced / np.maximum(magnitude, _EPSILON)

    restored = np.fft.irfft(spectrum * gains, n=_FRAME, axis=1) * window
    output = np.zeros(padded.size, dtype=np.float64)
    normalizer = np.zeros(padded.size, dtype=np.float64)
    squared = window * window
    for index in range(count):
        start = index * _HOP
        output[start : start + _FRAME] += restored[index]
        normalizer[start : start + _FRAME] += squared
    output /= np.maximum(normalizer, 1e-8)
    trimmed = output[_FRAME : _FRAME + samples.size]
    noise_dbfs = 20.0 * float(np.log10(max(float(np.mean(noise)), 1e-8)))
    return np.clip(trimmed, -1.0, 1.0).astype(np.float32), noise_dbfs


def enhance_window(
    samples: npt.NDArray[np.float32], settings: Settings
) -> tuple[npt.NDArray[np.float32], EnhancementReport]:
    """Apply the configured enhancement chain to an inference window."""

    working = samples
    denoised = False
    noise_dbfs = 0.0
    if settings.denoise_backend == "spectral_gate":
        working, noise_dbfs = spectral_gate(working, settings)
        denoised = True
    gain_db = 0.0
    if settings.agc_enabled:
        working, gain_db = normalize_loudness(working, settings)
    if working is samples:
        return samples, EnhancementReport()
    return working, EnhancementReport(
        gain_db=round(gain_db, 2),
        denoised=denoised,
        noise_floor_dbfs=round(noise_dbfs, 2),
    )
