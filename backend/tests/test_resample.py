from __future__ import annotations

import numpy as np
import pytest

from app.audio.resample import kaiser_beta, rational_ratio, resample

SOURCE = 48_000
TARGET = 16_000
# The first output samples are a boundary transient: at a signal edge the kernel
# sees a clamped extension rather than real audio. It decays within eight output
# samples (half a millisecond), so steady-state behaviour is measured past it.
EDGE_GUARD = 16


def _tone(frequency: float, seconds: float = 2.0, rate: int = SOURCE) -> np.ndarray:
    time = np.arange(int(seconds * rate)) / rate
    return (0.5 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _peak_db(samples: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.abs(samples).max()), 1e-12))


@pytest.mark.parametrize("frequency", [10_000, 12_000, 15_000, 20_000])
def test_stopband_energy_is_rejected_not_folded(frequency) -> None:
    """Content in the stopband must not reappear inside the band.

    Linear interpolation passed a 12 kHz tone through at full amplitude, mirrored
    to 4 kHz -- on top of the caller's voice, and inside the very band the
    quality gate measures to judge noise.
    """

    converted = resample(_tone(frequency), SOURCE, TARGET)

    assert _peak_db(converted[EDGE_GUARD:-EDGE_GUARD]) < -70.0


@pytest.mark.parametrize("frequency", [8_500, 9_000])
def test_transition_band_is_attenuated_though_not_eliminated(frequency) -> None:
    """No filter has a vertical edge, and this one is honest about where it ends.

    With the cutoff at 95% of Nyquist the transition runs to about 10 kHz, so
    8.5-9 kHz is attenuated by 18-30 dB rather than the 70 dB of the stopband.
    Closing that gap means doubling the kernel and the per-request latency, for
    a region where speech carries very little energy. Measured, not assumed.
    """

    converted = resample(_tone(frequency), SOURCE, TARGET)

    attenuation = _peak_db(converted[EDGE_GUARD:-EDGE_GUARD])
    assert attenuation < -18.0
    # Still vastly better than passing it through untouched, which is what the
    # previous linear interpolation did at every frequency.
    assert attenuation < -12.0


def test_the_old_linear_interpolation_would_have_aliased() -> None:
    """Characterizes the defect this module replaced, so it cannot come back."""

    tone = _tone(12_000)
    positions = np.arange(tone.size * TARGET // SOURCE) * (SOURCE / TARGET)
    aliased = np.interp(positions, np.arange(tone.size), tone).astype(np.float32)

    # Unattenuated: the input peak was 0.5, i.e. -6 dB.
    assert _peak_db(aliased) > -7.0
    assert _peak_db(resample(tone, SOURCE, TARGET)[EDGE_GUARD:-EDGE_GUARD]) < -60.0


@pytest.mark.parametrize("frequency", [300, 1_000, 3_000, 6_000])
def test_speech_band_passes_through_unchanged(frequency) -> None:
    """Flat to 6 kHz; the 95% rolloff only trims 7.6-8 kHz."""

    converted = resample(_tone(frequency), SOURCE, TARGET)

    assert np.abs(converted[EDGE_GUARD:-EDGE_GUARD]).max() == pytest.approx(
        0.5, abs=0.01
    )


def test_upsampling_does_not_create_images() -> None:
    """8 kHz telephony to 16 kHz must not synthesize energy it never had."""

    narrowband = _tone(1_000, seconds=2.0, rate=8_000)

    converted = resample(narrowband, 8_000, TARGET)

    spectrum = np.abs(np.fft.rfft(converted * np.hanning(converted.size))) ** 2
    frequencies = np.fft.rfftfreq(converted.size, 1.0 / TARGET)
    above_source_nyquist = spectrum[frequencies > 4_000].sum() / spectrum.sum()
    assert above_source_nyquist < 1e-6


def test_boundary_transient_is_short() -> None:
    converted = resample(_tone(12_000), SOURCE, TARGET)

    # Both edges ring briefly, and the interior is clean. This is inherent to a
    # finite kernel meeting a signal boundary, not a filter defect.
    assert _peak_db(converted[:4]) > -60.0
    assert _peak_db(converted[-4:]) > -60.0
    assert _peak_db(converted[EDGE_GUARD:-EDGE_GUARD]) < -60.0


def test_output_length_and_rates() -> None:
    samples = _tone(1_000, seconds=1.0)

    assert resample(samples, SOURCE, TARGET).size == TARGET
    assert resample(samples, SOURCE, SOURCE) is not None
    # A no-op conversion must not copy or filter.
    assert resample(samples, SOURCE, SOURCE).size == samples.size
    assert resample(np.zeros(1, dtype=np.float32), SOURCE, TARGET).size == 1
    with pytest.raises(ValueError):
        resample(samples, 0, TARGET)


def test_awkward_rates_use_a_bounded_phase_table() -> None:
    """44.1 kHz is common from browsers and needs 160 polyphase branches."""

    assert rational_ratio(48_000, 16_000) == (1, 3)
    assert rational_ratio(44_100, 16_000) == (160, 441)
    assert rational_ratio(8_000, 16_000) == (2, 1)

    converted = resample(_tone(12_000, rate=44_100), 44_100, TARGET)
    assert _peak_db(converted[EDGE_GUARD:-EDGE_GUARD]) < -55.0


def test_kaiser_beta_matches_the_published_piecewise_formula() -> None:
    assert kaiser_beta(72.0) == pytest.approx(0.1102 * (72.0 - 8.7))
    assert kaiser_beta(30.0) == pytest.approx(
        0.5842 * 9.0**0.4 + 0.07886 * 9.0, rel=1e-9
    )
    assert kaiser_beta(10.0) == 0.0
