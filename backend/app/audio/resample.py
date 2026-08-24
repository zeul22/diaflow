"""Band-limited sample-rate conversion.

Linear interpolation is not a resampler. Decimating 48 kHz to 16 kHz with
``np.interp`` leaves every component above the new 8 kHz Nyquist folded back into
the speech band at full amplitude: a 12 kHz tone reappears at 4 kHz. In a truck
cab or warehouse that means high-frequency machinery noise, beeps, and air
brakes are mirrored on top of the caller's voice, corrupting both the embedding
and the very SNR and spectral-flatness statistics the quality gate uses to judge
noise.

This module implements the standard fix used by production resamplers such as
libsoxr and FFmpeg's swresample: convolve with a windowed-sinc kernel whose
cutoff sits at the lower of the two Nyquist frequencies, evaluated at the exact
output positions. A Kaiser window sets the stopband attenuation, so aliasing is
pushed below the noise floor instead of into the passband.
"""

from __future__ import annotations

from math import ceil, gcd

import numpy as np
import numpy.typing as npt

# Eight zero crossings per side and ~72 dB of stopband rejection. Wider kernels
# buy attenuation nobody can hear at 16 kHz while costing latency per request.
DEFAULT_LOBES = 8
DEFAULT_ATTENUATION_DB = 72.0
# No filter has a vertical edge. Placing the cutoff exactly at the target Nyquist
# leaves the transition band *above* it, so content just past 8 kHz still folds
# back with only partial attenuation. Pulling the cutoff to 95% of Nyquist fits
# the whole transition inside the band being discarded anyway. libsoxr and
# swresample make the same trade; 7.6-8 kHz is negligible for a 16 kHz speech
# model and the mel filterbank barely weights it.
DEFAULT_ROLLOFF = 0.95
_BLOCK = 8_192


def kaiser_beta(attenuation_db: float) -> float:
    """Kaiser window parameter for a target stopband attenuation (Oppenheim)."""

    if attenuation_db > 50.0:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21.0:
        return 0.5842 * (attenuation_db - 21.0) ** 0.4 + 0.07886 * (
            attenuation_db - 21.0
        )
    return 0.0


def _kernel(
    offsets: npt.NDArray[np.float64],
    cutoff: float,
    half_width: float,
    beta: float,
) -> npt.NDArray[np.float64]:
    """Windowed-sinc weights for fractional offsets, in input-sample units."""

    scaled = 2.0 * cutoff * offsets
    sinc = np.sinc(scaled)
    # Kaiser window evaluated analytically so it can be sampled off-grid.
    ratio = np.clip(offsets / half_width, -1.0, 1.0)
    window = np.i0(beta * np.sqrt(np.maximum(0.0, 1.0 - ratio * ratio))) / np.i0(beta)
    return sinc * window


def resample(
    samples: npt.NDArray[np.float32],
    source_rate: int,
    target_rate: int,
    *,
    lobes: int = DEFAULT_LOBES,
    attenuation_db: float = DEFAULT_ATTENUATION_DB,
    rolloff: float = DEFAULT_ROLLOFF,
) -> npt.NDArray[np.float32]:
    """Convert ``samples`` to ``target_rate`` without folding out-of-band energy.

    The kernel is anti-aliasing when decimating and anti-imaging when
    interpolating, because in both directions the cutoff is the lower of the two
    Nyquist frequencies.
    """

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate == target_rate or array.size < 2:
        return array.astype(np.float32, copy=False)

    up, down = rational_ratio(source_rate, target_rate)
    step = source_rate / target_rate
    output_size = max(1, int(round(array.size / step)))
    # Cutoff in cycles per input sample: just under the lower Nyquist of the two
    # rates, so the transition band lands in the discarded region.
    cutoff = 0.5 * min(1.0, 1.0 / step) * rolloff
    half_width = lobes / (2.0 * cutoff)
    beta = kaiser_beta(attenuation_db)
    taps = int(ceil(half_width))
    offsets = np.arange(-taps, taps + 1, dtype=np.int64)

    # Output n sits at input position n*down/up, so its fractional offset is
    # (n*down mod up)/up: only ``up`` distinct phases exist. Evaluating the sinc
    # and Kaiser window once per phase instead of once per output sample is what
    # makes this a polyphase resampler rather than a very slow interpolator.
    phase_offsets = np.arange(up, dtype=np.float64) * down % up / up
    table = _kernel(
        phase_offsets[:, None] - offsets[None, :].astype(np.float64),
        cutoff,
        half_width,
        beta,
    )
    # Normalize each phase to unity DC gain once, here. Renormalizing a
    # *truncated* kernel per output sample instead would silently build a
    # different, much worse filter at the signal boundaries, which leaks exactly
    # the aliases this module exists to reject.
    table = (table / table.sum(axis=1, keepdims=True)).astype(np.float32)

    source = array.astype(np.float32, copy=False)
    output = np.empty(output_size, dtype=np.float32)
    limit = source.size - 1

    for begin in range(0, output_size, _BLOCK):
        end = min(begin + _BLOCK, output_size)
        counter = np.arange(begin, end, dtype=np.int64)
        base = counter * down // up
        weights = table[counter % up]
        indices = base[:, None] + offsets[None, :]
        # Clamped edge extension: the full symmetric kernel always applies, so
        # the filter keeps its stopband right to the first and last sample.
        gathered = source[np.clip(indices, 0, limit)]
        output[begin:end] = np.einsum("ij,ij->i", gathered, weights)

    return output


def rational_ratio(source_rate: int, target_rate: int) -> tuple[int, int]:
    """Reduced (up, down) conversion ratio, for documentation and tests."""

    divisor = gcd(source_rate, target_rate)
    return target_rate // divisor, source_rate // divisor
