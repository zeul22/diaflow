from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.models.base import RawAttributes
from app.models.ecapa import (
    dispersion_years,
    min_pairwise_cosine,
    sub_window_bounds,
)
from app.models.pool import EstimatorPool, as_pool


def test_sub_windows_are_equal_length_and_cover_the_segment() -> None:
    bounds = sub_window_bounds(80_000, 3, 32_000)

    assert bounds == ((0, 40_000), (20_000, 60_000), (40_000, 80_000))
    assert len({end - start for start, end in bounds}) == 1
    assert bounds[-1][1] == 80_000


def test_short_segments_fall_back_to_a_single_view() -> None:
    # Two seconds of audio cannot be carved into 2-second sub-windows.
    assert sub_window_bounds(32_000, 3, 32_000) == ((0, 32_000),)
    assert sub_window_bounds(80_000, 1, 32_000) == ((0, 80_000),)


def test_sub_window_count_is_honoured_when_it_fits() -> None:
    bounds = sub_window_bounds(160_000, 4, 32_000)

    assert len(bounds) == 4
    assert all(end - start >= 32_000 for start, end in bounds)
    assert bounds[-1][1] <= 160_000


def test_one_speaker_scores_higher_homogeneity_than_two() -> None:
    rng = np.random.default_rng(11)
    voice = rng.normal(size=192).astype(np.float32)
    other = rng.normal(size=192).astype(np.float32)
    same_speaker = np.stack([voice, voice + 0.05 * rng.normal(size=192)]).astype(
        np.float32
    )
    two_speakers = np.stack([voice, other]).astype(np.float32)

    assert min_pairwise_cosine(same_speaker) > 0.9
    assert min_pairwise_cosine(two_speakers) < 0.3


def test_homogeneity_is_unavailable_for_a_single_window() -> None:
    assert min_pairwise_cosine(np.zeros((1, 192), dtype=np.float32)) is None


def test_dispersion_requires_two_finite_estimates() -> None:
    assert dispersion_years([40.0]) is None
    assert dispersion_years([40.0, None]) is None
    assert dispersion_years([30.0, 50.0]) == pytest.approx(10.0)


class _Replica:
    name = "replica"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, samples):
        self.calls += 1
        return RawAttributes(gender_probabilities={"female": 0.5, "male": 0.5})

    def warmup(self) -> None:
        return None


def test_pool_hands_out_every_replica_before_blocking() -> None:
    async def scenario() -> None:
        pool = EstimatorPool([_Replica(), _Replica()])

        first = await pool.acquire(0.1)
        second = await pool.acquire(0.1)
        assert first is not second

        with pytest.raises(TimeoutError):
            await pool.acquire(0.05)

        pool.release(second)
        assert await pool.acquire(0.1) is second

    asyncio.run(scenario())


def test_single_estimator_is_wrapped_as_a_pool_of_one() -> None:
    replica = _Replica()

    pool = as_pool(replica)

    assert pool.size == 1
    assert pool.name == "replica"
    assert as_pool(pool) is pool


def test_ensemble_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="ENSEMBLE_WINDOWS"):
        replace(Settings(), ensemble_windows=0).validate()
    with pytest.raises(ValueError, match="ENSEMBLE_MIN_WINDOW_SECONDS"):
        replace(Settings(), ensemble_min_window_seconds=9.0).validate()
