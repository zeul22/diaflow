from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.audio.enhance import enhance_window, normalize_loudness, spectral_gate
from app.audio.jitter import ReorderBuffer, parse_sequenced_frame
from app.config import Settings
from app.main import create_app
from tests.conftest import FakeEstimator, speechlike_pcm, wav_bytes


def _frame(sequence: int, payload: bytes) -> bytes:
    return sequence.to_bytes(4, "big") + payload


def test_in_order_frames_pass_straight_through() -> None:
    buffer = ReorderBuffer(window=4)

    assert buffer.push(0, b"a") == [b"a"]
    assert buffer.push(1, b"b") == [b"b"]
    assert buffer.integrity().frames_lost == 0
    assert buffer.integrity().loss_ratio == 0.0


def test_a_swapped_pair_is_put_back_in_order() -> None:
    buffer = ReorderBuffer(window=4)

    assert buffer.push(0, b"a") == [b"a"]
    # Frame 2 arrives before frame 1 and must wait, not jump the queue.
    assert buffer.push(2, b"c") == []
    assert buffer.push(1, b"b") == [b"b", b"c"]
    assert buffer.integrity().frames_reordered == 1
    assert buffer.integrity().frames_lost == 0


def test_duplicates_are_dropped() -> None:
    buffer = ReorderBuffer(window=4)
    buffer.push(0, b"a")

    assert buffer.push(0, b"a") == []
    assert buffer.integrity().frames_duplicated == 1


def test_a_lost_frame_is_concealed_once_the_window_fills() -> None:
    buffer = ReorderBuffer(window=2)
    buffer.push(0, b"aaaa")

    # Frame 1 never arrives. Later frames queue until the window is exceeded,
    # then the gap is filled so the stream keeps moving.
    assert buffer.push(2, b"cccc") == []
    assert buffer.push(3, b"dddd") == []
    released = buffer.push(4, b"eeee")

    assert released[0] == b"aaaa"  # repeated previous frame conceals the gap
    integrity = buffer.integrity()
    assert integrity.frames_lost == 1
    assert integrity.frames_concealed == 1
    assert integrity.loss_ratio == pytest.approx(1 / 5)


def test_repetition_gives_way_to_silence_in_a_long_gap() -> None:
    buffer = ReorderBuffer(window=1, max_repeat_frames=1)
    buffer.push(0, b"\x10\x20")

    buffer.push(5, b"\x30\x40")
    released = buffer.flush()

    # One repeat, then silence: sustained repetition buzzes, and a
    # paralinguistic model would treat that as voice.
    assert released[0] == b"\x10\x20"
    assert released[1] == b"\x00\x00"
    assert buffer.integrity().frames_lost == 4


def test_a_late_frame_is_not_inserted_out_of_position() -> None:
    buffer = ReorderBuffer(window=1)
    buffer.push(0, b"aa")
    buffer.push(2, b"cc")
    buffer.push(3, b"dd")  # forces frame 1 to be declared lost

    assert buffer.push(1, b"bb") == []
    assert buffer.integrity().frames_duplicated == 1


def test_flush_releases_everything_held() -> None:
    buffer = ReorderBuffer(window=8)
    buffer.push(0, b"aa")
    buffer.push(3, b"dd")

    # Frames 1 and 2 are missing: both concealed by repetition (the default
    # allows two) before frame 3 is released.
    assert buffer.flush() == [b"aa", b"aa", b"dd"]
    assert buffer.integrity().frames_lost == 2


def test_frame_header_is_parsed_and_validated() -> None:
    assert parse_sequenced_frame(_frame(258, b"pcm")) == (258, b"pcm")
    with pytest.raises(ValueError):
        parse_sequenced_frame(b"\x00\x01")
    with pytest.raises(ValueError):
        ReorderBuffer().push(-1, b"x")


def test_sequenced_streaming_survives_reordering_end_to_end(settings) -> None:
    app = create_app(settings=settings, estimator=FakeEstimator())
    pcm = (speechlike_pcm(4.0) * 32767.0).astype("<i2").tobytes()
    step = 16_000 * 2 // 2  # half-second frames
    frames = [pcm[index : index + step] for index in range(0, len(pcm), step)]
    predictions = []

    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/analyze") as websocket,
    ):
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
                "framing": "seq32",
            }
        )
        order = list(range(len(frames)))
        order[2], order[3] = order[3], order[2]  # swap one pair in flight
        for index in order:
            websocket.send_bytes(_frame(index, frames[index]))
        websocket.send_json({"type": "end"})
        while True:
            message = websocket.receive_json()
            if message["type"] != "prediction":
                continue
            predictions.append(message)
            if message["is_final"]:
                break

    assert predictions[-1]["is_final"] is True
    assert predictions[-1]["age_bracket"]["prediction"] == "31-45"


def test_heavy_loss_is_reported_as_degraded_audio(settings) -> None:
    app = create_app(settings=settings, estimator=FakeEstimator())
    pcm = (speechlike_pcm(4.0) * 32767.0).astype("<i2").tobytes()
    step = 16_000 * 2 // 4
    frames = [pcm[index : index + step] for index in range(0, len(pcm), step)]

    with (
        TestClient(app) as client,
        client.websocket_connect("/ws/analyze") as websocket,
    ):
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
                "framing": "seq32",
            }
        )
        # Drop every third frame: well past WS_MAX_LOSS_RATIO.
        for index in range(len(frames)):
            if index % 3 == 1:
                continue
            websocket.send_bytes(_frame(index, frames[index]))
        websocket.send_json({"type": "end"})
        while True:
            message = websocket.receive_json()
            if message["type"] != "prediction":
                continue
            if message["is_final"]:
                final = message
                break

    assert final["audio_quality"] == "degraded"


def test_quiet_audio_is_normalized_toward_the_target_level() -> None:
    settings = Settings()
    # -30 dBFS: quiet but plainly speech, so above AGC_MIN_LEVEL_DBFS.
    quiet = (speechlike_pcm(2.0) * 0.1).astype(np.float32)

    louder, gain_db = normalize_loudness(quiet, settings)

    assert gain_db > 6.0
    assert np.abs(louder).max() > np.abs(quiet).max()
    assert np.abs(louder).max() <= 1.0


def test_near_silence_is_not_amplified_into_apparent_speech() -> None:
    settings = Settings()
    silence = (speechlike_pcm(2.0) * 0.0005).astype(np.float32)

    _, gain_db = normalize_loudness(silence, settings)

    assert gain_db == 0.0


def test_normalization_never_clips() -> None:
    settings = replace(Settings(), agc_target_dbfs=-1.0, agc_max_gain_db=40.0)
    loud = (speechlike_pcm(2.0) * 0.9).astype(np.float32)

    normalized, _ = normalize_loudness(loud, settings)

    assert np.abs(normalized).max() <= 1.0


def test_spectral_gate_reduces_stationary_noise() -> None:
    settings = replace(Settings(), denoise_backend="spectral_gate")
    rng = np.random.default_rng(5)
    speech = speechlike_pcm(3.0)
    noise = rng.normal(0.0, 0.05, speech.size).astype(np.float32)
    noisy = (speech + noise).astype(np.float32)

    cleaned, _ = spectral_gate(noisy, settings)

    # Measure the residual in the gaps between speech, where only noise lives.
    frame = 400
    frames = noisy[: noisy.size - noisy.size % frame].reshape(-1, frame)
    cleaned_frames = cleaned[: cleaned.size - cleaned.size % frame].reshape(-1, frame)
    energies = np.mean(frames * frames, axis=1)
    quietest = np.argsort(energies)[: max(1, len(energies) // 5)]
    before = float(np.mean(frames[quietest] ** 2))
    after = float(np.mean(cleaned_frames[quietest] ** 2))
    assert after < before


def test_enhancement_is_a_no_op_when_disabled() -> None:
    settings = replace(Settings(), agc_enabled=False, denoise_backend="none")
    samples = speechlike_pcm(2.0)

    result, report = enhance_window(samples, settings)

    assert result is samples
    assert report.gain_db == 0.0
    assert report.denoised is False


def test_enhancement_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="DENOISE_BACKEND"):
        replace(Settings(), denoise_backend="magic").validate()
    with pytest.raises(ValueError, match="AGC_TARGET_DBFS"):
        replace(Settings(), agc_target_dbfs=5.0).validate()
    with pytest.raises(ValueError, match="WS_MAX_LOSS_RATIO"):
        replace(Settings(), ws_max_loss_ratio=1.5).validate()
    with pytest.raises(ValueError, match="WS_REORDER_WINDOW_FRAMES"):
        replace(Settings(), ws_reorder_window_frames=0).validate()


def test_queue_wait_and_rejection_are_recorded(settings) -> None:
    """Queue wait is the autoscaling signal: it must be observed on both paths.

    Per-container capacity is bounded by memory bandwidth rather than cores, so
    neither replicas nor threads raise it much. Queue wait is what tells an
    autoscaler to add containers, and it rises before requests are shed.
    """

    from app.audio.types import SourceSpec
    from app.errors import ServiceBusyError
    from app.inference.service import AnalysisService
    from app.observability.metrics import Metrics

    metrics = Metrics()
    service = AnalysisService(
        settings=replace(settings, queue_timeout_seconds=0.01),
        estimator=FakeEstimator(),
        metrics=metrics,
    )

    def sample(name: str) -> float:
        for metric in metrics.registry.collect():
            for item in metric.samples:
                if item.name == name:
                    return item.value
        return -1.0

    payload = wav_bytes(speechlike_pcm(2.0))

    async def scenario() -> None:
        # A successful analysis records a short wait.
        await service.analyze(
            payload=payload,
            source=SourceSpec(encoding="wav"),
            contact_id=uuid4(),
        )
        assert sample("voice_attribute_queue_wait_seconds_count") == 1.0
        assert sample("voice_attribute_queue_rejections_total") == 0.0

        # With the only replica held, the next request is shed and counted.
        held = await service._pool.acquire(0.1)
        try:
            with pytest.raises(ServiceBusyError):
                await service.analyze(
                    payload=payload,
                    source=SourceSpec(encoding="wav"),
                    contact_id=uuid4(),
                )
        finally:
            service._pool.release(held)

        assert sample("voice_attribute_queue_rejections_total") == 1.0
        assert sample("voice_attribute_queue_wait_seconds_count") == 2.0

    asyncio.run(scenario())
