from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import _analysis_offset, _frame_bytes, create_app
from app.models.base import RawAttributes
from app.persistence import PersistenceService
from app.schemas import WebSocketStart
from tests.conftest import FakeEstimator, speechlike_pcm, wav_bytes
from tests.test_persistence import FakeMetadataStore, FakeObjectStore, audio_settings


def test_raw_wav_analysis_matches_contract(
    client, fake_estimator: FakeEstimator
) -> None:
    contact_id = uuid4()
    response = client.post(
        f"/analyze?contact_id={contact_id}",
        content=wav_bytes(speechlike_pcm()),
        headers={"Content-Type": "audio/wav", "X-Request-ID": "test-request"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["contact_id"] == str(contact_id)
    assert payload["gender"]["prediction"] == "male"
    assert payload["age_bracket"]["prediction"] == "31-45"
    assert 0.0 <= payload["gender"]["confidence"] <= 1.0
    assert 0.0 <= payload["age_bracket"]["confidence"] <= 1.0
    assert payload["audio_quality"] in {"good", "degraded"}
    assert payload["processing_ms"] >= 0
    assert response.headers["x-request-id"] == "test-request"
    assert fake_estimator.calls == 1
    assert fake_estimator.warmed


def test_narrowband_wav_is_never_reported_good(client) -> None:
    response = client.post(
        "/analyze",
        content=wav_bytes(speechlike_pcm(sample_rate=8_000), sample_rate=8_000),
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 200
    assert response.json()["audio_quality"] in {"degraded", "insufficient"}


def test_multipart_upload_uses_form_contact_id(client) -> None:
    contact_id = uuid4()
    response = client.post(
        "/analyze",
        files={"audio": ("caller.wav", wav_bytes(speechlike_pcm()), "audio/wav")},
        data={"contact_id": str(contact_id)},
    )

    assert response.status_code == 200, response.text
    assert response.json()["contact_id"] == str(contact_id)


def test_silence_returns_unknown_without_model_call(
    client, fake_estimator: FakeEstimator
) -> None:
    response = client.post(
        "/analyze",
        content=wav_bytes(np.zeros(3 * 16_000, dtype=np.float32)),
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_quality"] == "insufficient"
    assert payload["gender"] == {"prediction": "unknown", "confidence": 0.0}
    assert payload["age_bracket"] == {
        "prediction": "unknown",
        "confidence": 0.0,
    }
    assert fake_estimator.calls == 0


def test_invalid_contact_id_is_structured_error(client) -> None:
    response = client.post(
        "/analyze?contact_id=not-a-uuid",
        content=wav_bytes(speechlike_pcm()),
        headers={"Content-Type": "audio/wav"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_CONTACT_ID"
    assert response.json()["error"]["request_id"]


def test_non_audio_media_type_is_rejected(client) -> None:
    response = client.post(
        "/analyze", content=b"{}", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_metrics_and_health_endpoints(client) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "voice_attribute_http_requests_total" in metrics.text


def test_websocket_emits_progressive_and_final_predictions(client) -> None:
    contact_id = uuid4()
    raw_pcm = (speechlike_pcm(2.4) * 32767.0).astype("<i2").tobytes()
    one_second = 16_000 * 2

    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "contact_id": str(contact_id),
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        websocket.send_bytes(raw_pcm[:one_second])
        websocket.send_bytes(raw_pcm[one_second:])
        progressive = websocket.receive_json()
        assert progressive["type"] == "prediction"
        assert progressive["is_final"] is False
        assert progressive["contact_id"] == str(contact_id)

        websocket.send_json({"type": "end"})
        final = websocket.receive_json()
        assert final["type"] == "prediction"
        assert final["is_final"] is True
    assert final["sequence"] == progressive["sequence"] + 1


def test_progressive_updates_analyze_a_bounded_trailing_window() -> None:
    start = WebSocketStart(
        type="start", encoding="pcm_s16le", sample_rate=16_000, channels=1
    )
    limit_bytes = 10 * 16_000 * 2

    # Under the analysis window the whole session buffer is used as-is.
    assert _analysis_offset(limit_bytes, start, limit_bytes) == 0
    # Above it, only the trailing window is re-analyzed, so a long session costs
    # a constant amount per update instead of growing without bound.
    assert _analysis_offset(limit_bytes * 3, start, limit_bytes) == limit_bytes * 2

    stereo = start.model_copy(update={"channels": 2})
    offset = _analysis_offset(limit_bytes * 4 + 3, stereo, limit_bytes)
    assert offset % _frame_bytes(stereo) == 0


def test_progressive_emissions_back_off_over_a_long_session(
    client, fake_estimator: FakeEstimator
) -> None:
    raw_pcm = (speechlike_pcm(8.0) * 32767.0).astype("<i2").tobytes()
    one_second = 16_000 * 2
    predictions = []

    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        for index in range(8):
            websocket.send_bytes(raw_pcm[index * one_second : (index + 1) * one_second])
        websocket.send_json({"type": "end"})
        while True:
            message = websocket.receive_json()
            if message["type"] != "prediction":
                continue
            predictions.append(message)
            if message["is_final"]:
                break

    progressive = [item for item in predictions if not item["is_final"]]
    # A fixed one-second cadence would have re-run the model on every chunk.
    assert 2 <= len(progressive) < 8
    assert predictions[-1]["is_final"] is True
    assert fake_estimator.calls == len(predictions)


def test_a_changing_prediction_resets_the_progressive_cadence(settings) -> None:
    """Backoff must not make a moving estimate look stale.

    The alternating estimator changes its label on every call, so the interval
    is reset each time and the session emits more often than the stable case.
    """

    class AlternatingEstimator:
        name = "alternating-estimator"

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, samples):
            del samples
            self.calls += 1
            female = 0.92 if self.calls % 2 else 0.08
            return RawAttributes(
                gender_probabilities={"female": female, "male": 1.0 - female},
                age_years=40.0,
            )

        def warmup(self) -> None:
            return None

    def count_progressive(estimator) -> int:
        app = create_app(settings=settings, estimator=estimator)
        raw_pcm = (speechlike_pcm(8.0) * 32767.0).astype("<i2").tobytes()
        one_second = 16_000 * 2
        seen = 0
        with (
            TestClient(app) as test_client,
            test_client.websocket_connect("/ws/analyze") as websocket,
        ):
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16_000,
                    "channels": 1,
                }
            )
            for index in range(8):
                websocket.send_bytes(
                    raw_pcm[index * one_second : (index + 1) * one_second]
                )
            websocket.send_json({"type": "end"})
            while True:
                message = websocket.receive_json()
                if message["type"] != "prediction":
                    continue
                if message["is_final"]:
                    break
                seen += 1
        return seen

    assert count_progressive(AlternatingEstimator()) > count_progressive(
        FakeEstimator()
    )


def test_websocket_accepts_configured_browser_origin(client) -> None:
    with client.websocket_connect(
        "/ws/analyze", headers={"Origin": "http://localhost:3000"}
    ) as websocket:
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}


def test_websocket_accepts_originless_server_client(client) -> None:
    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        websocket.send_json({"type": "ping"})
        assert websocket.receive_json() == {"type": "pong"}


def test_websocket_rejects_unconfigured_browser_origin(client) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as exception,
        client.websocket_connect(
            "/ws/analyze", headers={"Origin": "https://attacker.example"}
        ),
    ):
        pass

    assert exception.value.code == 1008
    assert exception.value.reason == "WS_ORIGIN_FORBIDDEN"


def test_websocket_rejects_non_object_control_message(client) -> None:
    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        websocket.send_text("[]")
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["error"]["code"] == "WS_PROTOCOL_ERROR"


def test_websocket_requires_explicit_sample_rate(client) -> None:
    with client.websocket_connect("/ws/analyze") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "channels": 1,
            }
        )
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["error"]["code"] == "WS_PROTOCOL_ERROR"


def test_opted_in_rest_audio_is_manifested_and_deletable(
    settings, fake_estimator: FakeEstimator
) -> None:
    metadata = FakeMetadataStore()
    objects = FakeObjectStore()
    storage = PersistenceService(
        audio_settings(), metadata_store=metadata, object_store=objects
    )
    app = create_app(
        settings=settings,
        estimator=fake_estimator,
        persistence=storage,
    )
    audio = wav_bytes(speechlike_pcm())

    with TestClient(app) as persisted_client:
        response = persisted_client.post(
            "/analyze",
            content=audio,
            headers={
                "Content-Type": "audio/wav",
                "X-Persistence-Mode": "result_and_audio",
                "X-Consent-Reference": "consent-record-42",
            },
        )

        assert response.status_code == 200, response.text
        result = response.json()
        analysis_id = result["analysis_id"]
        assert result["persistence"]["status"] == "stored"
        assert result["persistence"]["chunks_stored"] == 1
        assert result["persistence"]["segments_stored"] == 1
        assert result["persistence"]["bytes_stored"] == len(audio)
        assert list(objects.objects.values()) == [audio]
        assert next(iter(metadata.sessions.values())).metadata == {
            "consent_reference_sha256": (
                "f14bcad92e2d14822a9399427db1fb0c8b41f3db899b6c12c9df46e97ff4ffae"
            )
        }

        detail = persisted_client.get(f"/analyses/{analysis_id}")
        assert detail.status_code == 200
        assert detail.headers["cache-control"] == "no-store"
        manifest = detail.json()
        assert manifest["segment_count"] == 1
        assert manifest["segments"][0]["logical_chunks"][0]["chunk_index"] == 0
        assert manifest["segments"][0]["sha256"]

        listing = persisted_client.get("/analyses")
        assert listing.status_code == 200
        assert listing.headers["cache-control"] == "no-store"
        assert listing.json()["items"][0]["analysis_id"] == analysis_id

        deleted = persisted_client.delete(f"/analyses/{analysis_id}")
        assert deleted.status_code == 200
        assert objects.objects == {}
        assert metadata.sessions == {}


def test_audio_retention_requires_consent_reference(
    settings, fake_estimator: FakeEstimator
) -> None:
    storage = PersistenceService(
        audio_settings(),
        metadata_store=FakeMetadataStore(),
        object_store=FakeObjectStore(),
    )
    app = create_app(
        settings=settings,
        estimator=fake_estimator,
        persistence=storage,
    )

    with TestClient(app) as persisted_client:
        response = persisted_client.post(
            "/analyze",
            content=wav_bytes(speechlike_pcm()),
            headers={
                "Content-Type": "audio/wav",
                "X-Persistence-Mode": "result_and_audio",
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONSENT_REFERENCE_REQUIRED"
    assert fake_estimator.calls == 0


def test_failed_retained_rest_request_deletes_partial_audio(settings) -> None:
    class FailingEstimator(FakeEstimator):
        def predict(self, samples: np.ndarray):
            del samples
            raise RuntimeError("model failed")

    metadata = FakeMetadataStore()
    objects = FakeObjectStore()
    storage = PersistenceService(
        audio_settings(), metadata_store=metadata, object_store=objects
    )
    app = create_app(
        settings=settings,
        estimator=FailingEstimator(),
        persistence=storage,
    )

    with TestClient(app, raise_server_exceptions=False) as persisted_client:
        response = persisted_client.post(
            "/analyze",
            content=wav_bytes(speechlike_pcm()),
            headers={
                "Content-Type": "audio/wav",
                "X-Persistence-Mode": "result_and_audio",
                "X-Consent-Reference": "consent-record-43",
            },
        )

    assert response.status_code == 500
    assert objects.objects == {}
    assert metadata.sessions == {}


def test_websocket_reports_physical_storage_progress(
    settings, fake_estimator: FakeEstimator
) -> None:
    metadata = FakeMetadataStore()
    objects = FakeObjectStore()
    storage = PersistenceService(
        audio_settings(), metadata_store=metadata, object_store=objects
    )
    app = create_app(
        settings=settings,
        estimator=fake_estimator,
        persistence=storage,
    )
    raw_pcm = (speechlike_pcm(2.4) * 32767.0).astype("<i2").tobytes()

    with TestClient(app) as persisted_client:
        with persisted_client.websocket_connect("/ws/analyze") as websocket:
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16_000,
                    "channels": 1,
                    "persistence_mode": "result_and_audio",
                    "consent_reference": "ws-consent-7",
                }
            )
            started = websocket.receive_json()
            assert started["type"] == "started"
            websocket.send_bytes(raw_pcm)
            progress = websocket.receive_json()
            assert progress["type"] == "storage"
            assert progress["persistence"]["chunks_received"] == 1
            assert progress["persistence"]["chunks_stored"] == 0
            assert progress["persistence"]["segments_stored"] == 2
            provisional = websocket.receive_json()
            assert provisional["type"] == "prediction"
            websocket.send_json({"type": "end"})
            final = websocket.receive_json()

        assert final["is_final"] is True
        assert final["persistence"]["status"] == "stored"
        assert final["persistence"]["chunks_stored"] == 1
        assert final["persistence"]["segments_stored"] == 3
        detail = persisted_client.get(f"/analyses/{final['analysis_id']}").json()
        assert detail["audio_bytes"] == len(raw_pcm)
        assert b"".join(objects.objects.values()) == raw_pcm
