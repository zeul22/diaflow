from __future__ import annotations

from uuid import uuid4

import numpy as np

from tests.conftest import FakeEstimator, speechlike_pcm, wav_bytes


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
