from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import Settings
from app.main import create_app
from app.ratelimit import RateLimiter, client_key
from tests.conftest import FakeEstimator


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings=settings, estimator=FakeEstimator()))


# --------------------------------------------------------------------------
# Token bucket
# --------------------------------------------------------------------------


def test_burst_is_allowed_then_throttled() -> None:
    limiter = RateLimiter(capacity=3.0, refill_per_second=1.0)

    assert [limiter.check("a", 0.0) for _ in range(3)] == [0.0, 0.0, 0.0]
    retry_after = limiter.check("a", 0.0)
    assert retry_after == pytest.approx(1.0)


def test_tokens_refill_over_time() -> None:
    limiter = RateLimiter(capacity=2.0, refill_per_second=1.0)
    limiter.check("a", 0.0)
    limiter.check("a", 0.0)
    assert limiter.check("a", 0.0) > 0.0

    # One second later exactly one token is available again.
    assert limiter.check("a", 1.0) == 0.0
    assert limiter.check("a", 1.0) > 0.0


def test_clients_are_independent() -> None:
    limiter = RateLimiter(capacity=1.0, refill_per_second=1.0)

    assert limiter.check("a", 0.0) == 0.0
    assert limiter.check("b", 0.0) == 0.0
    assert limiter.check("a", 0.0) > 0.0


def test_the_client_table_is_bounded() -> None:
    """An unbounded table keyed by client address is itself a DoS vector."""

    limiter = RateLimiter(capacity=1.0, refill_per_second=1.0, max_clients=100)

    for index in range(1_000):
        limiter.check(f"client-{index}", float(index))

    assert limiter.tracked_clients <= 100


@pytest.mark.parametrize(
    ("peer", "forwarded", "hops", "expected"),
    [
        # No trusted proxy: the header is ignored entirely, so a client cannot
        # forge its identity by sending one.
        ("10.0.0.1", "1.2.3.4", 0, "10.0.0.1"),
        # One trusted proxy appended the real client address.
        ("10.0.0.1", "1.2.3.4", 1, "1.2.3.4"),
        # A forged prefix is ignored; the rightmost trusted hop wins.
        ("10.0.0.1", "9.9.9.9, 1.2.3.4", 1, "1.2.3.4"),
        ("10.0.0.1", "9.9.9.9, 1.2.3.4, 10.0.0.9", 2, "1.2.3.4"),
        (None, None, 0, "unknown"),
    ],
)
def test_client_key_respects_trusted_hops(peer, forwarded, hops, expected) -> None:
    assert client_key(peer, forwarded, hops) == expected


# --------------------------------------------------------------------------
# Rate limiting over HTTP and WebSocket
# --------------------------------------------------------------------------


def test_requests_over_budget_get_429_with_retry_after(settings) -> None:
    limited = replace(settings, rate_limit_burst=2, rate_limit_requests_per_minute=60.0)

    with _client(limited) as client:
        assert client.get("/v1/analyses").status_code in (200, 503)
        client.get("/v1/analyses")
        response = client.get("/v1/analyses")

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "RATE_LIMITED"
    assert int(response.headers["retry-after"]) >= 1


def test_probes_and_metrics_are_never_throttled(settings) -> None:
    """Throttling a probe removes the container from service and blinds the
    scraper that would explain why."""

    limited = replace(settings, rate_limit_burst=1, rate_limit_requests_per_minute=60.0)

    with _client(limited) as client:
        client.get("/v1/analyses")  # spend the single token
        for _ in range(5):
            assert client.get("/healthz").status_code == 200
            assert client.get("/readyz").status_code == 200
            assert client.get("/metrics").status_code == 200


def test_rate_limiting_can_be_disabled(settings) -> None:
    unlimited = replace(settings, rate_limit_enabled=False, rate_limit_burst=1)

    with _client(unlimited) as client:
        for _ in range(5):
            assert client.get("/v1/analyses").status_code != 429


def test_websocket_over_budget_is_closed_before_accept(settings) -> None:
    """An over-budget client must not get a session that then holds a replica."""

    limited = replace(settings, rate_limit_burst=1, rate_limit_requests_per_minute=60.0)

    with _client(limited) as client:
        client.get("/v1/analyses")  # spend the single token
        with (
            pytest.raises(WebSocketDisconnect) as caught,
            client.websocket_connect("/v1/ws/analyze") as websocket,
        ):
            websocket.send_json(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16_000,
                    "channels": 1,
                }
            )
            websocket.receive_json()

    # 1013 "try again later" is the WebSocket analogue of HTTP 429.
    assert caught.value.code == 1013


def test_websocket_within_budget_still_connects(settings) -> None:
    with (
        _client(settings) as client,
        client.websocket_connect("/v1/ws/analyze") as websocket,
    ):
        websocket.send_json(
            {
                "type": "start",
                "encoding": "pcm_s16le",
                "sample_rate": 16_000,
                "channels": 1,
            }
        )
        websocket.send_bytes(b"\x00\x00" * 16_000)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "prediction"


# --------------------------------------------------------------------------
# Versioning
# --------------------------------------------------------------------------


def test_versioned_and_legacy_paths_both_serve(settings) -> None:
    with _client(settings) as client:
        versioned = client.get("/v1/persistence/capabilities")
        legacy = client.get("/persistence/capabilities")

    assert versioned.status_code == 200
    assert legacy.status_code == 200
    assert versioned.json() == legacy.json()


def test_only_the_versioned_surface_is_advertised(settings) -> None:
    """Legacy aliases keep integrations working but are not what /docs offers."""

    with _client(settings) as client:
        paths = set(client.get("/openapi.json").json()["paths"])

    assert "/v1/analyze" in paths
    assert "/analyze" not in paths
    assert all(path.startswith("/v1/") for path in paths)


def test_metric_labels_are_shared_across_versions(settings) -> None:
    """One dashboard has to span the migration, so the prefix is stripped."""

    with _client(settings) as client:
        client.get("/v1/analyses")
        client.get("/analyses")
        metrics = client.get("/metrics").text

    assert 'path="/analyses"' in metrics
    assert 'path="/v1/analyses"' not in metrics


# --------------------------------------------------------------------------
# Transport security headers
# --------------------------------------------------------------------------


def test_baseline_security_headers_are_always_present(settings) -> None:
    with _client(settings) as client:
        response = client.get("/readyz")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_hsts_is_only_sent_over_https_and_when_configured(settings) -> None:
    """Announcing HSTS on a plaintext response is ignored and untrue."""

    with _client(replace(settings, hsts_max_age_seconds=0)) as client:
        assert (
            "strict-transport-security"
            not in client.get("/readyz", headers={"X-Forwarded-Proto": "https"}).headers
        )

    with _client(replace(settings, hsts_max_age_seconds=31_536_000)) as client:
        plaintext = client.get("/readyz")
        forwarded = client.get("/readyz", headers={"X-Forwarded-Proto": "https"})

    assert "strict-transport-security" not in plaintext.headers
    assert forwarded.headers["strict-transport-security"] == "max-age=31536000"


def test_rate_limit_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="RATE_LIMIT_REQUESTS_PER_MINUTE"):
        replace(Settings(), rate_limit_requests_per_minute=0.0).validate()
    with pytest.raises(ValueError, match="RATE_LIMIT_BURST"):
        replace(Settings(), rate_limit_burst=0).validate()
    with pytest.raises(ValueError, match="TRUSTED_PROXY_HOPS"):
        replace(Settings(), trusted_proxy_hops=99).validate()
    with pytest.raises(ValueError, match="HSTS_MAX_AGE_SECONDS"):
        replace(Settings(), hsts_max_age_seconds=-1).validate()


def test_analysis_logs_carry_no_predicted_attributes(settings, caplog) -> None:
    """docs/PRIVACY.md promises logs contain no predictions.

    The identified language is an inference about the caller and was briefly
    logged per request; coverage now lives in an aggregate metric instead.
    """

    import json
    import logging

    from app.models.pool import EstimatorPool
    from tests.conftest import speechlike_pcm, wav_bytes
    from tests.test_language import FakeIdentifier

    app = create_app(
        settings=settings,
        estimator=FakeEstimator(),
        language_pool=EstimatorPool([FakeIdentifier(code="hi", confidence=0.9)]),
    )
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        payload = client.post(
            "/v1/analyze",
            content=wav_bytes(speechlike_pcm()),
            headers={"Content-Type": "audio/wav"},
        ).json()
        metrics = client.get("/metrics").text

    # The response carries the prediction; the logs must not.
    assert payload["language"]["prediction"] == "hi"
    analysis_events = [
        record for record in caplog.records if record.msg == "analysis_completed"
    ]
    assert analysis_events
    for record in analysis_events:
        event = getattr(record, "event_data", {})
        assert "language" not in event
        assert "gender" not in event
        assert "age_bracket" not in event
        assert "debug_age_years" not in event
        # Belt and braces: the predicted value must not appear under any key.
        assert "hi" not in json.dumps(event)

    # Coverage is still observable, in aggregate.
    assert 'voice_attribute_language_results_total{outcome="determined"}' in metrics


def test_successful_probes_are_not_logged_but_failures_are(settings, caplog) -> None:
    """A succeeding probe is the noisiest, least informative line in the log.

    The container healthcheck alone produces ~5,800 per day per container. They
    stay in metrics, where a rate belongs. A failing probe still logs, because it
    is what explains a container leaving the load balancer.
    """

    import logging

    with caplog.at_level(logging.INFO), _client(settings) as client:
        client.get("/healthz")
        client.get("/readyz")
        client.get("/metrics")
        client.get("/v1/analyses")

    logged = [
        getattr(record, "event_data", {}).get("path")
        for record in caplog.records
        if record.msg == "http_request_completed"
    ]
    assert "/analyses" in logged
    assert "/healthz" not in logged
    assert "/readyz" not in logged
    assert "/metrics" not in logged


def test_probe_traffic_is_still_counted_in_metrics(settings) -> None:
    with _client(settings) as client:
        client.get("/readyz")
        client.get("/readyz")
        metrics = client.get("/metrics").text

    assert 'voice_attribute_http_requests_total{path="/readyz",status="200"} 2.0' in (
        metrics
    )


def test_a_failing_readiness_probe_is_logged(settings, caplog) -> None:
    import logging

    app = create_app(settings=settings, estimator=FakeEstimator())
    # Before startup the analysis service is absent, so readiness reports 503.
    with caplog.at_level(logging.INFO):
        client = TestClient(app)
        response = client.get("/readyz")

    assert response.status_code == 503
    failures = [
        record
        for record in caplog.records
        if record.msg == "http_request_completed"
        and getattr(record, "event_data", {}).get("status") == 503
    ]
    assert failures
