from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.http_requests = Counter(
            "voice_attribute_http_requests_total",
            "HTTP requests completed",
            ("path", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "voice_attribute_http_request_duration_seconds",
            "End-to-end HTTP request duration",
            ("path",),
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            "voice_attribute_inference_duration_seconds",
            "Model inference duration",
            ("backend",),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
            registry=self.registry,
        )
        self.audio_quality = Counter(
            "voice_attribute_audio_quality_total",
            "Analyzed requests by quality gate result",
            ("quality",),
            registry=self.registry,
        )
        self.errors = Counter(
            "voice_attribute_errors_total",
            "Service errors by stable code",
            ("code",),
            registry=self.registry,
        )
        self.inference_inflight = Gauge(
            "voice_attribute_inference_inflight",
            "Model inferences currently running",
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
