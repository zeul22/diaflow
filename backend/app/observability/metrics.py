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
        # The autoscaling signal. Inference is memory-bandwidth-bound, so neither
        # replicas nor threads raise per-container capacity much: queue wait is
        # what actually indicates the service needs more containers, and it rises
        # before requests start being shed with SERVICE_BUSY.
        self.queue_wait = Histogram(
            "voice_attribute_queue_wait_seconds",
            "Time a request waited for a free model replica",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2),
            registry=self.registry,
        )
        self.queue_rejections = Counter(
            "voice_attribute_queue_rejections_total",
            "Requests shed because no replica became free within the deadline",
            registry=self.registry,
        )
        self.language_duration = Histogram(
            "voice_attribute_language_duration_seconds",
            "Language identification duration",
            ("backend",),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
            registry=self.registry,
        )
        # Aggregate coverage, so operators keep visibility into how often the
        # language is determined without a per-caller value reaching the logs.
        self.language_results = Counter(
            "voice_attribute_language_results_total",
            "Language identification outcomes",
            ("outcome",),
            registry=self.registry,
        )
        self.language_skipped = Counter(
            "voice_attribute_language_skipped_total",
            "Analyses that returned no language field",
            ("reason",),
            registry=self.registry,
        )
        self.speaker_homogeneity_downgrades = Counter(
            "voice_attribute_speaker_homogeneity_downgrades_total",
            "Analyses downgraded because sub-window embeddings disagreed on the speaker",
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "voice_attribute_rate_limited_total",
            "Requests rejected by the in-process rate limiter",
            ("transport",),
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
