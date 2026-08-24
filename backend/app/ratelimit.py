"""Per-client request rate limiting.

This is **defence in depth, not the primary control.** A limiter inside the
application cannot protect the application from traffic that has already reached
it: the connection, the TLS handshake, and the request body have all been paid
for by the time this code runs. The primary control belongs at the ingress or
WAF, which can drop traffic before it costs anything.

What an in-process limiter does buy is a bound on *expensive* work per client
even when the ingress is misconfigured or absent, which matters here because one
`/analyze` call is ~230 ms of memory-bandwidth-bound inference and the service
sheds load at only ~4 analyses/second per container.

Two properties this implementation is careful about:

* **The limit is per container, not global.** With N containers behind a load
  balancer a client can make N times the configured rate. A global limit needs
  shared state (Redis or the ingress); see docs/AUDIO_PIPELINE.md.
* **The client table is bounded.** Keying an unbounded dictionary by client
  address is itself a denial-of-service vector, so the table has a cap and evicts
  the least recently seen entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_seen: float


@dataclass(slots=True)
class RateLimiter:
    """Token bucket per client key.

    ``capacity`` is the burst a client may spend at once; ``refill_per_second``
    is the sustained rate it recovers at. A burst larger than the sustained rate
    is deliberate: real clients arrive unevenly, and a limiter that rejects the
    second of two adjacent requests is a worse experience than one that allows a
    short burst and then throttles.
    """

    capacity: float
    refill_per_second: float
    max_clients: int = 10_000
    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    def check(self, key: str, now: float) -> float:
        """Consume one token for ``key``.

        Returns ``0.0`` when the request is allowed, otherwise the number of
        seconds until a token is available, for a ``Retry-After`` header.
        """

        bucket = self._buckets.get(key)
        if bucket is None:
            self._evict_if_full(now)
            self._buckets[key] = _Bucket(tokens=self.capacity - 1.0, last_seen=now)
            return 0.0

        elapsed = max(0.0, now - bucket.last_seen)
        bucket.tokens = min(
            self.capacity, bucket.tokens + elapsed * self.refill_per_second
        )
        bucket.last_seen = now
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return 0.0
        if self.refill_per_second <= 0.0:
            return float("inf")
        return (1.0 - bucket.tokens) / self.refill_per_second

    def _evict_if_full(self, now: float) -> None:
        if len(self._buckets) < self.max_clients:
            return
        # Drop the stalest tenth in one pass rather than one entry per arrival,
        # so a flood of unique keys cannot turn every request into a full scan.
        victims = sorted(self._buckets, key=lambda key: self._buckets[key].last_seen)
        for key in victims[: max(1, self.max_clients // 10)]:
            del self._buckets[key]

    @property
    def tracked_clients(self) -> int:
        return len(self._buckets)


def client_key(peer: str | None, forwarded_for: str | None, trusted_hops: int) -> str:
    """Identify the client to rate limit.

    Behind a proxy the peer address is the proxy, so every caller would share one
    bucket. ``trusted_hops`` says how many proxies append to ``X-Forwarded-For``,
    and the address that many hops from the right is the first one those proxies
    did not add -- the earlier entries are client-supplied and forgeable.

    With ``trusted_hops`` at 0 the header is ignored entirely, which is correct
    when the service is reached directly or when uvicorn's ``--proxy-headers``
    has already rewritten the peer address.
    """

    if trusted_hops > 0 and forwarded_for:
        hops = [item.strip() for item in forwarded_for.split(",") if item.strip()]
        if hops:
            index = max(0, len(hops) - trusted_hops)
            return hops[index]
    return peer or "unknown"
