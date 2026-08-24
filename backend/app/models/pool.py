from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)


class Replica(Protocol):
    @property
    def name(self) -> str: ...

    def warmup(self) -> None: ...


ReplicaT = TypeVar("ReplicaT", bound=Replica)


class EstimatorPool(Generic[ReplicaT]):
    """A fixed set of model replicas, one per permitted concurrent inference.

    A single estimator instance serializes on its own lock, so a semaphore of
    two over one instance never produced two concurrent inferences: it only
    queued callers behind that lock while reporting a higher concurrency. The
    pool makes the trade explicit -- ``INFERENCE_CONCURRENCY`` is the replica
    count, and each replica costs another copy of the model's memory.
    """

    def __init__(self, estimators: Sequence[ReplicaT]) -> None:
        if not estimators:
            raise ValueError("An estimator pool needs at least one replica")
        self._estimators = tuple(estimators)
        self._available: asyncio.LifoQueue[ReplicaT] | None = None

    @property
    def name(self) -> str:
        return self._estimators[0].name

    @property
    def size(self) -> int:
        return len(self._estimators)

    def warmup(self) -> None:
        for estimator in self._estimators:
            estimator.warmup()

    def _queue(self) -> asyncio.LifoQueue[ReplicaT]:
        # Bound the queue to the running loop on first use: the pool is built in
        # a worker thread during startup, before the serving loop owns it.
        if self._available is None:
            queue: asyncio.LifoQueue[ReplicaT] = asyncio.LifoQueue()
            for estimator in self._estimators:
                queue.put_nowait(estimator)
            self._available = queue
        return self._available

    async def acquire(self, timeout: float) -> ReplicaT:
        """Take a replica, waiting at most ``timeout`` seconds.

        Raises ``TimeoutError`` so the caller can translate it into the existing
        503 busy contract rather than queueing without a deadline.
        """

        queue = self._queue()
        if queue.qsize() > 0:
            return queue.get_nowait()
        return await asyncio.wait_for(queue.get(), timeout=timeout)

    def release(self, estimator: ReplicaT) -> None:
        self._queue().put_nowait(estimator)


def as_pool(estimator: ReplicaT | EstimatorPool[ReplicaT]) -> EstimatorPool[ReplicaT]:
    if isinstance(estimator, EstimatorPool):
        return estimator
    return EstimatorPool([estimator])


def warn_on_thread_oversubscription(replicas: int, threads_per_replica: int) -> None:
    available = os.cpu_count() or 1
    requested = replicas * threads_per_replica
    if requested > available:
        logger.warning(
            "inference_threads_oversubscribed",
            extra={
                "event_data": {
                    "replicas": replicas,
                    "threads_per_replica": threads_per_replica,
                    "requested_threads": requested,
                    "available_cpus": available,
                }
            },
        )
