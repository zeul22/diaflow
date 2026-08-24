"""Reordering, duplicate suppression, and loss concealment for streamed audio.

A WebSocket runs over TCP, so within one connection bytes arrive in order. That
guarantee stops at the socket. Real logistics audio reaches this service through
a gateway that received RTP over a mobile network, and RTP loses and reorders
packets. If the gateway forwards what it received without repair, the stream this
service sees has gaps and swapped frames with no marker saying so -- and the
result is silently wrong rather than reported as degraded.

Sequenced framing lets a client label each frame so the damage becomes visible:
out-of-order frames are put back in place, duplicates are dropped, and gaps are
concealed *and counted* so the quality gate can downgrade a badly damaged
session instead of confidently describing a voice reconstructed from fragments.

This is deliberately a small reorder window, not a full adaptive jitter buffer:
holding audio longer to repair more damage costs exactly the latency the service
exists to avoid.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamIntegrity:
    """What the transport did to the audio, for logging and the quality gate."""

    frames_received: int = 0
    frames_duplicated: int = 0
    frames_reordered: int = 0
    frames_lost: int = 0
    frames_concealed: int = 0
    bytes_concealed: int = 0

    @property
    def expected(self) -> int:
        return self.frames_received + self.frames_lost

    @property
    def loss_ratio(self) -> float:
        if self.expected <= 0:
            return 0.0
        return self.frames_lost / self.expected


@dataclass(slots=True)
class ReorderBuffer:
    """Emit sequenced frames in order, concealing what never arrives.

    ``window`` frames may be held while waiting for a late one. A frame that
    arrives after its slot has already been released is dropped rather than
    inserted out of position, because feeding stale audio into a progressive
    estimate is worse than the gap it would fill.
    """

    window: int = 8
    max_repeat_frames: int = 2
    _pending: dict[int, bytes] = field(default_factory=dict)
    _next: int = 0
    _last_payload: bytes = b""
    _repeats: int = 0
    _frame_bytes: int = 0
    received: int = 0
    duplicated: int = 0
    reordered: int = 0
    lost: int = 0
    concealed: int = 0
    concealed_bytes: int = 0
    highest: int = -1

    def push(self, sequence: int, payload: bytes) -> list[bytes]:
        """Accept one sequenced frame, returning whatever is now in order."""

        if sequence < 0:
            raise ValueError("Frame sequence numbers cannot be negative")
        if sequence < self._next or sequence in self._pending:
            # Already released or already held: a retransmission or a duplicate.
            self.duplicated += 1
            return []
        if sequence < self.highest:
            self.reordered += 1
        self.highest = max(self.highest, sequence)
        self.received += 1
        self._pending[sequence] = payload
        if payload:
            self._frame_bytes = len(payload)
        return list(self._drain())

    def _drain(self) -> Iterator[bytes]:
        while self._pending:
            if self._next in self._pending:
                payload = self._pending.pop(self._next)
                self._next += 1
                self._last_payload = payload
                self._repeats = 0
                yield payload
                continue
            # The next frame is missing. Wait only until the window is full,
            # then declare it lost and keep the stream moving.
            if len(self._pending) <= self.window:
                return
            yield self._conceal()

    def _conceal(self) -> bytes:
        """Fill one missing frame and account for it."""

        self.lost += 1
        self._next += 1
        # Low-complexity concealment in the spirit of G.711 Appendix I: repeat
        # the previous frame briefly, then fall silent. Repeating for longer
        # produces a buzzing artifact that a paralinguistic model would happily
        # treat as voice.
        if self._last_payload and self._repeats < self.max_repeat_frames:
            self._repeats += 1
            self.concealed += 1
            self.concealed_bytes += len(self._last_payload)
            return self._last_payload
        filler = bytes(self._frame_bytes)
        if filler:
            self.concealed += 1
            self.concealed_bytes += len(filler)
        return filler

    def flush(self) -> list[bytes]:
        """Release everything still held, concealing any remaining gaps."""

        released: list[bytes] = []
        while self._pending:
            if self._next in self._pending:
                payload = self._pending.pop(self._next)
                self._next += 1
                self._last_payload = payload
                self._repeats = 0
                released.append(payload)
                continue
            released.append(self._conceal())
        return released

    def integrity(self) -> StreamIntegrity:
        return StreamIntegrity(
            frames_received=self.received,
            frames_duplicated=self.duplicated,
            frames_reordered=self.reordered,
            frames_lost=self.lost,
            frames_concealed=self.concealed,
            bytes_concealed=self.concealed_bytes,
        )


def parse_sequenced_frame(frame: bytes | bytearray) -> tuple[int, bytes]:
    """Split a ``seq32`` frame into its sequence number and audio payload.

    Framing is four bytes of big-endian sequence followed by the audio, chosen
    over a JSON envelope per chunk because it adds four bytes rather than
    doubling the message count on the hot path.
    """

    if len(frame) < 4:
        raise ValueError("A sequenced frame needs a 4-byte header")
    return int.from_bytes(bytes(frame[:4]), "big"), bytes(frame[4:])
