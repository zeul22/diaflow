#!/usr/bin/env python3
"""Exercise progressive and final predictions from a running WebSocket service."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct

import websockets


def synthetic_pcm(duration: float = 2.4, sample_rate: int = 16_000) -> bytes:
    frames = bytearray()
    for index in range(int(duration * sample_rate)):
        time = index / sample_rate
        phase = (
            2.0 * math.pi * (125.0 * time + 2.0 * math.sin(2.0 * math.pi * 0.8 * time))
        )
        envelope = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(2.0 * math.pi * 2.3 * time)) ** 2
        sample = envelope * (
            0.28 * math.sin(phase)
            + 0.13 * math.sin(2.0 * phase)
            + 0.08 * math.sin(3.0 * phase)
        )
        frames.extend(struct.pack("<h", max(-32768, min(32767, int(sample * 32767)))))
    return bytes(frames)


async def run(url: str) -> None:
    audio = synthetic_pcm()
    one_second = 16_000 * 2
    async with websockets.connect(url, ping_interval=None) as socket:
        await socket.send(
            json.dumps(
                {
                    "type": "start",
                    "encoding": "pcm_s16le",
                    "sample_rate": 16_000,
                    "channels": 1,
                }
            )
        )
        await socket.send(audio[:one_second])
        await socket.send(audio[one_second:])
        progressive = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
        await socket.send(json.dumps({"type": "end"}))
        final = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))

    if progressive.get("type") != "prediction" or progressive.get("is_final"):
        raise SystemExit("missing progressive prediction")
    if final.get("type") != "prediction" or not final.get("is_final"):
        raise SystemExit("missing final prediction")
    if final.get("sequence") != progressive.get("sequence", 0) + 1:
        raise SystemExit("prediction sequence is not monotonic")
    print(json.dumps({"progressive": progressive, "final": final}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/analyze")
    args = parser.parse_args()
    asyncio.run(run(args.url))


if __name__ == "__main__":
    main()
