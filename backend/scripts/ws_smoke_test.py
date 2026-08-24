#!/usr/bin/env python3
"""Exercise progressive and final predictions from a running WebSocket service."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct

import websockets


def synthetic_pcm(
    duration: float = 2.4,
    sample_rate: int = 16_000,
    encoding: str = "pcm_s16le",
) -> bytes:
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
        if encoding == "pcm_f32le":
            frames.extend(struct.pack("<f", sample))
        else:
            frames.extend(
                struct.pack("<h", max(-32768, min(32767, int(sample * 32767))))
            )
    return bytes(frames)


async def _next_prediction(socket, controls: list[dict]) -> dict:
    while True:
        payload = json.loads(await asyncio.wait_for(socket.recv(), timeout=10))
        if payload.get("type") == "prediction":
            return payload
        if payload.get("type") == "error":
            raise SystemExit(f"WebSocket service error: {payload}")
        if payload.get("type") in {"started", "storage", "pong"}:
            controls.append(payload)
            continue
        raise SystemExit(f"unexpected WebSocket message: {payload}")


async def run(
    url: str,
    encoding: str,
    persistence_mode: str,
    consent_reference: str,
) -> None:
    audio = synthetic_pcm(encoding=encoding)
    one_second = 16_000 * (4 if encoding == "pcm_f32le" else 2)
    controls: list[dict] = []
    async with websockets.connect(url, ping_interval=None) as socket:
        start = {
            "type": "start",
            "encoding": encoding,
            "sample_rate": 16_000,
            "channels": 1,
            "persistence_mode": persistence_mode,
        }
        if persistence_mode == "result_and_audio":
            start["consent_reference"] = consent_reference
        await socket.send(json.dumps(start))
        await socket.send(audio[:one_second])
        await socket.send(audio[one_second:])
        progressive = await _next_prediction(socket, controls)
        await socket.send(json.dumps({"type": "end"}))
        final = await _next_prediction(socket, controls)

    if progressive.get("type") != "prediction" or progressive.get("is_final"):
        raise SystemExit("missing progressive prediction")
    if final.get("type") != "prediction" or not final.get("is_final"):
        raise SystemExit("missing final prediction")
    if final.get("sequence") != progressive.get("sequence", 0) + 1:
        raise SystemExit("prediction sequence is not monotonic")
    if persistence_mode != "none":
        if not any(item.get("type") == "started" for item in controls):
            raise SystemExit("retained stream did not receive a started receipt")
        if final.get("persistence", {}).get("status") != "stored":
            raise SystemExit("retained stream did not receive a final stored receipt")
    if persistence_mode == "result_and_audio" and not any(
        item.get("type") == "storage" for item in controls
    ):
        raise SystemExit("retained audio stream did not receive storage progress")
    print(
        json.dumps(
            {"storage_messages": controls, "progressive": progressive, "final": final},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/analyze")
    parser.add_argument(
        "--encoding",
        choices=("pcm_s16le", "pcm_f32le"),
        default="pcm_s16le",
    )
    parser.add_argument(
        "--persistence-mode",
        choices=("none", "result", "result_and_audio"),
        default="none",
    )
    parser.add_argument("--consent-reference", default="synthetic-ws-smoke-test")
    args = parser.parse_args()
    asyncio.run(
        run(
            args.url,
            args.encoding,
            args.persistence_mode,
            args.consent_reference,
        )
    )


if __name__ == "__main__":
    main()
