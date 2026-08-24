#!/usr/bin/env python3
"""Generate a synthetic voiced signal and exercise POST /analyze."""

from __future__ import annotations

import argparse
import io
import json
import math
import struct
import urllib.request
import wave


def synthetic_wav(duration: float = 3.0, sample_rate: int = 16_000) -> bytes:
    output = io.BytesIO()
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
            + 0.04 * math.sin(4.0 * phase)
        )
        frames.extend(struct.pack("<h", max(-32768, min(32767, int(sample * 32767)))))
    with wave.open(output, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(frames)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/analyze",
        data=synthetic_wav(),
        headers={"Content-Type": "audio/wav"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.loads(response.read())
    print(json.dumps(payload, indent=2))
    if set(payload) != {
        "contact_id",
        "gender",
        "age_bracket",
        "processing_ms",
        "audio_quality",
    }:
        raise SystemExit("response did not match the expected contract")


if __name__ == "__main__":
    main()
