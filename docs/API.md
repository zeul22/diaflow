# API reference

Base URL in the supplied Compose deployment: `http://localhost:8000`.

The application has no built-in TLS, authentication, tenant authorization, or rate limiting. Put it behind an authenticated TLS ingress before accepting real caller audio.

## POST `/analyze`

Analyze one bounded audio body. The HTTP request body is consumed incrementally, but inference begins after the complete upload has arrived. The default maximum is 12 MiB of audio data, 30 decoded seconds, and a five-second inference window.

### Contact identifier

`contact_id` is optional and must be a UUID. Supply it by query parameter, `X-Contact-ID` header, or multipart text field. If omitted, the service creates a UUID. Conflicting values are rejected. This identifier is returned but is deliberately absent from application logs.

### Retention policy

Retention is per request and defaults to `none`, even when the deployment has PostgreSQL/S3 enabled:

| `X-Persistence-Mode` | Stored data | Extra requirement |
| --- | --- | --- |
| `none` | Nothing; request buffers are wiped best-effort. | None; this is the default. |
| `result` | Structured result and analysis metadata in PostgreSQL. | None. |
| `result_and_audio` | Result/manifest in PostgreSQL and encoded bytes in S3-compatible storage. | Non-empty `X-Consent-Reference`. Only its SHA-256 is stored. |

Multipart clients may use `persistence_mode` and `consent_reference` text fields instead. Conflicting header/field values are rejected. If explicitly requested persistence cannot be committed, the request fails with `STORAGE_UNAVAILABLE`; the service never silently returns a “stored” receipt.

### Container or compressed audio

Send a supported `audio/*` content type, such as WAV, MP3, FLAC, Ogg, or WebM. WAV is decoded natively when possible; other containers/codecs use the bundled FFmpeg process through stdin/stdout. Actual FFmpeg support depends on the Debian package build.

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?contact_id=123e4567-e89b-12d3-a456-426614174000' \
  -H 'Content-Type: audio/wav' \
  --data-binary @samples/caller.wav
```

HTTP `Content-Encoding` compression, including gzip, is rejected. Codec compression inside an audio container is supported.

### Multipart upload

The audio field may be named `audio` or `file`; exactly one non-empty audio part is required. Supported text fields are `contact_id`, `encoding`, `sample_rate`, `channels`, `persistence_mode`, and `consent_reference`.

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -F 'audio=@samples/caller.mp3;type=audio/mpeg' \
  -F 'contact_id=123e4567-e89b-12d3-a456-426614174000'
```

Multipart parsing is streaming and bounded. The parser accepts at most eight parts, 256 bytes per recognized text field, 2 KiB per header value, 256 KiB of total multipart overhead by default, and one audio part.

### Headerless telephony audio

Raw audio needs explicit encoding, rate, and channel metadata in the query, headers, or multipart fields. Query metadata takes precedence over headers, which take precedence over fields.

| Encoding value | Aliases | Bytes per sample | Typical rate |
| --- | --- | ---: | ---: |
| `pcm_s16le` | `pcm16`, `s16le` | 2 | 16000 |
| `pcm_s16be` | `s16be`; inferred by `audio/l16` | 2 | 16000 |
| `pcm_f32le` | `f32le` | 4 | 16000 |
| `mulaw` | `pcmu`, `mu-law`, `ulaw` | 1 | 8000 |
| `alaw` | `pcma`, `a-law` | 1 | 8000 |

`sample_rate` must be 8000–96000 and `channels` must be 1 or 2. Stereo is averaged to mono, then audio is resampled to 16 kHz. Raw sample alignment is validated.

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?encoding=mulaw&sample_rate=8000&channels=1' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @samples/caller.mulaw
```

The same metadata can be sent as `X-Audio-Encoding`, `X-Audio-Sample-Rate`, and `X-Audio-Channels` headers.

### Caller-only requirement

The payload must contain one contact speaker. There is no source separation, echo cancellation, or diarization. For inbound calls, route the remote caller leg; for outbound calls, route the callee/driver/customer leg. Do not send a mono mix containing the AI voice, dispatcher, hold music, or multiple people. A mixed signal may describe whichever voice dominates the selected speech window.

### Success response

```json
{
  "contact_id": "123e4567-e89b-12d3-a456-426614174000",
  "gender": {
    "prediction": "male",
    "confidence": 0.87
  },
  "age_bracket": {
    "prediction": "31-45",
    "confidence": 0.63
  },
  "processing_ms": 142,
  "audio_quality": "good"
}
```

With retained data, two optional fields are added:

```json
{
  "analysis_id": "7856a994-08e1-49a5-b31f-ffdbccf50345",
  "persistence": {
    "mode": "result_and_audio",
    "status": "stored",
    "chunks_received": 1,
    "chunks_stored": 1,
    "segments_stored": 1,
    "bytes_stored": 82413,
    "audio_expires_at": "2026-08-25T12:00:00Z",
    "result_expires_at": "2026-08-25T12:00:00Z"
  }
}
```

These fields extend the normal response; the abbreviated object above shows only the retention portion. Result-only sessions report zero audio chunks/segments/bytes and use their longer result TTL.

`gender.prediction` is `male`, `female`, or `unknown`. It is perceived binary voice presentation under the upstream labels, not gender identity. `age_bracket.prediction` is `18-30`, `31-45`, `46-60`, `60+`, or `unknown`. Ages regressed outside the service's plausible adult range of 18–120 become unknown.

Confidence is bounded to `[0,1]` but is not yet calibrated on logistics calls. If a prediction misses its configured threshold, the service returns `unknown` with confidence `0.0`; it does not expose the sub-threshold score. `processing_ms` covers decode, quality analysis, queueing, and inference inside the service.

`audio_quality` means:

- `good`: enough estimated speech and no configured degradation trigger.
- `degraded`: enough speech, but short/low-speech, noisy, clipped, low-frequency-heavy, or narrowband. Confidence is discounted.
- `insufficient`: too short, too little estimated voice, too quiet, or almost no speech. Both attributes are unknown and the model is skipped.

Quality decisions use heuristics, not transcription or speaker verification. A `good` label does not guarantee a correct attribute estimate.

### Request and response IDs

Send `X-Request-ID` with 1–80 ASCII letters, digits, `.`, `_`, or `-`. Otherwise the server generates one. It is returned in `X-Request-ID` and appears in logs/error responses. It is distinct from `contact_id`.

### Error response

```json
{
  "error": {
    "code": "INVALID_AUDIO",
    "message": "The audio payload could not be decoded",
    "request_id": "f34017c8-6c62-4f03-8f1c-d5db0ff6a91c"
  }
}
```

| HTTP | Stable code | Meaning |
| ---: | --- | --- |
| 408 | `REQUEST_BODY_TIMEOUT` | The next HTTP body chunk did not arrive before the idle deadline. |
| 413 | `PAYLOAD_TOO_LARGE` | Encoded payload or WebSocket buffer exceeds the byte limit. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Media type, explicit codec, or HTTP content encoding is unsupported. |
| 422 | `MISSING_AUDIO` | No audio body/part/chunk was provided. |
| 422 | `INVALID_AUDIO` | Decode failed, data is malformed/misaligned, or decoded duration is too long. |
| 422 | `INVALID_CONTACT_ID`, `CONFLICTING_CONTACT_ID` | Contact UUID is invalid or sources disagree. |
| 422 | `INVALID_PERSISTENCE_MODE`, `CONFLICTING_PERSISTENCE_MODE` | Retention policy is invalid or sources disagree. |
| 422 | `CONSENT_REFERENCE_REQUIRED`, `INVALID_CONSENT_REFERENCE`, `CONFLICTING_CONSENT_REFERENCE` | Audio retention lacks a valid opaque consent/policy reference, or supplied references disagree. |
| 422 | `INVALID_*`, `AUDIO_TOO_LONG`, multipart protocol codes | Metadata, size, duration, validation, or multipart constraints failed. |
| 503 | `SERVICE_BUSY` | The inference semaphore was not acquired before the queue timeout. Retry with bounded jitter. |
| 503 | `MODEL_UNAVAILABLE` | Startup/shutdown readiness prevents inference. |
| 503 | `PERSISTENCE_NOT_AVAILABLE`, `STORAGE_UNAVAILABLE`, `STORAGE_ERROR` | Opted-in retention is disabled or could not be committed. No success receipt is returned. |
| 500 | `INTERNAL_ERROR` | Unexpected failure; use `request_id` to correlate server logs. |

## WebSocket `/ws/analyze`

The WebSocket endpoint accepts headerless raw PCM/G.711 only; it does not accept streaming MP3, WAV, WebM, or other containers. Binary frames are cumulative for inference and are erased when the session ends unless that start message explicitly opts into audio retention. The first frame must be JSON text within ten seconds:

```json
{
  "type": "start",
  "contact_id": "123e4567-e89b-12d3-a456-426614174000",
  "encoding": "pcm_s16le",
  "sample_rate": 16000,
  "channels": 1,
  "persistence_mode": "none"
}
```

`contact_id` is optional. `encoding` and `sample_rate` are required; channels defaults to one. Encoding must be `pcm_s16le`, `pcm_s16be`, `pcm_f32le`, `mulaw`, or `alaw`. `persistence_mode` defaults to `none`; `result_and_audio` also requires `consent_reference`. Send binary audio frames next. After at least 1.25 seconds, the server can emit another cumulative prediction for each configured one second of new audio. Finish with the exact text control object `{"type":"end"}`. `{"type":"ping"}` produces `{"type":"pong"}`. The server closes sessions after 30 seconds without a frame or 120 seconds total by default, returning `WS_IDLE_TIMEOUT` or `WS_SESSION_TIMEOUT`.

For a retained session, the server first confirms its database identity:

```json
{
  "type": "started",
  "contact_id": "123e4567-e89b-12d3-a456-426614174000",
  "analysis_id": "7856a994-08e1-49a5-b31f-ffdbccf50345",
  "persistence": {"mode": "result_and_audio", "status": "pending"}
}
```

As network frames arrive it reports logical-versus-physical progress. Four typical 250 ms frames become about one one-second object, so `chunks_received` can lead `segments_stored`:

```json
{
  "type": "storage",
  "analysis_id": "7856a994-08e1-49a5-b31f-ffdbccf50345",
  "persistence": {
    "mode": "result_and_audio",
    "status": "pending",
    "chunks_received": 4,
    "chunks_stored": 4,
    "segments_stored": 1,
    "bytes_stored": 192000
  }
}
```

Prediction messages extend the REST response:

```json
{
  "type": "prediction",
  "sequence": 2,
  "is_final": true,
  "contact_id": "123e4567-e89b-12d3-a456-426614174000",
  "gender": {"prediction": "unknown", "confidence": 0.0},
  "age_bracket": {"prediction": "31-45", "confidence": 0.51},
  "processing_ms": 91,
  "audio_quality": "degraded"
}
```

Progressive predictions have `is_final=false`; the response to `end` is final and the server closes with code 1000. Retained predictions include the latest receipt, and the final receipt is `stored` only after the trailing object and PostgreSQL result commit. Each prediction re-analyzes the cumulative buffer, so it may change. Clients should treat only the final prediction as settled.

Browser `MediaRecorder` chunks are normally compressed WebM/Opus or MP4/AAC container fragments and must not be sent to this endpoint. The supplied live frontend uses an `AudioWorklet`, declares `pcm_f32le` at the browser's actual `AudioContext` rate, and batches mono raw samples into approximately 250 ms binary frames. See [STREAMING.md](STREAMING.md). A completed MediaRecorder blob can instead be uploaded to `POST /analyze`.

Create headerless input:

```bash
ffmpeg -y -i samples/caller.wav -f s16le -ac 1 -ar 16000 samples/caller.s16le
python -m pip install 'websockets>=13,<16'
```

Then run this streaming client from the repository root:

```python
import asyncio
import json
from pathlib import Path

import websockets


async def receive_predictions(socket):
    async for message in socket:
        payload = json.loads(message)
        print(json.dumps(payload, indent=2))
        if payload.get("type") in {"prediction", "error"} and (
            payload.get("is_final") or payload.get("type") == "error"
        ):
            return


async def main():
    audio = Path("samples/caller.s16le").read_bytes()
    async with websockets.connect("ws://localhost:8000/ws/analyze") as socket:
        await socket.send(json.dumps({
            "type": "start",
            "encoding": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        }))
        receiver = asyncio.create_task(receive_predictions(socket))
        for offset in range(0, len(audio), 16000):  # 0.5-second chunks
            await socket.send(audio[offset:offset + 16000])
            await asyncio.sleep(0.5)
        await socket.send(json.dumps({"type": "end"}))
        await receiver


asyncio.run(main())
```

WebSocket service errors are sent as `{"type":"error","error":{...}}`, followed by close code 1009 for size limits, 1013 for service unavailability, or 1008 for protocol/validation errors. Browser handshakes with an `Origin` outside `WS_ALLOWED_ORIGINS` are rejected before acceptance; clients without an `Origin` remain available for authenticated server-side telephony adapters. Any retained session that does not reach a final result deletes its partial objects and metadata.

## Stored-analysis endpoints

These endpoints exist to demonstrate the scalable manifest. They are bound to loopback by Compose but have no built-in tenant authorization; do not expose them publicly until authenticated tenant scoping is implemented.

| Method and path | Result |
| --- | --- |
| `GET /analyses?limit=20&contact_id=<uuid>` | Newest retained sessions as `{"items":[...]}`; limit is 1–100 and `before=<RFC3339>` provides a cursor. |
| `GET /analyses/{analysis_id}` | Full result plus physical object keys, SHA-256, byte ranges, and logical chunk source/segment offsets. It does not return audio bytes or a download URL. |
| `DELETE /analyses/{analysis_id}` | Immediately removes audio objects first and then metadata; returns the deleted ID. |

A live detail manifest contains entries like:

```json
{
  "sequence": 0,
  "object_key": "voice-attributes/v1/2026/08/24/<analysis>/segments/00000000-<random>.f32",
  "byte_start": 0,
  "byte_end": 192000,
  "byte_size": 192000,
  "sha256": "<64 lowercase hex characters>",
  "logical_chunks": [
    {
      "chunk_index": 0,
      "source_byte_start": 0,
      "source_byte_end": 48000,
      "segment_byte_start": 0,
      "segment_byte_end": 48000
    }
  ]
}
```

## Operational endpoints

| Method and path | Result |
| --- | --- |
| `GET /healthz` | `200 {"status":"ok"}` while the process can answer. |
| `GET /readyz` | HTTP 200 `ready` after model load/warmup; otherwise HTTP 503 `not_ready`. |
| `GET /persistence/capabilities` | Deployment maximum, default mode, retention periods, and storage roles; never returns credentials. |
| `GET /metrics` | Prometheus exposition format. |
| `GET /docs` | Interactive OpenAPI UI. |

Metrics include HTTP counts and latency, inference latency and in-flight work, quality counts, and errors by stable code. WebSocket sessions are logged but do not currently have a session counter or prediction-latency histogram.
