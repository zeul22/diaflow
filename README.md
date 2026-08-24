# Voice Contact Attribute Service

A FastAPI service that estimates an adult caller's perceived binary voice presentation and age bracket from uploaded, recorded, or live-streamed speech. It detects unusable audio, abstains when confidence is low, and defaults to no retention. A user can explicitly retain results or consent-linked audio through PostgreSQL metadata and S3-compatible object storage.

> The API field is named `gender` to satisfy the required contract. It is a binary acoustic estimate of how a voice presents to the training labels, not a determination of gender identity, sex, pronouns, or legal status. Do not use it for consequential, discriminatory, eligibility, pricing, employment, insurance, medical, or legal decisions.

[Setup](#quick-start) · [Design decisions](backend/docs/DESIGN.md) · [Model rationale](backend/docs/ADR-002-production-model-strategy.md) · [Known limitations](#known-limitations)

## What is implemented

- `POST /analyze` for raw HTTP bodies and streaming-parsed multipart uploads.
- `WS /ws/analyze` for progressive predictions over raw PCM, μ-law, or A-law chunks.
- A responsive React/Vite/SCSS web client for upload, record-then-analyze, raw-PCM live streaming, progressive results, and actionable error states.
- Per-request `none`, `result`, and consent-gated `result_and_audio` persistence modes; `none` is always the default.
- PostgreSQL manifests plus S3 audio objects, one-second physical WebSocket segments, logical chunk offsets/checksums, history, and delete-now controls.
- Native WAV/PCM/G.711 decoding plus FFmpeg fallback for common compressed containers and codecs.
- Quality gating for short, quiet, non-speech, noisy, clipped, low-frequency-heavy, and narrowband input.
- One pinned SpeechBrain ECAPA-TDNN encoder pass shared by Apache-2.0 griko SVM/SVR attribute heads.
- Confidence thresholds and an explicit `unknown` result instead of forced predictions.
- Request-size, decoded-duration, decode-time, queue-time, and inference-concurrency limits.
- JSON logs, request IDs, Prometheus metrics, liveness/readiness probes, and graceful structured errors.
- A multi-stage, non-root, read-only Docker image. Model weights are downloaded only while building the image.

Language/accent detection is not implemented in this version. A Common Voice evaluation harness is included for accuracy, coverage, confusion, and calibration checks.

## Repository layout

```text
.
├── backend/                  # FastAPI, models, persistence, tests, scripts, docs
├── frontend/                 # React, Vite, SCSS, Nginx proxy, UI tests
├── docker-compose.yml        # Complete local stack, run from this directory
├── docker-compose.wavlm.yml  # Owned WavLM deployment override
├── Makefile                  # Root-only development and verification commands
└── README.md                 # Project entry point
```

Run Docker Compose, Make, smoke, test, and development commands from the
repository root. The component READMEs are [backend/README.md](backend/README.md)
and [frontend/README.md](frontend/README.md).

## Quick start

Prerequisites are Docker with Compose v2, enough disk for the Python/PyTorch image, and internet access during the first image build. The build downloads only the publicly available, revision-pinned model artifacts; the running container is configured for offline model loading.

```bash
docker compose up --build
```

Compose starts the API, frontend, PostgreSQL, and a local S3-compatible object store. The first build is slower because it downloads CPU PyTorch wheels and model artifacts. Once startup logs contain `service_ready`, verify the service:

```bash
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/readyz
```

Both should return HTTP 200. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

Open the web interface at [http://localhost:3000](http://localhost:3000). Select or drag in an M4A, WAV, MP3, OGG, FLAC, or WebM recording. M4A can be uploaded directly—host FFmpeg is not required. The bundled decoder probes source codec/rate, and use of FFmpeg alone no longer marks a good recording degraded. Retention remains **None** unless the user changes it.

The local object-store console is [http://localhost:9001](http://localhost:9001) (`voice-local` / `voice-local-development-only`). It is a development visualization only; use managed S3 with IAM/KMS in production. PostgreSQL stays private to the Compose network; inspect it with `docker compose exec postgres psql -U voice_attributes -d voice_attributes`.

The **Record** mode captures a complete browser-supported clip and analyzes it through REST. **Live** uses an AudioWorklet to send raw microphone PCM in roughly 250 ms WebSocket frames and replaces provisional estimates until the final result arrives. Microphone access works over localhost for development and requires HTTPS/WSS when deployed. See [backend/docs/STREAMING.md](backend/docs/STREAMING.md) for the browser and real call-media designs.

Run a dependency-free synthetic contract smoke test from another terminal:

```bash
make smoke
```

Verify progressive streaming with the containerized smoke client:

```bash
make smoke-ws
```

Exercise WebSocket streaming through the same reverse proxy used by the browser:

```bash
make smoke-ws-ui
```

Verify the same REST contract through the frontend reverse proxy:

```bash
make smoke-ui
```

Prepare a 2–5 second, single-speaker test file by following [backend/samples/README.md](backend/samples/README.md), then submit it as a raw WAV body:

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?contact_id=123e4567-e89b-12d3-a456-426614174000' \
  -H 'Content-Type: audio/wav' \
  -H 'X-Request-ID: smoke-001' \
  --data-binary @backend/samples/caller.wav
```

Or use multipart form data:

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -F 'audio=@backend/samples/caller.wav;type=audio/wav' \
  -F 'contact_id=123e4567-e89b-12d3-a456-426614174000'
```

To retain the structured result but not audio:

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -H 'X-Persistence-Mode: result' \
  -F 'audio=@backend/samples/caller.wav;type=audio/wav'
```

To demonstrate consent-gated audio retention:

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -H 'X-Persistence-Mode: result_and_audio' \
  -H 'X-Consent-Reference: demo-approval-001' \
  -F 'audio=@backend/samples/caller.wav;type=audio/wav'
```

The retained response includes `analysis_id` and a `persistence` receipt. Inspect its object segments and logical chunk offsets with `GET /analyses/{analysis_id}`, list retained analyses with `GET /analyses`, or erase both objects and metadata with `DELETE /analyses/{analysis_id}`. The consent reference itself is never stored; only its SHA-256 is recorded.

A successful response has this shape:

```json
{
  "contact_id": "123e4567-e89b-12d3-a456-426614174000",
  "gender": {"prediction": "male", "confidence": 0.87},
  "age_bracket": {"prediction": "31-45", "confidence": 0.63},
  "processing_ms": 142,
  "audio_quality": "good"
}
```

Predictions and numbers above are illustrative. See [backend/docs/API.md](backend/docs/API.md) for raw telephony formats, response semantics, error codes, and a working WebSocket client.

## Telephony integration

Send only the remote contact/caller leg. This release does not diarize mixed calls. Feeding both the AI agent and caller into one channel can make the result describe the louder or more persistent speaker. Configure the carrier, SBC, media server, or recorder to provide split channels and route only the caller channel to this service.

For an 8 kHz μ-law telephony body:

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?encoding=mulaw&sample_rate=8000&channels=1' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @backend/samples/caller.mulaw
```

μ-law, A-law, and other 8 kHz sources are deliberately marked at least `degraded`; the service still predicts when there is enough speech but discounts confidence.

## Frontend development

The production frontend is built into a small Nginx container and proxies the analyzer, readiness, WebSocket, persistence-capability, and retained-analysis paths to FastAPI. Browser requests remain same-origin, so the backend does not need permissive CORS settings. Its permissions policy allows microphone access only to the same origin. The backend port and UI port are bound to loopback by default.

For Vite hot reload, keep the backend running and use the root Make targets in another terminal:

```bash
make frontend-install
make frontend-dev
```

Then open `http://localhost:5173`. Vite proxies `/api` to `http://127.0.0.1:8000`. The client keeps the selected file and result only in React memory: it does not use browser local storage, analytics, a service worker, or filename logging. Server-side retention occurs only when the visible per-request control is changed from `None`.

## Model and decision rationale

The selected pipeline runs the pinned `speechbrain/spkrec-ecapa-voxceleb` encoder once and applies the pinned `griko/gender_cls_svm_ecapa_voxceleb` and `griko/age_reg_svr_ecapa_voxceleb2` heads to the same 192-dimensional embedding. Model repositories declare Apache-2.0, which permits commercial use subject to its conditions. The service repository is MIT licensed. Model and dataset licenses are separate, and neither grants privacy, publicity, voice, or training-data rights; production adoption still requires legal review and preservation of required notices.

The runnable ECAPA stack is a baseline, not a claim that it is universally best. The current production target is an owned WavLM Base+ backbone with logistics-trained, calibrated joint heads, exported through the included `wavlm_onnx` adapter. Because those heads must be trained and evaluated rather than downloaded, Compose keeps ECAPA as the working default. audEERING devAIce is the best turnkey paid evaluation; its public weights cannot be used commercially. The ranked evidence and promotion gates are in [ADR-002](backend/docs/ADR-002-production-model-strategy.md); the original baseline record remains in [ADR-001](backend/docs/ADR-001-model-selection.md).

The Docker runtime includes ONNX Runtime, but it intentionally does not pretend that bare public WavLM backbone weights are a deployable demographic model. After training and exporting the owned graph described in ADR-002, place it at `backend/models/wavlm/model.onnx` and launch the supplied override:

```bash
WAVLM_MODEL_REVISION=logistics-v1 \
  docker compose -f docker-compose.yml -f docker-compose.wavlm.yml up --build
```

The override mounts the graph read-only and selects `wavlm_onnx`. Startup fails if the artifact is absent or its input/output contract is wrong; it never silently falls back to ECAPA.

During the Docker build, every model file is fetched from an immutable commit and checked against a hard-coded SHA-256. The griko heads arrive as joblib/pickle objects, which can execute code while loading. They are deserialized only in the disposable model-builder stage, validated against scikit-learn predictions, and exported to numeric `.npz`. The runtime image contains no joblib, pandas, or scikit-learn and loads the heads using `numpy.load(..., allow_pickle=False)`. Build in isolated CI: conversion removes runtime pickle exposure, not the risk at build time. The pinned upstream SpeechBrain PyTorch checkpoints also remain serialized model artifacts and are hash-verified.

## Confidence and quality

Gender confidence is the selected SVM class probability after normalization and quality discounting. Age is first regressed in years, mapped to an adult bracket, and assigned probability mass using a configurable ten-year residual sigma. Implausible estimates outside 18–120 abstain. A degraded signal multiplies both scores by `0.75`. Scores below their thresholds become `unknown` with confidence `0.0`; insufficient audio skips model inference entirely. Compressed audio is degraded only when probed source metadata is missing/narrowband or measured signal checks warrant it—not merely because FFmpeg decoded it.

These scores are not proof, and they have not yet been calibrated on logistics calls. Treat `unknown` as a normal outcome. Before launch, evaluate calibration and subgroup error on consented, caller-only recordings from target carriers, languages, devices, and noise conditions.

## Configuration

Docker Compose provides safe CPU defaults. Important environment variables are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `MODEL_BACKEND` | `ecapa` | Runnable `ecapa` baseline or an owned `wavlm_onnx` artifact. |
| `WAVLM_MODEL_PATH` | `/opt/models/wavlm/model.onnx` | Graph containing WavLM, owned heads, and calibration. |
| `MODEL_DEVICE` | `cpu` | Torch device, for example `cpu` or `cuda` in a GPU image. |
| `TORCH_THREADS` | `2` | Intra-op Torch CPU threads. |
| `INFERENCE_CONCURRENCY` | `1` | Concurrent inferences per process. The estimator itself is serialized. |
| `QUEUE_TIMEOUT_SECONDS` | `1.0` | Wait before returning retriable `SERVICE_BUSY`. |
| `INFERENCE_WINDOW_SECONDS` | `5.0` | Speech-evidence window selected from longer audio. |
| `MAX_AUDIO_SECONDS` | `30` | Maximum decoded or streamed duration. |
| `MAX_UPLOAD_BYTES` | `12582912` | Maximum audio payload bytes. |
| `MIN_AUDIO_SECONDS` | `1.25` | Minimum decoded duration. |
| `MIN_VOICED_SECONDS` | `0.65` | Minimum estimated voiced speech. |
| `GENDER_CONFIDENCE_THRESHOLD` | `0.60` | Abstention threshold after quality adjustment. |
| `AGE_CONFIDENCE_THRESHOLD` | `0.28` | Age-bracket abstention threshold. |
| `DEGRADED_CONFIDENCE_FACTOR` | `0.75` | Confidence multiplier for degraded audio. |
| `WS_EMIT_INTERVAL_SECONDS` | `1.0` | Minimum new-audio interval between progressive results. |
| `REQUEST_IDLE_TIMEOUT_SECONDS` | `10.0` | Maximum wait between HTTP request-body chunks. |
| `WS_IDLE_TIMEOUT_SECONDS` | `30.0` | Maximum wait between WebSocket frames after start. |
| `WS_MAX_SESSION_SECONDS` | `120.0` | Hard wall-clock limit for one WebSocket session. |
| `WS_ALLOWED_ORIGINS` | local UI origins | Comma-separated browser origins allowed to open WebSockets; origin-less server adapters remain supported. |
| `LOG_LEVEL` | `INFO` | JSON log level. |
| `PERSISTENCE_ENABLED` | `false` in code; `true` in Compose | Deployment maximum; every request still defaults to `none`. |
| `DATABASE_URL` | local Compose PostgreSQL | Result and audio-manifest metadata. |
| `S3_ENDPOINT_URL`, `S3_BUCKET` | local Compose object store | S3-compatible retained-audio location. |
| `AUDIO_RETENTION_HOURS` | `24` | TTL for result-and-audio sessions. |
| `RESULT_RETENTION_DAYS` | `30` | TTL for result-only sessions. |
| `STORAGE_SEGMENT_SECONDS` | `1.0` | Physical object target for live PCM chunks. |
| `STORAGE_OPERATION_TIMEOUT_SECONDS` | `5.0` | Per-attempt PostgreSQL/S3 operation timeout. |
| `STORAGE_WORKER_THREADS` | `4` | Dedicated blocking S3 workers per API replica. |

Invalid configuration fails startup. Keep the default one Uvicorn worker per container so the model is loaded once per replica; scale with more containers.

## Reliability and observability

- `/healthz` reports process liveness; `/readyz` reports model readiness.
- `/metrics` exports bounded-label Prometheus counters, histograms, and an in-flight gauge.
- Responses carry `X-Request-ID`. A valid caller-provided ID is accepted; otherwise one is generated.
- Logs include request path, status, timing, model timing, quality, and stable error codes. They do not include audio or `contact_id`.
- A full inference queue returns HTTP 503 with `SERVICE_BUSY`; clients should retry with jitter only while the call is still active.

The under-500 ms requirement is a deployment acceptance target, not a universal guarantee. Benchmark p50/p95 end-to-end latency on the intended CPU architecture and codec mix after model warmup. Compressed decoding, noisy clips, host contention, and progressive re-analysis change latency.

For 1,000 concurrent calls, use stateless regional API pods for ingestion and a separate GPU inference pool with dynamic micro-batching, deadline-aware queues, bounded per-call buffers, admission control, and autoscaling on queue depth, GPU utilization, and p95 latency. See [DESIGN.md](backend/docs/DESIGN.md).

## Tests

The test suite uses a deterministic fake estimator, so it does not need model downloads after the runtime image exists:

```bash
make test
```

It covers raw and multipart REST requests, insufficient audio, structured errors, health/metrics, progressive WebSocket results, audio decoding, quality gating, kernel-head conversion behavior, and confidence postprocessing.

Frontend checks use Vitest, React Testing Library, ESLint, and a production Vite build:

```bash
make frontend-test
make frontend-lint
make frontend-build
```

Run every backend and frontend quality gate with `make check`.

## Common Voice evaluation harness

Download and extract a [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) release after reviewing its terms. With the service running, point the harness at a metadata TSV and its `clips` directory:

```bash
python3 backend/scripts/evaluate_common_voice.py \
  --tsv /data/cv-corpus/en/test.tsv \
  --clips /data/cv-corpus/en/clips \
  --limit 500 \
  --url http://127.0.0.1:8000
```

The JSON report includes accuracy with unknown counted as error, coverage, accuracy among covered results, ten-bin expected calibration error, a correctness Brier score, confusion counts, request failures, and quality counts for both tasks. Common Voice adult age labels are decades rather than exact ages; the harness excludes `teens` and maps the remaining labels by midpoint, so bracket-boundary results are approximate. The script is sequential and intended for model evaluation, not load testing. Do not treat public-dataset scores as evidence of logistics-domain performance.

## Privacy and production checklist

With mode `none`, audio stays in memory only for the request or WebSocket session and is best-effort overwritten in `finally` blocks. With explicit `result_and_audio`, encoded bytes are written to S3-compatible storage and governed by the returned expiry/delete controls. Runtime inference remains local and never sends audio to a model host. This is data minimization, not a forensic zeroization guarantee.

Before production, add TLS at the ingress, authentication, tenant authorization, rate limits, network policies, request-body logging exclusions, crash-dump controls, and jurisdiction-specific consent/notice. Full controls and the threat boundary are in [backend/docs/PRIVACY.md](backend/docs/PRIVACY.md).

## Known limitations

- The default ECAPA/SVM/SVR model is a runnable baseline, not a logistics-domain production accuracy claim.
- Age and perceived voice-presentation estimates can be wrong or abstain, especially for short, noisy, narrowband, accented, multilingual, or out-of-domain speech.
- Mixed agent/caller audio is not diarized; integrations must send the caller channel only.
- Language and accent detection are not implemented.
- The local history API has no tenant authentication, and the bundled object store is for demonstration only.
- The WavLM production path requires an owned, evaluated ONNX artifact; public backbone weights alone do not provide the required heads or calibration.

See the [model card](backend/docs/MODEL_CARD.md) and [privacy checklist](backend/docs/PRIVACY.md) before production use.

## Documentation

- [API contract and setup examples](backend/docs/API.md)
- [Model-selection decision record](backend/docs/ADR-001-model-selection.md)
- [Production model rationale and ranked alternatives](backend/docs/ADR-002-production-model-strategy.md)
- [Opt-in PostgreSQL/S3 persistence decision](backend/docs/ADR-003-opt-in-persistence.md)
- [200-word design write-up](backend/docs/DESIGN.md)
- [Privacy and data lifecycle](backend/docs/PRIVACY.md)
- [Browser and telephony streaming design](backend/docs/STREAMING.md)
- [Model card and known limitations](backend/docs/MODEL_CARD.md)
- [Sample audio instructions](backend/samples/README.md)
- [Third-party model notices](backend/THIRD_PARTY_NOTICES.md)
- `backend/scripts/evaluate_common_voice.py` for public-dataset evaluation

## License

Service code is licensed under [MIT](LICENSE). Upstream weights and dependencies retain their own licenses. See the commercial-use analysis in [ADR-001](backend/docs/ADR-001-model-selection.md); it is documentation, not legal advice.
