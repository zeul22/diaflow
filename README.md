# Voice Contact Attribute Service

A request-scoped FastAPI service that estimates an adult caller's perceived binary voice presentation and age bracket from speech. It accepts uploaded or streamed audio, detects unusable or degraded audio, abstains when confidence is low, and exposes health, readiness, JSON logs, and Prometheus metrics.

> The API field is named `gender` to satisfy the required contract. It is a binary acoustic estimate of how a voice presents to the training labels, not a determination of gender identity, sex, pronouns, or legal status. Do not use it for consequential, discriminatory, eligibility, pricing, employment, insurance, medical, or legal decisions.

## What is implemented

- `POST /analyze` for raw HTTP bodies and streaming-parsed multipart uploads.
- `WS /ws/analyze` for progressive predictions over raw PCM, μ-law, or A-law chunks.
- A responsive React/Vite/SCSS web client for drag-and-drop upload, preview, results, and actionable error states.
- Native WAV/PCM/G.711 decoding plus FFmpeg fallback for common compressed containers and codecs.
- Quality gating for short, quiet, non-speech, noisy, clipped, low-frequency-heavy, and narrowband input.
- One pinned SpeechBrain ECAPA-TDNN encoder pass shared by Apache-2.0 griko SVM/SVR attribute heads.
- Confidence thresholds and an explicit `unknown` result instead of forced predictions.
- Request-size, decoded-duration, decode-time, queue-time, and inference-concurrency limits.
- JSON logs, request IDs, Prometheus metrics, liveness/readiness probes, and graceful structured errors.
- A multi-stage, non-root, read-only Docker image. Model weights are downloaded only while building the image.

Language/accent detection is not implemented in this version. A Common Voice evaluation harness is included for accuracy, coverage, confusion, and calibration checks.

## Quick start

Prerequisites are Docker with Compose v2, enough disk for the Python/PyTorch image, and internet access during the first image build. The build downloads only the publicly available, revision-pinned model artifacts; the running container is configured for offline model loading.

```bash
docker compose up --build
```

The first build is slower because it downloads CPU PyTorch wheels and model artifacts. Once startup logs contain `service_ready`, verify the service:

```bash
curl -fsS http://localhost:8000/healthz
curl -fsS http://localhost:8000/readyz
```

Both should return HTTP 200. Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

Open the web interface at [http://localhost:3000](http://localhost:3000). Select or drag in an M4A, WAV, MP3, OGG, FLAC, or WebM recording; the browser sends it directly to the service without storing it. Compressed recordings such as M4A are accepted through the backend's FFmpeg decoder and may be conservatively marked `degraded`.

Run a dependency-free synthetic contract smoke test from another terminal:

```bash
make smoke
```

After installing the project development dependencies, verify progressive
streaming as well:

```bash
make smoke-ws
```

Verify the same REST contract through the frontend reverse proxy:

```bash
make smoke-ui
```

Prepare a 2–5 second, single-speaker test file by following [samples/README.md](samples/README.md), then submit it as a raw WAV body:

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?contact_id=123e4567-e89b-12d3-a456-426614174000' \
  -H 'Content-Type: audio/wav' \
  -H 'X-Request-ID: smoke-001' \
  --data-binary @samples/caller.wav
```

Or use multipart form data:

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -F 'audio=@samples/caller.wav;type=audio/wav' \
  -F 'contact_id=123e4567-e89b-12d3-a456-426614174000'
```

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

Predictions and numbers above are illustrative. See [docs/API.md](docs/API.md) for raw telephony formats, response semantics, error codes, and a working WebSocket client.

## Telephony integration

Send only the remote contact/caller leg. This release does not diarize mixed calls. Feeding both the AI agent and caller into one channel can make the result describe the louder or more persistent speaker. Configure the carrier, SBC, media server, or recorder to provide split channels and route only the caller channel to this service.

For an 8 kHz μ-law telephony body:

```bash
curl -sS -X POST \
  'http://localhost:8000/analyze?encoding=mulaw&sample_rate=8000&channels=1' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @samples/caller.mulaw
```

μ-law, A-law, and other 8 kHz sources are deliberately marked at least `degraded`; the service still predicts when there is enough speech but discounts confidence.

## Frontend development

The production frontend is built into a small Nginx container and proxies only the analyzer, readiness, and WebSocket paths to FastAPI. Browser requests remain same-origin, so the backend does not need permissive CORS settings. The backend port and UI port are bound to loopback by default.

For Vite hot reload, keep the backend running and start the development server in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Then open `http://localhost:5173`. Vite proxies `/api` to `http://127.0.0.1:8000`. The client keeps the selected file and result only in React memory: it does not use local storage, analytics, a service worker, or filename logging.

## Model and decision rationale

The selected pipeline runs the pinned `speechbrain/spkrec-ecapa-voxceleb` encoder once and applies the pinned `griko/gender_cls_svm_ecapa_voxceleb` and `griko/age_reg_svr_ecapa_voxceleb2` heads to the same 192-dimensional embedding. Model repositories declare Apache-2.0, which permits commercial use subject to its conditions. The service repository is MIT licensed. Model and dataset licenses are separate, and neither grants privacy, publicity, voice, or training-data rights; production adoption still requires legal review and preservation of required notices.

The durable comparison—including audEERING 6/24-layer models, ChunkFormer, openSMILE/acoustic features, and two independent models—is in [ADR-001](docs/ADR-001-model-selection.md). The concise architecture write-up is in [DESIGN.md](docs/DESIGN.md), and operational model limitations are in [MODEL_CARD.md](docs/MODEL_CARD.md).

During the Docker build, every model file is fetched from an immutable commit and checked against a hard-coded SHA-256. The griko heads arrive as joblib/pickle objects, which can execute code while loading. They are deserialized only in the disposable model-builder stage, validated against scikit-learn predictions, and exported to numeric `.npz`. The runtime image contains no joblib, pandas, or scikit-learn and loads the heads using `numpy.load(..., allow_pickle=False)`. Build in isolated CI: conversion removes runtime pickle exposure, not the risk at build time. The pinned upstream SpeechBrain PyTorch checkpoints also remain serialized model artifacts and are hash-verified.

## Confidence and quality

Gender confidence is the selected SVM class probability after normalization and quality discounting. Age is first regressed in years, mapped to an adult bracket, and assigned probability mass using a configurable ten-year residual sigma. Implausible estimates outside 18–120 abstain. A degraded signal multiplies both scores by `0.75`. Scores below their thresholds become `unknown` with confidence `0.0`; insufficient audio skips model inference entirely. FFmpeg-fallback input is conservatively degraded because the service does not trust caller-omitted source-bandwidth metadata.

These scores are not proof, and they have not yet been calibrated on logistics calls. Treat `unknown` as a normal outcome. Before launch, evaluate calibration and subgroup error on consented, caller-only recordings from target carriers, languages, devices, and noise conditions.

## Configuration

Docker Compose provides safe CPU defaults. Important environment variables are:

| Variable | Default | Purpose |
| --- | ---: | --- |
| `MODEL_DEVICE` | `cpu` | Torch device, for example `cpu` or `cuda` in a GPU image. |
| `TORCH_THREADS` | `2` | Intra-op Torch CPU threads. |
| `INFERENCE_CONCURRENCY` | `1` | Concurrent inferences per process. The estimator itself is serialized. |
| `QUEUE_TIMEOUT_SECONDS` | `1.0` | Wait before returning retriable `SERVICE_BUSY`. |
| `INFERENCE_WINDOW_SECONDS` | `5.0` | Highest-energy inference window selected from longer audio. |
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
| `LOG_LEVEL` | `INFO` | JSON log level. |

Invalid configuration fails startup. Keep the default one Uvicorn worker per container so the model is loaded once per replica; scale with more containers.

## Reliability and observability

- `/healthz` reports process liveness; `/readyz` reports model readiness.
- `/metrics` exports bounded-label Prometheus counters, histograms, and an in-flight gauge.
- Responses carry `X-Request-ID`. A valid caller-provided ID is accepted; otherwise one is generated.
- Logs include request path, status, timing, model timing, quality, and stable error codes. They do not include audio or `contact_id`.
- A full inference queue returns HTTP 503 with `SERVICE_BUSY`; clients should retry with jitter only while the call is still active.

The under-500 ms requirement is a deployment acceptance target, not a universal guarantee. Benchmark p50/p95 end-to-end latency on the intended CPU architecture and codec mix after model warmup. Compressed decoding, noisy clips, host contention, and progressive re-analysis change latency.

For 1,000 concurrent calls, use stateless regional API pods for ingestion and a separate GPU inference pool with dynamic micro-batching, deadline-aware queues, bounded per-call buffers, admission control, and autoscaling on queue depth, GPU utilization, and p95 latency. See [DESIGN.md](docs/DESIGN.md).

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
python3 scripts/evaluate_common_voice.py \
  --tsv /data/cv-corpus/en/test.tsv \
  --clips /data/cv-corpus/en/clips \
  --limit 500 \
  --url http://127.0.0.1:8000
```

The JSON report includes accuracy with unknown counted as error, coverage, accuracy among covered results, ten-bin expected calibration error, a correctness Brier score, confusion counts, request failures, and quality counts for both tasks. Common Voice adult age labels are decades rather than exact ages; the harness excludes `teens` and maps the remaining labels by midpoint, so bracket-boundary results are approximate. The script is sequential and intended for model evaluation, not load testing. Do not treat public-dataset scores as evidence of logistics-domain performance.

## Privacy and production checklist

Audio stays in memory for the request or WebSocket session, is not written by application code, is never sent to a model host, and is best-effort overwritten and released in `finally` blocks. The container runs read-only as a non-root user with `/tmp` on tmpfs. Managed runtimes, kernel buffers, allocators, crash dumps, infrastructure logs, and observability agents can still copy memory, so this is data minimization rather than a forensic zeroization guarantee.

Before production, add TLS at the ingress, authentication, tenant authorization, rate limits, network policies, request-body logging exclusions, crash-dump controls, and jurisdiction-specific consent/notice. Full controls and the threat boundary are in [docs/PRIVACY.md](docs/PRIVACY.md).

## Documentation

- [API contract and examples](docs/API.md)
- [Model-selection decision record](docs/ADR-001-model-selection.md)
- [200-word design write-up](docs/DESIGN.md)
- [Privacy and data lifecycle](docs/PRIVACY.md)
- [Model card and limitations](docs/MODEL_CARD.md)
- [Sample audio instructions](samples/README.md)
- [Third-party model notices](THIRD_PARTY_NOTICES.md)
- `scripts/evaluate_common_voice.py` for public-dataset evaluation

## License

Service code is licensed under [MIT](LICENSE). Upstream weights and dependencies retain their own licenses. See the commercial-use analysis in [ADR-001](docs/ADR-001-model-selection.md); it is documentation, not legal advice.
