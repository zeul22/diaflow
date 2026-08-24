# Privacy and audio-data lifecycle

Caller audio is PII and can be biometric or otherwise sensitive data depending on jurisdiction and use. Age and perceived voice-presentation inference can also create sensitive inferred data. This document describes the application's controls, their boundaries, and deployment obligations; it is not legal advice.

## Data minimization and purpose

The service needs only a short caller-only speech segment, a UUID, codec metadata, and optional request ID. It does not need a caller name, telephone number, shipment data, transcript, address, or prior contact profile. Upstream systems should use an opaque, short-lived `contact_id`, send only enough speech to meet the quality threshold, and discard or separately govern any association between the UUID and a person.

Use the output only to make low-stakes conversational adaptation. Do not use inferred age or voice presentation for authentication, identification, surveillance, eligibility, access, pricing, routing priority, employment, credit, insurance, medical conclusions, legal status, fraud claims, or protected-class decisions. Provide notice and obtain consent where required. Offer a path that does not depend on attribute inference.

## Caller-only audio

This release has no diarization, source separation, or echo cancellation. The media layer must isolate the contact's remote leg:

- inbound call: send the external caller, not the AI/dispatcher return audio;
- outbound call: send the driver/customer/callee, not the agent prompt;
- dual-channel recording: select the known remote channel before upload;
- mixed mono recording: do not analyze it with this release.

Mixed audio can infer attributes of the wrong person and unnecessarily process bystanders. Channel isolation is a privacy and correctness control, not merely a quality optimization.

## Request lifecycle

Every request declares a retention mode. Omission means `none`. `result` retains only the structured inference; `result_and_audio` additionally requires a consent reference and stores only its SHA-256, never the reference text. Deployment enablement does not change the per-request default.

```text
socket chunks
    -> bounded bytearray
    -> native decoder or bounded FFmpeg stdin/stdout
    -> mono 16 kHz NumPy array
    -> quality statistics
    -> optional five-second inference window
    -> in-process local model
    -> aggregate response
    -> optional PostgreSQL result / S3 encoded-audio commit
    -> best-effort overwrite and release in finally
```

Application behavior:

- Mode `none` keeps REST and WebSocket input in request memory only. Application code creates no temporary audio files.
- Mode `result` writes the response and manifest to PostgreSQL but no audio object.
- Mode `result_and_audio` writes the original bounded REST body or coalesced live PCM segments to S3-compatible storage. PostgreSQL stores object keys, hashes, and exact logical chunk offsets; it never stores audio BLOBs.
- WebSocket audio is retained only for that active session, because progressive predictions use the cumulative signal.
- FFmpeg receives compressed bytes through stdin and returns float PCM through stdout. It is invoked without metadata copying.
- Runtime inference is local. `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, `ORT_DISABLE_TELEMETRY=1`, and baked/mounted model artifacts prevent model-host or ONNX Runtime telemetry uploads.
- Mutable encoded buffers are overwritten with zeros and cleared. Decoded and selected NumPy arrays are filled with zeros in `finally`, including error paths after decoding.
- The application does not cache waveforms or embeddings. Model outputs/contact UUIDs are durable only for an explicitly retained session.
- The API response returns `contact_id`; normal logs exclude it and all predictions.
- The browser requests microphone permission only after an explicit user action. A completed recording blob remains only in the current page state while selected. Live PCM is held in one roughly 250 ms application batch, a browser WebSocket queue bounded to 512 KiB before the client fails closed, and the backend's bounded active-session buffer. Mode changes, cancellation, errors, and teardown release browser tracks, nodes, arrays, sockets, timers, and object URLs on a best-effort basis.
- Live capture does not automatically reconnect or replay. Any retained REST or live request that does not reach a final result deletes its partial objects and metadata; a failed live session must be restarted explicitly.

## What zeroization does not guarantee

Python, NumPy, the ASGI server, TLS proxy, subprocess pipes, kernel socket buffers, allocators, copy-on-write memory, container runtime, swap, core dumps, and observability agents can create copies outside the arrays the application explicitly wipes. Immutable Python `bytes` received from framework/network boundaries cannot be overwritten in place. Best-effort clearing reduces lifetime but is not forensic erasure.

The Docker deployment limits exposure by running as a non-root user, dropping capabilities, setting a read-only root filesystem, and mounting `/tmp` as tmpfs. `/tmp` is used for SpeechBrain runtime files, not application audio. The Compose file does not configure host swap, core dumps, ingress logs, packet capture, or node-level agents; operators own those controls.

## Logging and observability

Structured logs contain UTC timestamp, severity, event, request ID, method, bounded path label, HTTP status, total duration, model name, inference duration, audio duration, and quality class. Analysis events additionally carry processing facts and signal statistics that describe the *pipeline*, not the speaker: language-identification duration, applied AGC gain in dB, whether denoising ran, the sub-window count, the age-model disagreement in years, sub-window embedding similarity, and whether more than one speaker was suspected. WebSocket session events carry framing mode and transport-integrity counts (frames lost, reordered, duplicated). Persistence lifecycle events additionally contain the random analysis/session UUID so operators can reconcile failed writes/deletes; treat it as pseudonymous metadata.

Logs do not intentionally include:

- request bodies or audio samples;
- HTTP query strings;
- `contact_id`;
- gender/age predictions or confidence;
- the identified language, which is an inference about the caller — coverage is tracked in the aggregate `voice_attribute_language_results_total` metric instead;
- the raw age estimate from `debug_age_years`, which is response-only and never logged or persisted;
- embeddings, waveform features, or codec payloads.

The distinction the analysis event draws is deliberate: a duration, a gain, or a model-disagreement spread cannot identify or characterize a caller, whereas a predicted attribute can. Anything that names what the model concluded about the person belongs in the response, and in aggregate metrics, not in a per-request log line.

Prometheus metrics use bounded labels for path, status, backend, quality, stable error code, rate-limit transport, and language outcome. They contain no per-caller identifier. Path labels have the `/v1` prefix stripped so a version migration does not split a time series. Unexpected exception logs include a stack trace; application exceptions do not include payload content, but dependency messages should still be reviewed for leakage.

The supplied frontend proxy disables Nginx access logging for the analyze endpoint, stored-analysis routes, and the WebSocket endpoint, under both the versioned `/api/v1/...` paths the UI calls and the deprecated unversioned aliases. Each endpoint is listed explicitly rather than proxying all of `/api/`: that allowlist is what keeps `/metrics`, `/docs`, and `/openapi.json` off the public proxy. Its remaining access logs record the normalized path only, not the query string. This prevents uploaded filenames, contact IDs, analysis IDs, and payloads from being written by the proxy's default request log. Container-platform log collection still needs access control and an appropriate retention policy.

Configure ingress, WAF, service mesh, APM, tracing, and support tooling to exclude request/response bodies, query strings, `X-Contact-ID`, WebSocket frames, and multipart content. Protect logs and metrics with access control and short, justified retention.

## Network and storage boundary

The application serves plaintext HTTP/WebSocket on port 8000. Terminate TLS and authenticate callers at a trusted ingress; use TLS or mTLS from ingress to service where threat modeling requires it. Apply namespace/network policy so only the media platform and monitoring scraper can connect. Do not expose `/metrics` publicly.

The supplied image is built with model files embedded under `/opt/models`. Audio is not embedded. The first build contacts Hugging Face and package repositories; runtime model inference stays offline. When persistence is deployment-enabled, PostgreSQL and S3-compatible storage are runtime dependencies only for opted-in sessions.

Compose binds the API, object console, and frontend to loopback while keeping PostgreSQL and the S3 API private to the Compose network. Its MinIO service is a local visualization, not the production storage recommendation: the upstream repository was archived in April 2026. Production should use an actively maintained managed S3 service with TLS/private endpoints, tenant-scoped IAM, bucket policies, versioning decisions, access logging that excludes sensitive metadata, and SSE-KMS. Object keys contain random analysis/segment UUIDs and dates, never contact IDs or filenames. Database volumes and backups also need encryption, access control, tested deletion, and bounded retention.

## Retention and deletion

Mode `none` retains application audio only for one request/session plus cleanup. Result-only sessions default to 30 days. Result-and-audio sessions default to 24 hours for both their object bytes and associated result. `DELETE /analyses/{analysis_id}` removes S3 objects before deleting PostgreSQL metadata and is idempotent at the storage-service layer. The local cleanup loop applies TTLs, and the Compose-owned bucket gets a coarse lifecycle backstop one day beyond the audio TTL. Production should run a separately scaled, retryable retention worker and configure an independently monitored S3 lifecycle rule as a second safety net.

The calling system may separately store `contact_id`, outputs, or a caller association and therefore must define its own retention, deletion, subject-access, and audit processes. Deleting this service's analysis does not delete external copies. Backups, replicas, object versions, and legal holds require an explicit erasure policy.

The app terminates an HTTP upload after ten seconds without a body chunk and a WebSocket after 30 seconds without a frame or 120 seconds total by default. It also enforces byte and 30-second audio limits, for streams as well as uploads, which bounds how much caller audio one session can put through the process at all. Keep stricter, independently enforced ingress idle and maximum connection durations before production; application timers are defense in depth, not protection against connections that never reach the process.

## Sensitive inference and fairness

The `gender` field is a binary acoustic classifier trained on `female`/`male` labels. Voice presentation does not establish gender identity, sex, pronouns, or how a person wishes to be addressed. Transgender, nonbinary, intersex, synthetic, converted, atypical, ill, or deliberately altered voices can be mislabeled. The system offers `unknown`; downstream dialog should use neutral language unless the caller supplies a preference.

Age is approximate. It is influenced by health, language, prosody, microphone, codec, acting, and noise. The model's celebrity/interview training domain differs from logistics calls, and adult bracket boundaries can magnify small regression errors. Do not infer minor status from this API.

Before use, evaluate consented caller-only data across lawful and relevant language, accent, geography, age, voice-presentation, carrier, handset, bandwidth, noise, and disability/health slices. Report both error and abstention coverage. Stop deployment where disparity or calibration is unacceptable.

## Production checklist

- Complete privacy, biometric/sensitive-data, employment, telecom-recording, and anti-discrimination legal review for each jurisdiction.
- Document purpose, lawful basis/consent, notices, opt-out, data controller/processor roles, and downstream retention.
- Use caller-only channel routing and test it continuously.
- Add authenticated TLS/mTLS ingress, tenant authorization, quotas, rate limiting, connection limits, and network policy.
- Enforce tenant-scoped history/delete authorization; the loopback demo endpoints have no built-in identity layer.
- Bind audio retention to a validated consent/policy record and keep the UI default at `none`.
- Disable request/response/WebSocket body capture throughout ingress, APM, tracing, support tools, and crash reporting.
- Disable or encrypt swap and core dumps; restrict node debugging and memory inspection.
- Bound HTTP/WebSocket idle and wall-clock lifetime at ingress as well as in the application.
- Use managed PostgreSQL/S3 with encryption, private networking, narrowly scoped credentials, backup/replica deletion, retention jobs, and audit controls.
- Audit model/data licenses and notices; inventory exact artifact hashes.
- Run security scanning, dependency/model provenance checks, incident response exercises, and privacy review on every model change.
- Monitor only aggregate performance, drift, quality, coverage, calibration, and subgroup safety; avoid raw-audio sampling in production.
