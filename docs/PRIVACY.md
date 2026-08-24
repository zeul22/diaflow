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

```text
socket chunks
    -> bounded bytearray
    -> native decoder or bounded FFmpeg stdin/stdout
    -> mono 16 kHz NumPy array
    -> quality statistics
    -> optional five-second inference window
    -> in-process local model
    -> aggregate response
    -> best-effort overwrite and release in finally
```

Application behavior:

- REST and WebSocket input is buffered in memory only. Application code creates no audio files.
- WebSocket audio is retained only for that active session, because progressive predictions use the cumulative signal.
- FFmpeg receives compressed bytes through stdin and returns float PCM through stdout. It is invoked without metadata copying.
- Runtime inference is local. `HF_HUB_OFFLINE=1` and baked model artifacts prevent model-host uploads.
- Mutable encoded buffers are overwritten with zeros and cleared. Decoded and selected NumPy arrays are filled with zeros in `finally`, including error paths after decoding.
- The application does not cache waveforms, embeddings, model outputs, or contact profiles.
- The API response returns `contact_id`; normal logs exclude it and all predictions.

## What zeroization does not guarantee

Python, NumPy, the ASGI server, TLS proxy, subprocess pipes, kernel socket buffers, allocators, copy-on-write memory, container runtime, swap, core dumps, and observability agents can create copies outside the arrays the application explicitly wipes. Immutable Python `bytes` received from framework/network boundaries cannot be overwritten in place. Best-effort clearing reduces lifetime but is not forensic erasure.

The Docker deployment limits exposure by running as a non-root user, dropping capabilities, setting a read-only root filesystem, and mounting `/tmp` as tmpfs. `/tmp` is used for SpeechBrain runtime files, not application audio. The Compose file does not configure host swap, core dumps, ingress logs, packet capture, or node-level agents; operators own those controls.

## Logging and observability

Structured logs contain UTC timestamp, severity, event, request ID, method, bounded path label, HTTP status, total duration, model name, inference duration, audio duration, and quality class. They do not intentionally include:

- request bodies or audio samples;
- HTTP query strings;
- `contact_id`;
- gender/age predictions or confidence;
- embeddings, waveform features, or codec payloads.

Prometheus metrics use bounded labels for path, status, backend, quality, and stable error code. They contain no per-caller identifier. Unexpected exception logs include a stack trace; application exceptions do not include payload content, but dependency messages should still be reviewed for leakage.

Configure ingress, WAF, service mesh, APM, tracing, and support tooling to exclude request/response bodies, query strings, `X-Contact-ID`, WebSocket frames, and multipart content. Protect logs and metrics with access control and short, justified retention.

## Network and storage boundary

The application serves plaintext HTTP/WebSocket on port 8000. Terminate TLS and authenticate callers at a trusted ingress; use TLS or mTLS from ingress to service where threat modeling requires it. Apply namespace/network policy so only the media platform and monitoring scraper can connect. Do not expose `/metrics` publicly.

The supplied image is built with model files embedded under `/opt/models`. Audio is not embedded. The first build contacts Hugging Face and package repositories; the running model does not require external services. Build logs and caches should be access-controlled, although they contain model artifacts rather than caller audio.

## Retention and deletion

The application's intended audio retention is the duration of one request/session plus cleanup. There is no application database and therefore no service-side deletion API. `contact_id`, output, and any caller association may be stored by the calling system; that system must define its own retention, deletion, subject-access, and audit processes.

The app terminates an HTTP upload after ten seconds without a body chunk and a WebSocket after 30 seconds without a frame or 120 seconds total by default. It also enforces byte and 30-second audio limits. Keep stricter, independently enforced ingress idle and maximum connection durations before production; application timers are defense in depth, not protection against connections that never reach the process.

## Sensitive inference and fairness

The `gender` field is a binary acoustic classifier trained on `female`/`male` labels. Voice presentation does not establish gender identity, sex, pronouns, or how a person wishes to be addressed. Transgender, nonbinary, intersex, synthetic, converted, atypical, ill, or deliberately altered voices can be mislabeled. The system offers `unknown`; downstream dialog should use neutral language unless the caller supplies a preference.

Age is approximate. It is influenced by health, language, prosody, microphone, codec, acting, and noise. The model's celebrity/interview training domain differs from logistics calls, and adult bracket boundaries can magnify small regression errors. Do not infer minor status from this API.

Before use, evaluate consented caller-only data across lawful and relevant language, accent, geography, age, voice-presentation, carrier, handset, bandwidth, noise, and disability/health slices. Report both error and abstention coverage. Stop deployment where disparity or calibration is unacceptable.

## Production checklist

- Complete privacy, biometric/sensitive-data, employment, telecom-recording, and anti-discrimination legal review for each jurisdiction.
- Document purpose, lawful basis/consent, notices, opt-out, data controller/processor roles, and downstream retention.
- Use caller-only channel routing and test it continuously.
- Add authenticated TLS/mTLS ingress, tenant authorization, quotas, rate limiting, connection limits, and network policy.
- Disable request/response/WebSocket body capture throughout ingress, APM, tracing, support tools, and crash reporting.
- Disable or encrypt swap and core dumps; restrict node debugging and memory inspection.
- Bound HTTP/WebSocket idle and wall-clock lifetime at ingress as well as in the application.
- Encrypt any downstream result storage; separate identity maps; apply deletion and access controls.
- Audit model/data licenses and notices; inventory exact artifact hashes.
- Run security scanning, dependency/model provenance checks, incident response exercises, and privacy review on every model change.
- Monitor only aggregate performance, drift, quality, coverage, calibration, and subgroup safety; avoid raw-audio sampling in production.
