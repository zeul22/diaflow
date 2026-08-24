# ADR-003: Opt-in analysis and audio persistence

- Status: Accepted for the local showcase; production requires tenant authentication
- Date: 2026-08-24
- Owners: Voice platform

## Decision

Store structured analysis metadata in PostgreSQL and retained audio in S3-compatible object storage. Never store voice bytes in PostgreSQL columns. A request explicitly selects one of:

- `none` — infer and wipe request buffers; write nothing (default);
- `result` — retain the structured result, model revision, and signal metadata;
- `result_and_audio` — retain the result and audio only with an explicit consent reference.

This changes the original blanket “no storage” behavior only for opted-in requests. The API hashes the consent reference before writing it and never logs the value. Production should connect to managed S3 with private networking, IAM, TLS, and SSE-KMS. Docker Compose uses MinIO only to make the object layout visible locally; MinIO's upstream repository was archived in April 2026, so that container is not the production recommendation.

## Data layout

PostgreSQL owns lifecycle state and queryable metadata. The implemented `persistence_sessions` table records contact, transport, source encoding, selected policy, model name, final result, object counters, status, and expiration. `persistence_audio_segments` records each immutable S3 object key, covered byte range, length, SHA-256, and a JSON array mapping every logical WebSocket chunk slice to its exact source and segment offsets. Keeping slices with their physical segment makes one object PUT and its manifest one commit; a future analytics pipeline can normalize slices into a dedicated high-volume chunk table without changing the object layout.

REST uploads are one source object with a safe codec-derived extension. Live 250 ms network chunks are logically recorded one-for-one but coalesced into approximately one-second segment objects. At 1,000 calls this reduces the object-write rate from roughly 4,000 to 1,000 PUTs/second while preserving exact chunk reconstruction. Object keys are random identifiers such as `v1/2026/08/24/<analysis-uuid>/segments/000001-<random>.bin`; they contain neither contact IDs nor uploaded filenames.

## Failure, deletion, and retention

If opted-in storage is unavailable, the request fails explicitly; the UI never claims data was retained when it was not. Any REST or WebSocket analysis that does not complete deletes its partial objects and metadata, so an error never strands retained audio without a returned analysis ID. Cancellation joins an in-flight object PUT before compensation. Deletion removes objects first and then hard-deletes metadata; repeating it has no additional storage effect and the HTTP API reports not found. Result-only sessions expire after 30 days by default. Result-and-audio sessions delete both together after 24 hours, avoiding an orphaned inferred result after its consent-scoped audio expires. Production uses one retention worker with retryable jobs and `FOR UPDATE SKIP LOCKED`; the Compose showcase uses an in-process cleanup task. The Compose-owned bucket also receives a coarse lifecycle backstop one day beyond the configured audio TTL; production S3 lifecycle policies remain a second safety net, not the only deletion mechanism.

History endpoints intentionally have no production security claim in this demo. Before exposing them beyond loopback, add authenticated tenant identity, tenant-scoped queries, authorization for deletion/download, audit events, per-tenant encryption keys and quotas, idempotency keys, and pagination. Do not globally deduplicate biometric voice content.
