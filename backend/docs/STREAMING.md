# Browser capture and real-time streaming

This service deliberately uses two microphone paths because a completed browser recording and a live media stream have different wire formats.

| Frontend mode | Browser source | Transport | Service input | Why |
| --- | --- | --- | --- | --- |
| Record | `MediaRecorder` | Multipart REST | WebM/Opus, Ogg/Opus, or MP4/AAC | The browser creates a complete, decodable container after recording stops. |
| Live | `AudioWorklet` | WebSocket binary frames | Headerless mono `pcm_f32le` at the browser's actual sample rate | The WebSocket contract needs independently appendable raw samples; MediaRecorder fragments are container-dependent and are not safe raw chunks. |

Microphone permission is requested only after an explicit button press. Capture works on `localhost`; production browser deployments require HTTPS so the socket automatically becomes WSS. Nginx permits the microphone for the same origin only. The frontend retains completed recording blobs only while they are selected and uses no browser local storage or analytics. Server retention is a separate visible control that defaults to `None`; audio retention requires acknowledgement plus an opaque consent reference. Live capture holds one roughly 250 ms application batch plus a bounded browser WebSocket queue (capped at 512 KiB before failing closed); browser, network-stack, and operating-system copies can live longer.

The microphone request asks for mono input with echo cancellation and noise suppression as browser-controlled `ideal` constraints, while preferring automatic gain control off. Browsers and operating systems may ignore those preferences or apply other DSP; that processing can change model features and must be included in device/domain evaluation.

## Live browser flow

1. Under the Start button's user gesture, the browser creates/resumes an `AudioContext` and requests a mono microphone track. Cancel remains available if the permission prompt is ignored; a track granted after cancellation is immediately stopped.
2. An `AudioWorklet` downmixes the incoming render blocks to one float channel.
3. The client opens `/api/ws/analyze` and first sends start JSON declaring `pcm_f32le`, one channel, the actual `AudioContext.sampleRate`, and the selected persistence mode.
4. Worklet output is grouped into approximately 250 ms little-endian binary frames. The backend resamples accepted 8–96 kHz input to 16 kHz.
5. Every retained session emits `started`. Sessions retaining audio then emit progressive `storage` receipts; result-only sessions proceed directly to predictions because no audio objects are written. For retained audio, the UI shows network chunks received, logical chunks represented in committed segments, one-second physical object segments, stored bytes, and expiry while capture continues.
6. The server emits a provisional prediction once there is enough speech, then backs off while the answer is not moving: the required interval of new audio grows by `WS_EMIT_BACKOFF` after each unchanged result, up to `WS_MAX_EMIT_INTERVAL_SECONDS`, and resets whenever the predicted labels change. Each provisional analysis reads a bounded trailing window (`WS_ANALYSIS_WINDOW_SECONDS`) instead of the whole cumulative buffer, so a long session costs a constant amount per update rather than growing quadratically. The UI replaces the cards in place and labels them provisional.
7. A session accepts at most `MAX_AUDIO_SECONDS` of streamed audio, which bounds the session buffer and the final `end` decode. The UI stops capture just under that limit so a normal session finishes cleanly rather than hitting `AUDIO_TOO_LONG`.
7. Stop flushes the aligned client buffer and sends the exact `{"type":"end"}` control. For audio-retained sessions, the service commits the trailing audio segment before the final result. For every retained mode, only after its selected storage work commits does the service send an `is_final: true` prediction with `persistence.status=stored`.

The client stops instead of accumulating audio when the WebSocket send buffer exceeds 512 KiB. It does not reconnect after capture begins because recovering a cumulative session would require retaining and replaying caller PII. All tracks, worklet nodes, pending arrays, audio contexts, sockets, timers, and preview object URLs are cleaned up on stop, cancel, error, mode change, and component teardown. Any retained session that does not reach a final result is deleted by the backend. Browser and operating-system internals can still make copies; these are lifetime-reduction controls, not forensic erasure.

Browser WebSockets are same-origin and the backend rejects any supplied `Origin` not listed in `WS_ALLOWED_ORIGINS`. Origin-less clients are reserved for authenticated server-to-server telephony adapters. Origin validation is not authentication: production ingress must still enforce tenant credentials, authorization, quotas, and rate limits.

## Real logistics-call integration

The web microphone is a local demonstration, not the recommended production call-media path. A carrier, SBC, or media server should provide the isolated remote caller leg to a small telephony adapter:

```text
carrier media WebSocket
    -> authenticate and validate provider envelope
    -> select caller-only track
    -> base64 decode / unwrap media payload
    -> aggregate 20 ms frames into about 200–250 ms
    -> service start JSON + binary PCM/G.711 frames
    -> progressive predictions to the call orchestrator
    -> end control when the caller leg ends
```

For a typical 8 kHz μ-law provider stream, start with `{"type":"start","encoding":"mulaw","sample_rate":8000,"channels":1}` and forward decoded binary μ-law bytes—not provider JSON or the base64 text itself. A provider-specific adapter should own signature verification, sequence-gap detection, timestamp ordering, track selection, and call-to-contact UUID mapping. Do not mix the AI/dispatcher return channel with caller audio.

The current server is transport-streaming but inference is cumulative: each progressive update re-analyzes the buffered signal, and the receive loop waits for that inference. This is suitable for the demo and bounded call snippets. At scale, split continuous frame ingestion from a latest-only inference scheduler, keep one bounded rolling speech window, avoid queuing stale intermediate predictions, and move inference to autoscaled GPU workers with admission control. Incremental VAD and embeddings are the longer-term optimization.

## Try it

Run `docker compose up --build`, open `http://localhost:3000`, select **Record** or **Live**, and grant microphone permission. The live view shows elapsed audio, actual sample rate, binary chunks sent, microphone level, prediction sequence, provisional/final state, and—in retained mode—database/object-storage progress.

The containerized synthetic WebSocket smoke test can exercise the same frontend proxy without installing Python packages on the host:

```bash
make smoke-ws-ui
```
