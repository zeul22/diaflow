# Sample audio for smoke testing

No real caller audio is committed. That avoids publishing voice PII and keeps licensing/consent explicit. Create or source a short, caller-only sample locally and do not commit it.

## Option 1: record yourself

Record 3–5 seconds of normal speech after giving yourself permission to use the recording for this test. Avoid names, phone numbers, addresses, tracking numbers, or other personal content. Convert it with FFmpeg:

```bash
ffmpeg -y -i /path/to/your-recording.m4a \
  -ac 1 -ar 16000 -c:a pcm_s16le backend/samples/caller.wav
```

On Linux with ALSA, you can record directly if `arecord` is available:

```bash
arecord -d 5 -t wav -f S16_LE -r 16000 -c 1 backend/samples/caller.wav
```

## Option 2: Mozilla Common Voice

Download a release from the [Mozilla Common Voice datasets page](https://commonvoice.mozilla.org/en/datasets). Review and follow the dataset release's license and terms, then choose one validated clip with a single adult speaker. Do not infer that metadata is perfect ground truth. Convert the downloaded clip:

```bash
ffmpeg -y -i /path/to/validated-clip.mp3 \
  -ac 1 -ar 16000 -c:a pcm_s16le backend/samples/caller.wav
```

Keep the clip outside version control unless its exact license and redistribution terms have been reviewed. Public availability does not remove voice-privacy or usage obligations.

## Check the sample

```bash
ffprobe -v error \
  -show_entries stream=codec_name,sample_rate,channels,duration \
  -of default=noprint_wrappers=1 backend/samples/caller.wav
```

Aim for one speaker, 2–5 seconds, mono, 16 kHz, audible speech, and minimal silence. The service also accepts other rates and common compressed formats, but this canonical file separates model behavior from decoder issues.

## REST smoke test

Start the service from the repository root:

```bash
docker compose up --build
```

For a privacy-safe transport smoke test that generates synthetic, speech-like audio in memory, run:

```bash
make smoke
```

This verifies the response contract, not real-speaker accuracy.

In another terminal:

```bash
curl -sS -X POST http://localhost:8000/analyze \
  -H 'Content-Type: audio/wav' \
  --data-binary @backend/samples/caller.wav
```

Expect HTTP 200 with the documented JSON schema. A result of `unknown` is valid, especially for short, noisy, narrowband, or out-of-domain speech. A smoke test proves integration, not prediction correctness.

## Raw WebSocket fixture

Create headerless signed 16-bit little-endian PCM for the client in [docs/API.md](../docs/API.md):

```bash
ffmpeg -y -i backend/samples/caller.wav \
  -f s16le -ac 1 -ar 16000 backend/samples/caller.s16le
```

To exercise narrowband telephony quality behavior, create an 8 kHz μ-law body:

```bash
ffmpeg -y -i backend/samples/caller.wav \
  -f mulaw -ac 1 -ar 8000 backend/samples/caller.mulaw

curl -sS -X POST \
  'http://localhost:8000/analyze?encoding=mulaw&sample_rate=8000&channels=1' \
  -H 'Content-Type: application/octet-stream' \
  --data-binary @backend/samples/caller.mulaw
```

The μ-law request should normally be `degraded` or `insufficient`, never `good`, because narrowband input is a configured degradation condition.
