# Audio and inference pipeline: techniques, alternatives, and trade-offs

Every signal-processing and inference decision from ingestion to a returned
attribute, why that technique was chosen, what the alternatives were, and what
each one costs. Numbers here are measured in the CPU container on this
repository's code, not quoted from literature. Where something is unmeasured it
says so.

Sections 1–7 cover the signal path. Sections 8–10 cover attribute inference, the
labelled data the confidence numbers depend on, and how this problem is
approached outside this repository.

Pipeline order, which itself is a decision:

```
ingest → decode → resample → quality gate → window select → enhance → inference
                                    ▲                            ▲
                    judges the untouched signal      applies only to the window
```

The quality gate deliberately runs **before** enhancement. If normalization ran
first, a near-silent recording would be amplified into looking usable and the
`insufficient` abstention would stop firing. Enhancement therefore only ever
touches the window handed to the model, never the signal being judged.

---

## 1. Sample-rate conversion

### What we do

Windowed-sinc polyphase resampling in [`app/audio/resample.py`](../app/audio/resample.py).
A Kaiser-windowed sinc kernel, cutoff at 95% of the lower Nyquist, evaluated at
the exact output positions with one precomputed kernel per polyphase branch.

### Why this was necessary

The previous implementation was `np.interp` — linear interpolation with no
anti-aliasing filter. That is not a resampler. Measured on this code:

| Input tone (48 kHz source → 16 kHz) | `np.interp` | windowed-sinc |
| --- | ---: | ---: |
| 12 kHz | **−6.0 dB (passed through, mirrored to 4 kHz)** | −95 dB |
| 15 kHz | −6.0 dB | −95 dB |
| 20 kHz | −6.0 dB | −99 dB |

A 12 kHz component reappeared at |12000 − 16000| = 4 kHz at full amplitude, in
the middle of the speech band. This mattered most on the **live browser path**,
where a 48 kHz microphone sends `pcm_f32le`: warehouse and cab noise above 8 kHz
— beeps, air brakes, tool whine, sibilance — was being folded on top of the
caller's voice, corrupting both the embedding and the SNR and spectral-flatness
statistics the quality gate uses to judge noise. FFmpeg-decoded uploads were
never affected because FFmpeg resamples properly at `-ar 16000`.

### Measured characteristic

| Band | Response |
| --- | ---: |
| ≤ 6 kHz | 0 dB (flat) |
| 7 kHz | −2.3 dB |
| 7.6 kHz | −6.0 dB (cutoff) |
| 8.5 kHz | −17.8 dB (transition) |
| 9 kHz | −29.6 dB (transition) |
| ≥ 10 kHz | −70 dB or better (stopband) |

Latency: **14.8 ms** for 10 s of 48 kHz audio, **13.6 ms** for 44.1 kHz. Boundary
transient at each signal edge decays within 8 output samples (0.5 ms).

### Design choices inside this technique

**Cutoff at 95% of Nyquist, not 100%.** No filter has a vertical edge. Placing
the cutoff exactly at 8 kHz leaves the transition band *above* it, so content
just past 8 kHz still folds back with only partial attenuation — measured at
−24.8 dB for 9 kHz. Pulling the cutoff to 7.6 kHz puts the whole transition
inside the band being discarded anyway, improving 9 kHz to −35.6 dB at the cost
of 7.6–8 kHz of bandwidth that a 16 kHz speech model barely weights. libsoxr and
swresample make the same trade.

**8 lobes, ~72 dB design attenuation.** Doubling the kernel would halve the
transition width and fully reject 9 kHz, at roughly double the latency. Speech
carries very little energy at 8–10 kHz, so the wider transition is the better
buy. `lobes` is a parameter if a deployment disagrees.

**Precomputed polyphase table.** The first implementation evaluated the sinc and
Kaiser window per output sample and took **337 ms** for 10 s of audio — 23× worse
than the final version. Output *n* sits at input position `n·down/up`, so only
`up` distinct fractional offsets exist (1 for 48→16 kHz, 160 for 44.1→16 kHz).
Computing the kernel once per phase instead of once per sample is the entire
difference between 337 ms and 14.8 ms.

**Clamped edge extension, normalized once.** An earlier version zero-padded and
renormalized the truncated kernel per output sample. That silently built a
*different, much worse* filter at the boundaries, which leaked the aliases the
module exists to reject — the whole-signal peak read −19.5 dB while the interior
was −86 dB. Normalizing the table once and clamping the input keeps the full
symmetric kernel in effect everywhere.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Windowed-sinc polyphase (chosen)** | Correct in both directions, arbitrary rates, no dependency, ~15 ms | Hand-written, so it needs its own tests | Adopted |
| `np.interp` (previous) | Trivial, fastest | Not a resampler; full-amplitude aliasing | Replaced |
| Shell out to FFmpeg for every conversion | Already a dependency, battle-tested resampler | A subprocess per progressive update — hundreds of ms and a process spawn on the hot path | Rejected for streaming; still used for compressed decode |
| `scipy.signal.resample_poly` | Well-tested, fast, ~10 lines to call | Adds SciPy (~40 MB) to a deliberately lean image, for one function | Rejected on image size |
| `soxr` / `samplerate` bindings | Best-in-class quality and speed | Another native dependency and licence to review | Rejected; revisit if resampling ever dominates the profile |
| FFT-domain resampling | O(N log N), exact for periodic signals | Circular-convolution edge artifacts; awkward for arbitrary rates | Rejected |
| Skip resampling; run models at native rate | No conversion at all | Both models are fixed at 16 kHz | Not possible |

### What is still not handled

Resampling quality is uniform across the file; there is no per-band adaptive
filtering, and none is warranted at this rate. The 8–10 kHz transition band is a
known, measured compromise rather than a bug.

---

## 2. Packet loss and reordering

### What we do

Opt-in sequenced framing in [`app/audio/jitter.py`](../app/audio/jitter.py). A
client sets `"framing": "seq32"` in the WebSocket start frame and prefixes every
audio frame with a 4-byte big-endian sequence number. The server keeps a bounded
reorder window (`WS_REORDER_WINDOW_FRAMES`, default 8), restores order, drops
duplicates, conceals gaps, and — most importantly — **counts** the damage. When
the loss ratio exceeds `WS_MAX_LOSS_RATIO` (default 0.15) the analysis is
reported as `degraded` rather than presented as a clean read.

### Why this is needed at all

A WebSocket runs over TCP, so within one connection bytes arrive in order. That
guarantee stops at the socket. Real logistics audio reaches this service through
a gateway that received **RTP over a mobile network**, and RTP loses and reorders
packets. If the gateway forwards what it received without repair, the stream here
has gaps and swapped frames with no marker saying so — and the result is
*silently wrong* rather than reported as degraded. A paralinguistic model has no
way to notice that 15% of a voice was reconstructed.

### Design choices inside this technique

**4-byte binary header, not a JSON envelope per chunk.** A control message before
each binary frame would double the message count on the hot path. Four bytes on a
1,500-byte frame is 0.3% overhead.

**Opt-in, not mandatory.** `framing: "raw"` remains the default so existing
clients and the bundled UI are unaffected. A gateway that already repairs RTP
should not pay for a second repair layer.

**Small reorder window, not an adaptive jitter buffer.** Holding audio longer
repairs more damage and costs exactly the latency this service exists to avoid.
Eight frames at 250 ms is 2 s of tolerance; a telephony jitter buffer would
adapt its depth to measured network delay, which only makes sense if this service
owned the RTP endpoint. It does not.

**A late frame is dropped, not inserted.** Once a slot has been released, a frame
that finally arrives for it is discarded. Feeding stale audio into a progressive
estimate is worse than the gap it would fill.

**Repeat-then-silence concealment.** A missing frame is filled by repeating the
previous frame for at most `max_repeat_frames` (2), then with silence. This is
the spirit of G.711 Appendix I. Sustained repetition produces a buzzing periodic
artifact, and a model trained to read voice quality would happily interpret that
as speech — so the concealment deliberately gives up and goes quiet rather than
manufacturing plausible-sounding voice.

**Loss feeds the quality gate.** This is the part that matters more than the
repair. Concealed audio is synthetic; past a threshold the estimate describes a
partly invented voice, and the caller is told through `audio_quality` instead of
being left to discover it.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Sequenced framing + bounded reorder (chosen)** | Detects and repairs; makes damage visible; opt-in; ~2 s tolerance | Requires client cooperation; not a real jitter buffer | Adopted |
| Trust TCP ordering, do nothing (previous) | Zero code | Blind to damage the gateway already inflicted; silently wrong results | Replaced |
| Full adaptive jitter buffer (RTP-style) | Best repair; adapts depth to network | Adds latency proportional to jitter; belongs in the gateway that owns the RTP session | Rejected |
| Model-based packet loss concealment (neural PLC) | Best perceptual fill | Another model, another forward pass, and it *fabricates* voice — the worst possible property for a measurement system | Rejected |
| Interpolate across the gap in the time domain | No repetition buzz | Smears the spectrum; the gate would read it as low-flatness "voice" | Rejected |
| Reject any session with loss | Simple, safe | Throws away usable calls; a 2% loss rate is fine | Rejected in favour of grading |
| Forward error correction / redundant frames | Repairs without retransmit | Doubles bandwidth; needs client and gateway support | Deferred |

### What is still not handled

No jitter *measurement* is exposed (arrival-time variance is not tracked, only
loss and reordering). No FEC. Sequence numbers are per-session and not
authenticated, so a hostile client can forge them — acceptable because the
endpoint has no authentication either, and both need addressing together.

---

## 3. Level normalization (AGC)

### What we do

`normalize_loudness` in [`app/audio/enhance.py`](../app/audio/enhance.py), **on by
default**. The window's speech-active RMS is estimated from the loudest third of
25 ms frames, then scaled toward `AGC_TARGET_DBFS` (−20) with the gain capped at
`AGC_MAX_GAIN_DB` (20) and a hard guarantee of no clipping.

### Why

Callers arrive at wildly different levels: a headset in an office, a speakerphone
on a dashboard, a handset in a warehouse. The heads were fitted on
utterance-level embeddings from reasonably levelled corpora, so handing them a
signal 25 dB quieter is avoidable variance. Normalization is cheap, and unlike
denoising it does not alter the spectral envelope the model reads.

### Design choices

**Speech-active RMS, not whole-window RMS.** Pauses between words would drag a
plain RMS down and over-amplify.

**A floor below which nothing is amplified.** Under `AGC_MIN_LEVEL_DBFS` (−45)
the gain is zero: the window is too quiet to be speech, and amplifying it would
only raise the noise floor and hand the model a louder version of nothing.

**Peak-safe.** If the computed gain would clip, it is backed off and the actual
applied gain is reported in the logs.

**Window-only, after the gate.** See the pipeline order above.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Fixed-target RMS normalization (chosen)** | Simple, predictable, reversible, no spectral change | Ignores perceptual loudness weighting | Adopted |
| Do nothing | No risk | Leaves a large, easily removed source of variance | Rejected |
| EBU R128 / ITU-R BS.1770 loudness | Perceptually correct; broadcast standard | K-weighting and gating machinery for a benefit the model cannot exploit | Rejected as over-engineering |
| Peak normalization | Trivial | One click or door slam sets the gain for the whole window | Rejected |
| Dynamic range compression | Evens out level *within* the window | Alters the temporal envelope, which is paralinguistic information | Rejected |
| Per-frame adaptive AGC | Tracks changing conditions | Same objection, worse: it flattens exactly the variation being measured | Rejected |
| Browser `autoGainControl` | Free | Opaque, device-dependent, and already explicitly disabled at capture for this reason | Rejected |

---

## 4. Noise reduction

### What we do

`spectral_gate` in [`app/audio/enhance.py`](../app/audio/enhance.py): spectral
subtraction with a per-bin noise estimate from the quietest frames and a spectral
floor. **Off by default** (`DENOISE_BACKEND=none`).

### Why it is off by default

This is the least intuitive decision in the pipeline, so it deserves the space.
Noise suppression helps a human listener and helps ASR. It measurably **hurts**
speaker and paralinguistic models, for two reasons: it attenuates the low-energy
spectral detail that carries voice identity, and it leaves behind artifacts
(musical noise, over-subtraction holes) that the model has never seen in
training. Every dB of noise it removes also removes some of the signal the model
reads.

The honest fix for noisy logistics audio is **training on noisy logistics
audio**, which is exactly what [ADR-002](ADR-002-production-model-strategy.md)
targets. Suppression is offered for evaluation, and for deployments that measure
a gain on their own data — never silently on. This repository has **not** measured
whether it helps or hurts on this domain, and that measurement is a prerequisite
for turning it on.

### Design choices

**Spectral subtraction with a floor**, rather than a hard gate: the floor
(`DENOISE_FLOOR_DB`, −18) caps how much of any bin can be removed, which is the
standard control for musical-noise artifacts. **Noise estimated from the 10th
percentile per bin**, so no separate noise-only recording is needed — this suits
the stationary sources that dominate the domain (engine, road, HVAC, fan) and
does nothing for transients like door slams.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Spectral subtraction, opt-in (chosen)** | No dependency, ~10 ms, effective on stationary noise, tunable | Musical noise; useless on transients; unmeasured effect on accuracy | Adopted, off by default |
| Nothing at all | Zero risk to the embedding | No option for deployments with measurably filthy audio | Rejected |
| On by default | Cleaner-sounding audio | Would silently degrade the model with no evidence it helps | Rejected |
| Wiener / MMSE-LSA filtering | Better artifact profile than plain subtraction | More parameters to tune blind, same fundamental objection | Deferred |
| RNNoise | Strong perceptual quality, small and fast | Native dependency; trained for perceptual quality, not for preserving paralinguistic cues | Rejected |
| Deep speech enhancement (DeepFilterNet, Demucs) | Best perceptual quality | Another model and forward pass; *generates* plausible speech, which is disqualifying for a measurement system | Rejected |
| Train the model on noise-augmented data | Fixes the actual problem; no inference cost | Needs the owned training pipeline | **The real answer**; see ADR-002 |

---

## 5. Echo cancellation

### What we do

Nothing server-side, and that is not a gap that can be closed here.

Acoustic echo cancellation subtracts a **known far-end reference** from the
near-end mixture — it is an adaptive filter that needs to know what the speaker
played in order to remove it from what the microphone heard. This service
receives one already-mixed channel and never sees what the agent played. There is
no reference to subtract, so anything labelled "echo cancellation" at this layer
would be guesswork dressed up as signal processing.

It has to happen where the reference exists:

- **Browser**: `echoCancellation: { ideal: true }` is already requested in
  [`microphone.js`](../../frontend/src/audio/microphone.js), so WebRTC's AEC runs
  at capture where it has the playback signal.
- **Telephony**: the gateway or PBX owns both directions and is where line echo
  cancellation belongs (G.168).

### Alternatives considered

| Option | Why not |
| --- | --- |
| Blind / single-channel echo suppression | Cannot distinguish echo from the caller's own speech; suppresses real voice |
| Require the far-end reference as a second channel | Would need a protocol change, double the bandwidth, and cooperation from an integrator who almost certainly already runs AEC |
| Detect echo and mark audio degraded | Plausible future addition; needs labelled echo data to set a threshold, which we do not have |

The residual risk is real and worth stating: if an integration sends a mixed
channel with agent audio in it, the pipeline will happily analyse whoever
dominates. The sub-window speaker-homogeneity check
([ADR-001](ADR-001-model-selection.md)) is the only defence, and it is a coarse
one.

---

## 6. Channel handling — known gap

### What we do

`_mix_channels` in [`app/audio/decoder.py`](../app/audio/decoder.py) averages a
stereo payload to mono. There is no way to select a channel.

### Why this is a gap, not just a simplification

Logistics telephony commonly delivers the two parties **already separated** —
caller left, agent right. Averaging them merges the caller with the agent and
destroys a separation the integration handed us for free. The documented
requirement ("send the caller channel only") is therefore advisory: the API
cannot enforce it and cannot honour a caller who has the separation available.

The downstream effect is at least visible rather than silent: the sub-window
speaker-homogeneity check ([ADR-001](ADR-001-model-selection.md)) will usually
mark such a segment `degraded` for holding more than one apparent speaker. That
is the correct report for a problem this stage created.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| Average to mono (current) | Trivial; correct for genuinely single-speaker stereo | Merges separated parties; unenforceable caller-only contract | Current behaviour |
| `channel_select=left\|right\|mix` parameter | Makes the caller-only requirement enforceable; ~30 lines plus tests | One more request parameter to document and validate | **Recommended; not implemented** |
| Always take the first channel | No new parameter | Wrong half the time, and silently so | Rejected |
| Diarize and pick the dominant speaker | Handles mono mixes too | A diarization model, another forward pass, and the service explicitly does not diarize | Rejected |
| Analyze both channels and return two results | Most informative | Changes the response contract, which is fixed | Rejected |

---

## 7. Streamed compressed codecs — known gap

### What we do

`WS /ws/analyze` accepts `pcm_s16le`, `pcm_s16be`, `pcm_f32le`, `mulaw`, and
`alaw`. Compressed containers (M4A, MP3, Opus, WebM, OGG, FLAC) are supported
only on `POST /analyze`.

### Why this is a gap

Compressed input works over HTTP; streaming works over WebSocket; streaming
*compressed* audio works on neither. G.711 over WebSocket covers the realistic
telephony case, and the bundled UI sidesteps the gap entirely by sending raw
float PCM from an AudioWorklet — but Opus over WebSocket is the natural choice
for a browser or a WebRTC-derived client, and it is unsupported.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| PCM/G.711 only (current) | No per-session decoder state; decode is pure and cheap | No Opus/WebM streaming client | Current behaviour |
| Persistent FFmpeg pipe per session | Reuses an existing dependency; handles every codec | A subprocess per concurrent session, with its own lifecycle, backpressure, and teardown failure modes; compressed frames cannot be sliced arbitrarily, so the trailing-window trick no longer applies | Deferred |
| `libopus` / `pyogg` bindings | Purpose-built, low latency, no subprocess | Another native dependency and licence review, for one codec | Deferred |
| Require clients to decode to PCM first (current position) | Keeps the server simple; browsers can already do this | Pushes work onto every integrator | Accepted for now |
| Accept WebRTC directly instead of a WebSocket | Native Opus, jitter buffer, and AEC all come for free | An entire media stack, and it would replace this transport rather than extend it | Out of scope; the right answer if this became a real telephony product |

Bandwidth is the honest counter-argument for revisiting this: raw 48 kHz float
PCM is roughly 1.5 Mbit/s where Opus would be ~24 kbit/s. That matters on a
mobile connection from a truck, which is exactly the target environment.

---

## 8. Measuring the age regressor: the debug estimate

### What we do

`EXPOSE_DEBUG_AGE_YEARS=true` adds a `debug_age_years` field carrying the
regressor's raw estimate. Off by default, **never persisted** (stripped in
`_model_payload`), and not part of the API contract.

### Why it exists

The service returns a bracket. Bracket labels can measure accuracy, confusion,
coverage and ECE — but **not** MAE, and not the residual spread. That is a
problem, because `AGE_RESIDUAL_SIGMA_YEARS` and the four abstention thresholds
are all supposed to be derived from exactly those two quantities. The evaluation
harness existed to inform constants it structurally could not measure. Twenty
lines of debug field closes that loop.

`AgeRegressionMetrics` in
[`evaluate_common_voice.py`](../scripts/evaluate_common_voice.py) now reports MAE,
bias, residual standard deviation overall and per expected bracket, and prints a
`suggested_AGE_RESIDUAL_SIGMA_YEARS`.

### Design choices

**The unclamped estimate, including out-of-range values.** A 14-year estimate is
`unknown` in the contract, but dropping it from the diagnostic would bias the
measured spread toward zero — precisely the statistic being measured.

**Decade label noise is subtracted in quadrature.** Common Voice publishes a
decade, not an age. Treating the midpoint as truth injects label noise with
standard deviation 10/√12 ≈ **2.887 years**, which is not negligible when the
model's own residual is around 8. The harness reports both the raw spread and the
spread with that term removed, and suggests the latter.

**Per-bracket spreads are reported separately.** If they differ widely, one
constant sigma is the wrong model and the residual should be heteroscedastic —
something the current postprocessing cannot express, and worth knowing before
committing to it.

**Off by default, never stored, and gated by name.** A point estimate of a
caller's age is a finer-grained personal inference than the bracket the contract
promises and consent covers. It is a diagnostic, so it is named `debug_`, and a
retained record must not contain it.

### Alternatives considered

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Debug field, off by default (chosen)** | Unblocks MAE and sigma measurement; zero cost when off | One more response field to explain | Adopted |
| Leave it unmeasurable (previous) | No new surface | Thresholds and sigma stay guesses forever | Replaced |
| Always expose it | Simpler | Ships a finer-grained personal inference to every caller by default | Rejected |
| A separate `/debug/analyze` endpoint | Cleanly outside the contract | Duplicates the whole pipeline, or refactors it, for a prototype diagnostic | Rejected |
| Log it instead of returning it | Never crosses the API boundary | Puts a per-caller age estimate in logs, which is worse for privacy than an opt-in field the harness reads and discards | Rejected |

---

## 9. Training and evaluation data

Nothing about the model can be *validated* without labelled audio. This is the
constraint behind every "unmeasured" row in this document.

### What is actually available

| Source | Age labels | Licence | Domain fit | Cost |
| --- | --- | --- | --- | --- |
| **Mozilla Common Voice** | Self-reported decade buckets | CC0 | Read speech, wideband, many languages | Free |
| **VoxCeleb2** | Scraped from Wikidata, noisy | Research-only | Celebrity interviews — already the current heads' training set | Free, non-commercial |
| **Fisher / Switchboard (LDC)** | Speaker metadata incl. age | LDC, paid | **Conversational telephone speech** — closest public match to the target domain | ~$1–3k non-member |
| **TIMIT (LDC)** | Speaker age | LDC, paid | Read, clean, small | ~$250 |
| **NIST SRE corpora** | Some editions | LDC, paid | Telephony, verification framing | Paid |
| **aGender** | Purpose-built 7-class age/gender | Restricted | German telephone; the Interspeech 2010 paralinguistic challenge corpus | Availability dated |

Licence terms change; verify before spending. Note that **no public corpus matches
the target domain** — logistics calls, specific carriers, specific accents, with
consented labels. That gap is structural, which is why
[ADR-002](ADR-002-production-model-strategy.md) concludes the answer is collecting
your own under an explicit consent and retention basis.

### Is measurement really needed?

Three different answers depending on the goal:

| Goal | Data needed? |
| --- | --- |
| Ship a prototype that abstains honestly | **No.** Everything is labelled uncalibrated and the service withholds rather than bluffs. |
| Stop the confidence numbers being decorative | **Some.** Common Voice is free and the harness already exists; this converts "unmeasured" into "measured on read speech". |
| Claim production accuracy in the target domain | **Yes, and only in-domain data will do.** Read-speech numbers do not transfer to a truck cab. |

The cheap middle step is the one worth taking: Common Voice plus
`EXPOSE_DEBUG_AGE_YEARS=true` yields a real gender ECE, a real bracket confusion
matrix, a measured MAE, and a fitted sigma — for the cost of a download. It will
not tell you how the service behaves over a mobile connection from a warehouse.

---

## 10. How voice age estimation is done at scale

### Current best-practice architecture

A self-supervised speech foundation model with a light head, fine-tuned:
wav2vec2, WavLM, or HuBERT. This is exactly why
[ADR-002](ADR-002-production-model-strategy.md) targets WavLM Base+ over the
present ECAPA stack — a speaker encoder is trained to *discard* the phonetic and
prosodic variation that age estimation needs, so it is working against itself.

Four techniques that separate a serious system from a naive one, none of which
the current baseline implements:

| Technique | Why it matters here |
| --- | --- |
| **Ordinal or distributional targets** — soft Gaussian labels, K−1 binary ordinal classifiers, or DEX-style expectation over a softmax | Age is ordinal with noisy labels. Regress-then-threshold is the direct cause of the "±8 years flips the bracket" problem |
| **Multi-task learning** (age + gender + speaker ID jointly) | Regularizes, and the tasks share structure |
| **Post-hoc calibration** — temperature scaling or isotonic regression on held-out speakers | Non-negotiable for any service that emits a confidence number |
| **Conformal prediction** | Produces honest intervals, which suits an abstaining API better than a point estimate plus a made-up sigma |

### Who ships this

**audEERING (devAIce)** is the clearest commercial vendor in exactly this space,
already ADR-002's "fastest buy-now" option — the public weights are CC-BY-NC, so
production use needs a paid licence. Contact-centre analytics platforms (Verint,
NICE, Genesys) and voice-fraud vendors such as Pindrop carry demographic or
voice-profiling features.

**The most informative observation: the hyperscalers deliberately do not sell
this.** Google, AWS and Azure speech APIs offer no voice age or gender inference,
and Microsoft *retired* face-based age/gender/emotion inference from Azure Face
in 2022 on Responsible AI grounds. When the largest providers examine demographic
inference from biometrics and choose not to productize it, that is a statement
about regulatory and reputational exposure rather than technical difficulty.

The EU AI Act points the same way: emotion recognition in workplace contexts is
prohibited, and biometric categorisation inferring protected attributes is
prohibited or high-risk. Inferring gender from voice sits uncomfortably close to
that boundary. Under GDPR, inferred demographics are personal data regardless of
whether the voice is used for identification. This is a product and legal
question, not an engineering one, and it belongs in the decision *before*
investing in data collection — which is why the [model card](MODEL_CARD.md)
forbids consequential use rather than leaving it to the integrator.

### Buy versus build, for this service

| Path | When it makes sense | Cost |
| --- | --- | --- |
| **Keep the abstaining baseline** | Prototype, low-stakes style adaptation | Already done |
| **Measure on Common Voice** | Immediately, to stop guessing constants | An afternoon |
| **Licence audEERING devAIce** | A product deadline and an acceptable licence fee | Commercial agreement |
| **Train WavLM + ordinal head + calibration** | Owning the domain, and having collected consented data | Months, plus a data-collection programme |
| **Don't ship the attribute at all** | The legal review says the exposure outweighs the benefit | The option the hyperscalers took |

---

## 11. Scaling: where the capacity actually goes

Measured in the CPU container, 11 CPUs visible. These are the numbers that
matter for capacity planning, and several of them contradict what the
configuration names suggest.

### Where a request spends its time

One 10-second progressive update, minimum of five runs:

| Stage | Time | Share |
| --- | ---: | ---: |
| Decode + resample | 14.3 ms | 5.8% |
| Quality gate | 2.1 ms | 0.8% |
| Window selection | 0.9 ms | 0.4% |
| Enhancement (AGC) | 0.1 ms | 0.0% |
| **Inference (3-window ECAPA)** | **230.2 ms** | **93.0%** |
| Total | 247.5 ms | |

**93% is one forward pass.** Every signal-processing technique in sections 1–7
combined is 7%. Optimizing decode, the resampler, or the quality gate is
therefore pointless; only the encoder matters.

### Inference does not scale with threads

| Torch threads | Latency | Speedup | CPU efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 391.1 ms | 1.00× | 100% |
| 2 | 221.9 ms | 1.76× | 88% |
| 4 | 206.7 ms | 1.89× | 47% |
| 8 | 220.9 ms | 1.77× | 22% |

It plateaus at two threads and gets *worse* at eight. ECAPA-TDNN at this size is
memory-bandwidth-bound, not core-starved, so `TORCH_THREADS=2` is the measured
sweet spot and raising it wastes cores that another container could use.

### Per-container capacity is ~4 analyses/second, however you arrange it

Eight concurrent `POST /analyze` requests:

| Configuration | 200 OK | 503 shed | p50 latency | Throughput | Memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 replica, 2 threads | 3 | 5 | 949 ms | 2.3/s | 692 MB |
| 4 in-process replicas | 7 | 1 | 1251 ms | 3.9/s | 1.2 GB |
| 4 uvicorn processes | 7 | 1 | 1111 ms | 4.0/s | 1.95 GB |

Three conclusions, none of them obvious from the configuration names:

1. **`INFERENCE_CONCURRENCY` buys availability, not speed.** Going from 1 to 4
   replicas converted five shed requests into one, but *raised* p50 latency from
   949 ms to 1251 ms. It is a knob for absorbing bursts, not for going faster.
2. **Processes are not better than in-process replicas here** — 4.0/s versus
   3.9/s, at 60% more memory. The initial hypothesis (that PyTorch's process-wide
   intra-op pool was the bottleneck) was **wrong**; measurement showed both
   arrangements hitting the same ceiling.
3. **The sub-500 ms objective holds for one in-flight request only.** Under eight
   concurrent requests `processing_ms` reaches 1.7 s. Load shedding via a 1-second
   queue deadline is what keeps that bounded, and it works.

### What actually raises capacity

In order of value per unit of effort:

| Lever | Effect | Cost | Status |
| --- | --- | --- | --- |
| **Fewer analyses per call** | Adaptive emit backoff already cut a 20 s session from ~19 updates to 8 — a 2.4× reduction in work per call, the single largest win so far | Configuration only | **Done** (§ streaming) |
| `ENSEMBLE_WINDOWS=1` | ~230 ms → ~130 ms, a 40% cut | Loses per-sample age spread (so age abstention stops being reachable) and the single-speaker check | Available, off |
| `LANGUAGE_BACKEND=none` | Saves ~200 ms per analysis | No language field | Default |
| **ONNX export + INT8 quantization** | Typically 2–3× on CPU for this model class; the largest remaining per-request win | Export and parity-validate the encoder; the last pickled artifact disappears too | **Not implemented**; see ADR-002 |
| Shorter `INFERENCE_WINDOW_SECONDS` | Compute scales with window length | Unmeasured accuracy cost | Available |
| **More containers** | Linear, and the only lever that is genuinely linear | Memory and orchestration | The recommended path |
| GPU micro-batching | The answer at four-figure concurrency; batching pays on GPU, barely on CPU | New deployment shape | ADR-002 target |

### Capacity and cost at 2,000 concurrent calls

Worked for 2,000 concurrent calls of 3 minutes average duration, at 65% target
utilization. **The workload shape dominates every other factor**, so it comes
first.

#### Step 1: how many analyses does a call actually need?

2,000 *concurrent* calls is only ~11 *new* calls per second. Whether that means
11 analyses/second or 522 depends entirely on the emit cadence:

| Shape | Analyses per call | Required throughput |
| --- | ---: | ---: |
| **A** Progressive, current defaults (1 s base, ×1.5, 4 s cap) | 47 | **522/s** |
| **B** Progressive, cost-tuned (2 s base, ×1.5, 8 s cap) | 25 | **278/s** |
| **C** One-shot per call | 1 | **11/s** |
| **D** One-shot + 2 stability re-checks | 3 | **33/s** |

A to C is a **47× range**. If the product needs one estimate to set an agent's
tone, the answer is 11 analyses/second and none of the rest of this section
matters. Decide this before provisioning anything.

#### Step 2: containers required (2 vCPU / 2 GB each)

| Configuration | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| CPU baseline, 3-window, language off (**measured** 4.0/s) | 187 | 100 | 4 | 12 |
| CPU, `ENSEMBLE_WINDOWS=1` (derived, 130 ms) | 105 | 56 | 3 | 7 |
| CPU + language ID (derived, 430 ms) | 350 | 186 | 8 | 23 |
| CPU + INT8 ONNX (**estimated** 2.5×) | 75 | 40 | 2 | 5 |
| GPU L4 + micro-batching (**estimate only**, unmeasured) | 9 | 5 | 1 | 1 |

#### Step 3: monthly compute cost

Approximate US-East on-demand rates: Fargate at $0.04048/vCPU-hour plus
$0.004445/GB-hour; EC2 `c7g.large` at roughly $0.046/hour under a one-year
compute savings plan. Rates move — verify before budgeting.

| Configuration | Shape | Fargate | EC2 Graviton, reserved |
| --- | --- | ---: | ---: |
| Baseline | A | $12,265 | $6,279 |
| Baseline | B | $6,559 | $3,358 |
| Baseline | **C** | **$262** | **$134** |
| Baseline | D | $787 | $403 |
| + language ID | A | $22,957 | $11,753 |
| + language ID | B | $12,200 | $6,246 |
| INT8 ONNX | A | $4,919 | $2,518 |
| INT8 ONNX | B | $2,624 | $1,343 |

#### Step 4: the cost that is easy to forget

Inference is not the only thing that scales. Audio has to arrive:

| Codec | Per call | Aggregate at 2,000 calls | Ingress volume |
| --- | ---: | ---: | ---: |
| **Raw f32 @48 kHz — what the browser sends today** | 1,536 kbit/s | **3.07 Gbit/s** | 33 TB/day |
| PCM s16 @16 kHz | 256 kbit/s | 0.51 Gbit/s | 5.5 TB/day |
| G.711 μ-law @8 kHz | 64 kbit/s | 0.13 Gbit/s | 1.4 TB/day |
| Opus @24 kbps — **unsupported over WebSocket today** | 24 kbit/s | 0.05 Gbit/s | 0.5 TB/day |

Three gigabits per second of sustained ingress, and 33 TB/day crossing the wire,
for a service whose entire compute bill at shape C is $134/month. Load-balancer
capacity units, NIC saturation, and client-side mobile data all sit on that line.

**This is the concrete price of the streaming-codec gap in section 7.** Opus is a
60× reduction. At prototype scale that gap is a footnote; at 2,000 calls it is
plausibly the largest single line item, and it would justify the persistent
decoder that section 7 defers.

#### Conclusions

1. **Fix the workload shape first.** A → C is worth more than every optimization
   in this document combined.
2. **If progressive updates are genuinely required**, tune the cadence and
   quantize: A-baseline $6,279/mo → B-INT8 $1,343/mo, a 4.7× reduction.
3. **Language identification roughly doubles the bill** (187 → 350 containers at
   shape A). It is off by default and cached once per session for this reason;
   at scale that default is worth real money.
4. **GPUs only pay at shape A.** Nine L4s beat 187 CPU containers, but at shape C
   a single small container suffices. Do not buy accelerators for 11 analyses/s.

Nothing here contradicts ADR-002, which already names GPU micro-batching as the
production answer. What is new is that the numbers are measured rather than
assumed, that two intuitive optimizations — more replicas, more threads — are
shown not to work, and that bandwidth rather than compute may set the bill.

#### Caveats on these numbers

- The 4.0 analyses/second baseline is measured on **Docker Desktop on Apple
  Silicon**, a virtualized environment, not a server. Re-measure on target
  hardware before committing budget. The *ratios* (thread plateau, replicas not
  helping, 93% of time in one forward pass) should travel; the absolute
  throughput may not.
- The **INT8 2.5× and GPU 100/s figures are estimates, not measurements.** They
  are marked as such in the tables and should not be used for procurement until
  measured.
- Costs cover compute and audio ingress only. They exclude the load balancer,
  observability, PostgreSQL and S3 (consent-gated, so usually a small fraction),
  NAT, and any GPU node minimums.

### Autoscaling signal

`voice_attribute_queue_wait_seconds` and `voice_attribute_queue_rejections_total`
exist for this. Queue wait rises *before* requests start being shed, which makes
it the correct scale-out trigger; rejections are the lagging alarm that capacity
was already exceeded. Scaling on CPU utilization would be misleading, because a
single memory-bandwidth-bound inference does not saturate the cores it is
starving.

---

## 12. Transport security, rate limiting, and API versioning

Three requests that arrive together and have three different correct answers:
one belongs entirely in the application, one only partly, and one not at all.

### 12.1 API versioning — belongs in the application

**What we do.** Business endpoints are served under `/v1`
(`/v1/analyze`, `/v1/analyses`, `/v1/analyses/{id}`,
`/v1/persistence/capabilities`, `/v1/ws/analyze`). The same router is mounted a
second time without the prefix, so every pre-existing path still works, and the
unversioned mount is hidden from the OpenAPI schema so `/docs` advertises only
the versioned surface. The bundled UI now calls `/v1`.

Probes and metrics (`/healthz`, `/readyz`, `/metrics`) stay **unversioned by
design**: they are operational contracts with the orchestrator and the scraper,
not part of the caller-facing API, and versioning them would mean editing
deployment manifests to ship an API change.

**Metric labels strip the prefix**, so `/v1/analyses` and `/analyses` share one
time series. A dashboard has to span the migration; two labels for one endpoint
would silently split every panel.

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Path prefix, legacy alias retained (chosen)** | Obvious in logs and dashboards; trivially routable at an ingress; no client breakage | Two mounts to keep in mind | Adopted |
| Header-based version negotiation | Cleaner URLs; content negotiation is the "correct" REST answer | Invisible in access logs, harder to route on, easy for a client to omit | Rejected |
| Break the existing paths | One surface, no duplication | Breaks every integration for cosmetic gain | Rejected |
| No versioning (previous) | Nothing to do | The first contract change becomes a breaking change | Replaced |

The next contract change is where this pays: `/v2` can ship beside `/v1` instead
of forcing every caller to move on the same day.

### 12.2 Rate limiting — belongs in the application, but only as depth

**What we do.** A token bucket per client in
[`app/ratelimit.py`](../app/ratelimit.py), enabled by default at 60 requests per
minute with a burst of 10. Over-budget HTTP requests get **429 with
`Retry-After`**; over-budget WebSocket handshakes are closed with **1013 "try
again later"** *before* `accept`, so a throttled client never holds a model
replica or a decode slot. `/healthz`, `/readyz` and `/metrics` are exempt.

**This is defence in depth, not the primary control.** By the time this code
runs, the connection, the TLS handshake and the request body have already been
paid for. The primary limit belongs at the ingress or WAF, which can drop traffic
before it costs anything. What the in-process limiter buys is a bound on
*expensive* work per client when the ingress is absent or misconfigured — and
that matters here specifically because one analysis is ~230 ms of
memory-bandwidth-bound inference against a ~4/second ceiling.

Two properties worth stating plainly:

- **The limit is per container.** With N containers behind a load balancer, a
  client gets N times the configured rate. At the 187 containers of shape A that
  is not a meaningful limit at all. A global limit needs shared state (Redis, or
  the ingress).
- **The client table is bounded** (`RATE_LIMIT_MAX_TRACKED_CLIENTS`, default
  10,000, evicting the stalest tenth). An unbounded dictionary keyed by client
  address is itself a denial-of-service vector, so the mitigation must not become
  the vulnerability.

**Client identity is the subtle part.** Behind a proxy the peer address is the
proxy, so every caller would share one bucket. `TRUSTED_PROXY_HOPS` says how many
proxies append to `X-Forwarded-For`; the address that many hops from the right is
the first one those proxies did not add, and everything to its left is
client-supplied and forgeable. At `0` the header is ignored entirely, which is
correct when the service is reached directly or when uvicorn's `--proxy-headers`
has already rewritten the peer address. **Getting this wrong fails in one of two
directions:** too low collapses every caller into one bucket, too high lets a
caller forge its identity and bypass the limit outright.

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **In-process token bucket (chosen)** | No dependency; works with no ingress; bounds expensive work per client | Per container, not global; cannot protect against traffic already accepted | Adopted as depth |
| Ingress / WAF limiting | Drops traffic before it costs anything; genuinely global | Infrastructure, not in this repository | **The primary control**; assumed present in production |
| Redis-backed shared bucket | Genuinely global across containers | A network round trip on every request, and a new hard dependency in the hot path | Deferred |
| Fixed-window counter | Simplest to reason about | Allows a double-rate burst across a window boundary | Rejected |
| Leaky-bucket queueing instead of rejecting | No client-visible errors | Adds latency to a real-time service, and queueing is what `SERVICE_BUSY` already sheds | Rejected |
| Per-tenant quotas | What a commercial deployment actually needs | Requires authentication, which does not exist yet | Blocked on auth |

### 12.3 TLS — does **not** belong in the application

The service still terminates plaintext HTTP, and that is the right default for a
container. TLS termination belongs at the ingress, load balancer, or service
mesh, which own certificate issuance, rotation, OCSP, and cipher policy. Moving
that into the application would mean shipping certificate lifecycle management
inside an inference service.

What the application *is* responsible for, and now does:

- **`X-Content-Type-Options: nosniff`** and **`Referrer-Policy: no-referrer`** on
  every response.
- **HSTS**, emitted only when `HSTS_MAX_AGE_SECONDS` is set **and** the request
  actually arrived over HTTPS (directly or per `X-Forwarded-Proto`). Announcing
  HSTS on a plaintext response is ignored by browsers and untrue besides, so it
  is off by default and conditional when on.
- **Being honest about the scheme it is behind**, which requires running uvicorn
  with `--proxy-headers --forwarded-allow-ips=<ingress CIDR>`. Without it the app
  believes every request is plaintext from the load balancer: HSTS never fires
  and rate limiting collapses to one bucket.

| Option | Benefits | Disadvantages | Verdict |
| --- | --- | --- | --- |
| **Terminate at the ingress (chosen)** | Certificate lifecycle handled by infrastructure built for it; standard container practice | Requires correct proxy-header configuration to avoid silent misbehaviour | Adopted |
| Terminate in uvicorn (`--ssl-keyfile`) | Useful for local HTTPS when testing microphone capture, which browsers gate behind a secure context | Certificate rotation becomes an application concern; no OCSP or modern cipher policy management | Supported for development only |
| mTLS between ingress and service | Real defence for east-west traffic | Belongs to a service mesh, not to this code | Recommended, out of scope |
| Application-level TLS in production | Nothing this deployment gains | Redundant behind an ingress that already terminates | Rejected |

### What this still does not give you

Rate limiting is not authentication, and neither is the WebSocket origin
allowlist — it rejects only when an `Origin` header is *present and unlisted*, so
a non-browser client that sends none is admitted. That is deliberate, so server
adapters work, but it means **the WebSocket endpoint has no access control**.

There is still no authentication, no tenant isolation, and no per-tenant quota.
Rate limiting narrows the blast radius of an anonymous caller; it does not
establish who anyone is. That remains the largest gap before real caller audio.

---

## Cross-cutting: what is measured versus assumed

| Claim | Status |
| --- | --- |
| Alias rejection ≥70 dB above 10 kHz | Measured, tested |
| Resampler latency 14.8 ms / 10 s of 48 kHz | Measured |
| Reorder, duplicate, loss accounting | Tested, including end-to-end over a WebSocket |
| Loss above threshold → `degraded` | Tested end-to-end |
| AGC does not clip, does not amplify silence | Tested |
| Spectral gate reduces stationary noise | Tested (energy in inter-speech gaps) |
| Age MAE and residual spread | **Measurable now** via `EXPOSE_DEBUG_AGE_YEARS` + Common Voice; not yet run |
| Whether denoising helps or hurts *accuracy* | **Unmeasured** — the reason it is off |
| Whether concealment thresholds are right for real networks | **Unmeasured** — no packet-loss traces |
| Whether AGC improves accuracy | **Unmeasured** — assumed from reduced variance |
| Per-container throughput (~4 analyses/s) | Measured, but on virtualized Apple Silicon rather than server hardware |
| INT8 quantization speedup (2.5×) and GPU throughput (100/s) | **Estimates** used in the cost tables; not measured |

The unmeasured rows need the same thing: a consented, labelled, in-domain
evaluation set. Until that exists these are defensible engineering defaults, not
validated ones.
