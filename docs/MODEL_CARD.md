# Model card: Runnable ECAPA baseline

## Summary

This service uses one pretrained SpeechBrain ECAPA-TDNN speaker encoder and two shallow models operating on its shared 192-dimensional embedding:

- a binary SVM for the upstream `female`/`male` voice labels;
- an SVR that estimates age in years, which service postprocessing converts to `18-30`, `31-45`, `46-60`, or `60+`.

The API can return `unknown` for either attribute and separately labels audio `good`, `degraded`, or `insufficient`. The service name `gender` follows the required contract; the model estimates perceived binary voice presentation, not gender identity or biological sex.

This is the Compose-default baseline and rollback path, not the final production-accuracy claim. [ADR-002](ADR-002-production-model-strategy.md) selects logistics-trained WavLM Base+ joint heads as the target after paired evaluation.

## Artifact identity

| Artifact | Revision | Declared license |
| --- | --- | --- |
| [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286) | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | Apache-2.0 |
| [`griko/gender_cls_svm_ecapa_voxceleb`](https://huggingface.co/griko/gender_cls_svm_ecapa_voxceleb/tree/25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64) | `25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64` | Apache-2.0 |
| [`griko/age_reg_svr_ecapa_voxceleb2`](https://huggingface.co/griko/age_reg_svr_ecapa_voxceleb2/tree/1d2356ac55f51fbd3f327f1b9260860decb21233) | `1d2356ac55f51fbd3f327f1b9260860decb21233` | Apache-2.0 |

Exact file SHA-256 values are maintained in `scripts/prepare_models.py` and copied into `/opt/models/model-metadata.json` in the image. See [ADR-001](ADR-001-model-selection.md) for license comparison and serialization controls. Model licenses do not resolve dataset, privacy, publicity, biometric, or voice-rights obligations.

## Intended use

- Low-stakes, best-effort adaptation of a logistics voice agent's conversational style.
- Caller-only, single-speaker, adult speech from active inbound or outbound calls.
- Short segments, ideally 2–5 seconds of speech, with explicit abstention when quality or confidence is poor.
- Offline inference in a controlled backend where results are transient or governed as sensitive inferred data.
- Evaluation baseline to be validated and recalibrated on consented target-domain recordings.

## Out-of-scope and prohibited use

- Identifying a speaker or matching them to a known person.
- Inferring gender identity, biological sex, pronouns, legal gender, or minor status.
- Authentication, authorization, surveillance, biometrics, fraud adjudication, or law enforcement.
- Eligibility, pricing, credit, insurance, employment, medical, legal, safety, or other consequential decisions.
- Mixed agent/caller audio, multi-party calls, hold music, or recordings without required notice/consent.
- Silent, unconsented, or indefinite profile enrichment and demographic databases.

## Input and preprocessing

The estimator consumes finite, mono, 16 kHz float32 PCM. The service accepts native WAV and raw PCM/G.711, while bundled FFmpeg handles common compressed containers and probes their original codec/rate. Stereo is averaged, and non-16 kHz input is linearly resampled. Samples are centered, clipped to `[-1,1]`, and capped at 30 seconds. For longer inputs, a speech-evidence score selects a contiguous five-second window, with a conservative energy fallback.

No speech enhancement, denoising, diarization, source separation, echo cancellation, transcription, language identification, or spoof detection is performed. Telephony integration must send only the caller/contact channel.

## Quality gate

The gate estimates duration, RMS level, voiced-frame coverage, SNR, clipping, spectral flatness, zero-crossing behavior, and low-frequency energy. Narrowband/8 kHz/G.711 audio is at least degraded.

Default insufficient triggers include duration below 1.25 seconds, estimated voiced speech below 0.65 seconds, speech ratio below 0.08, or RMS below -48 dBFS. Default degraded triggers include duration below two seconds, speech ratio below 0.22, SNR below 8 dB, clipping above 2%, low-frequency energy ratio above 0.55, or narrowband source. These are signal heuristics and may misclassify music, synthetic speech, unusual voices, or stationary noise.

Insufficient audio returns both outputs as unknown with zero confidence and skips the model. Degraded audio continues but multiplies confidence by `0.75`. FFmpeg use is not itself a degradation trigger; missing probed metadata and known narrowband codecs remain conservative triggers.

## Outputs and confidence

The SVM yields two scores converted to probabilities. The build converter reproduces the source classifier's Platt-style mapping when available; otherwise its fallback sigmoid is only a monotonic score. Runtime normalizes the two values, picks the maximum, applies the quality factor, and returns unknown below `0.60`.

The SVR yields a continuous age. Values below 18 or non-finite values become unknown. Otherwise the age is mapped to the required bracket. Confidence is not emitted by the source SVR: the service assumes a normal residual distribution with ten-year sigma, calculates the probability mass inside the selected bracket conditional on age at least 18, applies the quality factor, and returns unknown below `0.28`.

Therefore confidence is a model/postprocessing score, not a verified probability that the contact has an identity or age. It is not calibrated on logistics calls. Returning `0.0` for unknown means the result was withheld; it does not reveal the original sub-threshold score. Non-finite ages and regression outliers outside 18–120 also abstain.

## Training data and upstream evaluation

The SpeechBrain model card says the encoder was trained on VoxCeleb1 and VoxCeleb2 at 16 kHz mono for speaker verification. Its reported cleaned VoxCeleb1 test EER is 0.80%; that metric describes speaker verification, not age or voice-presentation accuracy.

The griko gender card says its SVM was trained on a balanced VoxCeleb2 speaker split with no speaker overlap and evaluated on VoxCeleb2, Mozilla Common Voice v10 English, and TIMIT. It reports 98.9%, 92.3%, and 99.6% accuracy respectively. It also states that VoxCeleb audio contains celebrity YouTube interviews and the classifier is binary.

The griko age card says the ECAPA-only SVR was trained on VoxCeleb2, with age metadata gathered from Wikidata and public sources, and reports 7.89 years MAE on its VoxCeleb2 test set. Age-label noise and time mismatch between a person's metadata and recording are plausible limitations.

All figures above are upstream-reported. This repository has not independently reproduced them and provides no logistics-call accuracy or calibration claim.

## Limitations and foreseeable failure modes

- Domain shift: celebrity interviews and clean read speech differ from truck cabs, warehouses, dispatch radios, packet loss, speakerphone, and moving vehicles.
- Demographic bias: representation, label conventions, recording conditions, languages, accents, regions, and age distribution can cause unequal error and abstention rates.
- Construct validity: a binary acoustic label cannot represent gender identity. Voices do not map reliably to identity categories.
- Age uncertainty: the upstream MAE is large relative to some API brackets; errors near 31, 46, and 60 are amplified by hard boundaries.
- Mixed speakers: without caller-channel isolation, the result may describe an agent, dispatcher, bystander, or whichever speaker dominates energy.
- Channel and noise: narrowband codecs, clipping, road/engine noise, music, reverberation, packet-loss concealment, and aggressive noise suppression change embeddings.
- Voice state: illness, fatigue, stress, smoking, vocal training, acting, pitch modification, and assistive devices can shift predictions.
- Synthetic or adversarial audio: cloned, converted, replayed, or deliberately manipulated speech is not detected.
- Children: the contract exposes adult brackets only. An under-18 regression becomes unknown, but this must not be used to determine whether someone is a minor.
- Confidence: the age score is heuristic, gender calibration is inherited/converted, and neither is validated for this domain.
- Progressive streaming: every update analyzes cumulative audio and can change; only the final result should be treated as settled.

## Evaluation requirements

Before production, build a consented, caller-only, speaker-disjoint dataset representing target carriers, languages, accents, devices, sites, vehicles, codecs, background conditions, durations, and call directions. Do not reuse operational recordings without an approved purpose and retention basis.

Report:

- gender confusion matrix, macro F1, balanced accuracy, coverage, and risk at each abstention threshold;
- age MAE plus bracket confusion matrix, macro F1, adjacent-versus-severe errors, and coverage;
- Brier score/ECE and reliability diagrams for each output;
- p50/p95/p99 decode, quality, queue, inference, and total latency after warmup;
- results sliced by codec, SNR, duration, carrier/device, language/accent, and lawful demographic groups;
- quality-gate false accept/reject behavior, multi-speaker contamination, synthetic speech, and drift.

Choose thresholds from error-versus-coverage and harm/cost curves. Recalibrate using held-out domain data. Keep evaluation labels self-reported where appropriate, distinguish label constructs, and document annotator uncertainty.

## Scaling and monitoring

The supplied CPU container serializes estimator access and defaults to one inference at a time. For 1,000 concurrent calls, separate ingestion from GPU inference, dynamically micro-batch five-second windows, use deadline-aware bounded queues, autoscale on queue depth/GPU utilization/p95 latency, and apply admission control before buffers or latency grow without bound. Regionalize processing to meet privacy and latency requirements.

Monitor aggregate quality distribution, unknown coverage, confidence histograms, latency, busy errors, and drift. Once delayed ground truth is lawfully available, monitor calibration and slice performance. Do not sample or retain raw production audio merely for dashboards.

## Known gaps in this release

- A Common Voice harness is included, but no dataset, reproduced score, logistics-domain holdout, or calibration artifact is bundled.
- No measured hardware-specific proof of the under-500 ms target.
- No language/accent field.
- No diarization or caller-channel verification.
- No ONNX, quantization, GPU batching, or incremental streaming embeddings.
- No built-in authentication, TLS, rate limiting, or tenant isolation.

## Change policy

Any model revision, checksum, preprocessing rule, bracket boundary, residual sigma, quality threshold, or confidence threshold is a model change. Rebuild the model card, rerun supply-chain checks and the full evaluation suite, compare coverage/error/calibration/latency, obtain privacy and license review, and canary before rollout.
