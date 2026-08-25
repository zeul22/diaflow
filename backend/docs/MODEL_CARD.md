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

Exact file SHA-256 values are maintained in `backend/scripts/prepare_models.py` and copied into `/opt/models/model-metadata.json` in the image. See [ADR-001](ADR-001-model-selection.md) for license comparison and serialization controls. Model licenses do not resolve dataset, privacy, publicity, biometric, or voice-rights obligations.

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

The estimator consumes finite, mono, 16 kHz float32 PCM. The service accepts native WAV and raw PCM/G.711, while bundled FFmpeg handles common compressed containers and probes their original codec/rate. Stereo is averaged, and non-16 kHz input is converted by a band-limited windowed-sinc resampler (linear interpolation previously folded out-of-band energy into the speech band). Samples are centered, clipped to `[-1,1]`, and capped at 30 seconds. For longer inputs, a speech-evidence score selects a contiguous five-second window, with a conservative energy fallback.

No diarization, source separation, echo cancellation, transcription, or spoof detection is performed. Level normalization is applied to the inference window only (after the quality gate has judged the untouched signal), denoising is available but off by default, and language identification is available but off by default. Telephony integration must send only the caller/contact channel.

## Quality gate

The gate estimates duration, RMS level, voiced-frame coverage, SNR, clipping, spectral flatness, zero-crossing behavior, and low-frequency energy. Narrowband/8 kHz/G.711 audio is at least degraded.

Default insufficient triggers include duration below 1.25 seconds, estimated voiced speech below 0.65 seconds, speech ratio below 0.08, or RMS below -48 dBFS. Default degraded triggers include duration below two seconds, speech ratio below 0.22, SNR below 8 dB, clipping above 2%, low-frequency energy ratio above 0.55, or narrowband source. These are signal heuristics and may misclassify music, synthetic speech, unusual voices, or stationary noise.

Insufficient audio returns both outputs as unknown with zero confidence and skips the model. Degraded audio continues, and rather than rescaling the reported confidence it requires the result to clear a stricter abstention threshold. FFmpeg use is not itself a degradation trigger; missing probed metadata and known narrowband codecs remain conservative triggers. A segment whose sub-window embeddings disagree about the speaker is also reported as degraded.

## Outputs and confidence

Confidence is the model's own score at every stage. Quality never multiplies it; quality selects which threshold it must clear (`0.60`/`0.75` for gender, `0.35`/`0.45` for age on good/degraded audio).

**Gender.** The pinned upstream SVM was fitted with `probability=False`, so it exposes no `predict_proba` and no Platt parameters exist to convert. The build exports an explicitly flagged uncalibrated margin sigmoid; the runtime refuses to load such an artifact unless `REQUIRE_CALIBRATED_GENDER=false`, which this baseline image sets and warns about at startup. **The baseline's gender confidence is therefore a monotonic function of the SVM decision margin, not a probability**, and its threshold is a margin cut-off. Runtime normalizes the two values and picks the maximum.

**Age.** The SVR yields a continuous age; values below 18, above 120, or non-finite become unknown. Otherwise the age is mapped to its bracket and confidence is the modeled normal mass inside that bracket, conditional on the bounded adult support 18–120. The sigma combines the assumed ten-year population residual, this sample's standard deviation across the sub-window ensemble, and an extrapolation term outside the 20–70 range where the upstream head has training support.

Two properties of the previous form are worth recording, because they are the reason the current one exists. Its fixed sigma made confidence a deterministic function of the distance to the nearest bracket edge — identical for every caller whose estimate landed on the same year, with a floor of `0.42` that sat above the configured threshold, so age never abstained on low confidence. And integrating the open `60+` bracket to infinity gave a 90-year estimate `0.999`, the API's highest confidence in the region where the model is least reliable. Ensemble spread now makes the score sample-specific and the threshold reachable; bounded support and extrapolation inflation cap the top bracket near `0.74` and make it decline as the estimate leaves the training range. Cross-bracket confidences remain incomparable, since `60+` is far wider than the others, and tail bias is still unmeasured.

**Language (optional).** When `LANGUAGE_BACKEND=voxlingua_ecapa`, a separate Apache-2.0 VoxLingua107 ECAPA classifier adds a `language` tag. Audio below `LANGUAGE_MIN_SECONDS` never reaches it, because short windows produce confident *wrong* answers rather than uncertain ones — three seconds of English measured Latin at 0.929. An answer is then accepted only when it clears a floor *and* leads the runner-up by a margin; with `LANGUAGE_ALLOWLIST` set the floor applies to the summed mass of the languages the deployment serves rather than to one class out of 107. The reported confidence is always the raw posterior, never renormalized. The model has no non-speech class and will confidently name a language for music, noise, or a test tone. Streaming sessions track the most recent confident detection, so a mid-call language switch is followed after a few seconds' lag and can be transiently wrong while the window holds both languages. It identifies a language and nothing else — never an accent, dialect, region, or nationality. See [ADR-004](ADR-004-language-identification.md).

Confidence is therefore a model/postprocessing score, not a verified probability that the contact has an identity or age, and it is not calibrated on logistics calls. Returning `0.0` for unknown means the result was withheld; it does not reveal the original sub-threshold score.

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
- Mixed speakers: the sub-window homogeneity check downgrades an obviously mixed segment to `degraded`, but it is a coarse similarity test rather than diarization. Without caller-channel isolation the result may still describe an agent, dispatcher, bystander, or whichever speaker dominates energy.
- Channel and noise: narrowband codecs, clipping, road/engine noise, music, reverberation, packet-loss concealment, and aggressive noise suppression change embeddings.
- Voice state: illness, fatigue, stress, smoking, vocal training, acting, pitch modification, and assistive devices can shift predictions.
- Synthetic or adversarial audio: cloned, converted, replayed, or deliberately manipulated speech is not detected.
- Children: the contract exposes adult brackets only. An under-18 regression becomes unknown, but this must not be used to determine whether someone is a minor.
- Confidence: the age score is a model of the residual rather than a measured one, the baseline's gender score has no calibration at all, and neither is validated for this domain.
- Progressive streaming: every update re-analyzes a trailing window and can change; only the final result should be treated as settled.

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

- No calibrated gender head. The pinned upstream classifier cannot supply one, so the baseline serves an uncalibrated margin score behind an explicit opt-in.
- `AGE_RESIDUAL_SIGMA_YEARS`, the age reliable range, and all four confidence thresholds are placeholders, not values derived from a coverage-versus-error curve.
- A Common Voice harness is included, but no dataset, reproduced score, logistics-domain holdout, or calibration artifact is bundled.
- No measured hardware-specific proof of the under-500 ms target.
- No accent or dialect field, and no plan for one: see [ADR-004](ADR-004-language-identification.md).
- Language identification is untested on telephony audio and has no non-speech reject class.
- No diarization or caller-channel verification.
- No bundled approved production WavLM ONNX artifact, quantization, GPU batching, or incremental streaming embeddings.
- No built-in authentication, TLS, rate limiting, or tenant isolation.

## Change policy

Any model revision, checksum, preprocessing rule, bracket boundary, residual sigma, quality threshold, or confidence threshold is a model change. Rebuild the model card, rerun supply-chain checks and the full evaluation suite, compare coverage/error/calibration/latency, obtain privacy and license review, and canary before rollout.
