# ADR-001: Shared ECAPA encoder with SVM/SVR attribute heads

- Status: Accepted as runnable baseline; production target superseded by ADR-002
- Date: 2026-08-24
- Owners: Voice platform
- Review trigger: domain evaluation failure, p95 latency above the service objective, material license change, or a better commercially usable model

## Context

The service must estimate an adult caller's age bracket and the binary voice-presentation label required by the API from roughly five seconds of caller-only audio. It must run locally, tolerate telephony codecs and logistics noise, target sub-500 ms end-to-end inference, abstain on bad input, permit commercial deployment, and avoid retaining audio. Reproducibility and supply-chain safety matter because public model repositories can change and two candidate heads are distributed as pickle-based joblib files.

The term `gender` is retained only for API compatibility. Available training labels are `female` and `male`; they describe a model's classification of acoustic voice presentation. They cannot establish identity, biological sex, pronouns, or legal gender.

## Decision drivers

1. A license compatible with a commercial logistics product.
2. One compact inference path for both attributes, preferably on CPU.
3. Public, inspectable weights and useful published cross-corpus evidence.
4. Direct continuous age output so product brackets can change without retraining.
5. Deterministic, offline deployment with immutable inputs and integrity checks.
6. Confidence access, explicit abstention, and quality-aware degradation.
7. A migration path to domain-trained or accelerated heads.

## Decision

Use a single SpeechBrain ECAPA-TDNN encoder and two shallow griko heads:

| Component | Immutable revision | Declared license | Role |
| --- | --- | --- | --- |
| [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286) | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` | Apache-2.0 | 16 kHz mono audio to a 192-dimensional speaker embedding. |
| [`griko/gender_cls_svm_ecapa_voxceleb`](https://huggingface.co/griko/gender_cls_svm_ecapa_voxceleb/tree/25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64) | `25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64` | Apache-2.0 | Binary SVM probabilities for the upstream `female`/`male` labels. |
| [`griko/age_reg_svr_ecapa_voxceleb2`](https://huggingface.co/griko/age_reg_svr_ecapa_voxceleb2/tree/1d2356ac55f51fbd3f327f1b9260860decb21233) | `1d2356ac55f51fbd3f327f1b9260860decb21233` | Apache-2.0 | Continuous age estimate from an SVR. |

Every downloaded file also has an expected SHA-256 in `scripts/prepare_models.py`. The generated image records sources, revisions, hashes, and conversion facts in `/opt/models/model-metadata.json`. Runtime model loading is offline.

The same embedding feeds both heads, avoiding a second neural backbone. The upstream gender card reports 98.9% VoxCeleb2, 92.3% Common Voice v10 English, and 99.6% TIMIT accuracy. The age card reports 7.89-year VoxCeleb2 test MAE. Those are upstream results, not this service's logistics-domain performance, and they must not be presented as production validation.

## Runtime pipeline

1. Stream the bounded request into memory; never create an audio file.
2. Decode natively when possible or pass compressed input through bounded FFmpeg stdin/stdout pipes.
3. Mix to mono, resample to 16 kHz, sanitize non-finite values, and cap duration.
4. Compute signal-quality statistics. Return `unknown` before inference when speech is insufficient.
5. Select the best speech-evidence five-second window from longer audio, with an energy fallback.
6. Run ECAPA once, then the SVM and SVR heads.
7. Map continuous age to an adult bracket, discount degraded audio, apply confidence thresholds, and abstain when needed.
8. Overwrite mutable audio buffers and arrays in `finally` blocks.

## Pickle conversion and supply-chain decision

The griko repositories publish scikit-learn transformers and estimators as joblib/pickle. Unpickling is code execution, so these objects never enter the runtime image. A disposable Docker build stage:

- downloads only fixed revisions and verifies SHA-256 before loading;
- deserializes with pinned joblib, pandas, scikit-learn, and NumPy versions;
- proves each preprocessing pipeline is affine and exports its matrix and bias;
- exports support vectors, dual coefficients, intercepts, kernel parameters, labels, and probability parameters;
- checks manual NumPy kernel decisions against scikit-learn on probe vectors;
- fits and parity-checks the SVC probability mapping when `predict_proba` is available;
- writes compressed numeric `.npz`, which runtime loads with `allow_pickle=False`.

This eliminates joblib/scikit-learn and head pickle loading in production. It does not make untrusted builds safe: the isolated builder still executes the source pickle. The SpeechBrain `.ckpt` files are also serialized PyTorch artifacts; they are immutable and hash-verified but still require a trusted, isolated build pipeline. Future work should prefer upstream safetensors or convert and validate the encoder too.

## Alternatives considered

| Option | Advantages | Rejection or deferral reason | Commercial-use posture |
| --- | --- | --- | --- |
| Selected ECAPA + griko SVM/SVR | One approximately 89 MB encoder pass; small heads; continuous age; simple CPU runtime; public cross-corpus gender evidence; easy head replacement. | Age error is material, confidence needs domain calibration, binary labels are reductive, and celebrity/interview data differs from logistics calls. | All three model pages declare Apache-2.0, which generally permits commercial use with license/notice obligations. |
| [audEERING Wav2Vec2 6-layer](https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender) | Joint age and three-way child/female/male output; approximately 90.8M parameters; stronger learned representation than handcrafted features. | Hard license blocker for this commercial service. It is also a heavier transformer path than the selected encoder and still needs domain calibration. | CC-BY-NC-SA-4.0 is noncommercial. A separate commercial license would be required. |
| [audEERING Wav2Vec2 24-layer](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender) | Joint model, published paper, broad pretraining, and all 24 layers fine-tuned. | Same hard license blocker; roughly 0.3B parameters/1.27 GB safetensors increases CPU latency, image size, and memory. | CC-BY-NC-SA-4.0 is noncommercial. A separate commercial license would be required. |
| [ChunkFormer multi-attribute classifier](https://huggingface.co/khanhld/chunkformer-gender-emotion-dialect-age-classification) | Apache-2.0, chunk-aware architecture, one model for age and gender plus dialect/emotion, and a direct five-class age output. | The 315 MB checkpoint and sparse model card provide no task accuracy, calibration, subgroup, noise, telephony, or cross-corpus results. The published ChunkFormer paper primarily evaluates long-form ASR, not this attribute checkpoint. Reconsider after independent evaluation and an immutable release. | Model page declares Apache-2.0; commercial use is plausible subject to notice and provenance review. |
| openSMILE or handcrafted acoustic features + local classifier | Very low latency and memory; interpretable pitch/formant/spectral features; trainable for 8 kHz telephony; no neural encoder required. | No ready production head for this contract; feature engineering, consented labels, robustness training, subgroup testing, and calibration become our responsibility. openSMILE itself introduces a commercial-license dependency. NumPy/librosa equivalents avoid that dependency but not the data burden. | The open-source [openSMILE license](https://github.com/audeering/opensmile/blob/master/LICENSE) restricts product use; a commercial license or a differently licensed implementation is required. |
| Separate end-to-end age and gender models | Each model can use its best architecture, data, window, calibration, and release cadence. Failure isolation is clearer. | Two encoders approximately double model memory and compute, reduce batch efficiency, complicate deployment, and risk contradictory quality behavior. Keep as an option if domain evaluation shows a material accuracy gain that justifies cost. | Depends on both models; the stricter license controls the combined service. |

## Why this option wins now

The selected stack is the smallest credible commercially compatible baseline with one shared learned representation and direct support for both outputs. ECAPA is designed to retain speaker characteristics, the griko heads are exactly aligned to its 192 features, and a continuous age estimate lets the API own bracket boundaries. The architecture makes abstention and future head replacement straightforward. It also keeps inference self-contained and avoids the audEERING noncommercial restriction and openSMILE commercial license.

This is an engineering baseline, not an assertion that public benchmark performance transfers to truck cabs, warehouses, mobile networks, accents, languages, or demographic groups. The sub-500 ms goal remains an acceptance benchmark on target hardware, not a result inferred from model size.

## Confidence and quality consequences

Gender confidence is the converted SVC probability for the winning upstream class, not a probability of identity. Age confidence is the modeled normal probability mass inside the selected bracket, conditional on adult age, using `AGE_RESIDUAL_SIGMA_YEARS=10`; it is not emitted by the SVR. `degraded` audio multiplies both by `0.75`. Gender below `0.60` and age below `0.28` become `unknown` with `0.0`. `insufficient` audio always returns both unknown without running the model.

These heuristics are intentionally conservative but are not calibrated on logistics calls. Release approval requires held-out, consented domain evaluation with accuracy/MAE, confusion matrices, Brier score or ECE, reliability diagrams, coverage-versus-error curves, carrier/codec/noise slices, and subgroup slices where lawful and ethically appropriate.

## Commercial licensing notes

Apache-2.0 permits commercial use, modification, and distribution under its conditions, including preserving applicable copyright, license, and NOTICE material and marking modified files. CC-BY-NC-SA-4.0 does not permit this intended commercial use without another grant. The repository's MIT license covers service code only. A model-card license is not a warranty of dataset provenance and grants no personality, biometric, privacy, publicity, or voice rights. Procurement/legal should review model, code, and training-data terms and retain an artifact bill of materials. This section is not legal advice.

## Consequences and follow-up

Positive consequences are one encoder call, modest artifacts, offline operation, a safe numeric runtime for the heads, and replaceable postprocessing. Negative consequences are a serialized estimator lock per replica, cumulative WebSocket re-inference, no diarization, binary-only voice labels, approximate age, and unproven logistics-domain calibration.

Before production:

1. Benchmark warmed p50/p95/p99 decode, queue, inference, and total latency on target codecs and hardware.
2. Evaluate caller-only logistics audio and set thresholds from coverage/error and business-cost curves.
3. Test demographic, language, accent, device, noise, bandwidth, and voice-conversion slices.
4. Replace heuristic confidence with held-out temperature/isotonic calibration and a calibrated age-error distribution.
5. Explore ONNX, quantization, GPU micro-batching, caller VAD/diarization, and incremental embedding reuse.
6. Revisit ChunkFormer or separate heads only with an equivalent immutable, licensed, independently measured candidate.
