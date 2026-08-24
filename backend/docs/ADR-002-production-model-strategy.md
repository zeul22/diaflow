# ADR-002: Production model strategy

- Status: Accepted
- Date: 2026-08-24
- Owners: Voice platform
- Supersedes: the claim in ADR-001 that ECAPA is the final production choice; ADR-001 remains the runnable-baseline record

## Decision in one sentence

Use an **owned, logistics-trained WavLM Base+ model with calibrated gender-presentation and ordinal age-bracket heads** as the production target; retain the pinned Apache-2.0 ECAPA stack as the reproducible baseline until that challenger passes the same-domain gates, and evaluate audEERING devAIce when a paid commercial license is acceptable.

There is no public checkpoint that is simultaneously the most accurate, commercially permissive, validated on noisy logistics calls, calibrated, and proven below 500 ms on our deployment hardware. Calling any public checkpoint “best in production” without those qualifiers would be misleading.

## Why the submitted M4A result was `unknown`

The result was not clean evidence that the gender head predicted incorrectly. The old pipeline marked *every* FFmpeg-decoded input as degraded merely because FFmpeg was used. That multiplied confidence by `0.75`; a configured `0.60` output threshold therefore required an upstream probability of at least `0.80`. M4A, MP3, and Opus were penalized even when the decoded signal was good. The age confidence was also a fixed Gaussian heuristic around an SVR estimate, not calibrated model confidence. In addition, the loudest five-second window could select truck or warehouse noise instead of speech.

Codec choice is now an observability fact, not an automatic quality failure. Narrow bandwidth, insufficient voiced speech, poor SNR, clipping, and other measured signal problems can still produce `degraded` or `insufficient`.

## Ranked options

| Option | Evidence and fit | Decision |
| --- | --- | --- |
| WavLM Base+ + our heads | A [2025 cross-dataset study](https://arxiv.org/html/2502.12007) reports 5.45-year age MAE and 99.81% gender accuracy on VoxCeleb2 for its multi-dataset WavLM Base+ system, while Common Voice gender is about 90.7%, demonstrating domain sensitivity. The [backbone](https://huggingface.co/microsoft/wavlm-base-plus) and [Microsoft implementation](https://github.com/microsoft/unilm/tree/master/wavlm) are available, but the paper's task heads are not. | **Production target.** Train direct ordinal brackets and gender-presentation jointly on consented logistics speech, calibrate on held-out speakers, export one pinned ONNX graph, and deploy through `MODEL_BACKEND=wavlm_onnx`. |
| audEERING devAIce | The [audEERING study](https://arxiv.org/abs/2306.16962) reports 7.1–10.8-year MAE and at least 91.1% three-class gender accuracy across evaluated datasets. The six-layer variant is the sensible latency/accuracy point. | **Fastest buy-now trial.** Use only under a [devAIce commercial agreement](https://www.audeering.com/products/devaice/). The [public weights](https://github.com/audeering/w2v2-age-gender-how-to) are CC-BY-NC-SA and cannot be shipped for this commercial use. |
| Current ECAPA + griko SVM/SVR | Small shared encoder, Apache-2.0 model cards, offline build, and already operational. Upstream reports 7.89-year age MAE; this is large relative to the API's bracket widths. | **Runnable baseline, not the final accuracy claim.** Keep for paired evaluation and rollback. |
| ECAPA + combined ANN age | The [model card](https://huggingface.co/griko/age_reg_ann_ecapa_librosa_combined) reports 6.93-year MAE using ECAPA plus 31 librosa features. | Challenger after exact feature/parity tests. It is not silently substituted because it changes preprocessing and has no logistics calibration. |
| ChunkFormer multi-attribute checkpoint | Apache-2.0 and direct age classes. | Do not promote: its card publishes no accuracy, calibration, codec, telephony, noise, or cross-corpus results for this checkpoint. |
| Vox-Profile WavLM Large and public audEERING | Strong research candidates. | Excluded from commercial deployment: their released model terms are noncommercial. |

Reported numbers use different datasets and splits and are not directly comparable leaderboards.

## Implemented migration seam

`MODEL_BACKEND` is deployment-scoped and accepts `ecapa` or `wavlm_onnx`; callers cannot choose a model per request. The WavLM adapter fails closed if its owned artifact is absent. Its ONNX contract is:

- input `audio`: one batch of 16 kHz mono float32 samples;
- output `gender_logits`: calibrated logits ordered `female`, `male`;
- output `age_logits`: calibrated ordinal logits ordered `18-30`, `31-45`, `46-60`, `60+`.

The graph must contain the pinned WavLM encoder, our two heads, and the approved calibration. Direct age-bracket probabilities bypass the baseline's made-up residual confidence. The artifact revision is included in the model name used for metrics and stored analysis metadata.

The shipped runtime includes ONNX Runtime. Put the approved artifact at `backend/models/wavlm/model.onnx`, set a traceable `WAVLM_MODEL_REVISION`, and start `docker-compose.yml` with the supplied `docker-compose.wavlm.yml` override. The graph is mounted read-only; missing or incompatible artifacts fail startup instead of falling back to the baseline.

## Promotion gates

Promote only after a speaker-disjoint, consented logistics evaluation beats the baseline with confidence intervals on age-bracket macro F1, severe/adjacent error rates, gender balanced accuracy, coverage-versus-error, classwise ECE/Brier/NLL, and lawful subgroup slices. Test clean and augmented AAC/M4A, Opus, PCMU/PCMA, packet loss, truck/warehouse noise, reverberation, device, carrier, duration, language, and accent slices. Also require warmed p95 end-to-end latency below 500 ms on five seconds of audio and stable memory under target concurrency.

At 1,000 concurrent calls, route sessions to GPU inference workers, gather 2–3 seconds of VAD-selected caller speech, micro-batch WavLM requests, cap queues, and emit a later 4–5 second stability update instead of rerunning the full transformer every second forever. Autoscale on queue delay and GPU utilization. Keep decode, storage, and API orchestration separate from inference workers.

These outputs estimate acoustic voice presentation and an approximate age range. They do not establish identity, pronouns, biological sex, or legal gender and must not drive consequential eligibility, pricing, employment, or service decisions.
