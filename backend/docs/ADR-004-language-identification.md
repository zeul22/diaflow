# ADR-004: Best-effort spoken-language identification

- Status: Accepted
- Date: 2026-08-24
- Owners: Voice platform
- Review trigger: a commercially usable model with published telephony/noise evaluation, a measured latency regression, or a product need for dialect or accent output

## Context

The service should offer a best-effort language or accent field so a voice agent can adapt. The existing pipeline already produces a 192-dimensional speaker embedding, so the cheapest imaginable option would be another shallow head on it.

That option does not exist. The ECAPA speaker encoder is trained to be *invariant* to what is being said: it discards the phonetic and prosodic content that identifies a language, because that content is nuisance variation for speaker verification. Language identification therefore needs its own encoder and its own forward pass.

## Decision

Add deployment-scoped language identification using [`speechbrain/lang-id-voxlingua107-ecapa`](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa) at revision `0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9`, declared Apache-2.0, pinned and SHA-256 verified in `backend/scripts/prepare_models.py` exactly like the attribute models. `LANGUAGE_BACKEND` accepts `none` (default) or `voxlingua_ecapa`. When disabled, the `language` response field is absent rather than null, so no caller has to distinguish "not configured" from "not determined".

**Language only, not accent.** The API exposes a language tag. It does not expose accent, dialect, region, or nationality, and the model cannot support those: VoxLingua107 has one class per language. A field that appeared to name a caller's accent would invite exactly the inferences the model card prohibits, and no commercially licensed accent classifier with published evaluation was found. The requirement asked for language *or* accent; this delivers language.

## Acceptance rule

Top-1 posterior alone is the wrong instrument across 107 classes. Measured on a 5-second English window from a real recording:

| Class | Posterior |
| --- | ---: |
| `en: English` | 0.4679 |
| `nl: Dutch` | 0.1269 |
| `la: Latin` | 0.1155 |
| `ca: Catalan` | 0.0580 |

The answer is correct and unambiguous — it leads the runner-up by 3.7x — yet a naive 0.5 threshold rejects it, because probability mass spreads across related languages. Two conditions are therefore required, and **what the floor is applied to depends on whether a label set is declared**:

| | Floor (`LANGUAGE_CONFIDENCE_THRESHOLD`, `0.50`) | Margin (`LANGUAGE_MARGIN_RATIO`, `2.0`) |
| --- | --- | --- |
| No allowlist | the top class's raw posterior | over the runner-up across all 107 |
| With allowlist | the **summed mass of the permitted languages** | over the runner-up among permitted |

Testing one class against the full 107-way distribution is too strict once 96 of those classes are impossible for the deployment: English at 0.45 is 48x chance level and was being discarded. Summing the permitted mass asks the question that actually matters — *does the model believe this is a language you serve?* — without renormalizing, because **the reported confidence remains the raw posterior** either way.

The floor was briefly lowered to `0.35` to admit one observed clip. That was a methodological error: a single sample cannot set an operating point. It is back at `0.50`, and the longer window plus the allowlist are what make that affordable. Both values still need re-deriving on domain audio.

The same clip scores 0.6411 over its full 6.8 seconds versus 0.4679 on the selected 5-second window, so the attribute pipeline's deliberately narrow window was costing language accuracy. Language identification is therefore given its own window; see the section above.

## Cost

Measured in the CPU container on 5 seconds of audio:

| Path | Attribute inference | Language | End-to-end `processing_ms` |
| --- | ---: | ---: | ---: |
| `LANGUAGE_BACKEND=none` | ~200 ms | — | 150–200 ms |
| `LANGUAGE_BACKEND=voxlingua_ecapa` | ~200 ms | ~165–205 ms | 425–465 ms |

Enabling it roughly doubles inference and consumes most of the sub-500 ms objective, which is why it is off by default and why `INFERENCE_CONCURRENCY` replicas now cost a second ~85 MB model each.

## Streaming: rechecked, not settled

An earlier revision identified the language **once** per session and froze it, on the assumption that a caller's language does not change mid-call. That assumption is wrong — code-switching is routine, and freezing left a session permanently reporting a language nobody was speaking any more.

Streaming sessions now recheck after `LANGUAGE_REFRESH_SECONDS` (default `3.0`) of new audio, and the `end` result always rechecks. A new confident answer replaces the previous one; an `unknown` window does **not** blank the field, because one bad window is far more often a bad window than a genuine loss of language. The reported value is therefore the most recent confident detection — "what is the caller speaking now", which is what a voice agent adapting its style needs.

Switches are detected within a few seconds, not instantly. Streaming a real English clip followed by a Thai one, with the switch at 6.0 s of audio:

| Configuration | Switch reported | Lag |
| --- | ---: | ---: |
| Defaults (`refresh 3.0`, `ws window 10.0`) | 10.6 s | ~4.6 s |
| `LANGUAGE_REFRESH_SECONDS=1.0`, `WS_ANALYSIS_WINDOW_SECONDS=5.0` | 8.9 s | ~2.9 s |

Three things add to that lag: the refresh interval, the trailing analysis window that still contains the old language, and the best-speech window selection inside it, which is chosen for speech evidence rather than recency. While the window straddles both languages the answer can be transiently wrong — the tighter run reported Welsh for one update between English and Thai. That is inherent to naming one language for a window containing two, and is a reason to treat a single update as provisional.

## Limitations

- **No non-speech reject class.** The model always names a language. A synthetic test tone was confidently classified as Latin. Hold music, engine noise, and silence will all produce a language, and only the existing quality gate stands between that and the caller.
- **Untested on telephony.** VoxLingua107 is automatically collected YouTube speech. No narrowband, codec, packet-loss, or logistics-noise evaluation is bundled or claimed.
- **Code-switching is detected, not modelled.** A switch is picked up after a few seconds, but the field holds one label at a time and can be transiently wrong while a window contains both languages. Simultaneous or sentence-level mixing has no representation in the contract.
- **107 classes include closely related pairs.** Confusions inside a language family are expected and are the reason for the margin rule.
- **Thresholds are uncalibrated**, exactly as with the attribute heads. The floor was initially tuned to admit a single observed clip at 0.468 -- a methodological error, since one sample cannot set an operating point. It is back at 0.50 now that longer windows make that affordable.
- **Accent shift is a documented weakness of language identification generally**, since the model keys on phonetics and prosody, which are exactly what an accent changes. An earlier draft of this ADR claimed Indian-accented English is frequently scored as Hindi; the one accented sample available contradicts that outright -- English scored 0.6411 against Hindi at 0.0010, a 640x margin. The observed failures were window length and 107-class spread, not accent. The general weakness stands as a caveat; the specific claim was unsupported and has been withdrawn.
- **VoxLingua107 ships superseded ISO tags** (`iw`, `jw`, `in`), which surfaced in the UI as "IW". They are normalised to `he`, `jv`, `id` at the boundary.

## Consequences

A language field is available without touching the attribute contract, and it is absent unless a deployment opts in. The image grows by roughly 85 MB whether or not the feature is enabled, because artifacts are baked in at build time. Language identification is not diarization: on a mixed segment it describes whichever speaker dominates, subject to the same homogeneity downgrade as the other attributes.

The output names a language spoken in a recording. It must not be used to infer nationality, ethnicity, immigration status, or location, and it must not drive consequential decisions.
