# Design write-up

**Approach.** FastAPI ingests bounded uploads and WebSocket PCM/G.711 chunks. Everything becomes mono 16 kHz float through a band-limited polyphase resampler. A quality gate measures usable speech, SNR, clipping, bandwidth and spectral shape, then abstains outright on unusable audio and requires a stricter confidence threshold on degraded audio rather than rescaling the number it reports. The selected window is encoded as overlapping sub-windows in one batched pass, which supplies per-sample age uncertainty and a single-speaker check.

**Model choice.** One pinned SpeechBrain ECAPA embedding feeds Apache-2.0 griko gender and age heads: the smallest commercially licensed stack that is reproducible, offline and CPU-friendly. Its 7.9-year age MAE against 13-year brackets is the real weakness, and the upstream gender head ships without calibration. Optional VoxLingua107 adds best-effort language.

**With more time.** Train WavLM Base+ on consented, noise- and codec-augmented logistics audio with a direct ordinal bracket head and held-out calibration; the ONNX adapter already fails closed until that artifact exists. Replacing heuristic confidence with measured calibration matters more than any accuracy gain.

**Scaling to 1,000 calls.** Measured: 93% of a request is one forward pass, roughly 4 analyses per second per container, and neither more threads nor more in-process replicas raise that. Scale horizontally to about 100 containers, autoscale on queue wait, reduce analyses per call, and adopt INT8 or GPU micro-batching to change the order of magnitude.
