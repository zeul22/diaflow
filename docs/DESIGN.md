# Design write-up

The service receives request-scoped audio through FastAPI REST or WebSocket endpoints. Native decoding handles WAV, PCM, mu-law, and A-law; FFmpeg handles compressed containers. Every signal becomes mono 16 kHz float PCM. A quality gate measures duration, voiced speech, loudness, clipping, spectral flatness, low-frequency energy, narrowband input, and estimated SNR. Insufficient speech returns unknown without inference; degraded speech lowers confidence.

One pinned SpeechBrain ECAPA-TDNN pass produces a 192-dimensional speaker embedding. Apache-2.0 griko SVM/SVR heads estimate binary voice presentation and age. Postprocessing maps age to adult brackets, derives bounded confidence, and applies abstention thresholds. We chose this shared encoder because it is smaller and cheaper than two neural backbones, commercially usable under declared licenses, and practical on CPU.

Next, we would evaluate consented logistics calls, add diarization, train calibrated noise-augmented heads, measure subgroup error, tune cost-based thresholds, export ONNX, quantize, and monitor drift. Language detection would remain optional.

For 1,000 concurrent calls, stateless API pods would buffer audio while a GPU tier batches embeddings. Deadline-aware queues, backpressure, and admission control would protect latency; autoscaling would follow queue depth, GPU utilization, and p95 duration. Telephony routing must supply caller-only channels. Regional deployment, encryption, memory limits, no body logging, and short-lived buffers preserve privacy.
