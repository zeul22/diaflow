from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.config import Settings

logger = logging.getLogger(__name__)

# The upstream label encoder stores entries as "en: English".
_LABEL_PATTERN = re.compile(r"^([A-Za-z-]{2,8})\s*:")

# VoxLingua107 uses superseded ISO 639-1 codes for three languages. Emitting
# them verbatim shows callers "IW" instead of Hebrew, so they are normalised to
# the current tags at the boundary.
_LEGACY_CODES = {"iw": "he", "jw": "jv", "in": "id", "ji": "yi"}


@dataclass(frozen=True, slots=True)
class LanguageEstimate:
    """Top-1 language identity for one window.

    ``code`` is the upstream ISO-639 style tag, not a locale: VoxLingua107
    labels a language, never a region, dialect, or accent.
    """

    code: str
    confidence: float


def parse_language_code(label: str) -> str:
    """Reduce an upstream label to its language tag.

    Returns ``"unknown"`` for anything unparseable so a changed upstream label
    format degrades into abstention rather than leaking a raw label into the API.
    """

    match = _LABEL_PATTERN.match(label.strip())
    if match is None:
        return "unknown"
    code = match.group(1).lower()
    return _LEGACY_CODES.get(code, code)


def decide_language(
    probabilities: npt.NDArray[np.float64],
    labels: Sequence[str],
    *,
    threshold: float,
    margin_ratio: float,
    allowed: Sequence[str] = (),
) -> LanguageEstimate:
    """Accept a language only when the top class also beats the runner-up.

    A bare posterior threshold is the wrong instrument across 107 classes: mass
    spreads over related languages, so a correct answer routinely scores below
    0.5 while still leading the runner-up several times over. Requiring both a
    floor and a margin keeps genuinely ambiguous audio abstaining without
    discarding a clear winner just because the absolute posterior is diffuse.

    ``allowed`` restricts the answer to the languages a deployment actually
    serves. Most of the model's absurd outputs come from classes no logistics
    caller will ever speak: measured on real English, a 3-second window scored
    Latin at 0.929 while English sat at 0.033. Removing implausible classes from
    contention fixes that, and the winner is then chosen among the rest.

    **The floor is still applied to the raw posterior, never to a renormalized
    one.** Renormalizing over a subset inflates confidence without adding any
    evidence -- that same 3-second window renormalizes to English at 0.945 on a
    raw score of 0.033. Restricting the candidates decides *which* language;
    the raw posterior decides whether there is enough evidence to say anything.

    The returned confidence is the raw posterior, not the margin.
    """

    if probabilities.ndim != 1 or probabilities.size < 2:
        return LanguageEstimate(code="unknown", confidence=0.0)
    if probabilities.size != len(labels):
        return LanguageEstimate(code="unknown", confidence=0.0)
    if not np.all(np.isfinite(probabilities)):
        return LanguageEstimate(code="unknown", confidence=0.0)

    codes = [parse_language_code(label) for label in labels]
    candidates = list(range(len(codes)))
    if allowed:
        permitted = {code.lower() for code in allowed}
        candidates = [index for index in candidates if codes[index] in permitted]
        if len(candidates) < 2:
            return LanguageEstimate(code="unknown", confidence=0.0)

    ranked = sorted(candidates, key=lambda index: probabilities[index], reverse=True)
    best = float(probabilities[ranked[0]])
    runner_up = float(probabilities[ranked[1]])
    code = codes[ranked[0]]
    if code == "unknown":
        return LanguageEstimate(code="unknown", confidence=0.0)

    if allowed:
        # With a declared label set the floor asks a better question: does the
        # model believe this is one of the languages this deployment serves at
        # all? Judging a single class against the full 107-way distribution is
        # too strict once 96 of those classes are known to be impossible --
        # English at 0.45 is 48x chance level, yet a flat 0.50 floor rejects it.
        # Summing the permitted mass keeps the test honest without renormalizing,
        # because the reported confidence is still the raw posterior.
        evidence = float(sum(probabilities[index] for index in candidates))
    else:
        evidence = best
    if evidence < threshold:
        return LanguageEstimate(code="unknown", confidence=0.0)
    if best < margin_ratio * runner_up:
        return LanguageEstimate(code="unknown", confidence=0.0)
    return LanguageEstimate(code=code, confidence=min(1.0, max(0.0, best)))


class VoxLinguaLanguageIdentifier:
    """SpeechBrain VoxLingua107 ECAPA language identification.

    This is a second encoder, not a second head: the speaker embedding used for
    age and gender is trained to discard exactly the phonetic content language
    identification depends on. Enabling it therefore costs another forward pass,
    which is why it is deployment-scoped rather than always on.

    The model identifies a language. It does not identify an accent, a dialect,
    a region, or a nationality, and its output must not be used to infer any of
    those about a caller.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            import torch
            from speechbrain.inference.classifiers import EncoderClassifier
        except ImportError as exc:  # pragma: no cover - exercised by container startup
            raise RuntimeError(
                "Language identification requires the project's 'model' dependencies"
            ) from exc

        self._torch = torch
        self._lock = threading.Lock()
        self._device = settings.model_device
        self._sample_rate = settings.target_sample_rate
        self._threshold = settings.language_confidence_threshold
        self._margin_ratio = settings.language_margin_ratio
        self._allowed = settings.language_allowlist
        model_root = Path(settings.model_root) / "language"
        if not (model_root / "hyperparams.yaml").is_file():
            raise FileNotFoundError(f"Missing language model under {model_root}")

        cache_dir = Path("/tmp/speechbrain-language")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._classifier = EncoderClassifier.from_hparams(
            source=str(model_root),
            savedir=str(cache_dir),
            run_opts={"device": self._device},
            # Keep startup strictly offline against the pinned, hash-verified
            # files instead of the Hugging Face path in the upstream YAML.
            overrides={"pretrained_path": str(model_root)},
        )
        self._labels = [
            str(self._classifier.hparams.label_encoder.ind2lab[index])
            for index in range(len(self._classifier.hparams.label_encoder.ind2lab))
        ]

    @property
    def name(self) -> str:
        return "speechbrain-voxlingua107-ecapa"

    def identify(self, samples: npt.NDArray[np.float32]) -> LanguageEstimate:
        waveform = np.ascontiguousarray(samples, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Expected a non-empty mono waveform")
        with self._lock, self._torch.inference_mode():
            tensor = self._torch.from_numpy(waveform).unsqueeze(0).to(self._device)
            lengths = self._torch.ones(1, device=self._device)
            log_probabilities, _, _, _ = self._classifier.classify_batch(
                tensor, lengths
            )
            # ``classify_batch`` returns log posteriors over all 107 classes; the
            # runner-up is needed for the margin rule, not just the winner.
            posteriors = (
                self._torch.exp(log_probabilities.reshape(-1))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            del tensor, lengths, log_probabilities
        return decide_language(
            posteriors,
            self._labels,
            threshold=self._threshold,
            margin_ratio=self._margin_ratio,
            allowed=self._allowed,
        )

    def warmup(self) -> None:
        logger.info("language_model_warmup_started")
        signal = np.zeros(self._sample_rate, dtype=np.float32)
        with suppress(ValueError):
            self.identify(signal)
        signal.fill(0.0)
        logger.info("language_model_warmup_completed")
