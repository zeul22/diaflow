from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class RawAttributes:
    gender_probabilities: Mapping[str, float]
    age_years: float | None = None
    # A domain model can predict the API's ordinal brackets directly.  The
    # legacy ECAPA/SVR backend continues to populate ``age_years`` instead.
    age_bracket_probabilities: Mapping[str, float] | None = None

    # Per-sample evidence measured across an encoder's sub-window ensemble.
    # ``age_spread_years`` is the dispersion of the per-window age estimates and
    # is what makes age confidence carry sample-specific information instead of
    # being a fixed function of the distance to a bracket edge.
    # ``speaker_homogeneity`` is the lowest pairwise cosine similarity between
    # sub-window embeddings; a low value means the segment probably does not
    # contain a single speaker. A backend that produces one view leaves both
    # ``None`` and postprocessing falls back to the configured residual only.
    age_spread_years: float | None = None
    speaker_homogeneity: float | None = None
    ensemble_windows: int = 1


class AttributeEstimator(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, samples: npt.NDArray[np.float32]) -> RawAttributes: ...

    def warmup(self) -> None: ...
