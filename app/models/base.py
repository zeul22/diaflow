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


class AttributeEstimator(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, samples: npt.NDArray[np.float32]) -> RawAttributes: ...

    def warmup(self) -> None: ...
