from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.models.base import RawAttributes


def _scalar(array: npt.NDArray[np.generic]) -> object:
    return array.reshape(-1)[0].item()


def _kernel(
    sample: npt.NDArray[np.float64],
    support_vectors: npt.NDArray[np.float64],
    kind: str,
    gamma: float,
    coef0: float,
    degree: int,
) -> npt.NDArray[np.float64]:
    dot = support_vectors @ sample
    if kind == "linear":
        return dot
    if kind == "rbf":
        squared = np.sum((support_vectors - sample) ** 2, axis=1)
        return np.exp(-gamma * squared)
    if kind == "poly":
        return np.power(gamma * dot + coef0, degree)
    if kind == "sigmoid":
        return np.tanh(gamma * dot + coef0)
    raise ValueError(f"Unsupported exported kernel '{kind}'")


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exp_value = math.exp(max(value, -60.0))
    return exp_value / (1.0 + exp_value)


class KernelHeadBundle:
    """Safe NumPy runtime for build-time converted scikit-learn heads."""

    def __init__(self, artifact_path: Path) -> None:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing converted head artifact: {artifact_path}")
        with np.load(artifact_path, allow_pickle=False) as data:
            self.gender_transform_matrix = data["gender_transform_matrix"].astype(
                np.float64
            )
            self.gender_transform_bias = data["gender_transform_bias"].astype(
                np.float64
            )
            self.gender_support_vectors = data["gender_support_vectors"].astype(
                np.float64
            )
            self.gender_dual_coef = data["gender_dual_coef"].astype(np.float64)
            self.gender_intercept = float(_scalar(data["gender_intercept"]))
            self.gender_kernel = str(_scalar(data["gender_kernel"]))
            self.gender_gamma = float(_scalar(data["gender_gamma"]))
            self.gender_coef0 = float(_scalar(data["gender_coef0"]))
            self.gender_degree = int(_scalar(data["gender_degree"]))
            self.gender_probability_slope = float(
                _scalar(data["gender_probability_slope"])
            )
            self.gender_probability_intercept = float(
                _scalar(data["gender_probability_intercept"])
            )
            self.gender_positive_label = str(_scalar(data["gender_positive_label"]))
            self.gender_negative_label = str(_scalar(data["gender_negative_label"]))

            self.age_transform_matrix = data["age_transform_matrix"].astype(np.float64)
            self.age_transform_bias = data["age_transform_bias"].astype(np.float64)
            self.age_support_vectors = data["age_support_vectors"].astype(np.float64)
            self.age_dual_coef = data["age_dual_coef"].astype(np.float64)
            self.age_intercept = float(_scalar(data["age_intercept"]))
            self.age_kernel = str(_scalar(data["age_kernel"]))
            self.age_gamma = float(_scalar(data["age_gamma"]))
            self.age_coef0 = float(_scalar(data["age_coef0"]))
            self.age_degree = int(_scalar(data["age_degree"]))
        self._validate()

    def _validate(self) -> None:
        if self.gender_transform_matrix.shape[0] != 192:
            raise ValueError("Gender head expects a 192-dimensional ECAPA embedding")
        if self.age_transform_matrix.shape[0] != 192:
            raise ValueError("Age head expects a 192-dimensional ECAPA embedding")
        if self.gender_support_vectors.shape[0] != self.gender_dual_coef.size:
            raise ValueError("Gender support-vector artifact is inconsistent")
        if self.age_support_vectors.shape[0] != self.age_dual_coef.size:
            raise ValueError("Age support-vector artifact is inconsistent")
        if {self.gender_positive_label, self.gender_negative_label} != {
            "female",
            "male",
        }:
            raise ValueError("Gender artifact labels must be female and male")

    @staticmethod
    def _transform(
        embedding: npt.NDArray[np.float64],
        matrix: npt.NDArray[np.float64],
        bias: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        return embedding @ matrix + bias

    def predict(self, embedding: npt.NDArray[np.float32]) -> RawAttributes:
        vector = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if vector.size != 192 or not np.all(np.isfinite(vector)):
            raise ValueError("ECAPA embedding is invalid")

        gender_features = self._transform(
            vector, self.gender_transform_matrix, self.gender_transform_bias
        )
        gender_values = _kernel(
            gender_features,
            self.gender_support_vectors,
            self.gender_kernel,
            self.gender_gamma,
            self.gender_coef0,
            self.gender_degree,
        )
        decision = float(gender_values @ self.gender_dual_coef + self.gender_intercept)
        positive_probability = _sigmoid(
            self.gender_probability_slope * decision + self.gender_probability_intercept
        )
        positive_probability = float(np.clip(positive_probability, 0.001, 0.999))
        gender_probabilities = {
            self.gender_positive_label: positive_probability,
            self.gender_negative_label: 1.0 - positive_probability,
        }

        age_features = self._transform(
            vector, self.age_transform_matrix, self.age_transform_bias
        )
        age_values = _kernel(
            age_features,
            self.age_support_vectors,
            self.age_kernel,
            self.age_gamma,
            self.age_coef0,
            self.age_degree,
        )
        age_years = float(age_values @ self.age_dual_coef + self.age_intercept)
        if not math.isfinite(age_years):
            age_years = None
        return RawAttributes(
            gender_probabilities=gender_probabilities,
            age_years=age_years,
        )
