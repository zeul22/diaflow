from __future__ import annotations

import numpy as np

from app.models.kernel_heads import KernelHeadBundle


def test_converted_kernel_heads_load_without_pickle(tmp_path) -> None:
    identity = np.eye(192, dtype=np.float64)
    zero = np.zeros(192, dtype=np.float64)
    artifact = tmp_path / "heads.npz"
    np.savez_compressed(
        artifact,
        gender_transform_matrix=identity,
        gender_transform_bias=zero,
        gender_support_vectors=np.zeros((1, 192), dtype=np.float64),
        gender_dual_coef=np.asarray([0.0]),
        gender_intercept=np.asarray([2.0]),
        gender_kernel=np.asarray(["linear"]),
        gender_gamma=np.asarray([1.0]),
        gender_coef0=np.asarray([0.0]),
        gender_degree=np.asarray([3]),
        gender_probability_slope=np.asarray([1.0]),
        gender_probability_intercept=np.asarray([0.0]),
        gender_positive_label=np.asarray(["male"]),
        gender_negative_label=np.asarray(["female"]),
        age_transform_matrix=identity,
        age_transform_bias=zero,
        age_support_vectors=np.zeros((1, 192), dtype=np.float64),
        age_dual_coef=np.asarray([0.0]),
        age_intercept=np.asarray([40.0]),
        age_kernel=np.asarray(["linear"]),
        age_gamma=np.asarray([1.0]),
        age_coef0=np.asarray([0.0]),
        age_degree=np.asarray([3]),
    )

    prediction = KernelHeadBundle(artifact).predict(np.zeros(192, dtype=np.float32))
    assert prediction.age_years == 40.0
    assert prediction.gender_probabilities["male"] > 0.8
