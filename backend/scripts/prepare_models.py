#!/usr/bin/env python3
"""Download pinned model artifacts and convert sklearn heads to safe NumPy data.

The source head files are joblib/pickle objects. They are downloaded by immutable
revision and SHA-256, deserialized only in the disposable Docker build stage, and
converted into arrays consumed by the runtime without pickle or scikit-learn.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ECAPA_REPO = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
GENDER_REPO = "griko/gender_cls_svm_ecapa_voxceleb"
GENDER_REVISION = "25f3e5a3c1c172dceeb723d8061e3e80ba6c8d64"
AGE_REPO = "griko/age_reg_svr_ecapa_voxceleb2"
AGE_REVISION = "1d2356ac55f51fbd3f327f1b9260860decb21233"
LANGUAGE_REPO = "speechbrain/lang-id-voxlingua107-ecapa"
LANGUAGE_REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"

ECAPA_FILES = {
    "hyperparams.yaml": "6f78854fa04ba59e761437b76a2575d3aba5e5016de3e9b69f0c9a5077fb1a41",
    "embedding_model.ckpt": "0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2",
    "mean_var_norm_emb.ckpt": "cd70225b05b37be64fc5a95e24395d804231d43f74b2e1e5a513db7b69b34c33",
    "classifier.ckpt": "fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535",
    "label_encoder.txt": "e13c3a167bb4112685670ee896d20e2b565af16b3a4ceeaa8689fa4d22adb8b9",
}

# The separate 107-language ECAPA classifier. Its encoder is trained for
# language discrimination and cannot be replaced by the speaker embedding, so
# enabling language identification costs a second forward pass.
LANGUAGE_FILES = {
    "hyperparams.yaml": "88fec9791a8416a152fb10834327e18d38e5bf7a351e9b714e08cdc4af05de6f",
    "embedding_model.ckpt": "ab750d5c06d713477045fa798fab5d33e959dbc0dfe4de510a9a47844c79a19a",
    "classifier.ckpt": "a50d9024ff58d317031c9787d4c6c614d454a87a8ef32f9d36338cd3ff57adbc",
    "label_encoder.txt": "9f566d83c4f19168be4a0bf86c0c7dac7d3264a95105bcbf33a7c32b83ccc17f",
    "normalizer.ckpt": "c369e01dfa2e0d84c6b116f33c7b94f1fe28c061642086538e93cde3d97c26ef",
}

GENDER_FILES = {
    "scaler.joblib": "4e44e58d1e6602f61913b53f65bcc328e400a4c33a97410c543a5f1e2f357651",
    "svm_model.joblib": "74badd2f209f7bb09e511b85da387ca383e2734e3d092474edcff3b3549f2bfa",
}

AGE_FILES = {
    "scaler.joblib": "41515fd50d331ccbc06750cef97d38c63660d07c3406c7566092916534a54b19",
    "model.joblib": "91e2be1f70d14cbff46fc26b84e521b6887a101ace32e573af90a2b248d30369",
}

FEATURE_NAMES = [f"{index}_speechbrain_embedding" for index in range(192)]


def _download(
    repo: str, revision: str, filename: str, destination: Path, sha256: str
) -> None:
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "diaflow-model-builder/0.1"}
    )
    digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as output,
    ):
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            output.write(chunk)
    actual = digest.hexdigest()
    if actual != sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {repo}/{filename}: expected {sha256}, got {actual}"
        )


def _as_dense(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def _linearize_scaler(scaler: Any) -> tuple[np.ndarray, np.ndarray]:
    zero = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    basis = np.eye(len(FEATURE_NAMES), dtype=np.float64)
    frame = pd.DataFrame(np.concatenate([zero, basis]), columns=FEATURE_NAMES)
    transformed = _as_dense(scaler.transform(frame))
    bias = transformed[0]
    matrix = transformed[1:] - bias

    rng = np.random.default_rng(7)
    probe = rng.normal(size=(8, len(FEATURE_NAMES)))
    expected = _as_dense(scaler.transform(pd.DataFrame(probe, columns=FEATURE_NAMES)))
    actual = probe @ matrix + bias
    if not np.allclose(expected, actual, rtol=1e-9, atol=1e-9):
        raise RuntimeError("The fitted scaler is not an affine transform")
    return matrix, bias


def _kernel(model: Any, samples: np.ndarray) -> np.ndarray:
    support_vectors = np.asarray(model.support_vectors_, dtype=np.float64)
    kind = str(model.kernel)
    dot = samples @ support_vectors.T
    if kind == "linear":
        return dot
    if kind == "rbf":
        sample_norm = np.sum(samples * samples, axis=1, keepdims=True)
        support_norm = np.sum(support_vectors * support_vectors, axis=1)[None, :]
        squared = np.maximum(0.0, sample_norm + support_norm - 2.0 * dot)
        return np.exp(-float(model._gamma) * squared)
    if kind == "poly":
        return np.power(
            float(model._gamma) * dot + float(model.coef0), int(model.degree)
        )
    if kind == "sigmoid":
        return np.tanh(float(model._gamma) * dot + float(model.coef0))
    raise RuntimeError(f"Unsupported sklearn kernel: {kind}")


def _manual_decision(model: Any, samples: np.ndarray) -> np.ndarray:
    kernel_values = _kernel(model, samples)
    dual = np.asarray(model.dual_coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    return kernel_values @ dual + intercept


def _validate_kernel_model(model: Any, *, classifier: bool) -> None:
    support = np.asarray(model.support_vectors_, dtype=np.float64)
    step = max(1, support.shape[0] // 256)
    probe = support[::step][:256]
    expected = (
        np.asarray(model.decision_function(probe), dtype=np.float64).reshape(-1)
        if classifier
        else np.asarray(model.predict(probe), dtype=np.float64).reshape(-1)
    )
    actual = _manual_decision(model, probe)
    if not np.allclose(expected, actual, rtol=1e-7, atol=1e-7):
        maximum_error = float(np.max(np.abs(expected - actual)))
        raise RuntimeError(
            f"Kernel conversion parity failed (max error {maximum_error})"
        )


def _probability_calibration(
    model: Any, *, allow_uncalibrated: bool
) -> tuple[float, float, str, float | None, bool]:
    support = np.asarray(model.support_vectors_, dtype=np.float64)
    step = max(1, support.shape[0] // 2_000)
    probe = support[::step][:2_000]
    try:
        probabilities = np.asarray(model.predict_proba(probe), dtype=np.float64)[:, 1]
    except (AttributeError, RuntimeError) as exc:
        # A margin sigmoid is monotonic in the decision value and nothing more.
        # Serving it behind a probability threshold would make GENDER_CONFIDENCE_
        # THRESHOLD meaningless, so the build fails closed instead of shipping a
        # number that only looks like a probability.
        if not allow_uncalibrated:
            raise RuntimeError(
                "The source gender classifier exposes no predict_proba, so no "
                "probability calibration can be recovered. Rebuild from a "
                "calibrated classifier, or pass --allow-uncalibrated-gender to "
                "produce an evaluation-only artifact."
            ) from exc
        return math.log(3.0), 0.0, "margin_sigmoid", None, False

    decisions = np.asarray(model.decision_function(probe), dtype=np.float64).reshape(-1)
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logits = np.log(probabilities / (1.0 - probabilities))
    design = np.column_stack([decisions, np.ones_like(decisions)])
    slope, intercept = np.linalg.lstsq(design, logits, rcond=None)[0]
    predicted = 1.0 / (1.0 + np.exp(-(slope * decisions + intercept)))
    max_error = float(np.max(np.abs(predicted - probabilities)))
    if max_error > 0.01:
        raise RuntimeError(
            f"SVC probability conversion parity failed (max error {max_error})"
        )
    return float(slope), float(intercept), "libsvm_platt", max_error, True


def _label(value: Any, labels: list[str]) -> str:
    if isinstance(value, int | np.integer):
        return labels[int(value)]
    normalized = str(value).lower()
    if normalized not in labels:
        raise RuntimeError(f"Unexpected gender class label {value!r}")
    return normalized


def _export_heads(
    temporary: Path, output: Path, *, allow_uncalibrated: bool
) -> dict[str, Any]:
    gender_scaler = joblib.load(temporary / "gender" / "scaler.joblib")
    gender_model = joblib.load(temporary / "gender" / "svm_model.joblib")
    age_scaler = joblib.load(temporary / "age" / "scaler.joblib")
    age_model = joblib.load(temporary / "age" / "model.joblib")

    if len(gender_model.classes_) != 2:
        raise RuntimeError("Gender SVM must be binary")
    _validate_kernel_model(gender_model, classifier=True)
    _validate_kernel_model(age_model, classifier=False)
    gender_matrix, gender_bias = _linearize_scaler(gender_scaler)
    age_matrix, age_bias = _linearize_scaler(age_scaler)
    slope, probability_intercept, calibration, calibration_error, calibrated = (
        _probability_calibration(gender_model, allow_uncalibrated=allow_uncalibrated)
    )
    labels = ["female", "male"]
    negative_label = _label(gender_model.classes_[0], labels)
    positive_label = _label(gender_model.classes_[1], labels)

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "attribute_heads.npz",
        gender_transform_matrix=gender_matrix,
        gender_transform_bias=gender_bias,
        gender_support_vectors=np.asarray(
            gender_model.support_vectors_, dtype=np.float64
        ),
        gender_dual_coef=np.asarray(gender_model.dual_coef_, dtype=np.float64).reshape(
            -1
        ),
        gender_intercept=np.asarray(gender_model.intercept_, dtype=np.float64),
        gender_kernel=np.asarray([str(gender_model.kernel)]),
        gender_gamma=np.asarray([float(gender_model._gamma)]),
        gender_coef0=np.asarray([float(gender_model.coef0)]),
        gender_degree=np.asarray([int(gender_model.degree)]),
        gender_probability_slope=np.asarray([slope]),
        gender_probability_intercept=np.asarray([probability_intercept]),
        gender_probability_calibrated=np.asarray([int(calibrated)]),
        gender_positive_label=np.asarray([positive_label]),
        gender_negative_label=np.asarray([negative_label]),
        age_transform_matrix=age_matrix,
        age_transform_bias=age_bias,
        age_support_vectors=np.asarray(age_model.support_vectors_, dtype=np.float64),
        age_dual_coef=np.asarray(age_model.dual_coef_, dtype=np.float64).reshape(-1),
        age_intercept=np.asarray(age_model.intercept_, dtype=np.float64),
        age_kernel=np.asarray([str(age_model.kernel)]),
        age_gamma=np.asarray([float(age_model._gamma)]),
        age_coef0=np.asarray([float(age_model.coef0)]),
        age_degree=np.asarray([int(age_model.degree)]),
    )
    return {
        "gender_probability_calibration": calibration,
        "gender_probability_calibrated": calibrated,
        "gender_probability_conversion_max_error": calibration_error,
        "gender_support_vectors": int(gender_model.support_vectors_.shape[0]),
        "age_support_vectors": int(age_model.support_vectors_.shape[0]),
    }


def prepare(output: Path, *, allow_uncalibrated: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="voice-attribute-models-") as temp_name:
        temporary = Path(temp_name)
        for filename, digest in GENDER_FILES.items():
            _download(
                GENDER_REPO,
                GENDER_REVISION,
                filename,
                temporary / "gender" / filename,
                digest,
            )
        for filename, digest in AGE_FILES.items():
            _download(
                AGE_REPO,
                AGE_REVISION,
                filename,
                temporary / "age" / filename,
                digest,
            )
        conversion = _export_heads(
            temporary, output, allow_uncalibrated=allow_uncalibrated
        )

    ecapa_output = output / "ecapa"
    for filename, digest in ECAPA_FILES.items():
        _download(
            ECAPA_REPO,
            ECAPA_REVISION,
            filename,
            ecapa_output / filename,
            digest,
        )

    language_output = output / "language"
    for filename, digest in LANGUAGE_FILES.items():
        _download(
            LANGUAGE_REPO,
            LANGUAGE_REVISION,
            filename,
            language_output / filename,
            digest,
        )

    metadata = {
        "format_version": 1,
        "sources": {
            "ecapa": {
                "repo": ECAPA_REPO,
                "revision": ECAPA_REVISION,
                "files": ECAPA_FILES,
            },
            "gender": {
                "repo": GENDER_REPO,
                "revision": GENDER_REVISION,
                "files": GENDER_FILES,
            },
            "age": {
                "repo": AGE_REPO,
                "revision": AGE_REVISION,
                "files": AGE_FILES,
            },
            "language": {
                "repo": LANGUAGE_REPO,
                "revision": LANGUAGE_REVISION,
                "files": LANGUAGE_FILES,
            },
        },
        "conversion": conversion,
    }
    (output / "model-metadata.json").write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-uncalibrated-gender",
        action="store_true",
        help=(
            "Export a gender head whose probabilities are an uncalibrated margin "
            "sigmoid. The runtime then refuses to load it unless "
            "REQUIRE_CALIBRATED_GENDER=false. Evaluation only."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    prepare(args.output, allow_uncalibrated=args.allow_uncalibrated_gender)


if __name__ == "__main__":
    main()
