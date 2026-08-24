from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from app.config import Settings
from app.models.factory import create_estimator
from app.models.wavlm import WavlmAttributeEstimator


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeOnnxSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self):
        return [_Node("audio")]

    def get_outputs(self):
        return [_Node("gender_logits"), _Node("age_logits")]

    def run(self, outputs, feeds):
        assert outputs == ["gender_logits", "age_logits"]
        assert feeds["audio"].dtype == np.float32
        assert feeds["audio"].shape == (1, 16_000)
        self.calls += 1
        return np.array([[0.0, 2.0]]), np.array([[0.0, 0.5, 3.0, -1.0]])


def test_wavlm_adapter_maps_logits_to_exact_api_outputs() -> None:
    session = FakeOnnxSession()
    estimator = WavlmAttributeEstimator(
        replace(Settings(), wavlm_model_revision="logistics-v3"), session=session
    )

    result = estimator.predict(np.zeros(16_000, dtype=np.float32))

    assert estimator.name == "wavlm-base-plus-domain-logistics-v3"
    assert result.gender_probabilities["male"] > 0.85
    assert result.age_bracket_probabilities is not None
    assert (
        max(
            result.age_bracket_probabilities,
            key=result.age_bracket_probabilities.get,
        )
        == "46-60"
    )
    assert sum(result.age_bracket_probabilities.values()) == pytest.approx(1.0)
    assert session.calls == 1


def test_wavlm_adapter_rejects_wrong_artifact_contract() -> None:
    session = FakeOnnxSession()
    session.get_inputs = lambda: [_Node("input_values")]

    with pytest.raises(RuntimeError, match="named 'audio'"):
        WavlmAttributeEstimator(Settings(), session=session)


def test_factory_fails_closed_when_owned_wavlm_artifact_is_missing(tmp_path) -> None:
    settings = replace(
        Settings(),
        model_backend="wavlm_onnx",
        wavlm_model_path=tmp_path / "missing.onnx",
        warmup_model=False,
    )

    with pytest.raises(FileNotFoundError, match="owned WavLM"):
        create_estimator(settings)
