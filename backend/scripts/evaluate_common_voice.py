#!/usr/bin/env python3
"""Evaluate a running service against an extracted Common Voice release.

Example:
  python3 backend/scripts/evaluate_common_voice.py \
    --tsv /data/cv-corpus/en/test.tsv --clips /data/cv-corpus/en/clips --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import statistics
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

AGE_MIDPOINTS = {
    "twenties": 25,
    "thirties": 35,
    "fourties": 45,
    "forties": 45,
    "fifties": 55,
    "sixties": 65,
    "seventies": 75,
    "eighties": 85,
    "nineties": 95,
}


# Common Voice publishes a decade, not an age. Treating the midpoint as truth
# adds label noise with a standard deviation of 10/sqrt(12) for a uniform decade,
# which is large enough to matter when the whole point is estimating the model's
# own residual spread -- so it is subtracted back out below.
DECADE_LABEL_NOISE_STD = 10.0 / math.sqrt(12.0)


def age_midpoint(label: str) -> int | None:
    return AGE_MIDPOINTS.get(label.strip().lower())


def age_bracket(label: str) -> str | None:
    midpoint = AGE_MIDPOINTS.get(label.strip().lower())
    if midpoint is None:
        return None
    if midpoint < 31:
        return "18-30"
    if midpoint < 46:
        return "31-45"
    if midpoint < 60:
        return "46-60"
    return "60+"


def gender_label(value: str) -> str | None:
    normalized = value.strip().lower()
    mapping = {
        "male": "male",
        "male_masculine": "male",
        "female": "female",
        "female_feminine": "female",
    }
    return mapping.get(normalized)


@dataclass
class TaskMetrics:
    total: int = 0
    covered: int = 0
    correct: int = 0
    covered_correct: int = 0
    calibration: list[tuple[float, int]] = field(default_factory=list)
    confusion: Counter[tuple[str, str]] = field(default_factory=Counter)

    def add(self, expected: str | None, prediction: dict[str, object]) -> None:
        if expected is None:
            return
        predicted = str(prediction["prediction"])
        confidence = float(prediction["confidence"])
        self.total += 1
        is_correct = int(predicted == expected)
        self.correct += is_correct
        self.confusion[(expected, predicted)] += 1
        if predicted != "unknown":
            self.covered += 1
            self.covered_correct += is_correct
            self.calibration.append((confidence, is_correct))

    def summary(self) -> dict[str, object]:
        bins = 10
        ece = 0.0
        brier = 0.0
        for confidence, correct in self.calibration:
            brier += (confidence - correct) ** 2
        for bin_index in range(bins):
            low, high = bin_index / bins, (bin_index + 1) / bins
            members = [
                item
                for item in self.calibration
                if low <= item[0] < high or (bin_index == bins - 1 and item[0] == 1.0)
            ]
            if not members:
                continue
            mean_confidence = sum(item[0] for item in members) / len(members)
            accuracy = sum(item[1] for item in members) / len(members)
            ece += (
                len(members)
                / max(1, len(self.calibration))
                * abs(mean_confidence - accuracy)
            )
        return {
            "samples": self.total,
            "coverage": self.covered / self.total if self.total else 0.0,
            "accuracy_with_unknown_as_error": self.correct / self.total
            if self.total
            else 0.0,
            "selective_accuracy": self.covered_correct / self.covered
            if self.covered
            else 0.0,
            "ece_10_bin": ece,
            "brier_correctness": brier / len(self.calibration)
            if self.calibration
            else 0.0,
            "confusion": {
                f"{expected}->{predicted}": count
                for (expected, predicted), count in sorted(self.confusion.items())
            },
        }


@dataclass
class AgeRegressionMetrics:
    """Statistics of the raw age estimate, not just the bracket decision.

    Requires ``EXPOSE_DEBUG_AGE_YEARS=true`` on the service. Bracket labels alone
    cannot measure MAE or the residual spread, which is why every constant these
    numbers inform was previously a guess.
    """

    residuals: list[float] = field(default_factory=list)
    by_bracket: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add(self, expected_midpoint: int, predicted_years: float, bracket: str) -> None:
        self.residuals.append(predicted_years - expected_midpoint)
        self.by_bracket[bracket].append(predicted_years - expected_midpoint)

    def summary(self) -> dict[str, object]:
        if len(self.residuals) < 2:
            return {
                "samples": len(self.residuals),
                "note": (
                    "No raw age estimates were returned. Start the service with "
                    "EXPOSE_DEBUG_AGE_YEARS=true to measure MAE and residual spread."
                ),
            }
        mae = sum(abs(value) for value in self.residuals) / len(self.residuals)
        bias = statistics.fmean(self.residuals)
        spread = statistics.pstdev(self.residuals)
        # The measured spread includes decade-midpoint label noise. Removing it in
        # quadrature gives a less pessimistic estimate of the model's own residual.
        model_variance = spread**2 - DECADE_LABEL_NOISE_STD**2
        model_spread = math.sqrt(model_variance) if model_variance > 0.0 else 0.0
        return {
            "samples": len(self.residuals),
            "mae_years_vs_decade_midpoint": round(mae, 3),
            "bias_years": round(bias, 3),
            "residual_std_years": round(spread, 3),
            "residual_std_years_excluding_label_noise": round(model_spread, 3),
            "residual_std_by_expected_bracket": {
                bracket: round(statistics.pstdev(values), 3)
                for bracket, values in sorted(self.by_bracket.items())
                if len(values) > 1
            },
            "suggested_AGE_RESIDUAL_SIGMA_YEARS": round(max(model_spread, 1.0), 1),
            "note": (
                "Read speech, not logistics telephony. A positive bias means the "
                "regressor reads older than the label. Per-bracket spreads that "
                "differ widely argue for a heteroscedastic sigma rather than one "
                "constant."
            ),
        }


def multipart_audio(path: Path) -> tuple[bytes, str]:
    boundary = f"----voice-attributes-{uuid.uuid4().hex}"
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    prefix = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="audio"; filename="{path.name}"\r\n'
        f"Content-Type: {media_type}\r\n\r\n"
    ).encode()
    suffix = f"\r\n--{boundary}--\r\n".encode()
    return prefix + path.read_bytes() + suffix, boundary


def analyze(base_url: str, clip: Path) -> dict[str, object]:
    body, boundary = multipart_audio(clip)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    gender = TaskMetrics()
    age = TaskMetrics()
    age_regression = AgeRegressionMetrics()
    quality = Counter()
    failed = 0
    attempted = 0
    with args.tsv.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source, delimiter="\t"):
            expected_gender = gender_label(row.get("gender", ""))
            expected_age = age_bracket(row.get("age", ""))
            expected_midpoint = age_midpoint(row.get("age", ""))
            if expected_gender is None and expected_age is None:
                continue
            clip = args.clips / row["path"]
            if not clip.is_file():
                continue
            if args.limit and attempted >= args.limit:
                break
            attempted += 1
            try:
                result = analyze(args.url, clip)
            except (OSError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
                failed += 1
                print(f"warning: {clip.name}: {exc}")
                continue
            gender.add(expected_gender, result["gender"])
            age.add(expected_age, result["age_bracket"])
            predicted_years = result.get("debug_age_years")
            if expected_midpoint is not None and isinstance(
                predicted_years, (int, float)
            ):
                age_regression.add(
                    expected_midpoint, float(predicted_years), expected_age or "unknown"
                )
            quality[str(result["audio_quality"])] += 1

    print(
        json.dumps(
            {
                "attempted": attempted,
                "request_failures": failed,
                "gender": gender.summary(),
                "age_bracket": age.summary(),
                "age_regression": age_regression.summary(),
                "audio_quality": dict(quality),
                "note": "Adult Common Voice decade labels are mapped by midpoint; teens are excluded. Calibrate on a logistics-domain holdout before deployment.",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
