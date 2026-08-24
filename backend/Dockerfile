# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS model-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install \
    "joblib==1.4.2" \
    "numpy==2.1.3" \
    "pandas==2.2.3" \
    "scikit-learn==1.5.2"

COPY backend/scripts/prepare_models.py /usr/local/bin/prepare_models.py
# The pinned griko SVM was fitted with probability=False, so it exposes no
# predict_proba and no Platt parameters exist to convert. The exported gender
# score is therefore monotonic in the decision value and is NOT a calibrated
# probability. The flag makes that fact explicit in the artifact and in
# /opt/models/model-metadata.json instead of silently shipping a fallback
# sigmoid that looks like one. Remove it once a calibrated head is available.
RUN python /usr/local/bin/prepare_models.py --output /opt/models \
      --allow-uncalibrated-gender


FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    ORT_DISABLE_TELEMETRY=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_BACKEND=ecapa \
    MODEL_ROOT=/opt/models \
    MODEL_DEVICE=cpu \
    TORCH_THREADS=2

# This baseline image knowingly serves the uncalibrated gender score exported
# above. A production image built on a calibrated head must drop this so the
# runtime fails closed rather than serving a score shaped like a probability.
ENV REQUIRE_CALIBRATED_GENDER=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /nonexistent --shell /usr/sbin/nologin app \
    && mkdir -p /licenses \
    && cp /usr/share/common-licenses/Apache-2.0 /licenses/Apache-2.0.txt

WORKDIR /srv/app

COPY backend/pyproject.toml backend/README.md ./

# Pin CPU wheels so x86 images do not pull CUDA libraries. The extra remains
# architecture-portable on Docker's supported Linux targets. Installing the
# dependency metadata before application sources keeps this expensive layer
# cached during ordinary source edits.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.8.0" "torchaudio==2.8.0" \
    && pip install ".[model,wavlm]"

COPY LICENSE /licenses/service-MIT.txt
COPY backend/THIRD_PARTY_NOTICES.md /licenses/THIRD_PARTY_NOTICES.md
COPY backend/app ./app
RUN pip install --force-reinstall --no-deps .

COPY --from=model-builder /opt/models /opt/models

USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=2).read()"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]


FROM runtime AS test

USER root
COPY backend/tests ./tests
COPY backend/scripts ./scripts
RUN pip install ".[dev]"
ENV RUFF_CACHE_DIR=/tmp/ruff-cache \
    PYTEST_ADDOPTS="-q -p no:cacheprovider"
USER app

CMD ["pytest", "-q"]
