# Backend service

This directory contains the FastAPI inference service, model adapters, audio
pipeline, persistence layer, migrations, backend tests, evaluation utilities,
and backend-specific documentation.

Run commands from the repository root so Docker Compose, build contexts, and
the frontend proxy use the same configuration:

```bash
docker compose up --build
make test
make lint
make smoke
make smoke-ws
```

The Python package remains named `app` inside containers and local backend
environments. PostgreSQL stores retained result/manifests; consent-gated audio
is written to S3-compatible object storage. Retention defaults to `none` for
every request.

Repository documentation:

- [Complete setup and root commands](../README.md#quick-start)
- [Backend API](docs/API.md)
- [System design](docs/DESIGN.md)
- [Audio pipeline techniques and trade-offs](docs/AUDIO_PIPELINE.md)
- [Production model decision](docs/ADR-002-production-model-strategy.md)
- [Language identification decision](docs/ADR-004-language-identification.md)
- [Persistence decision](docs/ADR-003-opt-in-persistence.md)
- [Known model limitations](docs/MODEL_CARD.md)
- [Privacy and production controls](docs/PRIVACY.md)
