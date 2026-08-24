# Frontend application

This directory contains the React, Vite, and SCSS client plus its Nginx
same-origin API/WebSocket proxy.

Use the repository-root commands; no directory change is required:

```bash
make frontend-install
make frontend-dev
make frontend-test
make frontend-lint
make frontend-build
```

The production UI is started with the complete stack:

```bash
docker compose up --build
```

See the [root README](../README.md) for setup, privacy defaults, model rationale,
and known limitations. Streaming behavior is documented in
[the backend streaming guide](../backend/docs/STREAMING.md).
