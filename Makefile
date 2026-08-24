.PHONY: build up test-image test lint smoke smoke-ui smoke-ws smoke-ws-ui frontend-install frontend-test frontend-lint frontend-build frontend-dev check

TEST_IMAGE := voice-contact-attributes:test

build:
	docker compose build

up:
	docker compose up --build

test-image:
	docker build -f backend/Dockerfile --target test -t $(TEST_IMAGE) .

test: test-image
	docker run --rm $(TEST_IMAGE)

lint: test-image
	docker run --rm --entrypoint ruff $(TEST_IMAGE) check app tests scripts
	docker run --rm --entrypoint ruff $(TEST_IMAGE) format --check app tests scripts

smoke:
	python3 backend/scripts/smoke_test.py --url http://127.0.0.1:8000

smoke-ui:
	python3 backend/scripts/smoke_test.py --url http://127.0.0.1:3000/api

smoke-ws: test-image
	docker run --rm --add-host=host.docker.internal:host-gateway --entrypoint python3 $(TEST_IMAGE) scripts/ws_smoke_test.py --url ws://host.docker.internal:8000/ws/analyze

smoke-ws-ui: test-image
	docker run --rm --add-host=host.docker.internal:host-gateway --entrypoint python3 $(TEST_IMAGE) scripts/ws_smoke_test.py --url ws://host.docker.internal:3000/api/ws/analyze --encoding pcm_f32le

frontend-install:
	npm --prefix frontend ci

frontend-test: frontend-install
	npm --prefix frontend test

frontend-lint: frontend-install
	npm --prefix frontend run lint

frontend-build: frontend-install
	npm --prefix frontend run build

frontend-dev:
	npm --prefix frontend run dev

check: test lint frontend-test frontend-lint frontend-build
