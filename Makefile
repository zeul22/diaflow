.PHONY: build up test lint smoke smoke-ui smoke-ws frontend-install frontend-test frontend-lint frontend-build frontend-dev check

build:
	docker compose build

up:
	docker compose up --build

test:
	docker build --target test -t voice-contact-attributes:test .
	docker run --rm voice-contact-attributes:test

lint:
	docker build --target test -t voice-contact-attributes:test .
	docker run --rm --entrypoint ruff voice-contact-attributes:test check app tests scripts
	docker run --rm --entrypoint ruff voice-contact-attributes:test format --check app tests scripts

smoke:
	python3 scripts/smoke_test.py --url http://127.0.0.1:8000

smoke-ui:
	python3 scripts/smoke_test.py --url http://127.0.0.1:3000/api

smoke-ws:
	python3 scripts/ws_smoke_test.py --url ws://127.0.0.1:8000/ws/analyze

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
