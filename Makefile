.PHONY: install dev test lint format up down ingest web-build web-test

install:
	uv sync --all-groups
	cd web && npm install

dev:
	uv run policy-rag serve --reload --port 8000

ingest:
	uv run policy-rag ingest corpus/

test:
	uv run pytest -q --cov=src
	cd web && npm test

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

web-build:
	cd web && npm run build

web-test:
	cd web && npm test

up:
	docker compose up --build -d

down:
	docker compose down -v
