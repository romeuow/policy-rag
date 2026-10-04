# ---- Stage 1: build the frontend -------------------------------------------
FROM node:24-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---- Stage 2: resolve Python dependencies with uv --------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# ---- Stage 3: runtime image (no uv, non-root) ------------------------------
FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app --home /app app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=web --chown=app:app /web/dist /app/web/dist
COPY --chown=app:app corpus/ /app/corpus/
COPY --chown=app:app alembic.ini /app/
COPY --chown=app:app alembic/ /app/alembic/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 WEB_DIST_DIR=/app/web/dist CORPUS_DIR=/app/corpus
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["policy-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
