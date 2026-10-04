"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from policy_rag import __version__
from policy_rag.api import routes_admin, routes_chat, routes_meta
from policy_rag.api.container import Container, build_container
from policy_rag.config import Settings, get_settings
from policy_rag.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    container: Container | None = None,
    ingest_on_startup: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    container = container or build_container(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if ingest_on_startup and settings.app_mode == "demo":
            container.ingest_corpus()
        logger.info("app_started", extra={"mode": settings.app_mode, "version": __version__})
        yield

    app = FastAPI(title="policy-rag", version=__version__, lifespan=lifespan)
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(routes_meta.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_admin.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"path": request.url.path})
        container.metrics.inc("errors_total")
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    _mount_frontend(app, Path(settings.web_dist_dir))
    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve the built SPA when `web/dist` exists; unknown non-API paths fall back to index."""
    index = dist / "index.html"
    if not index.is_file():
        return
    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        candidate = (dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(index)
