"""Command-line entry point: `policy-rag ingest <dir>` and `policy-rag serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from policy_rag.api.container import build_container
from policy_rag.config import get_settings
from policy_rag.logging import configure_logging


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = build_container(settings)
    directory = Path(args.path)
    if not directory.is_dir():
        print(f"error: {directory} is not a directory", file=sys.stderr)
        return 2
    docs = container.documents.ingest_directory(directory)
    summary = {
        "mode": settings.app_mode,
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "version": d.version, "chunks": d.chunk_count}
            for d in docs
        ],
        "total_chunks": container.store.count(),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if settings.app_mode == "demo":
        print(
            "note: APP_MODE=demo uses an in-memory store; the API re-ingests CORPUS_DIR on start.",
            file=sys.stderr,
        )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "policy_rag.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_config=None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="policy-rag", description="Policy RAG assistant CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Parse, chunk and embed every document in a folder")
    ingest.add_argument("path", help="Directory containing .md/.txt/.pdf files")
    ingest.set_defaults(func=cmd_ingest)

    serve = sub.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
