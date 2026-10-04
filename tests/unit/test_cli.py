import json

import pytest

from policy_rag import cli
from policy_rag.config import Settings, get_settings
from tests.conftest import CORPUS_DIR


@pytest.fixture(autouse=True)
def _demo_settings(monkeypatch: pytest.MonkeyPatch):
    get_settings.cache_clear()
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(app_mode="demo", log_level="ERROR"))


def test_ingest_command_prints_summary(capsys):
    assert cli.main(["ingest", str(CORPUS_DIR)]) == 0
    out = capsys.readouterr().out
    summary = json.loads(out)
    assert summary["mode"] == "demo"
    assert len(summary["documents"]) == 7
    assert summary["total_chunks"] > 20


def test_ingest_command_rejects_missing_directory(tmp_path, capsys):
    assert cli.main(["ingest", str(tmp_path / "nope")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_serve_command_delegates_to_uvicorn(monkeypatch: pytest.MonkeyPatch):
    import uvicorn

    called = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: called.update(kw))
    assert cli.main(["serve", "--port", "9999"]) == 0
    assert called["port"] == 9999 and called["factory"] is True
