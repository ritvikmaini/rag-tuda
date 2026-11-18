"""The folders the indexer reads and writes, and the folder of its secrets, come from the settings: the compose paths
are the defaults, the environment moves them (a native run puts them under .rag/)."""

import json
from pathlib import Path

from indexer import build, gates
from indexer.settings import load_settings

CI = str(Path(__file__).resolve().parents[2] / "deploy" / "env" / "ci-example.yaml")


def test_the_defaults_are_the_compose_paths(profile):
    paths = load_settings({"RAG_PROFILE": profile, "RAG_OVERLAY": CI}).overlay.paths
    assert paths.model_dump() == {
        "crawl_dir": "/app/data/crawl",
        "manifest_dir": "/app/data/vector_database_contents/documents",
        "state_dir": "/app/data/vector_database_contents",
        "extracted_dir": "/app/data/extracted",
        "pdf_dir": "/app/data/pdfs",
        "skipped_pdfs_log": "/app/data/skipped_pdfs_page_limit.json",
    }


def test_paths_from_settings(settings, moved, fakes, monkeypatch):
    context = build.context(settings, "web", None, False)
    assert str(context.state.root) == moved["state_dir"] and str(context.cache_dir) == moved["extracted_dir"]
    assert dict(context.paths) == moved

    seen = {}
    monkeypatch.setattr(build, "extract_pdfs", lambda **kw: seen.update(extract=kw) or {})
    monkeypatch.setattr(build, "pdf_documents", lambda items, hashes, **kw: seen.update(documents=kw) or ([], 0))
    pdf = build.PdfItem("/p.pdf", "https://x.example/a.pdf", "h1", {})
    build.build_corpus(settings, {"web": fakes.Source([pdf])}, embedding=fakes.Embedding(None))
    assert seen["extract"]["extracted_dir"] == moved["extracted_dir"]
    assert seen["extract"]["skipped_log"] == moved["skipped_pdfs_log"]
    assert seen["documents"]["extracted_dir"] == moved["extracted_dir"]

    state_dir = settings.overlay.paths.state_dir
    fakes.stats.write_text(json.dumps({"parents": 7}), encoding="utf-8")
    assert gates.previous_stats(state_dir) == {"parents": 7}
    assert gates.record_build_stats(state_dir, {"parents": 1, "children": 2}) is True
    assert json.loads(fakes.stats.read_text(encoding="utf-8"))["parents"] == 1


def test_the_secrets_folder_is_set_by_the_environment(tmp_path, profile):
    secrets = tmp_path / "elsewhere"
    secrets.mkdir()
    (secrets / "pg_vector_password").write_text("pw\n", encoding="utf-8")
    environ = {"RAG_PROFILE": profile, "RAG_OVERLAY": CI, "RAG_SECRETS_DIR": str(secrets)}
    assert load_settings(environ).overlay.database.password.get_secret_value() == "pw"
