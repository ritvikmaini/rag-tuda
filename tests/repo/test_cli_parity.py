"""The command line (standard library only) agrees with the services it starts: the window, the sampling defaults, the
pins and the environments it hands them are what the services' own loaders accept. These tests import the workspace
packages, so they run with the full workspace and not in the command line's own job."""

import re
from pathlib import Path

import yaml

from rag_tuda import cards, classes, runtime, stack

ROOT = Path(__file__).resolve().parents[2]
S4B = cards.named("s-qwen3-4b-instruct-2507")
M35 = cards.named("m-qwen3.6-35b-a3b")
GROQ = cards.named("l-groq-gpt-oss-120b")


def test_the_command_line_computes_the_window_the_backend_does():
    from rag_core.settings.card import effective_window, load_card

    shipped = sorted(cards.CARDS_DIR.glob("*.toml"))
    assert len(shipped) == 10
    for path in shipped:
        assert cards.window(cards.load(path)) == effective_window(load_card(path))


def test_the_command_line_knows_the_class_windows_and_reads_a_card_that_leaves_its_window_out(tmp_path):
    from rag_core.settings.card import CLASS_DEFAULTS, effective_window, load_card

    assert cards.CLASS_WINDOW_TOKENS == {name: d["budget"]["window_tokens"] for name, d in CLASS_DEFAULTS.items()}
    for path in sorted(cards.CARDS_DIR.glob("*.toml")):
        text = re.sub(r"^window_tokens = .*\n", "", path.read_text(encoding="utf-8"), flags=re.M)
        assert "window_tokens =" not in text
        bare = tmp_path / path.name
        bare.write_text(text, encoding="utf-8")
        assert cards.window(cards.load(bare)) == effective_window(load_card(bare)), path.name


def test_the_sampling_defaults_are_the_backends():
    from rag_core.settings.card import Sampling

    assert runtime.SAMPLING == {"temperature": Sampling().temperature, "top_p": Sampling().top_p}


def test_the_backend_environment_loads_with_the_services_settings(tmp_path):
    """The values the native backend gets are accepted by the settings loader (an unknown or missing key would stop it)."""
    from assistant_api.settings import load_settings

    secrets = tmp_path / "secrets"
    secrets.mkdir()
    for name in ("pgdatabase", "vector_user", "pg_vector_password", "backend_token"):
        (secrets / name).write_text("x", encoding="utf-8")
    llm = runtime.resolve(M35, "native")
    env = runtime.backend_env(llm, root=str(ROOT), secrets=str(secrets), profile="example")
    settings = load_settings(env)
    assert settings.card.answer.model == "rag-m-qwen3.6-35b-a3b"
    assert settings.card.answer.url.startswith("http://127.0.0.1")
    assert settings.overlay.database.port == 55432 and settings.overlay.http.cors_domain == ""


def test_the_indexer_environment_loads_with_the_indexers_settings(tmp_path):
    from indexer.settings import load_settings

    overlay = runtime.overlay_text(str(ROOT), "example", {"gesetze": "documents"})
    assert overlay is not None
    (tmp_path / "overlay.yaml").write_text(overlay, encoding="utf-8")
    env = runtime.indexer_env(
        root=str(ROOT),
        rag=str(tmp_path),
        secrets=str(tmp_path),
        profile="example",
        overlay=str(tmp_path / "overlay.yaml"),
    )
    settings = load_settings(env)
    assert settings.endpoints["gesetze"].dir == str(ROOT / "profiles" / "example" / "documents")
    assert settings.overlay.paths.crawl_dir == str(tmp_path / "crawl") and settings.overlay.database.port == 55432


def test_a_hosted_cards_environment_loads_with_the_services_settings(tmp_path):
    from assistant_api.settings import load_settings

    for name in ("pgdatabase", "vector_user", "pg_vector_password", "backend_token"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    env = runtime.backend_env(runtime.resolve(GROQ, "native"), root=str(ROOT), secrets=str(tmp_path), profile="example")
    card = load_settings(env).card
    assert card.answer.provider == "groq" and card.answer.model == "openai/gpt-oss-120b"
    assert card.analysis_endpoint.model == "rag-l-groq-gpt-oss-120b-analysis"


def test_the_pins_are_the_ones_the_profile_and_the_images_pin():
    profile = yaml.safe_load((ROOT / "profiles" / "example" / "profile.yaml").read_text(encoding="utf-8"))["retrieval"]
    embedding, reranker = stack.PINNED["embedding"], stack.PINNED["reranker"]
    assert (embedding.repo, embedding.revision) == (profile["embedding"]["model"], profile["embedding"]["revision"])
    assert (reranker.repo, reranker.revision) == (profile["rerank"]["model"], profile["rerank"]["revision"])
    for name, pin in stack.PINNED.items():
        dockerfile = (ROOT / "model_servers" / name / "Dockerfile").read_text(encoding="utf-8")
        baked = dict(re.findall(r"^ENV (\w+)=(.+)$", dockerfile, re.M))
        for key, value in pin.env.items():
            assert baked[key].strip("'") == value, key
    assert {m.name for m in classes.FIXED_MODELS} >= {embedding.repo, reranker.repo}
