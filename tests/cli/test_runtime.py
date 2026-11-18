import json
from pathlib import Path

import pytest

from rag_tuda import cards, runtime

ROOT = Path(__file__).resolve().parents[2]


def card(name: str) -> dict:
    return cards.named(name)


S4B = card("s-qwen3-4b-instruct-2507")
M35 = card("m-qwen3.6-35b-a3b")
GROQ = card("l-groq-gpt-oss-120b")
XL = card("xl-vllm-example")


def test_a_native_ollama_card_gets_a_derived_model_with_the_cards_window():
    llm = runtime.resolve(S4B, "native")
    assert llm.derived == llm.served == "rag-s-qwen3-4b-instruct-2507"
    assert llm.base_tag == "qwen3:4b-instruct-2507-q4_K_M" and llm.pulls == (llm.base_tag,)
    assert (llm.answer_url, llm.answer_model) == ("http://127.0.0.1:11434/v1", llm.derived)
    assert llm.ollama_api == "http://127.0.0.1:11434" and llm.window == 8000 and llm.warms
    assert runtime.derived_model_body(llm, S4B) == {
        "model": "rag-s-qwen3-4b-instruct-2507",
        "from": "qwen3:4b-instruct-2507-q4_K_M",
        "parameters": {"num_ctx": 8000, "temperature": 0.1, "top_p": 0.9},
        "stream": False,
    }


def test_the_window_is_the_cards_effective_window():
    assert runtime.resolve(M35, "native").window == 16000
    assert runtime.derived_model_body(runtime.resolve(M35, "native"), M35)["parameters"]["num_ctx"] == 16000


def test_the_cards_sampling_and_extras_go_into_the_derived_model():
    tuned = {**S4B, "sampling": {"temperature": 0.3, "top_p": 0.8, "extras": {"top_k": 20, "repeat_penalty": 1.05}}}
    body = runtime.derived_model_body(runtime.resolve(tuned, "native"), tuned)
    assert body["parameters"] == {
        "num_ctx": 8000,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.05,
    }


def test_the_sampling_defaults_are_the_backends():
    from rag_core.settings.card import Sampling

    assert runtime.SAMPLING == {"temperature": Sampling().temperature, "top_p": Sampling().top_p}


def test_the_model_option_changes_the_base_not_the_served_name():
    llm = runtime.resolve(S4B, "native", model="mistral-small3.2:24b")
    assert llm.derived == "rag-s-qwen3-4b-instruct-2507" and llm.base_tag == "mistral-small3.2:24b"
    assert llm.pulls == ("mistral-small3.2:24b",)
    assert runtime.derived_model_body(llm, S4B)["from"] == "mistral-small3.2:24b"


def test_a_foreign_server_is_not_managed():
    llm = runtime.resolve(M35, "native", llm_url="http://gpu.lan:8080/v1")
    assert (llm.url, llm.derived, llm.base_tag, llm.pulls, llm.ollama_api) == (
        "http://gpu.lan:8080/v1",
        None,
        None,
        (),
        None,
    )
    assert (llm.served, llm.answer_model) == (None, "qwen3.6:35b-a3b")
    assert runtime.derived_model_body(llm, M35) is None
    named = runtime.resolve(M35, "native", llm_url="http://gpu.lan:8080/v1", model="my-model")
    assert (named.served, named.answer_model) == ("my-model", "my-model")


def test_a_hosted_card_keeps_its_answer_remote_and_runs_its_analysis_on_ollama():
    llm = runtime.resolve(GROQ, "native")
    assert (llm.provider, llm.derived, llm.warms) == ("groq", None, False)
    assert llm.pulls == ("qwen3:4b-instruct-2507-q4_K_M",) and llm.ollama_api == "http://127.0.0.1:11434"


def test_vllm_has_no_native_server_of_its_own():
    with pytest.raises(ValueError, match="llm-url"):
        runtime.resolve(XL, "native")
    assert runtime.resolve(XL, "native", llm_url="http://10.0.0.5:8000/v1").url == "http://10.0.0.5:8000/v1"
    assert runtime.resolve(XL, "compose").derived is None


def test_the_compose_stack_serves_the_cards_own_model():
    llm = runtime.resolve(S4B, "compose")
    assert (llm.derived, llm.served, llm.url, llm.ollama_api) == (None, None, None, None)
    assert llm.pulls == ("qwen3:4b-instruct-2507-q4_K_M",)
    for kwargs in ({"llm_url": "http://x/v1"}, {"model": "m"}):
        with pytest.raises(ValueError, match="native"):
            runtime.resolve(S4B, "compose", **kwargs)
    with pytest.raises(ValueError, match="hosted"):
        runtime.resolve(GROQ, "compose")


def test_a_bad_mode():
    with pytest.raises(ValueError):
        runtime.resolve(S4B, "swarm")


def test_the_backend_environment():
    llm = runtime.resolve(S4B, "native")
    env = runtime.backend_env(llm, root="/repo", secrets="/repo/.rag/secrets/backend", profile="example")
    assert env == {
        "RAG_PROFILE": "/repo/profiles/example/profile.yaml",
        "RAG_SECRETS_DIR": "/repo/.rag/secrets/backend",
        "RAG_CARDS_DIR": "/repo/models/cards",
        "RAG__CARD": "s-qwen3-4b-instruct-2507",
        "RAG__LLM_MODEL": "rag-s-qwen3-4b-instruct-2507",
        "RAG__DATABASE__HOST": "127.0.0.1",
        "RAG__DATABASE__PORT": "55432",
        "RAG__MODEL_SERVERS__EMBEDDING_URL": "http://127.0.0.1:8701",
        "RAG__MODEL_SERVERS__RERANKER_URL": "http://127.0.0.1:8702",
        "RAG__COMPUTE__SLOT_WAIT_S": "30",
        "RAG__TIMEOUTS__RETRIEVAL_S": "10",
        "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
        "RAG__TIMEOUTS__RERANKER_S": "15",
        "RAG__HTTP__CORS_DOMAIN": "localhost",
        "RAG__TRACING__URL": "",
        "OPENAI_API_KEY": "not-needed",
        "GUNICORN_BIND": "127.0.0.1:8700",
        "UVICORN_WORKERS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    foreign = runtime.resolve(S4B, "native", llm_url="http://gpu.lan/v1", model="m")
    moved = runtime.backend_env(foreign, root="/repo", secrets="/s", profile="example")
    assert (moved["RAG__LLM_URL"], moved["RAG__LLM_MODEL"]) == ("http://gpu.lan/v1", "m")


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
    assert settings.overlay.database.port == 55432 and settings.overlay.http.cors_domain == "localhost"


def test_the_indexer_gets_the_six_folders_under_rag():
    env = runtime.indexer_env(
        root="/repo", rag="/repo/.rag", secrets="/repo/.rag/secrets/indexer", profile="p", overlay=None
    )
    assert {k: v for k, v in env.items() if k.startswith("RAG__PATHS__")} == {
        "RAG__PATHS__CRAWL_DIR": "/repo/.rag/crawl",
        "RAG__PATHS__PDF_DIR": "/repo/.rag/crawl/pdfs",
        "RAG__PATHS__EXTRACTED_DIR": "/repo/.rag/crawl/extracted",
        "RAG__PATHS__STATE_DIR": "/repo/.rag/state",
        "RAG__PATHS__MANIFEST_DIR": "/repo/.rag/state/documents",
        "RAG__PATHS__SKIPPED_PDFS_LOG": "/repo/.rag/state/skipped_pdfs_page_limit.json",
    }
    assert env["RAG_SECRETS_DIR"] == "/repo/.rag/secrets/indexer" and "RAG_OVERLAY" not in env
    with_overlay = runtime.indexer_env(root="/r", rag="/r/.rag", secrets="/s", profile="p", overlay="/o.yaml")
    assert with_overlay["RAG_OVERLAY"] == "/o.yaml"


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


def test_the_overlay_is_json_and_empty_without_folder_sources():
    assert runtime.overlay_text("/r", "p", {}) is None
    assert json.loads(runtime.overlay_text("/r", "p", {"gesetze": "documents"}) or "") == {
        "connectors": {"gesetze": {"dir": "/r/profiles/p/documents"}}
    }


def test_the_migrate_environment():
    assert runtime.migrate_env("/s") == {
        "RAG_SECRETS_DIR": "/s",
        "RAG_MIGRATE_HOST": "127.0.0.1",
        "RAG_MIGRATE_PORT": "55432",
    }


def test_compose_env_names_the_card_the_window_the_token_and_the_roles_folder():
    llm = runtime.resolve(M35, "compose")
    text = runtime.compose_env_text(llm, token="tok", roles_dir="/repo/.rag/secrets/roles")
    assert dict(line.split("=", 1) for line in text.splitlines()) == {
        "RAG__CARD": "m-qwen3.6-35b-a3b",
        "OLLAMA_CONTEXT_LENGTH": "16000",
        "BACKEND_REFRESH_TOKEN": "tok",
        "RAG_ROLES_DIR": "/repo/.rag/secrets/roles",
    }
