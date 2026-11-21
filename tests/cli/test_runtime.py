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


def test_the_native_backend_serves_the_page_of_the_checkout_it_was_started_from():
    llm = runtime.resolve(S4B, "native")
    env = runtime.backend_env(llm, root="/elsewhere/rag", secrets="/s", profile="example")
    assert env["RAG_PAGE_DIR"] == "/elsewhere/rag/web/page"


def test_the_backend_environment():
    llm = runtime.resolve(S4B, "native")
    env = runtime.backend_env(llm, root="/repo", secrets="/repo/.rag/secrets/backend", profile="example")
    assert env == {
        "RAG_PROFILE": "/repo/profiles/example/profile.yaml",
        "RAG_SECRETS_DIR": "/repo/.rag/secrets/backend",
        "RAG_CARDS_DIR": "/repo/models/cards",
        "RAG_PAGE_DIR": "/repo/web/page",
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
        "RAG__TRACING__URL": "",
        "OPENAI_API_KEY": "not-needed",
        "GUNICORN_BIND": "127.0.0.1:8700",
        "UVICORN_WORKERS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    foreign = runtime.resolve(S4B, "native", llm_url="http://gpu.lan/v1", model="m")
    moved = runtime.backend_env(foreign, root="/repo", secrets="/s", profile="example")
    assert (moved["RAG__LLM_URL"], moved["RAG__LLM_MODEL"]) == ("http://gpu.lan/v1", "m")


def test_the_indexer_gets_the_six_folders_under_rag():
    env = runtime.indexer_env(
        root="/repo", rag="/repo/.rag", secrets="/repo/.rag/secrets/indexer", profile="p", overlay=None
    )
    assert {k: v for k, v in env.items() if k.startswith("RAG__PATHS__")} == {
        "RAG__PATHS__CRAWL_DIR": "/repo/.rag/crawl",
        "RAG__PATHS__PDF_DIR": "/repo/.rag/crawl/pdfs",
        "RAG__PATHS__EXTRACTED_DIR": "/repo/.rag/crawl/extracted",
        "RAG__PATHS__STATE_DIR": "/repo/.rag/state",
        "RAG__PATHS__SKIPPED_PDFS_LOG": "/repo/.rag/state/skipped_pdfs_page_limit.json",
        "RAG__PATHS__USER_SOURCES": "/repo/.rag/sources.json",
    }
    assert env["RAG_SECRETS_DIR"] == "/repo/.rag/secrets/indexer" and "RAG_OVERLAY" not in env
    with_overlay = runtime.indexer_env(root="/r", rag="/r/.rag", secrets="/s", profile="p", overlay="/o.yaml")
    assert with_overlay["RAG_OVERLAY"] == "/o.yaml"


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


def test_compose_env_names_the_card_the_window_the_token_the_roles_folder_and_the_profile():
    llm = runtime.resolve(M35, "compose")
    text = runtime.compose_env_text(llm, token="tok", roles_dir="/repo/.rag/secrets/roles")
    assert dict(line.split("=", 1) for line in text.splitlines()) == {
        "RAG__CARD": "m-qwen3.6-35b-a3b",
        "OLLAMA_CONTEXT_LENGTH": "16000",
        "BACKEND_REFRESH_TOKEN": "tok",
        "RAG_ROLES_DIR": "/repo/.rag/secrets/roles",
        "RAG_PROFILE": "example",
    }
    other = runtime.compose_env_text(llm, token="t", roles_dir="/r", profile="p", overlay="/r/overlay.yaml")
    assert "RAG_PROFILE=p\n" in other and "RAG_INDEXER_OVERLAY=/r/overlay.yaml\n" in other


def test_a_remote_card_without_its_own_analysis_block_is_refused():
    bare = {k: v for k, v in GROQ.items() if k != "analysis"}
    with pytest.raises(
        ValueError, match=r"l-groq-gpt-oss-120b: the answer provider groq is remote.*\[analysis\] block"
    ):
        runtime.resolve(bare, "native")
    assert runtime.resolve(S4B, "native").provider == "ollama"  # a local card needs none


def test_an_analysis_model_other_than_the_answers_gets_its_own_derived_model_and_window():
    llm = runtime.resolve(GROQ, "native")
    assert (
        llm.analysis_derived == "rag-l-groq-gpt-oss-120b-analysis"
        and llm.analysis_base == "qwen3:4b-instruct-2507-q4_K_M"
    )
    assert llm.analysis_window == 8000  # min(its 262144, the card's request window 8000)
    assert runtime.analysis_model_body(llm, GROQ) == {
        "model": "rag-l-groq-gpt-oss-120b-analysis",
        "from": "qwen3:4b-instruct-2507-q4_K_M",
        "parameters": {"num_ctx": 8000, "temperature": 0.1, "top_p": 0.9},
        "stream": False,
    }
    env = runtime.backend_env(llm, root="/r", secrets="/s", profile="example")
    assert env["RAG__LLM_ANALYSIS_MODEL"] == "rag-l-groq-gpt-oss-120b-analysis" and "RAG__LLM_MODEL" not in env
    assert "GROQ_API_KEY" not in env  # the key travels in the spawn environment, never in the plan
    small = {**GROQ, "analysis": {**GROQ["analysis"], "context_window": 4096}}
    assert runtime.resolve(small, "native").analysis_window == 4096


def test_no_analysis_model_is_made_when_it_shares_the_answers_or_is_not_ours():
    shared = {**S4B, "analysis": S4B["answer"]}
    assert runtime.resolve(shared, "native").analysis_derived is None
    assert runtime.resolve(S4B, "native").analysis_derived is None
    foreign = runtime.resolve(GROQ, "native", llm_url="http://gpu.lan/v1")
    assert foreign.analysis_derived is None and runtime.analysis_model_body(foreign, GROQ) is None


def test_the_compose_warm_up_targets_the_compose_model_server():
    assert runtime.resolve(S4B, "compose").answer_url == "http://ollama:11434/v1"
    assert runtime.resolve(XL, "compose").answer_url == "http://vllm:8000/v1"
    assert runtime.resolve(S4B, "compose").answer_model == "qwen3:4b-instruct-2507-q4_K_M"
