import pytest

from rag_tuda import runtime, classes

S, M, L, XL = classes.CLASSES
TOKEN = "/tmp/token"

COMMON = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def env(tier, mode="native", **kw):
    return runtime.env(tier, mode, token_file=TOKEN, **kw)


def test_native_m():
    assert env(M) == {
        **COMMON,
        "RAG__LLM__URL": "http://127.0.0.1:11434/v1",
        "RAG__LLM__MODEL": "lcr-m",
        "RAG__LLM__ANALYSIS_MODEL": "lcr-m",
        "RAG__LLM__CONTEXT_WINDOW": "16000",
        "RAG__LLM__FLAVOUR": "ollama",
        "RAG__DATABASE__HOST": "127.0.0.1",
        "RAG__DATABASE__PORT": "55432",
        "RAG__MODEL_SERVERS__EMBEDDING_URL": "http://127.0.0.1:8701",
        "RAG__MODEL_SERVERS__RERANKER_URL": "http://127.0.0.1:8702",
        "RAG__HTTP__TOKEN_FILE": TOKEN,
    }


def test_compose_m_uses_service_names():
    e = env(M, "compose")
    assert e["RAG__LLM__URL"] == "http://ollama:11434/v1"
    assert e["RAG__DATABASE__HOST"] == "pgvector" and e["RAG__DATABASE__PORT"] == "5432"
    assert e["RAG__MODEL_SERVERS__EMBEDDING_URL"] == "http://embedding:8000"
    assert e["RAG__MODEL_SERVERS__RERANKER_URL"] == "http://reranker:8000"
    assert e["RAG__LLM__FLAVOUR"] == "ollama"
    assert not any("127.0.0.1" in v for v in e.values())


@pytest.mark.parametrize("mode", ["native", "compose"])
@pytest.mark.parametrize("cls,name,ctx", [(S, "lcr-s", "8000"), (M, "lcr-m", "16000")])
def test_ollama_classes(cls, name, ctx, mode):
    e = env(cls, mode)
    assert (e["RAG__LLM__MODEL"], e["RAG__LLM__ANALYSIS_MODEL"], e["RAG__LLM__CONTEXT_WINDOW"]) == (name, name, ctx)
    assert e["RAG__LLM__FLAVOUR"] == "ollama"
    assert {k: e[k] for k in COMMON} == COMMON


def test_all_values_are_strings():
    assert all(isinstance(v, str) for v in env(S).values())


def test_xl_is_vllm_only():
    e = env(XL, "compose")
    assert e["RAG__LLM__FLAVOUR"] == "vllm"
    assert e["RAG__LLM__MODEL"] == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    assert e["RAG__LLM__ANALYSIS_MODEL"] == e["RAG__LLM__MODEL"]
    assert e["RAG__LLM__URL"] == "http://vllm:8000/v1"
    assert e["RAG__LLM__CONTEXT_WINDOW"] == "64000"
    assert runtime.derived_model_body(runtime.resolve(XL, "compose")) is None
    with pytest.raises(ValueError, match="vLLM"):
        runtime.resolve(XL, "compose", flavour="ollama")


def test_vllm_flavour_on_a_local_class():
    e = env(M, "compose", flavour="vllm")
    assert (e["RAG__LLM__MODEL"], e["RAG__LLM__URL"]) == ("Qwen/Qwen3.6-35B-A3B", "http://vllm:8000/v1")


def test_vllm_has_no_native_server_of_its_own():
    with pytest.raises(ValueError, match="llm-url|compose"):
        env(XL, "native")
    assert env(XL, "native", llm_url="http://10.0.0.5:8000/v1")["RAG__LLM__URL"] == "http://10.0.0.5:8000/v1"


def test_llm_url_override_means_a_foreign_server():
    e = env(M, llm_url="http://gpu.lan:8080/v1")
    assert e["RAG__LLM__URL"] == "http://gpu.lan:8080/v1"
    assert e["RAG__LLM__FLAVOUR"] == "openai"
    assert e["RAG__LLM__MODEL"] == "Qwen/Qwen3.6-35B-A3B"
    assert runtime.derived_model_body(runtime.resolve(M, "native", llm_url="http://gpu.lan:8080/v1")) is None


def test_model_override_with_own_server():
    e = env(M, llm_url="http://gpu.lan/v1", model="my-model")
    assert e["RAG__LLM__MODEL"] == e["RAG__LLM__ANALYSIS_MODEL"] == "my-model"


def test_model_override_on_ollama_changes_the_base_not_the_served_name():
    llm = runtime.resolve(M, "native", model="mistral-small3.2:24b")
    assert llm.model == "lcr-m" and llm.base_tag == "mistral-small3.2:24b"
    assert runtime.derived_model_body(llm)["from"] == "mistral-small3.2:24b"


def test_context_window_override():
    assert env(S, context_window=16384)["RAG__LLM__CONTEXT_WINDOW"] == "16384"
    body = runtime.derived_model_body(runtime.resolve(S, "native", context_window=16384))
    assert body["parameters"]["num_ctx"] == 16384


def test_openai_flavour_needs_a_url():
    with pytest.raises(ValueError, match="llm-url"):
        env(M, flavour="openai")


def test_bad_arguments():
    with pytest.raises(ValueError):
        env(M, "swarm")
    with pytest.raises(ValueError):
        env(M, flavour="llamafile")
    with pytest.raises(ValueError):
        env(M, context_window=0)


@pytest.mark.parametrize(
    "cls,name,tag,ctx",
    [
        (S, "lcr-s", "qwen3:4b-instruct-2507-q4_K_M", 8000),
        (M, "lcr-m", "qwen3.6:35b-a3b", 16000),
    ],
)
def test_derived_model_body(cls, name, tag, ctx):
    assert runtime.derived_model_body(runtime.resolve(cls, "native")) == {
        "model": name,
        "from": tag,
        "parameters": {"num_ctx": ctx, "top_k": 50, "repeat_penalty": 1.05},
    }


def test_derived_model_body_values_are_passed_in():
    body = runtime.derived_model_body(runtime.resolve(M, "native"), top_k=20, repeat_penalty=1.1)
    assert body["parameters"] == {"num_ctx": 16000, "top_k": 20, "repeat_penalty": 1.1}


def test_derived_model_is_the_same_in_both_modes():
    assert runtime.derived_model_body(runtime.resolve(M, "native")) == runtime.derived_model_body(
        runtime.resolve(M, "compose")
    )
