import pytest

from rag_tuda import runtime, tiers

T1, T2, T3, T3P = tiers.TIERS
TOKEN = "/tmp/token"

COMMON = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


def env(tier, mode="native", **kw):
    return runtime.env(tier, mode, token_file=TOKEN, **kw)


def test_native_t2():
    assert env(T2) == {
        **COMMON,
        "RAG__LLM__URL": "http://127.0.0.1:11434/v1",
        "RAG__LLM__MODEL": "lcr-t2",
        "RAG__LLM__ANALYSIS_MODEL": "lcr-t2",
        "RAG__LLM__CONTEXT_WINDOW": "8192",
        "RAG__LLM__FLAVOUR": "ollama",
        "RAG__DATABASE__HOST": "127.0.0.1",
        "RAG__DATABASE__PORT": "55432",
        "RAG__MODEL_SERVERS__EMBEDDING_URL": "http://127.0.0.1:8701",
        "RAG__MODEL_SERVERS__RERANKER_URL": "http://127.0.0.1:8702",
        "RAG__HTTP__TOKEN_FILE": TOKEN,
    }


def test_compose_t2_uses_service_names():
    e = env(T2, "compose")
    assert e["RAG__LLM__URL"] == "http://ollama:11434/v1"
    assert e["RAG__DATABASE__HOST"] == "pgvector" and e["RAG__DATABASE__PORT"] == "5432"
    assert e["RAG__MODEL_SERVERS__EMBEDDING_URL"] == "http://embedding:8000"
    assert e["RAG__MODEL_SERVERS__RERANKER_URL"] == "http://reranker:8000"
    assert e["RAG__LLM__FLAVOUR"] == "ollama"
    assert not any("127.0.0.1" in v for v in e.values())


@pytest.mark.parametrize("mode", ["native", "compose"])
@pytest.mark.parametrize("tier,name,ctx", [(T1, "lcr-t1", "8192"), (T2, "lcr-t2", "8192"), (T3, "lcr-t3", "32768")])
def test_ollama_tiers(tier, name, ctx, mode):
    e = env(tier, mode)
    assert (e["RAG__LLM__MODEL"], e["RAG__LLM__ANALYSIS_MODEL"], e["RAG__LLM__CONTEXT_WINDOW"]) == (name, name, ctx)
    assert e["RAG__LLM__FLAVOUR"] == "ollama"
    assert {k: e[k] for k in COMMON} == COMMON


def test_all_values_are_strings():
    assert all(isinstance(v, str) for v in env(T1).values())


def test_t3_plus_is_vllm_only():
    e = env(T3P, "compose")
    assert e["RAG__LLM__FLAVOUR"] == "vllm"
    assert e["RAG__LLM__MODEL"] == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
    assert e["RAG__LLM__ANALYSIS_MODEL"] == e["RAG__LLM__MODEL"]
    assert e["RAG__LLM__URL"] == "http://vllm:8000/v1"
    assert e["RAG__LLM__CONTEXT_WINDOW"] == "32768"
    assert runtime.derived_model_body(runtime.resolve(T3P, "compose")) is None
    with pytest.raises(ValueError, match="vLLM"):
        runtime.resolve(T3P, "compose", flavour="ollama")


def test_vllm_flavour_on_a_lower_tier():
    e = env(T2, "compose", flavour="vllm")
    assert (e["RAG__LLM__MODEL"], e["RAG__LLM__URL"]) == ("Qwen/Qwen3-14B-AWQ", "http://vllm:8000/v1")


def test_vllm_has_no_native_server_of_its_own():
    with pytest.raises(ValueError, match="llm-url|compose"):
        env(T3P, "native")
    assert env(T3P, "native", llm_url="http://10.0.0.5:8000/v1")["RAG__LLM__URL"] == "http://10.0.0.5:8000/v1"


def test_llm_url_override_means_a_foreign_server():
    e = env(T2, llm_url="http://gpu.lan:8080/v1")
    assert e["RAG__LLM__URL"] == "http://gpu.lan:8080/v1"
    assert e["RAG__LLM__FLAVOUR"] == "openai"
    assert e["RAG__LLM__MODEL"] == "Qwen/Qwen3-14B-AWQ"
    assert runtime.derived_model_body(runtime.resolve(T2, "native", llm_url="http://gpu.lan:8080/v1")) is None


def test_model_override_with_own_server():
    e = env(T2, llm_url="http://gpu.lan/v1", model="my-model")
    assert e["RAG__LLM__MODEL"] == e["RAG__LLM__ANALYSIS_MODEL"] == "my-model"


def test_model_override_on_ollama_changes_the_base_not_the_served_name():
    llm = runtime.resolve(T2, "native", model="mistral-small3.2:24b")
    assert llm.model == "lcr-t2" and llm.base_tag == "mistral-small3.2:24b"
    assert runtime.derived_model_body(llm)["from"] == "mistral-small3.2:24b"


def test_context_window_override():
    assert env(T1, context_window=16384)["RAG__LLM__CONTEXT_WINDOW"] == "16384"
    body = runtime.derived_model_body(runtime.resolve(T1, "native", context_window=16384))
    assert body["parameters"]["num_ctx"] == 16384


def test_openai_flavour_needs_a_url():
    with pytest.raises(ValueError, match="llm-url"):
        env(T2, flavour="openai")


def test_bad_arguments():
    with pytest.raises(ValueError):
        env(T2, "swarm")
    with pytest.raises(ValueError):
        env(T2, flavour="llamafile")
    with pytest.raises(ValueError):
        env(T2, context_window=0)


@pytest.mark.parametrize(
    "tier,name,tag,ctx",
    [
        (T1, "lcr-t1", "qwen3:4b-instruct-2507-q4_K_M", 8192),
        (T2, "lcr-t2", "qwen3:14b", 8192),
        (T3, "lcr-t3", "qwen3:30b-a3b-instruct-2507-q4_K_M", 32768),
    ],
)
def test_derived_model_body(tier, name, tag, ctx):
    assert runtime.derived_model_body(runtime.resolve(tier, "native")) == {
        "model": name,
        "from": tag,
        "parameters": {"num_ctx": ctx, "top_k": 50, "repeat_penalty": 1.05},
    }


def test_derived_model_body_values_are_passed_in():
    body = runtime.derived_model_body(runtime.resolve(T2, "native"), top_k=20, repeat_penalty=1.1)
    assert body["parameters"] == {"num_ctx": 8192, "top_k": 20, "repeat_penalty": 1.1}


def test_derived_model_is_the_same_in_both_modes():
    assert runtime.derived_model_body(runtime.resolve(T2, "native")) == runtime.derived_model_body(
        runtime.resolve(T2, "compose")
    )
