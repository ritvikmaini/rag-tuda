"""Where things run: the mapping from (class, mode, overrides) to the environment the services get and to the request that
creates the derived Ollama model. Pure functions; nothing here touches the machine."""

from dataclasses import dataclass

from rag_tuda.classes import SizeClass

FLAVOURS = ("ollama", "vllm", "openai")
MODES = ("native", "compose")
NATIVE = {
    "ollama": "http://127.0.0.1:11434/v1",
    "db_host": "127.0.0.1",
    "db_port": "55432",
    "embedding": "http://127.0.0.1:8701",
    "reranker": "http://127.0.0.1:8702",
}
COMPOSE = {
    "ollama": "http://ollama:11434/v1",
    "vllm": "http://vllm:8000/v1",
    "db_host": "pgvector",
    "db_port": "5432",
    "embedding": "http://embedding:8000",
    "reranker": "http://reranker:8000",
}
PRIVACY = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}


@dataclass(frozen=True)
class Llm:
    flavour: str
    url: str
    model: str  # the name the backend sends; for Ollama the derived model
    context_window: int
    base_tag: str | None  # the Ollama tag the derived model is made from (ollama flavour only)


def resolve(
    cls: SizeClass,
    mode: str,
    *,
    model: str | None = None,
    llm_url: str | None = None,
    flavour: str | None = None,
    context_window: int | None = None,
) -> Llm:
    """The LLM settings. Default flavour: Ollama when the class has an Ollama model, else vLLM; `llm_url` means a server we do
    not manage (flavour openai unless given). `model` replaces the class's model: the base of the derived model for Ollama,
    the served model name otherwise."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}: {mode!r}")
    flavour = flavour or ("openai" if llm_url else "ollama" if cls.ollama_tag else "vllm")
    if flavour not in FLAVOURS:
        raise ValueError(f"flavour must be one of {FLAVOURS}: {flavour!r}")
    ctx = context_window if context_window is not None else cls.window_tokens
    if ctx <= 0:
        raise ValueError("the context window must be positive")
    if flavour == "ollama":
        if not cls.ollama_tag:
            raise ValueError(f"{cls.name} has no Ollama model: it runs on vLLM (or pass --llm-url)")
        return Llm(
            "ollama",
            llm_url or (NATIVE if mode == "native" else COMPOSE)["ollama"],
            f"lcr-{cls.name.lower()}",
            ctx,
            model or cls.ollama_tag,
        )
    url = COMPOSE["vllm"] if flavour == "vllm" and mode == "compose" and not llm_url else llm_url
    if not url:
        raise ValueError(
            "this flavour needs --llm-url (vLLM runs in compose mode, or on a server you pass with --llm-url)"
        )
    return Llm(flavour, url, model or cls.vllm_model, ctx, None)


def env(
    cls: SizeClass,
    mode: str,
    *,
    token_file: str,
    model: str | None = None,
    llm_url: str | None = None,
    flavour: str | None = None,
    context_window: int | None = None,
) -> dict[str, str]:
    """The environment for the backend and the indexer. One model serves answers and the analysis step."""
    llm = resolve(cls, mode, model=model, llm_url=llm_url, flavour=flavour, context_window=context_window)
    where = NATIVE if mode == "native" else COMPOSE
    return {
        "RAG__LLM__URL": llm.url,
        "RAG__LLM__MODEL": llm.model,
        "RAG__LLM__ANALYSIS_MODEL": llm.model,
        "RAG__LLM__CONTEXT_WINDOW": str(llm.context_window),
        "RAG__LLM__FLAVOUR": llm.flavour,
        "RAG__DATABASE__HOST": where["db_host"],
        "RAG__DATABASE__PORT": where["db_port"],
        "RAG__MODEL_SERVERS__EMBEDDING_URL": where["embedding"],
        "RAG__MODEL_SERVERS__RERANKER_URL": where["reranker"],
        "RAG__HTTP__TOKEN_FILE": token_file,
        **PRIVACY,
    }


def derived_model_body(llm: Llm, *, top_k: int = 50, repeat_penalty: float = 1.05) -> dict | None:
    """The body for `POST /api/create`, or None when this setup has no derived Ollama model (vLLM, a foreign server)."""
    if llm.base_tag is None:
        return None
    return {
        "model": llm.model,
        "from": llm.base_tag,
        "parameters": {"num_ctx": llm.context_window, "top_k": top_k, "repeat_penalty": repeat_penalty},
    }
