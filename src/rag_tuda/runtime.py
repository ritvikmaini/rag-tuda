"""Where things run: the mapping from (card, mode, overrides) to the language model setup, the environment each native process
gets and the request that creates the derived Ollama model. Pure functions; nothing here touches the machine."""

import json
from dataclasses import dataclass
from pathlib import PurePath

from rag_tuda import cards

MODES = ("native", "compose")
REMOTE = ("groq",)  # providers whose endpoint is not on this machine
OLLAMA = "ollama"
SERVICE_URLS = {"ollama": "http://ollama:11434/v1", "vllm": "http://vllm:8000/v1"}  # the compose stack's model servers
PORTS = {"backend": 8700, "embedding": 8701, "reranker": 8702, "postgres": 55432}
LOOPBACK = "127.0.0.1"
PRIVACY = {"HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"}
# rag_core.settings.card.Sampling's defaults (tests/cli/test_runtime.py compares them): a card that omits [sampling]
SAMPLING = {"temperature": 0.1, "top_p": 0.9}
# the assistant's technical sections that have no default in its settings model (the stack's values, deploy/compose.yaml)
BACKEND_SECTIONS = {
    "RAG__COMPUTE__SLOT_WAIT_S": "30",
    "RAG__TIMEOUTS__RETRIEVAL_S": "10",
    "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
    "RAG__TIMEOUTS__RERANKER_S": "15",
    "RAG__HTTP__CORS_DOMAIN": "",  # empty: CORS off, the backend serves the page itself
    "RAG__TRACING__URL": "",  # empty: tracing is off
}


@dataclass(frozen=True)
class Llm:
    card: str
    provider: str  # of the answer endpoint
    window: int
    url: str | None  # RAG__LLM_URL: only for a server the user names with --llm-url
    served: str | None  # RAG__LLM_MODEL: the derived model, or the model name of the user's server
    derived: str | None  # the Ollama model this tool creates (native, an Ollama answer)
    base_tag: str | None  # the Ollama model the derived one is made from
    pulls: tuple[str, ...]  # the Ollama models the card needs on the machine
    answer_url: str  # where the answer endpoint listens
    answer_model: str  # the model name a request to it carries
    ollama_api: str | None  # the Ollama server this tool talks to (native, a card that uses Ollama)
    analysis_derived: str | None = None  # the Ollama model made for an analysis model other than the answer's
    analysis_base: str | None = None
    analysis_window: int = 0

    @property
    def warms(self) -> bool:
        """Whether a warm-up request is sent: the answer model runs on this machine or on a server the user runs."""
        return self.provider not in REMOTE


def resolve(card: dict, mode: str, *, model: str | None = None, llm_url: str | None = None) -> Llm:
    """The language model setup for a loaded card. `model` replaces the Ollama model the derived one is made from (with
    `llm_url`: it names the model of that server); `llm_url` is a server we do not manage."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}: {mode!r}")
    name, answer, analysis = card["name"], card["answer"], card.get("analysis")
    provider = answer["provider"]
    if (llm_url or model) and mode == "compose":
        raise ValueError(
            "--llm-url and --model belong to the native stack: the compose stack serves the card's own model"
        )
    if provider in REMOTE and not analysis:
        raise ValueError(f"{name}: the answer provider {provider} is remote: the card needs its own [analysis] block")
    if provider in REMOTE and mode == "compose":
        raise ValueError(f"{name}: the compose stack runs Ollama or vLLM cards; run a hosted card natively")
    if provider == "vllm" and mode == "native" and not llm_url:
        raise ValueError(f"{name} runs on vLLM: pass its server with --llm-url, or use the compose stack on NVIDIA")
    managed = provider == OLLAMA and mode == "native" and not llm_url
    pulls = [model or answer["model"]] if provider == OLLAMA and not llm_url else []
    if analysis and analysis["provider"] == OLLAMA and not llm_url:
        pulls.append(analysis["model"])
    derived = f"rag-{name}" if managed else None
    served = derived or (model if llm_url else None)
    # an analysis model other than the answer's is made again with its own window; one on the answer's model shares its runner
    own_analysis = bool(
        analysis
        and analysis["provider"] == OLLAMA
        and mode == "native"
        and not llm_url
        and analysis["model"] != answer["model"]
    )
    ollama = answer if provider == OLLAMA else analysis if analysis and analysis["provider"] == OLLAMA else None
    return Llm(
        card=name,
        provider=provider,
        window=cards.window(card),
        url=llm_url,
        served=served,
        derived=derived,
        base_tag=(model or answer["model"]) if managed else None,
        pulls=tuple(dict.fromkeys(pulls)),
        answer_url=llm_url or (SERVICE_URLS.get(provider, answer["url"]) if mode == "compose" else answer["url"]),
        answer_model=served or answer["model"],
        ollama_api=ollama["url"].removesuffix("/v1") if ollama and mode == "native" and not llm_url else None,
        analysis_derived=f"rag-{name}-analysis" if own_analysis else None,
        analysis_base=analysis["model"] if own_analysis and analysis else None,
        analysis_window=min(analysis["context_window"], cards.window_tokens(card)) if own_analysis and analysis else 0,
    )


def parameters(card: dict, window: int) -> dict:
    """The derived model's parameters: the window, the card's sampling (or the backend's defaults) and the card's extras."""
    sampling = card.get("sampling", {})
    return {
        "num_ctx": window,
        "temperature": sampling.get("temperature", SAMPLING["temperature"]),
        "top_p": sampling.get("top_p", SAMPLING["top_p"]),
        **sampling.get("extras", {}),
    }


def derived_model_body(llm: Llm, card: dict) -> dict | None:
    """The body for `POST /api/create`, or None when this setup has no derived Ollama model."""
    if llm.derived is None or llm.base_tag is None:
        return None
    return {"model": llm.derived, "from": llm.base_tag, "parameters": parameters(card, llm.window), "stream": False}


def analysis_model_body(llm: Llm, card: dict) -> dict | None:
    """The body for `POST /api/create` of the analysis model: the card's analysis model with the smaller of its own window and
    the card's request window, or None when the analysis runs on the answer's model or on a server we do not manage."""
    if llm.analysis_derived is None or llm.analysis_base is None:
        return None
    return {
        "model": llm.analysis_derived,
        "from": llm.analysis_base,
        "parameters": parameters(card, llm.analysis_window),
        "stream": False,
    }


def endpoint(name: str) -> str:
    return f"http://{LOOPBACK}:{PORTS[name]}"


def database_env() -> dict[str, str]:
    return {"RAG__DATABASE__HOST": LOOPBACK, "RAG__DATABASE__PORT": str(PORTS["postgres"])}


def backend_env(llm: Llm, *, root: str, secrets: str, profile: str) -> dict[str, str]:
    """The backend process: the card, the secret folder, the model servers and the loopback bind."""
    env = {
        "RAG_PROFILE": str(PurePath(root, "profiles", profile, "profile.yaml")),
        "RAG_SECRETS_DIR": secrets,
        "RAG_CARDS_DIR": str(PurePath(root, "models", "cards")),
        "RAG_PAGE_DIR": str(PurePath(root, "web", "page")),  # the page the backend serves at /
        "RAG__CARD": llm.card,
        **database_env(),
        "RAG__MODEL_SERVERS__EMBEDDING_URL": endpoint("embedding"),
        "RAG__MODEL_SERVERS__RERANKER_URL": endpoint("reranker"),
        **BACKEND_SECTIONS,
        "OPENAI_API_KEY": "not-needed",  # the chat client wants a key; the local servers ignore it
        "GUNICORN_BIND": f"{LOOPBACK}:{PORTS['backend']}",
        "UVICORN_WORKERS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    if llm.url:
        env["RAG__LLM_URL"] = llm.url
    if llm.served:
        env["RAG__LLM_MODEL"] = llm.served
    if llm.analysis_derived:
        env["RAG__LLM_ANALYSIS_MODEL"] = llm.analysis_derived
    return env


def indexer_paths(rag: str) -> dict[str, str]:
    """The indexer's folders under `.rag/`: crawled pages, PDFs and the text extracted from them live in `crawl/`."""
    return {
        "RAG__PATHS__CRAWL_DIR": str(PurePath(rag, "crawl")),
        "RAG__PATHS__PDF_DIR": str(PurePath(rag, "crawl", "pdfs")),
        "RAG__PATHS__EXTRACTED_DIR": str(PurePath(rag, "crawl", "extracted")),
        "RAG__PATHS__STATE_DIR": str(PurePath(rag, "state")),
        "RAG__PATHS__SKIPPED_PDFS_LOG": str(PurePath(rag, "state", "skipped_pdfs_page_limit.json")),
        "RAG__PATHS__USER_SOURCES": str(PurePath(rag, "sources.json")),  # the folders `rag add` registered
    }


def indexer_env(*, root: str, rag: str, secrets: str, profile: str, overlay: str | None) -> dict[str, str]:
    env = {
        "RAG_PROFILE": str(PurePath(root, "profiles", profile, "profile.yaml")),
        "RAG_SECRETS_DIR": secrets,
        **database_env(),
        "RAG__MODEL_SERVERS__EMBEDDING_URL": endpoint("embedding"),
        **indexer_paths(rag),
        "PYTHONUNBUFFERED": "1",
    }
    if overlay:
        env["RAG_OVERLAY"] = overlay
    return env


def migrate_env(secrets: str) -> dict[str, str]:
    return {"RAG_SECRETS_DIR": secrets, "RAG_MIGRATE_HOST": LOOPBACK, "RAG_MIGRATE_PORT": str(PORTS["postgres"])}


def overlay_text(root: str, profile: str, folders: dict[str, str]) -> str | None:
    """The indexer's overlay file (JSON is YAML): where the profile's folder sources read their files; None without any."""
    if not folders:
        return None
    connectors = {name: {"dir": str(PurePath(root, "profiles", profile, folder))} for name, folder in folders.items()}
    return json.dumps({"connectors": connectors}, indent=2) + "\n"


def compose_env_text(
    llm: Llm, *, token: str, roles_dir: str, profile: str = "example", overlay: str | None = None
) -> str:
    """`.rag/compose.env`: what deploy/compose.yaml reads from its env file."""
    values = {
        "RAG__CARD": llm.card,
        "OLLAMA_CONTEXT_LENGTH": str(llm.window),
        "BACKEND_REFRESH_TOKEN": token,
        "RAG_ROLES_DIR": roles_dir,
        "RAG_PROFILE": profile,
    }
    if overlay:
        values["RAG_INDEXER_OVERLAY"] = overlay
    return "".join(f"{key}={value}\n" for key, value in values.items())
