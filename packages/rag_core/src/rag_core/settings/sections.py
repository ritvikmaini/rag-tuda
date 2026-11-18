"""Overlay (O) sections both apps read: the database and the model servers, plus the assistant's technical sections. The
defaults are the stack's values."""

from pydantic import Field, SecretStr

from rag_core.settings import OverlaySection, secret


class DatabaseSection(OverlaySection):
    host: str = "pgvector"
    port: int = Field(default=5432, gt=0, lt=65536)
    name: SecretStr | None = secret("pgdatabase")
    user: SecretStr | None = secret("vector_user")
    password: SecretStr | None = secret("pg_vector_password")


class ModelServersSection(OverlaySection):
    embedding_url: str = Field(default="http://embedding_service:8001", pattern=r"^https?://\S+[^/]$")
    # the OpenAI-compatible LLM endpoint and the reranker's base URL
    llm_url: str = Field(default="http://vllm:8000/v1", pattern=r"^https?://\S+[^/]$")
    reranker_url: str = Field(default="http://reranker_service:8003", pattern=r"^https?://\S+[^/]$")


# --- the assistant's technical sections ---------------------------------------------------------------------------------
# They come from RAG__<SECTION>__<KEY> in the assistant's compose environment; the overlay file (deploy/env/ci.yaml is one) is the
# indexer's.


class Compute(OverlaySection):
    slots: int = Field(ge=1)  # one GPU pipeline slot
    slot_wait_s: float = Field(gt=0)  # then the busy message


class Timeouts(OverlaySection):
    request_deadline_s: float = Field(gt=0)
    request_idle_s: float = Field(gt=0)
    analysis_s: float = Field(gt=0)
    retrieval_s: float = Field(gt=0)
    parent_lookup_s: float = Field(gt=0)
    reranker_s: float = Field(gt=0)
    llm_read_s: float = Field(gt=0)


class Http(OverlaySection):
    cors_domain: str  # origins on this domain (and its subdomains), localhost and 127.0.0.1 get CORS headers


class Tracing(OverlaySection):
    url: str  # the Langfuse server; empty: tracing is off


class Server(OverlaySection):
    token: SecretStr | None = secret("backend_token")  # the compose secret
