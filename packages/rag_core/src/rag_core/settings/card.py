"""Model cards: one TOML file per model setup holds the model's facts and every workaround the system needs for it. The
card is validated here (the command line reads the same files with tomllib); a key the card leaves out takes its size class's
default, and the context window a request may fill is effective_window()."""

import tomllib
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rag_core.settings import SettingsError

Provider = Literal["ollama", "vllm", "lmstudio", "groq"]
REMOTE_PROVIDERS = frozenset({"groq"})
SizeClass = Literal["S", "M", "L", "XL"]
Reasoning = Literal["off", "low", "medium", "high"]
MAX_LOGIT_BIAS = 1024
DEFAULT_CARDS_DIR = Path("/app/cards")  # where the images carry the shipped cards


class CardModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Endpoint(CardModel):
    provider: Provider
    url: str = Field(pattern=r"^https?://\S+[^/]$")
    model: str = Field(min_length=1)
    context_window: int = Field(gt=0)  # the model's own window


class Limits(CardModel):
    tpm: int = Field(default=0, ge=0)  # tokens per minute; 0 = none (Groq free tier: 8000)


class Budget(CardModel):
    window_tokens: int = Field(gt=0)  # total per request: system + history + sources + question + answer
    answer_tokens: int = Field(gt=0)
    final_k: int = Field(gt=0)
    full_sources: int = Field(ge=0)
    history_turns: int = Field(ge=0)
    history_tokens: int = Field(gt=0)
    max_queries: int = Field(gt=0)
    pre_rerank_k: int = Field(gt=0)
    rerank_queries: int = Field(gt=0)


class Timeouts(CardModel):
    deadline_s: float = Field(default=240, gt=0)
    idle_s: float = Field(default=90, gt=0)
    llm_read_s: float = Field(default=120, gt=0)
    analysis_s: float = Field(default=40, gt=0)
    retries: int = Field(default=1, ge=0)


class ReasoningLevels(CardModel):
    answer: Reasoning = "off"
    analysis: Reasoning = "off"


class Sampling(CardModel):
    temperature: float = Field(default=0.1, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)
    extras: dict[str, Any] = {}  # sent only where the provider accepts them


class Gated(CardModel):
    iterative_retrieval: bool
    version_table: bool


class Quirks(CardModel):
    strip_cjk: bool = False  # drop CJK, kana and Hangul from the streamed answer
    logit_bias: dict[int, float] = {}  # token id -> bias, sent with German answers only
    bad_words: tuple[str, ...] = ()  # vLLM bad_words, sent with German answers only

    @field_validator("logit_bias")
    @classmethod
    def _bias_size(cls, value: dict[int, float]) -> dict[int, float]:
        # vLLM's sampler rejects more than 1024 logit-bias entries inside the engine core, and the rejection kills it
        # (EngineDeadError, the container restarts): such a card never loads
        if len(value) > MAX_LOGIT_BIAS:
            raise ValueError(f"logit_bias has {len(value)} entries; vLLM accepts at most {MAX_LOGIT_BIAS}")
        return value


# the starting values per size class; a card's own value wins key by key
CLASS_DEFAULTS: dict[str, dict[str, dict[str, int | bool]]] = {
    "S": {
        "budget": dict(
            window_tokens=8000,
            answer_tokens=700,
            final_k=8,
            full_sources=3,
            history_turns=2,
            history_tokens=1200,
            max_queries=3,
            pre_rerank_k=20,
            rerank_queries=1,
        ),
        "gated": dict(iterative_retrieval=False, version_table=False),
    },
    "M": {
        "budget": dict(
            window_tokens=16000,
            answer_tokens=1200,
            final_k=14,
            full_sources=6,
            history_turns=4,
            history_tokens=3000,
            max_queries=6,
            pre_rerank_k=30,
            rerank_queries=2,
        ),
        "gated": dict(iterative_retrieval=True, version_table=True),
    },
    "L": {  # the provider's cap is the window
        "budget": dict(
            window_tokens=8000,
            answer_tokens=1000,
            final_k=10,
            full_sources=4,
            history_turns=3,
            history_tokens=1500,
            max_queries=6,
            pre_rerank_k=30,
            rerank_queries=2,
        ),
        "gated": dict(iterative_retrieval=True, version_table=True),
    },
    "XL": {  # full_sources 25 = every source of final_k
        "budget": dict(
            window_tokens=64000,
            answer_tokens=2000,
            final_k=25,
            full_sources=25,
            history_turns=8,
            history_tokens=8000,
            max_queries=8,
            pre_rerank_k=40,
            rerank_queries=2,
        ),
        "gated": dict(iterative_retrieval=True, version_table=True),
    },
}


class Card(CardModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9.-]*$")
    size_class: SizeClass = Field(alias="class")
    structured_output: Literal["json_schema", "json_object", "prompt"]
    chars_per_token: float = Field(gt=0)  # calibrated from the provider's usage; vLLM counts with /tokenize
    prompt_overlay: Literal["small", "large"]
    slots: int = Field(ge=1)  # pipeline slots; more than 1 only where the server batches or the provider allows
    download_gb: float | None = Field(default=None, gt=0)  # what `rag doctor` announces for a local model
    answer: Endpoint
    analysis: Endpoint | None = None  # omitted = the answer's; required when the answer provider is remote
    limits: Limits = Limits()
    budget: Budget
    timeouts: Timeouts = Timeouts()
    reasoning: ReasoningLevels = ReasoningLevels()
    sampling: Sampling = Sampling()
    gated: Gated
    quirks: Quirks = Quirks()

    @model_validator(mode="before")
    @classmethod
    def _class_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        given = cast("dict[str, Any]", data)
        defaults = CLASS_DEFAULTS.get(str(given.get("class", given.get("size_class"))))
        if defaults is None:
            return given  # an unknown class: validation names it
        # a section that is no table (budget = 5) is left to the field validation, which names it
        merged: dict[str, Any] = {}
        for section, values in defaults.items():
            own = given.get(section, {})
            merged[section] = {**values, **own} if isinstance(own, dict) else own
        return {**given, **merged}

    @model_validator(mode="after")
    def _remote_names_its_analysis(self) -> "Card":
        if self.answer.provider in REMOTE_PROVIDERS and self.analysis is None:
            raise ValueError(
                f"the answer provider {self.answer.provider} is remote: the card needs its own analysis block"
            )
        return self

    @property
    def analysis_endpoint(self) -> Endpoint:
        return self.analysis or self.answer


def effective_window(card: Card) -> int:
    """min(budget.window_tokens, the model's window, tokens per minute when limited): the window the context guard receives.
    The guard subtracts the answer tokens and its safety margin itself."""
    windows = [card.budget.window_tokens, card.answer.context_window]
    if card.limits.tpm:
        windows.append(card.limits.tpm)
    return min(windows)


def with_local_url(card: Card, url: str) -> Card:
    """The card with url as the url of each endpoint whose provider is local (ollama, vllm, lmstudio); a remote endpoint keeps
    its own. Cards name the host's address of a local server, a container reaches it by its service name."""

    def swap(endpoint: Endpoint) -> Endpoint:
        if endpoint.provider in REMOTE_PROVIDERS:
            return endpoint
        try:
            return Endpoint.model_validate({**endpoint.model_dump(), "url": url})
        except ValidationError:
            raise SettingsError(
                "RAG__LLM_URL: expected an http(s) URL without a trailing slash (environment)"
            ) from None

    return card.model_copy(update={"answer": swap(card.answer), "analysis": card.analysis and swap(card.analysis)})


def with_local_model(card: Card, model: str) -> Card:
    """The card with model as the model of each local endpoint that serves the card's answer model (an analysis endpoint on the
    same model shares the one loaded runner); an endpoint on another model keeps its own, and a card whose answer is remote is
    returned as it is. The command line names the server's derived model (the answer model with the card's window) this way."""
    answer = card.answer
    if answer.provider in REMOTE_PROVIDERS:
        return card

    def swap(endpoint: Endpoint) -> Endpoint:
        if endpoint.provider in REMOTE_PROVIDERS or endpoint.model != answer.model:
            return endpoint
        try:
            return Endpoint.model_validate({**endpoint.model_dump(), "model": model})
        except ValidationError:
            raise SettingsError("RAG__LLM_MODEL: expected a model name (environment)") from None

    return card.model_copy(update={"answer": swap(answer), "analysis": card.analysis and swap(card.analysis)})


def with_local_analysis_model(card: Card, model: str) -> Card:
    """The card with model as the model of its own local analysis endpoint (the command line names the derived model it makes
    for an analysis model other than the answer's); a card without an analysis block, or with a remote one, is returned as it is."""
    analysis = card.analysis
    if analysis is None or analysis.provider in REMOTE_PROVIDERS:
        return card
    try:
        return card.model_copy(update={"analysis": Endpoint.model_validate({**analysis.model_dump(), "model": model})})
    except ValidationError:
        raise SettingsError("RAG__LLM_ANALYSIS_MODEL: expected a model name (environment)") from None


def load_card(name_or_path: str | Path, cards_dir: str | Path = DEFAULT_CARDS_DIR) -> Card:
    """The card named name_or_path (a file in cards_dir), or the file at that path when it is a Path or ends in .toml; an
    unreadable file or an invalid key raises SettingsError naming the file and the key, never a value."""
    text = str(name_or_path)
    path = (
        Path(name_or_path)
        if isinstance(name_or_path, Path) or text.endswith(".toml")
        else Path(cards_dir) / f"{text}.toml"
    )
    source = f"card {path}"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SettingsError(f"{source}: cannot be read ({exc.strerror})") from None
    except tomllib.TOMLDecodeError as exc:
        raise SettingsError(f"{source}: not valid TOML ({exc})") from None
    try:
        return Card.model_validate(data)
    except ValidationError as exc:
        problems = [
            f"{'.'.join(str(part) for part in error['loc']) or '(root)'}: {error['msg']} ({source})"
            for error in exc.errors(include_url=False, include_input=False)
        ]
        raise SettingsError("; ".join(problems)) from None
