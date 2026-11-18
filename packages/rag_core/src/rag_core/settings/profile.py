"""The profile's sections: what decides the index, then the assistant's sections and keys. Every model forbids unknown
keys, except a source entry, whose connector-specific keys the connector validates."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_core.settings import ProfileModel

Audience = Literal["internal", "public"]


class Collection(ProfileModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,40}$")
    audiences: list[Audience] = Field(min_length=1)


class Embedding(ProfileModel):
    model: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    dimensions: int = Field(gt=0)
    query_prefix: str
    passage_prefix: str


class Rerank(ProfileModel):
    model: str
    revision: str
    threshold: float = Field(ge=0.0, le=1.0)  # 0.50 with parent_text, 0.15 without
    parent_text: bool
    min_docs: int = Field(ge=0)
    # An out-of-scope question whose best passage (top-1 cross_encoder_score) reaches this is answered from the documents;
    # null = off, the question is declined without a search
    scope_fallback_min: float | None = Field(default=None, ge=0.0, le=1.0)


class Retrieval(ProfileModel):
    embedding: Embedding
    fts_language: str
    semantic_k: int = Field(gt=0)
    keyword_k: int = Field(gt=0)
    fts_extra_stopwords: tuple[str, ...]
    max_queries: int = Field(gt=0)
    final_k: int = Field(gt=0)
    semantic_max_distance: float | None
    rerank: Rerank


class Chunking(ProfileModel):
    mode: Literal["recursive", "structure"]
    parent_chars: int = Field(gt=0)
    parent_overlap: int = Field(ge=0)
    child_chars: int = Field(gt=0)
    child_overlap: int = Field(ge=0)
    children: Literal["recursive", "sentences"]
    header: Literal["title", "tier1"]


class CollectionIngest(ProfileModel):
    min_build_ratio: float = Field(ge=0, le=1)


class Extraction(ProfileModel):
    processing: Literal["fast", "full"]
    max_ocr_pages: int = Field(ge=0)
    ocr_pages_per_run: int = Field(ge=0)
    clean_markup: bool
    min_text_length: int = Field(ge=0)


class ScopeVocabulary(ProfileModel):
    """ingest.scope: the words that make a document's own scope statement a limit for the source header.
    Stems are lowercase; the matcher folds the text."""

    groups: dict[
        str, tuple[str, ...]
    ]  # label -> stems naming a person group (the addressee, a statement, the page-1 heading)
    areas: tuple[str, ...]  # stems naming an employer or area: in the addressee or a statement only, never from a title
    compilations: tuple[str, ...]  # title stems of documents that bundle acts: no statement is read in them

    @model_validator(mode="after")
    def _lowercase_stems(self) -> "ScopeVocabulary":
        stems = [s for group in self.groups.values() for s in group] + list(self.areas) + list(self.compilations)
        if any(not group for group in self.groups.values()) or any(
            not s or s != s.lower() or s != s.strip() for s in stems
        ):
            raise ValueError("ingest.scope: every group has stems, and every stem is lowercase, trimmed and non-empty")
        return self


class Source(BaseModel):
    """One entry of ingest.sources: the generic keys; the connector's own keys stay in model_extra for the connector."""

    model_config = ConfigDict(extra="allow", frozen=True)
    connector: str
    audience: Audience
    min_documents: int = Field(default=0, ge=0)
    min_doc_ratio: float | None = Field(default=None, ge=0, le=1)


class Ingest(ProfileModel):
    chunking: Chunking
    collections: dict[str, CollectionIngest]
    extraction: Extraction
    scope: ScopeVocabulary
    sources: dict[str, Source]  # in build order


# --- the assistant's sections ----------------------------------------------------------------------------------------
# Every key is required (an answer-affecting value has no code default); a nullable key is written as null.


class Identity(ProfileModel):
    bot_name: str
    language: str
    locale: str
    date_format: str  # strftime format; %A is the weekday's name from day_names (rag_core.domain.dates)
    day_names: tuple[str, str, str, str, str, str, str]  # Monday first

    @field_validator("day_names")
    @classmethod
    def _plain_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if any("%" in name for name in names):
            raise ValueError("a day name may not contain %")
        return names


class Ranking(ProfileModel):
    recency_types: list[str] = []  # document types whose newer sources win; none by default
    recency_bonus_per_year: float = Field(ge=0.0)
    recency_max_bonus: float = Field(ge=0.0)
    in_range_bonus: float
    in_range_by_year: bool
    question_years: bool
    no_recency_in_range: bool
    relevance_floor: float
    source_min_score: float
    max_per_document: int = Field(ge=0)  # 0 = no cap

    @model_validator(mode="after")
    def _coupled(self) -> "Ranking":
        if self.recency_max_bonus < self.recency_bonus_per_year:
            raise ValueError("recency_max_bonus must be at least recency_bonus_per_year")
        return self


class History(ProfileModel):
    turns: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    max_age_hours: float = Field(ge=0.0)  # 0 = no limit
    position: Literal["after_system", "first"]  # "first" puts the turns before the system message


class Guard(ProfileModel):
    max_question_chars: int = Field(gt=0)  # = the page's QUESTION_LIMIT
    context_window: int = Field(gt=0)  # = vLLM's --max-model-len; /tokenize's max_model_len wins at run time
    safety_margin: int = Field(ge=0)
    source_window_chars: int = Field(ge=0)
    full_sources: int = Field(ge=0)


class Sampling(ProfileModel):
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    frequency_penalty: float | None
    presence_penalty: float | None


MAX_LOGIT_BIAS = 1024


class Generation(ProfileModel):
    model: str
    analysis_model: str
    temperature: float
    analysis_temperature: float
    max_tokens: int = Field(gt=0)
    analysis_max_tokens: int = Field(gt=0)
    disable_thinking: bool  # Qwen3's chat_template_kwargs.enable_thinking = false
    sampling: Sampling
    logit_bias: dict[int, float] = {}  # token id -> bias, sent with German answers only
    bad_words: tuple[str, ...] = ()  # vLLM bad_words, sent with German answers only

    @field_validator("logit_bias")
    @classmethod
    def _bias_size(cls, value: dict[int, float]) -> dict[int, float]:
        # vLLM's sampler rejects more than 1024 logit-bias entries inside the engine core, and the rejection kills it
        # (EngineDeadError, the container restarts): such a profile never loads
        if len(value) > MAX_LOGIT_BIAS:
            raise ValueError(f"logit_bias has {len(value)} entries; vLLM accepts at most {MAX_LOGIT_BIAS}")
        return value

    @model_validator(mode="after")
    def _one_llm(self) -> "Generation":
        if self.analysis_model != self.model:
            raise ValueError("analysis_model must be the answer model (one LLM serves the stack)")
        return self


class Formats(ProfileModel):
    sources_event: Literal["before", "seen", "cited"]


class StatusGroup(ProfileModel):
    """One status group (a row of the group table); the words are lowercase."""

    label: str  # the group's name in the answer texts
    side: str  # the groups of one side are contrasted with those of the other sides, never a subgroup with its parent
    part_of: str | None = None  # the parent group of a subgroup (a key of the table)
    ask: tuple[str, ...] = ()  # question words: at a word start or the end of a compound ("Verwaltungsbeamtin")
    ask_self: tuple[str, ...] = ()  # question words that count only after "als", "ich bin", ... (ambiguous words)
    source: tuple[
        str, ...
    ] = ()  # regular expressions, matched at a word start in a source's scope, path, title, file name


class StatusGroups(ProfileModel):
    """How an answer tells status groups (Beamte with Anwärter, Tarifbeschäftigte with Auszubildende) apart."""

    groups: dict[str, StatusGroup] = {}
    filter: bool = (
        False  # a question naming its own group drops the sources of the other sides (when any source remains)
    )
    compare: bool = False  # a question comparing groups gets a prompt line asking for each group's rule
    # a general question whose best sources carry two sides gets a prompt line to separate the groups where they differ
    reminder: Literal["off", "differ"] = "off"
    # a German answer cited only from one side's sources gets a closing note on whom the rule is for
    note: Literal["off", "classic"] = "off"
    # only the best-ranked context sources count for the reminder: a far-down source of the other side does not
    # make a process question group-specific
    reminder_sources: int = 5
    overrides: dict[str, str] = {}  # title substring (any case) -> the group key it is for, or "alle" for none

    @field_validator("reminder", "note", mode="before")
    @classmethod
    def _unquoted_off(cls, value: object) -> object:
        return "off" if value is False else value  # YAML reads a bare off as false

    @model_validator(mode="after")
    def _table_is_consistent(self) -> "StatusGroups":
        for key, group in self.groups.items():
            parent = self.groups.get(group.part_of) if group.part_of else None
            if group.part_of and (parent is None or parent.side != group.side or parent.part_of):
                raise ValueError(f"status group {key}: part_of must name a top group of the same side")
        for part, key in self.overrides.items():
            if key != "alle" and key not in self.groups:
                raise ValueError(f"status group override {part!r}: unknown group {key!r}")
        return self


class Profile(ProfileModel):
    collections: list[Collection] = Field(min_length=1)
    retrieval: Retrieval
    ingest: Ingest
    identity: Identity
    ranking: Ranking
    history: History
    guard: Guard
    generation: Generation
    formats: Formats
    status_groups: StatusGroups = StatusGroups()
