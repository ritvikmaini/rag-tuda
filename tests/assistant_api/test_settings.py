"""The assistant's settings: the profile holds the effective values, the technical values come from the environment, the
messages are the expected texts, and no source may set a key of another layer."""

from pathlib import Path

import pytest
import yaml

from assistant_api.runtime import Runtime
from assistant_api.settings import load_settings
from rag_core.domain.source_meta import SourceLabels
from rag_core.settings import SettingsError

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "profiles" / "example" / "profile.yaml"

# setting key -> (its path in the profile, its effective value)
EXPECTED = {
    "BOT_NAME": ("identity.bot_name", "Beispiel-Assistent"),
    "COLLECTION_PREFIX": ("collections.0.name", "rag"),
    "EMBED_QUERY_PREFIX": ("retrieval.embedding.query_prefix", "query: "),
    "FTS_LANGUAGE": ("retrieval.fts_language", "german"),
    "PARENT_PLUS_RERANKER_CHILD_VECTOR_COUNT": ("retrieval.semantic_k", 30),
    "BM25_K": ("retrieval.keyword_k", 10),
    "FTS_EXTRA_STOPWORDS": ("retrieval.fts_extra_stopwords", []),
    "MAX_SEARCH_QUERIES": ("retrieval.max_queries", 6),
    "VECTOR_COUNT": ("retrieval.final_k", 14),
    "SEMANTIC_MAX_DISTANCE": ("retrieval.semantic_max_distance", None),
    "CROSS_ENCODER_THRESHOLD": ("retrieval.rerank.threshold", 0.5),
    "RERANK_PARENT_TEXT": ("retrieval.rerank.parent_text", True),
    "MIN_RERANKED_DOCS": ("retrieval.rerank.min_docs", 3),
    "CROSS_ENCODER_RERANKING_MODEL_NAME": ("retrieval.rerank.model", "BAAI/bge-reranker-v2-m3"),
    "RECENCY_TYPES": ("ranking.recency_types", []),
    "RECENCY_BONUS_PER_YEAR": ("ranking.recency_bonus_per_year", 0.01),
    "RECENCY_MAX_BONUS": ("ranking.recency_max_bonus", 0.05),
    "IN_RANGE_BONUS": ("ranking.in_range_bonus", 0.2),
    "IN_RANGE_BY_YEAR": ("ranking.in_range_by_year", True),
    "QUESTION_YEARS": ("ranking.question_years", True),
    "NO_RECENCY_IN_RANGE": ("ranking.no_recency_in_range", True),
    "RELEVANCE_FLOOR": ("ranking.relevance_floor", 0.0),
    "SOURCE_MIN_SCORE": ("ranking.source_min_score", 0.0),
    "MAX_PARENTS_PER_DOC": ("ranking.max_per_document", 0),
    "HISTORY_TURNS": ("history.turns", 2),
    "HISTORY_MAX_TOKENS": ("history.max_tokens", 1200),
    "HISTORY_MAX_AGE_HOURS": ("history.max_age_hours", 12),
    "HISTORY_POSITION": ("history.position", "after_system"),
    "MAX_QUESTION_CHARS": ("guard.max_question_chars", 2000),
    "LLM_CONTEXT_WINDOW": ("guard.context_window", 8192),
    "CONTEXT_SAFETY_MARGIN": ("guard.safety_margin", 128),
    "SOURCE_WINDOW_CHARS": ("guard.source_window_chars", 300),
    "FULL_SOURCES": ("guard.full_sources", 3),
    "MODEL_NAME": ("generation.model", "Qwen/Qwen3-14B-AWQ"),
    "FUNCTION_CALLING_MODEL_NAME": ("generation.analysis_model", "Qwen/Qwen3-14B-AWQ"),
    "RAG_TEMPERATURE": ("generation.temperature", 0.1),
    "MAX_TOKENS": ("generation.max_tokens", 900),
    "ANALYSIS_MAX_TOKENS": ("generation.analysis_max_tokens", 400),
    "DISABLE_THINKING_MODE": ("generation.disable_thinking", True),
    "TOP_K": ("generation.sampling.top_k", 50),
    "TOP_P": ("generation.sampling.top_p", 0.1),
    "REPETITION_PENALTY": ("generation.sampling.repetition_penalty", 1.05),
    "FREQUENCY_PENALTY": ("generation.sampling.frequency_penalty", None),
    "PRESENCE_PENALTY": ("generation.sampling.presence_penalty", None),
    "SOURCES_EVENT": ("formats.sources_event", "before"),
}
# values the profile names beyond that table: the analysis temperature, the date locale and format, the reranker's revision
NAMED = {
    "generation.analysis_temperature": 0,
    "generation.bad_words": [],
    "identity.language": "de",
    "identity.locale": "de_DE",
    "identity.date_format": "%A, der %d.%m.%Y %H:%M:%S",
    "identity.day_names": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
    "retrieval.rerank.revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "retrieval.rerank.scope_fallback_min": None,
}
# the stack's technical values (deploy/compose.yaml's backend environment; a repo test holds the file to them)
STACK = {
    "RAG__DATABASE__HOST": "pgvector",
    "RAG__DATABASE__PORT": "5432",
    "RAG__MODEL_SERVERS__LLM_URL": "http://vllm:8000/v1",
    "RAG__MODEL_SERVERS__EMBEDDING_URL": "http://embedding_service:8001",
    "RAG__MODEL_SERVERS__RERANKER_URL": "http://reranker_service:8003",
    "RAG__COMPUTE__SLOTS": "1",
    "RAG__COMPUTE__SLOT_WAIT_S": "30",
    "RAG__TIMEOUTS__REQUEST_DEADLINE_S": "120",
    "RAG__TIMEOUTS__REQUEST_IDLE_S": "45",
    "RAG__TIMEOUTS__ANALYSIS_S": "20",
    "RAG__TIMEOUTS__RETRIEVAL_S": "10",
    "RAG__TIMEOUTS__PARENT_LOOKUP_S": "5",
    "RAG__TIMEOUTS__RERANKER_S": "15",
    "RAG__TIMEOUTS__LLM_READ_S": "60",
    "RAG__HTTP__CORS_DOMAIN": "example.org",
    "RAG__TRACING__URL": "",
}
MESSAGES = {
    "guard.not_ready": "Der Assistent wird gerade aktualisiert und ist in wenigen Minuten wieder für Sie da.",
    "guard.busy": "Gerade sind sehr viele Anfragen gleichzeitig eingegangen. Bitte versuchen Sie es in einem Moment noch einmal.",
    "guard.error": "Entschuldigung, bei der Beantwortung ist ein technischer Fehler aufgetreten. Bitte versuchen Sie es in einem "
    "Moment noch einmal.",
    "guard.timeout": "Die Antwort dauert gerade ungewöhnlich lange, deshalb habe ich abgebrochen. Bitte stellen Sie Ihre Frage in "
    "ein paar Minuten noch einmal.",
    "variant.refused": "Diese Umwandlung ist nur für die letzte Antwort mit Quellen möglich.",
    "answer.unavailable": "Entschuldigung, der KI-Assistent ist derzeit leider nicht verfügbar. Bitte versuchen Sie es später "
    "erneut.",
    "answer.context_too_long": "Entschuldigung, die Frage und die gefundenen Dokumente waren zusammen zu umfangreich für eine "
    "Antwort. Bitte stellen Sie die Frage kürzer oder beginnen Sie ein neues Gespräch.",
    "docs.none": "Hinweis: Für die Antwort wurden keine Quellen benutzt!",
}


@pytest.fixture
def secrets(tmp_path: Path) -> Path:
    """The database secret files the Database section reads; values are placeholders."""
    for name, value in (
        ("vector_user", "rag_app"),
        ("pg_vector_password", "placeholder"),
        ("pgdatabase", "vectordb"),
    ):
        (tmp_path / name).write_text(value, encoding="utf-8")
    return tmp_path


def environment(**changes: str) -> dict[str, str]:
    env = {"RAG_PROFILE": str(PROFILE), **STACK}
    env.update(changes)
    return {k: v for k, v in env.items() if v != "<unset>"}


def get(obj: object, path: str) -> object:
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)  # type: ignore[index]
    return list(obj) if isinstance(obj, tuple) else obj


def leaves(data: object, prefix: str = "") -> dict[str, object]:
    if isinstance(data, dict):
        out: dict[str, object] = {}
        for key, value in data.items():
            out.update(leaves(value, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: list(data) if isinstance(data, tuple) else data}


def test_the_profile_holds_its_effective_values(secrets: Path) -> None:
    profile = load_settings(environment(), secrets_dir=secrets).profile
    assert {key: get(profile, path) for key, (path, _) in EXPECTED.items()} == {
        key: value for key, (_, value) in EXPECTED.items()
    }
    assert {path: get(profile, path) for path in NAMED} == NAMED


def test_every_assistant_profile_value_is_explained(secrets: Path) -> None:
    profile = load_settings(environment(), secrets_dir=secrets).profile
    sections = ("identity", "ranking", "history", "guard", "generation", "formats")
    found = leaves({name: getattr(profile, name).model_dump() for name in sections})
    found = {k: v for k, v in found.items() if not k.startswith("generation.logit_bias.")}  # token ids, not named keys
    found.update(
        {
            f"retrieval.{k}": v
            for k, v in leaves(profile.retrieval.model_dump(exclude={"embedding", "fts_language"})).items()
        }
    )
    assert set(found) == {
        path
        for path, _ in EXPECTED.values()
        if path.split(".")[0] in sections + ("retrieval",)
        and not path.startswith(("retrieval.embedding", "retrieval.fts_language"))
    } | set(NAMED)


def test_the_environment_supplies_the_technical_values(secrets: Path) -> None:
    overlay = load_settings(environment(), secrets_dir=secrets).overlay
    assert (overlay.compute.slots, overlay.compute.slot_wait_s) == (1, 30)
    assert (
        overlay.timeouts.request_deadline_s,
        overlay.timeouts.request_idle_s,
        overlay.timeouts.analysis_s,
        overlay.timeouts.retrieval_s,
        overlay.timeouts.parent_lookup_s,
        overlay.timeouts.reranker_s,
        overlay.timeouts.llm_read_s,
    ) == (120, 45, 20, 10, 5, 15, 60)
    assert (overlay.model_servers.llm_url, overlay.model_servers.reranker_url) == (
        "http://vllm:8000/v1",
        "http://reranker_service:8003",
    )
    assert (overlay.database.host, overlay.database.port) == ("pgvector", 5432)
    assert (overlay.http.cors_domain, overlay.tracing.url) == ("example.org", "")


def test_a_profile_key_in_the_environment_is_refused(secrets: Path) -> None:
    with pytest.raises(SettingsError, match="retrieval"):
        load_settings(environment(RAG__RETRIEVAL__FINAL_K="10"), secrets_dir=secrets)


def test_a_missing_technical_value_is_refused(secrets: Path) -> None:
    with pytest.raises(SettingsError, match="compute"):
        load_settings(environment(RAG__COMPUTE__SLOTS="<unset>"), secrets_dir=secrets)


def test_the_slot_wait_ends_before_the_silence_limit(secrets: Path) -> None:
    with pytest.raises(SettingsError, match="slot_wait_s"):
        load_settings(environment(RAG__COMPUTE__SLOT_WAIT_S="45"), secrets_dir=secrets)


def test_the_token_comes_from_the_secret_file(secrets: Path) -> None:
    assert load_settings(environment(), secrets_dir=secrets).overlay.server.token is None
    (secrets / "backend_token").write_text("t0k\n", encoding="utf-8")
    token = load_settings(environment(), secrets_dir=secrets).overlay.server.token
    assert token is not None and token.get_secret_value() == "t0k"


def test_the_messages_are_the_expected_texts(secrets: Path) -> None:
    messages = load_settings(environment(), secrets_dir=secrets).messages
    assert {path: get(messages, path) for path in MESSAGES} == MESSAGES
    old = (
        f"Ihre Nachricht ist leider zu lang für mich (höchstens {f'{2000:,}'.replace(',', '.')} Zeichen). Bitte kürzen Sie "
        "Ihre Frage auf das Wesentliche oder stellen Sie mehrere kürzere Fragen nacheinander, dann helfe ich Ihnen gern weiter."
    )
    assert messages.question.too_long.format(limit="2.000") == old
    assert sorted(load_settings(environment(), secrets_dir=secrets).prompts) == [
        "direct-reply",
        "no-docs-found",
        "question-rewriting",
        "rag-generation",
    ]


def test_the_coupled_profile_values_are_checked() -> None:
    from rag_core.settings.profile import Generation, Ranking

    data = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="analysis_model"):
        Generation.model_validate(dict(data["generation"], analysis_model="other"))
    with pytest.raises(ValueError, match="recency_max_bonus"):
        Ranking.model_validate(dict(data["ranking"], recency_bonus_per_year=0.1, recency_max_bonus=0.05))


def test_the_example_messages_carry_the_labels(secrets: Path) -> None:
    messages = load_settings(environment(), secrets_dir=secrets).messages
    assert messages.labels.source_labels() == SourceLabels(
        source_types={},
        origins={"website": "Gesetze im Internet"},
        hosts={},
        scope="Geltungsbereich laut Dokument",
    )


def test_the_runtime_flattens_the_texts_and_leaves_the_labels_out(secrets: Path) -> None:
    rt = Runtime.__new__(Runtime)  # only the settings property reads are needed
    rt.settings = load_settings(environment(), secrets_dir=secrets)
    assert MESSAGES.items() <= rt.messages.items()  # the runtime's table keeps its dict[str, str]
    assert all(isinstance(text, str) and not key.startswith("labels.") for key, text in rt.messages.items())


def test_a_logit_bias_of_more_than_1024_tokens_is_refused(tmp_path: Path) -> None:
    from rag_core.settings import load_profile
    from rag_core.settings.profile import Profile

    source = Path(__file__).resolve().parents[2] / "profiles" / "example" / "profile.yaml"
    profile = yaml.safe_load(source.read_text(encoding="utf-8"))
    file = tmp_path / "profile.yaml"
    profile["generation"]["logit_bias"] = {i: -100 for i in range(1024)}
    file.write_text(yaml.safe_dump(profile), encoding="utf-8")
    assert len(load_profile(Profile, file).generation.logit_bias) == 1024
    profile["generation"]["logit_bias"] = {i: -100 for i in range(1025)}
    file.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(SettingsError, match="1024"):
        load_profile(Profile, file)
