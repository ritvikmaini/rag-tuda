"""The fakes' logic: which prompt a call carries, the per-test script, the streamed pieces, /tokenize's count, the embedder's
vectors, the reranker's scores and order."""

import json
import math

from fastapi.testclient import TestClient
from hypothesis import given, strategies as st

from rag_testing import llm, scoring, serve

ANALYSIS = [
    {"role": "system", "content": 'Antworte als JSON: {"standalone_question": "..."}'},
    {"role": "user", "content": "Gesprächsverlauf:\nKein Verlauf\n\nNachricht: Wo ist die Garage?"},
]
ANSWER = [
    {"role": "system", "content": "2. Belege jede Aussage mit der Nummer ihrer Quelle."},
    {"role": "user", "content": "# Quellen\n[1] Text\n\n# Frage\nWo ist die Garage?"},
]
DIRECT = [
    {"role": "system", "content": 'Die Nachricht wurde als "about_bot" eingeordnet.'},
    {"role": "user", "content": "Was kannst du?"},
]


def test_every_call_is_classified_by_its_system_message_and_its_question_found():
    assert [llm.purpose(m) for m in (ANALYSIS, ANSWER, DIRECT, [{"role": "user", "content": "x"}])] == [
        "analysis",
        "answer",
        "direct",
        "unknown",
    ]
    assert llm.question(ANALYSIS, "analysis") == llm.question(ANSWER, "answer") == "Wo ist die Garage?"
    assert llm.question(DIRECT, "direct") == "Was kannst du?"


def test_the_first_fitting_rule_answers_and_the_analysis_keeps_its_defaults():
    rules = [
        {"purpose": "answer", "match": "Kantine", "reply": "nein"},
        {"purpose": "analysis", "match": "Garage", "reply": {"intent": "conversation"}},
        {"purpose": "answer", "system": "Belege", "reply": "ja [1]"},
    ]
    assert llm.reply(llm.pick(rules, "answer", ANSWER), "answer", ANSWER) == "ja [1]"
    analysis = json.loads(llm.reply(llm.pick(rules, "analysis", ANALYSIS), "analysis", ANALYSIS))
    assert analysis == {
        "standalone_question": "Wo ist die Garage?",
        "intent": "conversation",
        "search_queries": [],
        "language": "Deutsch",
        "date_from": None,
        "date_to": None,
        "answer_shape": "text",
    }
    assert llm.reply(llm.pick([], "direct", DIRECT), "direct", DIRECT) == llm.DEFAULT_REPLIES["direct"]


@given(st.text(alphabet=st.characters(categories=["L", "N", "P", "Zs"]) | st.sampled_from("\n\t[]"), max_size=200))
def test_the_streamed_pieces_join_to_the_reply(text):
    assert "".join(llm.pieces(text)) == text


def test_tokenize_counts_words_of_a_prompt_or_of_messages_plus_four_each():
    assert llm.count_tokens({"prompt": "eins zwei  drei"}) == 3
    assert (
        llm.count_tokens({"messages": [{"role": "user", "content": "eins zwei"}, {"role": "system", "content": ""}]})
        == 10
    )


def test_a_vector_has_768_unit_dimensions_and_ignores_the_query_prefix():
    one = scoring.vector("Fahrradgarage im Fixturhof")
    assert len(one) == scoring.DIMENSION and math.isclose(sum(v * v for v in one), 1.0)
    assert scoring.vector("query: Fahrradgarage im Fixturhof") == one != scoring.vector("Fahrradgarage")


def test_the_rerank_score_is_the_share_of_query_words():
    assert scoring.score("Wo ist die Garage", "Die Garage ist hinten.") == 0.75


def test_the_fake_reranker_answers_v2_like_the_real_one():
    client = TestClient(serve.reranker_app())
    docs = ["Die Fahrradgarage hat Stellplätze", "Rathaus", "Die Fahrradgarage hat Stellplätze"]
    body = client.post(
        "/v2/rerank", json={"queries": ["Fahrradgarage Stellplätze", "Rathaus"], "documents": docs}
    ).json()
    assert body == {"scores": scoring.scores(["Fahrradgarage Stellplätze", "Rathaus"], docs)}
    assert body["scores"][0] == body["scores"][2] and len(body["scores"]) == 3
    assert client.post("/v2/rerank", json={"queries": [], "documents": docs}).status_code == 422
    assert client.post("/v2/rerank", json={"queries": ["x"], "documents": []}).json() == {"scores": []}
