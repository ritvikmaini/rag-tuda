"""The fake servers through FastAPI's test client: the LLM streams as vLLM does and as the OpenAI client reads it, records
every call and answers from the script; the embedder and the reranker keep the services' contracts."""

import json

import pytest
import uvicorn
from fastapi.testclient import TestClient

from rag_testing import llm, serve

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


def test_the_llm_server_streams_like_vllm_and_records_every_call():
    client = TestClient(serve.llm_app())
    client.post("/_script", json={"rules": [{"purpose": "answer", "reply": "Die Garage [1]."}]})
    with client.stream("POST", "/v1/chat/completions", json={"messages": ANSWER, "stream": True}) as resp:
        body = resp.read().decode()
    events = [line[6:] for line in body.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Die Garage [1]."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    plain = client.post("/v1/chat/completions", json={"messages": DIRECT}).json()
    assert plain["choices"][0]["message"]["content"] == llm.DEFAULT_REPLIES["direct"]
    assert [(c["purpose"], c["stream"]) for c in client.get("/_calls").json()] == [("answer", True), ("direct", False)]
    client.post("/_script", json={"rules": [{"purpose": "direct", "fail": 500}]})
    assert client.post("/v1/chat/completions", json={"messages": DIRECT}).status_code == 500
    client.post("/_reset")
    assert client.get("/_calls").json() == []
    assert client.post("/tokenize", json={"prompt": "a b"}).json() == {"count": 2, "max_model_len": 8192}


def test_the_embedder_and_reranker_servers_keep_the_services_contracts():
    vectors = TestClient(serve.embedder_app()).post("/embed", json={"texts": ["a", "b"]}).json()["embeddings"]
    assert [len(v) for v in vectors] == [768, 768]
    client = TestClient(serve.reranker_app())
    ranked = client.post("/v2/rerank", json={"queries": ["a b"], "documents": ["a"]})
    assert ranked.json() == {"scores": [0.5]}
    assert client.post("/rerank-multi-query", json={"queries": ["a b"], "documents": ["a"]}).status_code == 404


@pytest.mark.parametrize("name", ["llm", "embedder", "reranker", "minisite", "minisite-intern"])
def test_every_fake_can_be_named_on_the_command_line(name, monkeypatch):
    started = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: started.append((app, kw["port"])))
    serve.main([name, "--port", "8099"])
    assert started and started[0][1] == 8099
