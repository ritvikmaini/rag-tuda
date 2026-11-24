"""The reranker server by import: settings from the environment, the pinned load (fake torch, Hugging Face and
sentence-transformers), the two scoring modes, length-sorted scoring, the rerank endpoint, one GPU job at a time, the client-left stop, the
start-up reference pair, the routes, the fatal CUDA exit and the device choice. No GPU and no model."""

import asyncio
import dataclasses
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import types

import fastapi
import pytest
from fastapi.testclient import TestClient

from reranker_server import main

REV = "d" * 40
SETTINGS = main.Settings("BAAI/bge-reranker-v2-m3", REV)
REF = {"query": "Wann endet die Kernarbeitszeit?", "document": "Die Kernarbeitszeit endet um 15:00 Uhr."}


def overlap(query, doc):
    """the fake cross-encoder: the share of the query's words found in the passage"""
    words = set(query.lower().split())
    return len(words & set(doc.lower().split())) / len(words)


def fake_predict(pairs, batch_size, convert_to_numpy, calls=None, gate=None):
    if calls is not None:
        calls.append([tuple(p) for p in pairs])
    if gate is not None:
        gate.wait(5)
    return [overlap(q, d) for q, d in pairs]


def loaded(**kwargs):
    return main.Loaded(lambda pairs, **kw: fake_predict(pairs, **kw, **kwargs), "float16", "cuda")


def reference_file(tmp_path, score=None, revision=REV):
    path = tmp_path / "reference.json"
    data = {
        "model": SETTINGS.model,
        "revision": revision,
        **REF,
        "score": overlap(REF["query"], REF["document"]) if score is None else score,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def client(tmp_path, settings=SETTINGS, fake=None, **reference):
    app = main.create_app(
        settings, load=lambda s: fake or loaded(), reference_path=reference_file(tmp_path, **reference)
    )
    return TestClient(app)


# --- settings and the pinned load -----------------------------------------------------------------------------------------
def test_settings_come_from_the_environment():
    env = {"RERANKER_MODEL": SETTINGS.model, "RERANKER_REVISION": REV}
    assert main.Settings.from_env(env) == SETTINGS
    assert SETTINGS.max_length == 1024  # the default
    assert main.Settings.from_env({**env, "MAX_LENGTH": "512"}).max_length == 512
    assert main.Settings.from_env({**env, "MAX_LENGTH": ""}) == SETTINGS
    with pytest.raises(ValueError, match="MAX_LENGTH is not a positive integer"):
        main.Settings.from_env({**env, "MAX_LENGTH": "0"})
    assert SETTINGS.mode == "cross_encoder"
    assert main.Settings.from_env({**env, "RERANK_MODE": "qwen_yes_no"}).mode == "qwen_yes_no"
    assert main.Settings.from_env({**env, "RERANK_MODE": ""}) == SETTINGS
    with pytest.raises(ValueError, match="RERANK_MODE=yes_no is not one of cross_encoder, qwen_yes_no"):
        main.Settings.from_env({**env, "RERANK_MODE": "yes_no"})
    assert main.Settings.from_env(
        {**env, "RERANKER_PREDICT_BATCH_SIZE": "8", "MULTI_QUERY_WAIT_S": "1.5"}
    ) == main.Settings(SETTINGS.model, REV, 8, 1.5)
    with pytest.raises(ValueError, match="missing in the environment: RERANKER_REVISION"):
        main.Settings.from_env({"RERANKER_MODEL": SETTINGS.model})
    with pytest.raises(ValueError, match="RERANKER_REVISION is not a 40-character commit hash"):
        main.Settings.from_env({**env, "RERANKER_REVISION": "main"})


@pytest.fixture(autouse=True)
def no_device_override(monkeypatch):
    monkeypatch.delenv("RAG_DEVICE", raising=False)


def fake_libraries(monkeypatch, cuda, half_fails=False, mps=False):
    calls = {}

    class Param:
        def __init__(self, dtype):
            self.dtype = dtype

    class Model:
        def __init__(self, dtype="torch.float32"):
            self.dtype = dtype

        def half(self):
            if half_fails:
                raise RuntimeError("no fp16")
            return Model("torch.float16")

        def to(self, device):
            calls.setdefault("to", []).append((device, self.dtype))
            return self

        def parameters(self):
            return iter([Param(self.dtype)])

    class CrossEncoder:
        def __init__(self, path, max_length, device):
            calls["cross_encoder"] = (path, max_length, device)
            self.model = Model()

        def predict(self, pairs, batch_size, convert_to_numpy):
            return [0.5] * len(pairs)

    def snapshot_download(repo_id, revision, local_files_only):
        calls["snapshot"] = (repo_id, revision, local_files_only)
        return "/hf/hub/snapshots/" + revision

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=CrossEncoder))
    return calls


def test_the_model_loads_from_its_pinned_snapshot_at_the_set_length_in_fp16(monkeypatch):
    calls = fake_libraries(monkeypatch, cuda=True)
    model = main.load_model(SETTINGS)
    assert calls["snapshot"] == (SETTINGS.model, REV, True)
    assert calls["cross_encoder"] == (
        "/hf/hub/snapshots/" + REV,
        1024,
        "cpu",
    )  # s-t 4 would put fp32 weights on the GPU
    assert calls["to"] == [("cuda", "torch.float16")]  # the GPU only ever holds the fp16 weights
    assert (model.dtype, model.device) == ("float16", "cuda")
    calls = fake_libraries(monkeypatch, cuda=True)
    main.load_model(dataclasses.replace(SETTINGS, max_length=512))
    assert calls["cross_encoder"][1] == 512


def test_a_failed_fp16_conversion_keeps_fp32_and_the_cpu_stays_fp32(monkeypatch):
    calls = fake_libraries(monkeypatch, cuda=True, half_fails=True)
    assert main.load_model(SETTINGS).dtype == "float32"
    assert calls["to"] == [("cuda", "torch.float32")]  # fp32 still goes to the GPU
    calls = fake_libraries(monkeypatch, cuda=False)
    assert (main.load_model(SETTINGS).device, main.load_model(SETTINGS).dtype) == ("cpu", "float32")
    assert "to" not in calls


def test_fp16_on_mps(monkeypatch):
    calls = fake_libraries(monkeypatch, cuda=False, mps=True)
    model = main.load_model(SETTINGS)
    assert calls["to"] == [("mps", "torch.float16")] and (model.dtype, model.device) == ("float16", "mps")


# --- the qwen yes/no mode ------------------------------------------------------------------------------------------------
class FakeQwenTokenizer:
    """words are tokens: 0 is "no", 1 is "yes", the rest are numbered as they come; pad ids are 99, on the left"""

    def __init__(self, padding_side="left"):
        self.vocabulary, self.padding_side = {}, padding_side

    def _ids(self, text):
        return [self.vocabulary.setdefault(word, 2 + len(self.vocabulary)) for word in text.split()]

    def convert_tokens_to_ids(self, token):
        return {"no": 0, "yes": 1}[token]

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return self._ids(text)

    def __call__(self, texts, padding, truncation, return_attention_mask, max_length):
        assert (padding, truncation, return_attention_mask) == (False, "longest_first", False)
        return {"input_ids": [self._ids(text)[:max_length] for text in texts]}

    def pad(self, inputs, padding, return_tensors):
        import torch

        assert (padding, return_tensors) == (True, "pt")
        longest = max(map(len, inputs["input_ids"]))
        gap = {
            "left": lambda ids: [99] * (longest - len(ids)) + ids,
            "right": lambda ids: ids + [99] * (longest - len(ids)),
        }
        rows = [gap[self.padding_side](ids) for ids in inputs["input_ids"]]
        mask = [[int(i != 99) for i in row] for row in rows]

        class Batch(dict):
            def to(self, device):
                return self

        return Batch(input_ids=torch.tensor(rows), attention_mask=torch.tensor(mask))


class FakeQwenModel:
    """says yes (logit +6, against 0 for no) when the query's key word is also in the document, else no (-6); a row that
    does not end on a real token (right padding) makes it say no, as a model reading a pad position would be wrong"""

    device = "cpu"

    def __init__(self, key_id):
        self.key_id, self.seen = key_id, []

    def __call__(self, input_ids, attention_mask, logits_to_keep):
        import torch

        self.seen.append((input_ids, attention_mask, logits_to_keep))
        logits = torch.zeros(len(input_ids), 1, 3)
        for row, ids in enumerate(input_ids):
            logits[row, -1, 1] = 6.0 if (ids == self.key_id).sum() >= 2 and ids[-1] != 99 else -6.0
        return types.SimpleNamespace(logits=logits)


def test_qwen_yes_no_scores_in_unit_interval():
    tokenizer = FakeQwenTokenizer()
    model = FakeQwenModel(key_id=tokenizer._ids("urlaub")[0])
    predict = main.qwen_yes_no_predict(tokenizer, model, max_length=64)
    pairs = [
        ("urlaub", "urlaub beantragen bei der personalabteilung"),
        ("urlaub", "die mensa"),
        ("urlaub", "so viel urlaub wie noch nie"),
    ]
    scores = predict(pairs, batch_size=2, convert_to_numpy=True)
    yes_if_relevant = 1 / (1 + math.exp(-6))  # softmax over [no=0, yes=6], taken at yes
    assert scores == pytest.approx([yes_if_relevant, 1 - yes_if_relevant, yes_if_relevant])
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert [len(ids) for ids, _, _ in model.seen] == [2, 1]
    assert len(model.seen) == 2 and all(
        keep == 1 for _, _, keep in model.seen
    )  # batches of 2, last-position logits only


def test_qwen_scores_do_not_depend_on_the_tokenizers_padding_side():
    pairs = [("urlaub", "urlaub beantragen bei der personalabteilung heute noch"), ("urlaub", "urlaub")]
    for side in ("left", "right"):
        tokenizer = FakeQwenTokenizer(padding_side=side)
        predict = main.qwen_yes_no_predict(tokenizer, FakeQwenModel(key_id=tokenizer._ids("urlaub")[0]), max_length=200)
        assert tokenizer.padding_side == "left"
        assert predict(pairs, batch_size=2) == pytest.approx([1 / (1 + math.exp(-6))] * 2), side


def test_the_qwen_prompt_wraps_each_pair_and_pads_on_the_left():
    tokenizer = FakeQwenTokenizer()
    model = FakeQwenModel(key_id=-1)
    predict = main.qwen_yes_no_predict(tokenizer, model, max_length=1000)
    predict([("a", "b"), ("a", "b c d e f")], batch_size=8)
    ((ids, mask, _),) = model.seen
    prefix, suffix = tokenizer._ids(main.QWEN_PREFIX), tokenizer._ids(main.QWEN_SUFFIX)
    short, long = ids.tolist()
    assert long[: len(prefix)] == prefix and long[-len(suffix) :] == suffix
    assert short[-len(suffix) :] == suffix and short[:4] == [99] * 4  # the shorter row is padded on the left
    assert mask.tolist()[0][:4] == [0] * 4 and sum(mask.tolist()[1]) == len(long)
    sent = main.QWEN_PAIR.format(instruction=main.QWEN_INSTRUCTION, query="a", document="b")
    assert sent.startswith("<Instruct>: Given a web search query") and "\n<Query>: a\n<Document>: b" in sent


def test_a_qwen_pair_is_cut_to_the_length_setting_with_the_prompt_kept():
    tokenizer = FakeQwenTokenizer()
    model = FakeQwenModel(key_id=-1)
    prompt = len(tokenizer._ids(main.QWEN_PREFIX)) + len(tokenizer._ids(main.QWEN_SUFFIX))
    main.qwen_yes_no_predict(tokenizer, model, max_length=prompt + 10)([("q", " ".join(["w"] * 50))], batch_size=1)
    assert model.seen[0][0].shape[1] == prompt + 10


def test_qwen_scores_go_through_the_length_sorted_call():
    tokenizer = FakeQwenTokenizer()
    predict = main.qwen_yes_no_predict(tokenizer, FakeQwenModel(key_id=tokenizer._ids("urlaub")[0]), max_length=200)
    pairs = [("urlaub", "x " * 20 + "urlaub"), ("urlaub", "nichts"), ("urlaub", "urlaub")]
    scores = main.length_sorted_predict(predict, pairs, 2)
    assert [round(score) for score in scores] == [1, 0, 1]  # in the input order


def fake_qwen_libraries(monkeypatch, cuda=False, mps=False):
    calls = fake_libraries(monkeypatch, cuda=cuda, mps=mps)
    sys.modules["torch"].float16, sys.modules["torch"].float32 = "torch.float16", "torch.float32"

    class Tokenizer(FakeQwenTokenizer):
        @classmethod
        def from_pretrained(cls, path, padding_side, local_files_only):
            calls["tokenizer"] = (path, padding_side, local_files_only)
            return cls()

    class CausalLM:
        device = "cpu"

        def __init__(self, dtype):
            self.dtype = dtype

        @classmethod
        def from_pretrained(cls, path, torch_dtype, local_files_only):
            calls["causal_lm"] = (path, torch_dtype, local_files_only)
            return cls(torch_dtype)

        def to(self, device):
            calls["moved_to"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

        def parameters(self):
            return iter([types.SimpleNamespace(dtype=self.dtype)])

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForCausalLM=CausalLM)
    )
    return calls


@pytest.mark.parametrize(
    "cuda, mps, device, dtype",
    [(True, False, "cuda", "float16"), (False, True, "mps", "float16"), (False, False, "cpu", "float32")],
)
def test_the_qwen_mode_loads_a_causal_model_in_the_device_dtype(monkeypatch, cuda, mps, device, dtype):
    calls = fake_qwen_libraries(monkeypatch, cuda=cuda, mps=mps)
    model = main.load_model(dataclasses.replace(SETTINGS, mode="qwen_yes_no"))
    path = "/hf/hub/snapshots/" + REV
    assert calls["snapshot"] == (SETTINGS.model, REV, True) and "cross_encoder" not in calls
    assert calls["tokenizer"] == (path, "left", True)  # the answer position must be the last
    assert calls["causal_lm"] == (path, "torch." + dtype, True) and calls["moved_to"] == device and calls["eval"]
    assert (model.dtype, model.device) == (dtype, device)


# --- length-sorted scoring ----------------------------
def test_pairs_are_scored_shortest_first_and_returned_in_input_order():
    calls = []

    def predict(pairs, batch_size, convert_to_numpy):
        calls.append((list(pairs), batch_size, convert_to_numpy))
        return [float(len(q) + len(d)) for q, d in pairs]  # a score that names its pair

    pairs = [("q", "x" * 900), ("q", "y" * 10), ("query", "z" * 300), ("q", "w" * 10)]
    assert main.length_sorted_predict(predict, pairs, 16) == [901.0, 11.0, 305.0, 11.0]  # input order
    ((sent, batch_size, as_numpy),) = calls
    assert [len(q) + len(d) for q, d in sent] == [11, 11, 305, 901] and batch_size == 16 and as_numpy is True
    assert sent[:2] == [("q", "y" * 10), ("q", "w" * 10)]  # ties keep their order


def test_no_pairs_no_call():
    assert main.length_sorted_predict(lambda *a, **k: 1 / 0, [], 32) == []


# --- the rerank endpoint ---------------------------------------------------------------------------------------------------
QUERIES = ["kernarbeitszeit endet", "urlaub beantragen"]
DOCS = ["die kernarbeitszeit endet um 15 uhr", "urlaub beantragen bei der personalabteilung", "die mensa", "die mensa"]


def test_the_old_endpoint_is_gone(tmp_path):
    with client(tmp_path) as c:
        assert c.post("/rerank-multi-query", json={"queries": ["q"], "documents": ["x"]}).status_code == 404


def test_v2_scores_are_aligned_to_the_documents_duplicates_included(tmp_path):
    with client(tmp_path) as c:
        assert c.post("/v2/rerank", json={"queries": QUERIES, "documents": DOCS}).json() == {
            "scores": [1.0, 1.0, 0.0, 0.0]
        }
        assert c.post("/v2/rerank", json={"queries": QUERIES, "documents": []}).json() == {"scores": []}
        r = c.post("/v2/rerank", json={"queries": [], "documents": DOCS})
        assert (r.status_code, r.json()) == (422, {"detail": "queries: at least one query is needed"})


def test_one_query_per_predict_call_with_the_configured_batch_size(tmp_path):
    calls, seen = [], []

    def predict(pairs, batch_size, convert_to_numpy):
        seen.append(batch_size)
        return fake_predict(pairs, batch_size, convert_to_numpy, calls=calls)

    settings = main.Settings(SETTINGS.model, REV, predict_batch_size=8)
    with client(tmp_path, settings=settings, fake=main.Loaded(predict, "float16", "cuda")) as c:
        calls.clear()
        c.post("/v2/rerank", json={"queries": QUERIES, "documents": DOCS[:2]})
    assert [sorted({q for q, _ in call}) for call in calls] == [[QUERIES[0]], [QUERIES[1]]] and set(seen) == {8}


@pytest.mark.parametrize(
    "device, configured, batch_size",
    [("mps", None, 4), ("cuda", None, 32), ("cpu", None, 32), ("mps", 16, 16), ("cuda", 8, 8)],
)
def test_the_batch_size_is_the_devices_own_unless_configured(tmp_path, device, configured, batch_size):
    seen = []

    def predict(pairs, batch_size, convert_to_numpy):
        seen.append(batch_size)
        return fake_predict(pairs, batch_size, convert_to_numpy)

    settings = main.Settings(SETTINGS.model, REV, predict_batch_size=configured)
    with client(tmp_path, settings=settings, fake=main.Loaded(predict, "float16", device)) as c:
        c.post("/v2/rerank", json={"queries": QUERIES[:1], "documents": DOCS[:2]})
    assert seen == [batch_size] * 2  # the start-up reference pair, then the request


def test_a_waiting_request_gets_503_while_the_gpu_job_runs(tmp_path):
    gate, calls, answers = threading.Event(), [], []
    settings = main.Settings(SETTINGS.model, REV, wait_s=0.2)
    fake = main.Loaded(
        lambda pairs, **kw: fake_predict(pairs, **kw, calls=calls, gate=gate if pairs[0][0] == "held" else None),
        "float16",
        "cuda",
    )
    with client(tmp_path, settings=settings, fake=fake) as c:
        held = threading.Thread(
            target=lambda: answers.append(
                c.post("/v2/rerank", json={"queries": ["held"], "documents": ["x"]}).status_code
            )
        )
        held.start()
        deadline = time.monotonic() + 5
        while not any(call[0][0] == "held" for call in calls):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        busy = c.post("/v2/rerank", json={"queries": ["q"], "documents": ["x"]})
        gate.set()
        held.join(5)
    assert (busy.status_code, busy.json()) == (503, {"detail": "Reranker busy."}) and answers == [200]


def test_a_failing_model_answers_500_and_frees_the_gpu_job(tmp_path):
    def predict(pairs, batch_size, convert_to_numpy):
        if pairs[0][0] == "boom":
            raise RuntimeError("CUDA error")
        return [0.25] * len(pairs)

    with client(tmp_path, fake=main.Loaded(predict, "float16", "cuda"), score=0.25) as c:
        r = c.post("/v2/rerank", json={"queries": ["boom"], "documents": ["x"]})
        assert (r.status_code, r.json()) == (500, {"detail": "Reranking inference failed: CUDA error"})
        assert c.post("/v2/rerank", json={"queries": ["fine"], "documents": ["x"]}).json() == {"scores": [0.25]}


def test_a_job_stops_between_queries_when_its_client_has_left():
    calls, answers = [], iter([False, True])

    async def disconnected():
        return next(answers)

    fake = loaded(calls=calls)
    best = asyncio.run(
        main.best_scores(fake, 32, ["a b", "c d", "e f"], ["a", "c"], disconnected, on_fatal=lambda: None)
    )
    assert best == [0.5, 0.0] and len(calls) == 1  # the first query only

    async def gone():
        return True

    assert asyncio.run(main.best_scores(fake, 32, ["a"], ["a"], gone, on_fatal=lambda: None)) is None


def test_v2_answers_no_scores_when_the_client_left_before_the_first_query(tmp_path, monkeypatch):
    calls = []

    async def gone(self):
        return True

    with client(tmp_path, fake=loaded(calls=calls)) as c:
        calls.clear()  # the start-up reference check
        monkeypatch.setattr(fastapi.Request, "is_disconnected", gone)
        r = c.post("/v2/rerank", json={"queries": QUERIES, "documents": DOCS})
    assert (r.status_code, r.json()) == (200, {"scores": []}) and calls == []


# --- readiness, identity, liveness, removed routes -------------------------------------------------------------------------
def test_ready_when_the_reference_pair_scores_as_before(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["difference"] == 0.0


def test_not_ready_when_the_reference_score_moved_or_belongs_to_another_revision(tmp_path):
    with client(tmp_path, score=overlap(REF["query"], REF["document"]) - 0.002) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["difference"] == pytest.approx(0.002)
    with client(tmp_path, revision="e" * 40) as c:
        assert c.get("/readyz").json()["detail"] == "the reference was taken with %s at %s" % (SETTINGS.model, "e" * 40)


def test_the_reference_pair_is_scored_once_never_per_probe(tmp_path):
    calls = []
    with client(tmp_path, fake=loaded(calls=calls)) as c:
        for _ in range(5):
            assert c.get("/readyz").status_code == 200
    assert calls == [[(REF["query"], REF["document"])]]


def test_info_liveness_and_the_routes_that_are_gone(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/info").json() == {
            "model": SETTINGS.model,
            "revision": REV,
            "max_length": 1024,
            "dtype": "float16",
            "device": "cuda",
            "mode": "cross_encoder",
        }
    with client(tmp_path, settings=dataclasses.replace(SETTINGS, max_length=256)) as c:
        assert c.get("/info").json()["max_length"] == 256
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/health").status_code == 404
        gone = [c.post("/rerank", json={"query": "q", "documents": ["x"]}).status_code, c.get("/metrics").status_code]
        assert gone + [c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404] * 5


def test_reference_file_by_device(tmp_path):
    base = reference_file(tmp_path)
    assert main.reference_for(base, "mps") == base  # only the base exists
    own = tmp_path / "reference.mps.json"
    own.write_text("{}", encoding="utf-8")
    assert main.reference_for(base, "mps") == own
    assert main.reference_for(base, "cuda") == base and main.reference_for(base, "cpu") == base


def test_the_start_up_check_reads_the_reference_of_the_device_class(tmp_path):
    reference_file(tmp_path)
    (tmp_path / "reference.mps.json").write_text(
        json.dumps({"model": SETTINGS.model, "revision": REV, **REF, "score": 0.0}), encoding="utf-8"
    )
    for device, status in (("cuda", 200), ("mps", 503)):
        fake = main.Loaded(loaded().predict, "float16", device)
        app = main.create_app(SETTINGS, load=lambda s: fake, reference_path=tmp_path / "reference.json")
        with TestClient(app) as c:
            assert c.get("/readyz").status_code == status


# --- the fatal CUDA exit and the device choice ---------------------------------------------------
def fake_torch_errors(monkeypatch):
    """torch with the two error classes the rule tells apart (torch 2.11: both are RuntimeErrors)"""
    torch = types.ModuleType("torch")
    torch.AcceleratorError = type("AcceleratorError", (RuntimeError,), {})
    torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def failing_client(tmp_path, error, stops):
    def predict(pairs, batch_size, convert_to_numpy):
        if pairs[0][0] == "boom":
            raise error
        return [0.25] * len(pairs)

    app = main.create_app(
        SETTINGS,
        load=lambda s: main.Loaded(predict, "float16", "cuda"),
        reference_path=reference_file(tmp_path, score=0.25),
        stop=lambda: stops.append(1),
    )
    return TestClient(app)


def test_an_accelerator_error_answers_500_reports_not_ready_and_stops_the_process_once(tmp_path, monkeypatch):
    torch, stops = fake_torch_errors(monkeypatch), []
    with failing_client(tmp_path, torch.AcceleratorError("device-side assert"), stops) as c:
        r = c.post("/v2/rerank", json={"queries": ["boom"], "documents": ["x"]})
        assert (r.status_code, r.json()) == (500, {"detail": "Reranking inference failed: device-side assert"})
        ready = c.get("/readyz")
        assert ready.status_code == 503 and ready.json()["status"] == "not ready" and ready.json()["ready"] is False
        assert ready.json()["detail"] == "fatal CUDA error (AcceleratorError): the server restarts"
        assert c.post("/v2/rerank", json={"queries": ["boom"], "documents": ["x"]}).status_code == 500
    assert stops == [1]


def test_other_failures_stay_500_and_ready(tmp_path, monkeypatch):
    torch, stops = fake_torch_errors(monkeypatch), []
    for error in (torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("CUDA error")):
        with failing_client(tmp_path, error, stops) as c:
            assert c.post("/v2/rerank", json={"queries": ["boom"], "documents": ["x"]}).status_code == 500
            assert c.get("/readyz").status_code == 200
    assert stops == []


def test_without_torch_nothing_is_fatal(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert main.is_fatal(RuntimeError("CUDA error")) is False


def test_stop_self_asks_uvicorn_to_stop_at_once_and_ends_the_process_after_the_grace(monkeypatch):
    sent, exited, timers = [], [], []

    class FakeTimer:
        """Records what stop_self arms; fires nothing, the test runs the target by hand."""

        daemon = False  # threading.Timer's default: only a daemon timer lets a clean exit through

        def __init__(self, interval, function, args=()):
            self.interval, self.function, self.args, self.started = interval, function, args, False
            timers.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(main.threading, "Timer", FakeTimer)
    monkeypatch.setattr(main.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(main.os, "_exit", lambda code: exited.append(code))
    main.stop_self()
    assert sent == [(os.getpid(), signal.SIGTERM)] and exited == []  # nothing ends the process at once
    (timer,) = timers
    assert timer.started and timer.daemon is True and timer.interval == main.FATAL_GRACE_S
    timer.function(*timer.args)
    assert exited == [1]


@pytest.mark.parametrize(
    "env, cuda, mps, device",
    [
        ({}, True, True, "cuda"),
        ({}, False, True, "mps"),
        ({}, False, False, "cpu"),
        ({"RAG_DEVICE": ""}, False, True, "mps"),  # empty counts as unset
        ({"RAG_DEVICE": "cpu"}, True, True, "cpu"),  # the setting wins over the order
        ({"RAG_DEVICE": "mps"}, True, True, "mps"),
        ({"RAG_DEVICE": " CUDA "}, True, False, "cuda"),
    ],
)
def test_device_order(env, cuda, mps, device):
    assert main.pick_device(env, cuda, mps) == device


def test_a_device_that_is_not_available_or_not_known_is_an_error_never_a_fallback():
    with pytest.raises(RuntimeError, match="RAG_DEVICE=cuda but cuda is not available"):
        main.pick_device({"RAG_DEVICE": "cuda"}, False, True)
    with pytest.raises(RuntimeError, match="RAG_DEVICE=mps but mps is not available"):
        main.pick_device({"RAG_DEVICE": "mps"}, True, False)
    with pytest.raises(ValueError, match="RAG_DEVICE=tpu is not one of cuda, mps, cpu"):
        main.pick_device({"RAG_DEVICE": "tpu"}, True, True)


def test_a_missing_device_stops_the_load_before_any_download(monkeypatch):
    monkeypatch.setenv("RAG_DEVICE", "cuda")
    calls = fake_libraries(monkeypatch, cuda=False, mps=True)
    with pytest.raises(RuntimeError, match="RAG_DEVICE=cuda"):
        main.load_model(SETTINGS)
    assert "snapshot" not in calls
    calls = fake_libraries(monkeypatch, cuda=True)
    assert main.load_model(SETTINGS).device == "cuda" and "snapshot" in calls


def test_the_device_setting_picks_the_device_the_model_loads_on(monkeypatch):
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    calls = fake_libraries(monkeypatch, cuda=True, mps=True)
    assert main.load_model(SETTINGS).device == "cpu" and "to" not in calls


def test_a_failing_predict_logs_no_query_text(tmp_path, caplog):
    def predict(pairs, batch_size, convert_to_numpy):
        if pairs[0][0].startswith("Personalnummer"):
            raise RuntimeError("CUDA error")
        return [0.25] * len(pairs)

    with client(tmp_path, fake=main.Loaded(predict, "float16", "cuda"), score=0.25) as c:
        with caplog.at_level(logging.INFO):
            r = c.post("/v2/rerank", json={"queries": ["Personalnummer 4711 Beihilfe"], "documents": ["x"]})
    assert r.status_code == 500
    failures = [rec for rec in caplog.records if rec.levelno == logging.ERROR]
    assert failures and all("4711" not in rec.getMessage() + (rec.exc_text or "") for rec in caplog.records)
    assert "(1 documents)" in failures[0].getMessage()


def test_the_real_torch_accelerator_error_is_a_runtime_error_but_not_an_out_of_memory_error():
    import torch  # the real one, from the lock (2.11.0); the fakes above carry its names

    assert issubclass(torch.AcceleratorError, RuntimeError)
    assert not issubclass(torch.AcceleratorError, torch.OutOfMemoryError)
    assert not issubclass(torch.OutOfMemoryError, torch.AcceleratorError)  # an OOM is never fatal
