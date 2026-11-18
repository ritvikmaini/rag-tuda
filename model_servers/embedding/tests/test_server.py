"""The embedding server by import: settings from the environment, the pinned load (fake torch, Hugging Face and
langchain-huggingface), the start-up reference check, the routes, the batch worker and its errors, the fatal CUDA exit and
the device choice. No GPU and no model: every test runs with fakes."""

import dataclasses
import json
import os
import signal
import sys
import threading
import time
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from embedding_server import main

REV = "a" * 40
CODE = "b" * 40
KWARGS = {"use_memory_efficient_attention": False, "unpad_inputs": False}
SETTINGS = main.Settings("Snowflake/snowflake-arctic-embed-m-v2.0", REV, CODE, KWARGS)


def vector(text):
    """a deterministic 4-dimensional vector per text (the fake model)"""
    return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0, float(text.count(" "))]


def fake_loaded(embed=None, calls=None):
    def default(texts):
        if calls is not None:
            calls.append(list(texts))
        return [vector(t) for t in texts]

    return main.Loaded(embed or default, 4, "float16", "cuda", lambda: None, 512)


def reference_file(tmp_path, texts=("query: eins", "zwei drei"), vectors=None, revision=REV):
    path = tmp_path / "reference.json"
    data = {
        "model": SETTINGS.model,
        "revision": revision,
        "texts": list(texts),
        "vectors": vectors if vectors is not None else [vector(t) for t in texts],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def client(tmp_path, loaded=None, **reference):
    app = main.create_app(
        SETTINGS, load=lambda s: loaded or fake_loaded(), reference_path=reference_file(tmp_path, **reference)
    )
    return TestClient(app)


# --- settings -----------------------------------------------------------------------------------------------------------
ENV = {
    "EMBEDDING_MODEL": SETTINGS.model,
    "EMBEDDING_REVISION": REV,
    "EMBEDDING_CODE_REVISION": CODE,
    "EMBEDDING_CONFIG_KWARGS": json.dumps(KWARGS),
}


def test_settings_come_from_the_environment():
    assert main.Settings.from_env(ENV) == SETTINGS
    assert main.Settings.from_env({**ENV, "EMBED_RELEASE_AFTER_TEXTS": "8"}).release_after_texts == 8
    assert (SETTINGS.max_seq_length, SETTINGS.dimensions) == (1024, None)  # the defaults
    sized = main.Settings.from_env({**ENV, "MAX_SEQ_LENGTH": "256", "DIMENSIONS": "1024"})
    assert (sized.max_seq_length, sized.dimensions) == (256, 1024)
    assert main.Settings.from_env({**ENV, "MAX_SEQ_LENGTH": "", "DIMENSIONS": ""}) == SETTINGS


@pytest.mark.parametrize(
    "change, message",
    [
        ({"EMBEDDING_REVISION": ""}, "missing in the environment: EMBEDDING_REVISION"),
        ({"EMBEDDING_REVISION": "main"}, "EMBEDDING_REVISION is not a 40-character commit hash"),
        ({"EMBEDDING_CODE_REVISION": "A" * 40}, "EMBEDDING_CODE_REVISION is not a 40-character commit hash"),
        ({"EMBEDDING_CONFIG_KWARGS": "[]"}, "EMBEDDING_CONFIG_KWARGS is not a JSON object"),
        ({"MAX_SEQ_LENGTH": "0"}, "MAX_SEQ_LENGTH is not a positive integer"),
        ({"DIMENSIONS": "-8"}, "DIMENSIONS is not a positive integer"),
    ],
)
def test_settings_refuse_a_missing_or_unpinned_value(change, message):
    with pytest.raises(ValueError, match=message):
        main.Settings.from_env({**ENV, **change})


# --- the pinned load ------------------------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def no_device_override(monkeypatch):
    monkeypatch.delenv("RAG_DEVICE", raising=False)


def fake_libraries(monkeypatch, cuda, half_fails=False, mps=False, model_max_length=8192):
    calls = {}

    class Param:
        def __init__(self, dtype):
            self.dtype = dtype

    class Client:
        max_seq_length = model_max_length
        dtype = "torch.float32"

        def half(self):
            if half_fails:
                raise RuntimeError("no fp16")
            self.dtype = "torch.float16"
            return self

        def parameters(self):
            return iter([Param(self.dtype)])

        def get_sentence_embedding_dimension(self):
            return 768

    class HuggingFaceEmbeddings:
        def __init__(self, model_name, model_kwargs):
            calls["model_name"], calls["model_kwargs"] = model_name, model_kwargs
            self._client = Client()

        def embed_documents(self, texts):
            return [[3.0, 4.0] + [0.0] * 766 for _ in texts]

    def snapshot_download(repo_id, revision, local_files_only):
        calls["snapshot"] = (repo_id, revision, local_files_only)
        return "/hf/hub/snapshots/" + revision

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: calls.setdefault("emptied", True))
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setitem(
        sys.modules, "langchain_huggingface", types.SimpleNamespace(HuggingFaceEmbeddings=HuggingFaceEmbeddings)
    )
    return calls


def test_the_model_loads_from_its_pinned_snapshot_with_the_pinned_remote_code(monkeypatch):
    calls = fake_libraries(monkeypatch, cuda=True)
    loaded = main.load_model(SETTINGS)
    assert calls["snapshot"] == (SETTINGS.model, REV, True)
    assert calls["model_name"] == "/hf/hub/snapshots/" + REV
    assert calls["model_kwargs"] == {
        "device": "cuda",
        "trust_remote_code": True,
        "local_files_only": True,
        "config_kwargs": {"use_memory_efficient_attention": False, "unpad_inputs": False, "code_revision": CODE},
        "model_kwargs": {"code_revision": CODE},
    }
    assert (loaded.dimension, loaded.dtype, loaded.device) == (768, "float16", "cuda")


def test_the_input_length_is_the_setting_and_never_above_the_models_own(monkeypatch):
    fake_libraries(monkeypatch, cuda=True)
    assert main.load_model(SETTINGS).embed.__self__._client.max_seq_length == 1024  # the model's 8192 capped
    assert (
        main.load_model(dataclasses.replace(SETTINGS, max_seq_length=256)).embed.__self__._client.max_seq_length == 256
    )
    fake_libraries(monkeypatch, cuda=True, model_max_length=512)
    assert main.load_model(SETTINGS).embed.__self__._client.max_seq_length == 512  # the model's own limit holds


def test_mrl_truncates_and_renormalises():
    assert main.truncate_and_renormalise([[3.0, 4.0, 12.0], [0.0, 2.0, 5.0]], 2) == [[0.6, 0.8], [0.0, 1.0]]
    assert main.truncate_and_renormalise([[0.0, 0.0, 1.0]], 2) == [[0.0, 0.0]]  # no division by zero


def test_the_dimensions_setting_cuts_the_models_vectors_and_reports_the_size(monkeypatch):
    fake_libraries(monkeypatch, cuda=True)
    whole = main.load_model(SETTINGS)
    assert whole.dimension == 768 and whole.embed(["x"]) == [[3.0, 4.0] + [0.0] * 766]  # untouched without the setting
    cut = main.load_model(dataclasses.replace(SETTINGS, dimensions=2))
    assert cut.dimension == 2 and cut.embed(["x", "y"]) == [[0.6, 0.8], [0.6, 0.8]]
    assert main.load_model(dataclasses.replace(SETTINGS, dimensions=768)).dimension == 768
    with pytest.raises(ValueError, match="DIMENSIONS=1024 is above the model's 768 dimensions"):
        main.load_model(dataclasses.replace(SETTINGS, dimensions=1024))


def test_a_failed_fp16_conversion_keeps_fp32(monkeypatch):
    fake_libraries(monkeypatch, cuda=True, half_fails=True)
    assert main.load_model(SETTINGS).dtype == "float32"


def test_no_fp16_on_the_cpu(monkeypatch):
    fake_libraries(monkeypatch, cuda=False)
    loaded = main.load_model(SETTINGS)
    assert (loaded.device, loaded.dtype) == ("cpu", "float32")


def test_fp16_on_mps(monkeypatch):
    calls = fake_libraries(monkeypatch, cuda=False, mps=True)
    loaded = main.load_model(SETTINGS)
    assert (loaded.device, loaded.dtype) == ("mps", "float16") and calls["model_kwargs"]["device"] == "mps"


# --- readiness, identity, liveness ----------------------------------------------------------------------------------------
def test_ready_when_the_reference_vectors_come_back(tmp_path):
    with client(tmp_path) as c:
        r = c.get("/readyz")
        assert r.status_code == 200 and r.json()["status"] == "ready" and r.json()["texts"] == 2
        assert r.json()["lowest_cosine"] == pytest.approx(1.0)


def test_not_ready_when_a_vector_moved(tmp_path):
    moved = [vector("query: eins"), [1.0, 0.0, 0.0, 0.0]]
    with client(tmp_path, vectors=moved) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["status"] == "not ready" and r.json()["lowest_cosine"] < 0.9999


def test_not_ready_when_the_vectors_have_another_size_than_the_reference(tmp_path):
    with client(tmp_path, vectors=[[1.0, 2.0], [3.0, 4.0]]) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and "number of dimensions" in r.json()["detail"]


def test_not_ready_when_the_reference_belongs_to_another_revision(tmp_path):
    with client(tmp_path, revision="c" * 40) as c:
        r = c.get("/readyz")
        assert r.status_code == 503 and r.json()["detail"] == "the reference was taken with %s at %s" % (
            SETTINGS.model,
            "c" * 40,
        )


def test_the_reference_is_embedded_once_never_per_probe(tmp_path):
    calls = []
    with client(tmp_path, loaded=fake_loaded(calls=calls)) as c:
        for _ in range(5):
            assert c.get("/readyz").status_code == 200
    assert calls == [["query: eins", "zwei drei"]]


def test_info_names_the_model_its_revisions_and_shape(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/info").json() == {
            "model": SETTINGS.model,
            "revision": REV,
            "code_revision": CODE,
            "dimension": 4,
            "dtype": "float16",
            "device": "cuda",
            "max_seq_length": 512,
        }


def test_info_reports_the_length_in_force_and_the_cut_dimension(tmp_path, monkeypatch):
    for model_limit, setting, served in ((8192, 256, 256), (512, 1024, 512)):  # the model's own limit caps the setting
        fake_libraries(monkeypatch, cuda=True, model_max_length=model_limit)
        settings = dataclasses.replace(SETTINGS, max_seq_length=setting, dimensions=2)
        app = main.create_app(settings, load=main.load_model, reference_path=reference_file(tmp_path))
        with TestClient(
            app
        ) as c:  # not ready (the fake model's vectors are not the reference), /info answers all the same
            info = c.get("/info").json()
        assert (info["max_seq_length"], info["dimension"]) == (served, 2)


def test_liveness_on_healthz_and_no_docs(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/health").status_code == 404
        assert [c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]


def test_metrics_are_the_queue_gauges(tmp_path):
    with client(tmp_path) as c:
        assert c.get("/metrics").json() == {
            "queue_size": 0,
            "max_queue_size": 500,
            "batch_size": 32,
            "batch_timeout_ms": 50.0,
            "device": "cuda",
        }


def test_reference_file_by_device(tmp_path):
    base = reference_file(tmp_path)
    assert main.reference_for(base, "mps") == base  # only the base exists
    own = tmp_path / "reference.mps.json"
    own.write_text("{}", encoding="utf-8")
    assert main.reference_for(base, "mps") == own
    assert main.reference_for(base, "cuda") == base and main.reference_for(base, "cpu") == base


def test_the_start_up_check_reads_the_reference_of_the_device_class(tmp_path):
    reference_file(tmp_path)
    moved = [vector("query: eins"), [1.0, 0.0, 0.0, 0.0]]
    (tmp_path / "reference.mps.json").write_text(
        json.dumps({"model": SETTINGS.model, "revision": REV, "texts": ["query: eins", "zwei drei"], "vectors": moved}),
        encoding="utf-8",
    )
    for device, status in (("cuda", 200), ("mps", 503)):
        loaded = main.Loaded(fake_loaded().embed, 4, "float16", device, lambda: None, 512)
        app = main.create_app(SETTINGS, load=lambda s: loaded, reference_path=tmp_path / "reference.json")
        with TestClient(app) as c:
            assert c.get("/readyz").status_code == status


# --- /embed -----------------------------------------------------------------------------------------------------------------
def test_embed_answers_in_input_order_and_nothing_for_no_text(tmp_path):
    with client(tmp_path) as c:
        assert c.post("/embed", json={"texts": ["b b", "a"]}).json() == {"embeddings": [vector("b b"), vector("a")]}
        assert c.post("/embed", json={"texts": []}).json() == {"embeddings": []}


def wait_until(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_queued_requests_share_one_call_and_each_gets_its_slice(tmp_path):
    release, calls, answers = threading.Event(), [], {}

    def embed(texts):
        calls.append(list(texts))
        if texts == ["x"]:
            release.wait(5)
        return [vector(t) for t in texts]

    with client(tmp_path, loaded=fake_loaded(embed=embed)) as c:
        calls.clear()  # the start-up reference check

        def post(name, texts):
            answers[name] = c.post("/embed", json={"texts": texts}).json()["embeddings"]

        first = threading.Thread(target=post, args=("first", ["x"]))
        first.start()
        wait_until(lambda: calls == [["x"]])
        others = [threading.Thread(target=post, args=(n, t)) for n, t in (("b", ["b1", "b2"]), ("c", ["c1"]))]
        for t in others:
            t.start()
        wait_until(lambda: c.app.state.queue.qsize() == 2)
        release.set()
        for t in [first, *others]:
            t.join(5)
    assert calls[1] in (["b1", "b2", "c1"], ["c1", "b1", "b2"])  # one call for both queued requests
    assert answers == {"first": [vector("x")], "b": [vector("b1"), vector("b2")], "c": [vector("c1")]}


def test_a_failing_model_answers_500_with_its_message(tmp_path):
    def embed(texts):
        if texts == ["boom"]:
            raise RuntimeError("CUDA out of memory")
        return [vector(t) for t in texts]

    with client(tmp_path, loaded=fake_loaded(embed=embed)) as c:
        r = c.post("/embed", json={"texts": ["boom"]})
        assert (r.status_code, r.json()) == (500, {"detail": "CUDA out of memory"})


def test_a_slow_model_answers_504(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULT_TIMEOUT", 0.2)
    release = threading.Event()

    def embed(texts):
        if texts == ["slow"]:
            release.wait(5)
        return [vector(t) for t in texts]

    with client(tmp_path, loaded=fake_loaded(embed=embed)) as c:
        r = c.post("/embed", json={"texts": ["slow"]})
        release.set()
        assert (r.status_code, r.json()) == (504, {"detail": "Embedding request timed out."})


def test_a_full_queue_answers_503(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MAX_QUEUE_SIZE", 1)
    monkeypatch.setattr(main, "QUEUE_PUT_TIMEOUT", 0.1)
    release, calls, codes = threading.Event(), [], []

    def embed(texts):
        calls.append(texts)
        if texts == ["held"]:
            release.wait(5)
        return [vector(t) for t in texts]

    with client(tmp_path, loaded=fake_loaded(embed=embed)) as c:

        def post(text):
            codes.append(c.post("/embed", json={"texts": [text]}).status_code)

        held = threading.Thread(target=post, args=("held",))
        held.start()
        wait_until(lambda: ["held"] in calls)
        queued = threading.Thread(target=post, args=("queued",))
        queued.start()
        wait_until(lambda: c.app.state.queue.qsize() == 1)
        r = c.post("/embed", json={"texts": ["refused"]})
        release.set()
        for t in (held, queued):
            t.join(5)
    assert (r.status_code, r.json()) == (503, {"detail": "Embedding service overloaded. Please retry later."})
    assert sorted(codes) == [200, 200]


def test_a_timed_out_request_leaves_the_rest_of_its_batch_their_vectors(tmp_path, monkeypatch):
    release, calls, answers = threading.Event(), [], {}

    def embed(texts):
        calls.append(list(texts))
        if texts == ["held"]:
            release.wait(5)
        return [vector(t) for t in texts]

    with client(tmp_path, loaded=fake_loaded(embed=embed)) as c:
        calls.clear()  # the start-up reference check

        def post(name, texts):
            r = c.post("/embed", json={"texts": texts})
            answers[name] = (r.status_code, r.json())

        held = threading.Thread(target=post, args=("held", ["held"]))
        held.start()
        wait_until(lambda: calls == [["held"]])
        monkeypatch.setattr(main, "RESULT_TIMEOUT", 0.2)
        post("t", ["t"])  # answers 504; its job stays queued behind the held one
        monkeypatch.setattr(main, "RESULT_TIMEOUT", 60.0)
        others = [threading.Thread(target=post, args=(n, t)) for n, t in (("b", ["b1", "b2"]), ("c", ["c1"]))]
        for t in others:
            t.start()
        wait_until(lambda: c.app.state.queue.qsize() == 3)
        release.set()
        for t in [held, *others]:
            t.join(5)
        post("later", ["l"])
    assert not any(t.is_alive() for t in [held, *others])
    assert answers["held"] == (200, {"embeddings": [vector("held")]})
    assert calls[1] in (["t", "b1", "b2", "c1"], ["t", "c1", "b1", "b2"])  # the timed-out text is still embedded
    assert answers["t"] == (504, {"detail": "Embedding request timed out."})
    assert answers["b"] == (200, {"embeddings": [vector("b1"), vector("b2")]})
    assert answers["c"] == (200, {"embeddings": [vector("c1")]})
    assert answers["later"] == (200, {"embeddings": [vector("l")]})


def test_the_model_embeds_through_embed_documents_which_turns_newlines_into_spaces(monkeypatch):
    seen = []

    class SentenceTransformer:  # langchain-huggingface (real) builds and calls this
        max_seq_length = 512

        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, **kwargs):
            seen.append(list(texts))
            return np.array([vector(t) for t in texts])

        def parameters(self):
            return iter([types.SimpleNamespace(dtype="torch.float32")])

        def get_sentence_embedding_dimension(self):
            return 4

    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=lambda *a, **k: "/m"))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=SentenceTransformer)
    )
    loaded = main.load_model(SETTINGS)
    assert loaded.embed(["Satz eins.\nSatz zwei."]) == [vector("Satz eins. Satz zwei.")]
    assert seen == [["Satz eins. Satz zwei."]]


def test_the_worker_releases_cached_memory_at_the_configured_threshold(tmp_path):
    released = []
    base = fake_loaded()
    loaded = main.Loaded(base.embed, 4, "float16", "cuda", lambda: released.append(True), 512)
    app = main.create_app(
        dataclasses.replace(SETTINGS, release_after_texts=3),
        load=lambda s: loaded,
        reference_path=reference_file(tmp_path),
    )
    with TestClient(app) as c:
        assert c.post("/embed", json={"texts": ["a", "b"]}).status_code == 200
        assert released == []  # 2 texts: below the threshold
        assert c.post("/embed", json={"texts": ["a", "b", "c"]}).status_code == 200
        assert released == [True]


# --- the fatal CUDA exit and the device choice ---------------------------------------------------
def fake_torch_errors(monkeypatch):
    """torch with the two error classes the rule tells apart (torch 2.11: both are RuntimeErrors)"""
    torch = types.ModuleType("torch")
    torch.AcceleratorError = type("AcceleratorError", (RuntimeError,), {})
    torch.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def failing_client(tmp_path, error, stops):
    def embed(texts):
        if texts == ["boom"]:
            raise error
        return [vector(t) for t in texts]

    app = main.create_app(
        SETTINGS,
        load=lambda s: fake_loaded(embed=embed),
        reference_path=reference_file(tmp_path),
        stop=lambda: stops.append(1),
    )
    return TestClient(app)


def test_an_accelerator_error_answers_500_reports_not_ready_and_stops_the_process_once(tmp_path, monkeypatch):
    torch, stops = fake_torch_errors(monkeypatch), []
    with failing_client(tmp_path, torch.AcceleratorError("device-side assert"), stops) as c:
        r = c.post("/embed", json={"texts": ["boom"]})
        assert (r.status_code, r.json()) == (500, {"detail": "device-side assert"})
        ready = c.get("/readyz")
        assert ready.status_code == 503 and ready.json()["status"] == "not ready" and ready.json()["ready"] is False
        assert ready.json()["detail"] == "fatal CUDA error (AcceleratorError): the server restarts"
        assert c.post("/embed", json={"texts": ["boom"]}).status_code == 500
    assert stops == [1]


def test_other_failures_stay_500_and_ready(tmp_path, monkeypatch):
    torch, stops = fake_torch_errors(monkeypatch), []
    for error in (torch.OutOfMemoryError("CUDA out of memory"), RuntimeError("CUDA error")):
        with failing_client(tmp_path, error, stops) as c:
            assert c.post("/embed", json={"texts": ["boom"]}).status_code == 500
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
    fake_libraries(monkeypatch, cuda=True, mps=True)
    assert main.load_model(SETTINGS).device == "cpu"


def test_the_real_torch_accelerator_error_is_a_runtime_error_but_not_an_out_of_memory_error():
    import torch  # the real one, from the lock (2.11.0); the fakes above carry its names

    assert issubclass(torch.AcceleratorError, RuntimeError)
    assert not issubclass(torch.AcceleratorError, torch.OutOfMemoryError)
    assert not issubclass(torch.OutOfMemoryError, torch.AcceleratorError)  # an OOM is never fatal
