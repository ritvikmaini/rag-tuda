"""The begin -> write -> verify -> publish -> housekeep flow (build.py), run with fake steps. A failed publish marks the generation failed; once the publish has committed nothing
raises any more, so the caller still writes the stats."""

import asyncio
from types import SimpleNamespace

import pytest

from indexer import build
from indexer.index import generations as gen

TARGET = gen.Target.for_collection("example")
PARENTS = ["p1", "p2"]


class Conn:
    def close(self):
        pass


def flow(monkeypatch, fail=None):
    """write_and_publish over fake steps; fail maps a step name to the exception it raises. Returns (run, calls, seen)."""
    calls, seen = [], {}

    def step(name, result=None):
        def fn(conn, *args, **kwargs):
            calls.append(name)
            seen[name] = (args, kwargs)
            if fail and name in fail:
                raise fail[name]
            return result

        fn.__name__ = name
        return fn

    for name, result in (
        ("begin", (2, None)),
        ("verify", {"children": 4, "parents": 2}),
        ("document_changes", {"added": 0, "changed": 1, "removed": 0}),
        ("publish", (1, 0.2)),
        ("annotate", None),
        ("mark_failed", None),
        ("housekeep", {"dropped_tables": [], "errors": []}),
    ):
        monkeypatch.setattr(gen, name, step(name, result))

    def write(conn, embed, tables, parents, profile, *, generation, cache_key):
        calls.append("write")
        seen["write"] = (embed, tables, parents, profile, generation, cache_key)
        return {"children": 4, "parents": 2, "embeddings": {"cached": 3, "computed": 1}}

    monkeypatch.setattr(build, "write_v2", write)
    monkeypatch.setattr(build.cache, "seed", step("seed", 0))

    def run(settings, swap=True, prov=None):
        return asyncio.run(
            build.write_and_publish(
                settings,
                TARGET,
                lambda application_name: Conn(),
                SimpleNamespace(embed_passages="embed"),
                PARENTS,
                swap=swap,
                build_minutes=1.5,
                prov=prov,
            )
        )

    return run, calls, seen


def test_a_clean_publish_reports_no_warnings_and_records_the_timings(db_settings, monkeypatch):
    run, calls, seen = flow(monkeypatch)
    reply = run(db_settings)
    assert calls == ["begin", "seed", "write", "verify", "document_changes", "publish", "annotate", "housekeep"]
    assert reply["swapped"] is True and reply["previous_generation"] == 1 and "warnings" not in reply
    assert set(build.TIMINGS) <= set(reply) and reply["publish_s"] == 0.2
    timings = seen["annotate"][1]["timings"]
    assert set(timings) == {"build_minutes", "index_s", "publish_s"} and timings["build_minutes"] == 1.5


def test_the_writer_gets_the_parents_the_v2_names_and_the_profile(db_settings, monkeypatch):
    run, _, seen = flow(monkeypatch)
    run(db_settings)
    embed, tables, parents, profile = seen["write"][:4]
    assert embed == "embed" and parents is PARENTS and profile is db_settings.profile
    assert tables == {"children": "example_g0002_chunks", "parents": "example_g0002_sections"}
    assert seen["write"][4] == 2 and seen["write"][5] == (
        "example",
        "Snowflake/snowflake-arctic-embed-m-v2.0",
        "95c2741480856aa9666782eb4afe11959938017f",
        "",
        768,
    )


def test_the_builds_provenance_goes_into_the_generation_row(db_settings, monkeypatch):
    run, _, seen = flow(monkeypatch)
    prov = {"git_commit": "abc", "config_hash": "h", "embedding_model": "m", "embedding_revision": "r"}
    run(db_settings, prov=prov)
    assert seen["begin"][0] == (TARGET, prov)


def test_a_staging_build_is_annotated_and_never_published(db_settings, monkeypatch):
    run, calls, _ = flow(monkeypatch)
    reply = run(db_settings, swap=False)
    assert calls == ["begin", "seed", "write", "verify", "document_changes", "annotate"]
    assert reply["swapped"] is False and reply["staging_table"] == "example_g0002" and "publish_s" not in reply


def test_the_reply_names_the_generation_table_and_the_registry_keeps_the_counts(db_settings, monkeypatch):
    run, _, seen = flow(monkeypatch)
    live = run(db_settings)
    assert live["generation_table"] == "example_g0002" and "staging_table" not in live
    assert live["index_counts"] == {"children": 4, "parents": 2}
    assert seen["annotate"][1]["counts"] == {"children": 4, "parents": 2}
    staged = run(db_settings, swap=False)
    assert staged["staging_table"] == staged["generation_table"] == "example_g0002"
    assert not {"publish_s", "previous_generation", "housekeeping"} & set(staged)


def test_a_failed_publish_marks_the_generation_failed_and_raises(db_settings, monkeypatch):
    error = gen.GenerationError("publish of generation 2 failed, the views are unchanged")
    run, calls, _ = flow(monkeypatch, {"publish": error})
    with pytest.raises(gen.GenerationError, match="the views are unchanged"):
        run(db_settings)
    assert calls == ["begin", "seed", "write", "verify", "document_changes", "publish", "mark_failed"]


def test_a_failed_check_after_the_commit_is_a_published_generation_with_a_warning(db_settings, monkeypatch):
    unchecked = gen.PublishedUnchecked(
        "generation 2 is published, but the check after the commit failed: no row", 1, 0.3
    )
    run, calls, _ = flow(monkeypatch, {"publish": unchecked})
    reply = run(db_settings)
    assert "mark_failed" not in calls and calls[-2:] == ["annotate", "housekeep"]
    assert reply["swapped"] is True and reply["previous_generation"] == 1 and reply["publish_s"] == 0.3
    assert reply["warnings"] == [str(unchecked)]


@pytest.mark.parametrize("step", ["annotate", "housekeep"])
def test_a_failure_after_the_publish_is_reported_not_raised(db_settings, monkeypatch, step):
    run, calls, _ = flow(monkeypatch, {step: RuntimeError("cannot drop table")})
    reply = run(db_settings)
    assert "mark_failed" not in calls and calls[-2:] == ["annotate", "housekeep"]
    assert reply["swapped"] is True and reply["generation"] == 2
    assert reply["warnings"] == [f"{step}: RuntimeError: cannot drop table"]


def test_the_reply_carries_the_embedding_counts(db_settings, monkeypatch):
    run, _, _ = flow(monkeypatch)
    for swap in (True, False):
        assert run(db_settings, swap=swap)["embeddings"] == {"cached": 3, "computed": 1}


def test_the_cache_key_names_the_served_revision(db_settings, monkeypatch):
    run, calls, seen = flow(monkeypatch)
    run(db_settings, prov={"embedding_revision": "a" * 40})
    assert seen["write"][5][2] == "a" * 40 and seen["seed"][0] == (seen["write"][5],)
    run(db_settings, prov=None)  # a server without /info (the harness's fake): the profile's revision
    assert seen["write"][5][2] == "95c2741480856aa9666782eb4afe11959938017f"
    assert calls.index("seed") < calls.index("write")


def test_the_reply_carries_the_document_changes(db_settings, monkeypatch):
    run, calls, _ = flow(monkeypatch)
    for swap in (True, False):
        assert run(db_settings, swap=swap)["changes"] == {"added": 0, "changed": 1, "removed": 0}
    # before the publish: against the live generation
    assert calls.index("document_changes") == calls.index("verify") + 1
