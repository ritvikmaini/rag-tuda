"""What the settings and the build hand every connector besides the folders: the build options (source set, resume), the
crawl contact of the user's environment and the profile's folder."""

from pathlib import Path

from indexer import build
from indexer.settings import load_settings

CI = str(Path(__file__).resolve().parents[2] / "deploy" / "env" / "ci-example.yaml")


def test_the_contact_is_optional_and_comes_from_the_environment(tmp_path, profile):
    environ = {"RAG_PROFILE": profile, "RAG_OVERLAY": CI}
    secrets = str(tmp_path / "secrets")
    assert load_settings(environ, secrets_dir=secrets).overlay.crawl.contact is None
    mine = load_settings({**environ, "RAG__CRAWL__CONTACT": "someone@example.invalid"}, secrets_dir=secrets)
    assert mine.overlay.crawl.contact == "someone@example.invalid"
    assert mine.config_hash() == load_settings(environ, secrets_dir=secrets).config_hash()  # not part of the index


def test_the_routes_come_from_the_overlay_and_are_not_part_of_the_index(tmp_path, profile):
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("crawl: {routes: {www.faculty.test: 'http://minisite:8091'}}\n", encoding="utf-8")
    secrets = str(tmp_path / "secrets")
    plain = load_settings({"RAG_PROFILE": profile, "RAG_OVERLAY": CI}, secrets_dir=secrets)
    routed = load_settings({"RAG_PROFILE": profile, "RAG_OVERLAY": str(overlay)}, secrets_dir=secrets)
    assert (
        routed.overlay.crawl.routes == {"www.faculty.test": "http://minisite:8091"} and not plain.overlay.crawl.routes
    )
    assert build.context(routed, "gesetze", None, False).options["routes"] == routed.overlay.crawl.routes


def test_a_connector_context_carries_the_options_the_contact_and_the_profile_folder(settings, profile):
    bare = build.context(settings, "web", None, False)
    assert dict(bare.options) == {
        "sources": "core",
        "resume": False,
        "contact": None,
        "routes": {},
        "profile_dir": str(Path(profile).parent),
    }
    given = build.context(settings, "web", None, False, {"sources": "full", "resume": True})
    assert (given.options["sources"], given.options["resume"]) == ("full", True)


def test_the_build_gives_every_source_the_options(settings, fakes):
    sources = {name: fakes.Source() for name in ("web", "pdfs", "maps")}
    build.build_corpus(settings, sources, embedding=fakes.Embedding(None), options={"sources": "full", "resume": False})
    assert [s.asked[0].options["sources"] for s in sources.values()] == ["full"] * 3
