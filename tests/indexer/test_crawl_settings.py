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


def test_a_connector_context_carries_the_options_the_contact_and_the_profile_folder(settings, profile):
    bare = build.context(settings, "web", None, False)
    assert dict(bare.options) == {
        "sources": "core",
        "resume": False,
        "contact": None,
        "profile_dir": str(Path(profile).parent),
    }
    given = build.context(settings, "web", None, False, {"sources": "full", "resume": True})
    assert (given.options["sources"], given.options["resume"]) == ("full", True)


def test_the_build_gives_every_source_the_options(settings, fakes):
    sources = {name: fakes.Source() for name in ("web", "pdfs", "maps")}
    build.build_corpus(settings, sources, embedding=fakes.Embedding(None), options={"sources": "full", "resume": False})
    assert [s.asked[0].options["sources"] for s in sources.values()] == ["full"] * 3
