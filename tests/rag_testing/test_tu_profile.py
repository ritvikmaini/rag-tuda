"""The harness's copy of the TU profile: only the two hosts the mini-site stands for stay in the allow-list, and everything
else in the profile is the real one."""

from pathlib import Path

from rag_core.settings import load_profile
from rag_core.settings.profile import Profile, Roles, Sources
from rag_testing import tu_profile

REAL = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"


def made(tmp_path: Path) -> Path:
    tu_profile.write(REAL, tmp_path / "profile")
    return tmp_path / "profile"


def test_the_crawl_is_allowed_the_two_hosts_the_minisite_stands_for_and_no_other_and_no_statutes(tmp_path):
    ours, real = load_profile(Sources, made(tmp_path) / "sources.yaml"), load_profile(Sources, REAL / "sources.yaml")
    assert set(ours.hosts) == set(tu_profile.HOSTS) and not ours.laws
    assert ours.hosts == {host: real.hosts[host] for host in tu_profile.HOSTS}  # their entries, as they are


def test_the_roles_are_the_real_ones(tmp_path):
    assert load_profile(Roles, made(tmp_path) / "roles.yaml") == load_profile(Roles, REAL / "roles.yaml")


def test_the_profile_differs_from_the_real_one_by_the_statute_source_and_the_minimum(tmp_path):
    ours, real = load_profile(Profile, made(tmp_path) / "profile.yaml"), load_profile(Profile, REAL / "profile.yaml")
    assert (
        list(ours.ingest.sources) == ["tu_web"]
        and ours.ingest.sources["tu_web"].min_documents == tu_profile.MIN_DOCUMENTS
    )
    expected = real.model_dump()
    expected["ingest"]["sources"] = {
        "tu_web": {**real.ingest.sources["tu_web"].model_dump(), "min_documents": tu_profile.MIN_DOCUMENTS}
    }
    assert ours.model_dump() == expected


def test_the_real_files_are_left_as_they_were(tmp_path):
    before = {p.name: p.read_bytes() for p in REAL.iterdir() if p.is_file()}
    made(tmp_path)
    assert {p.name: p.read_bytes() for p in REAL.iterdir() if p.is_file()} == before
