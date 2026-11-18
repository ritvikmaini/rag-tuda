"""The access seams: one anonymous principal, a filter that allows everything; a filter that denies drops the section (the
seam is live). No SQL predicate."""

from dataclasses import dataclass

from rag_core.domain.access import ANONYMOUS, AllowAll, Principal, allowed


@dataclass(frozen=True)
class S:
    section_id: str


class DenyB:
    def allows(self, principal: Principal, section: S) -> bool:
        return section.section_id != "b"


SECTIONS = {sid: S(sid) for sid in ("a", "b")}


def test_the_principal_is_anonymous_and_allow_all_allows_it():
    assert ANONYMOUS == Principal() and ANONYMOUS.name == "anonymous"
    assert allowed(SECTIONS, ANONYMOUS, AllowAll()) == SECTIONS


def test_a_denying_filter_drops_the_section():
    assert list(allowed(SECTIONS, ANONYMOUS, DenyB())) == ["a"]
