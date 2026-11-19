"""The status groups of the TU profile (bachelor, master, promotion under the students, one group per programme): what a
question says about them in German and English, which group, programme and PO versions a source carries in its typed
columns, and the prompt lines the answer gets from both."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from rag_core.domain.status_groups import (
    COMPARE,
    GENERAL,
    SELF,
    SUBJECT,
    Note,
    Scope,
    compare_labels,
    join_labels,
    label_of_side,
    named_sides,
    note_for,
    question_scope,
    scope_lines,
    source_group,
    source_programme,
    source_versions,
    sources_differ,
    versions_of,
)
from rag_core.settings import load_profile
from rag_core.settings.profile import Roles

TU = Path(__file__).resolve().parents[2] / "profiles" / "tu-informatik"
GROUPS = load_profile(Roles, TU / "roles.yaml").groups


@dataclass(frozen=True)
class Wording:
    scope_compare: str
    scope_differ: str
    scope_versions: str
    version_table: str


TEXTS = Wording("compare: {groups}", "differ", "versions: {versions}", "table")
BACHELOR = {"degree": ["bachelor"], "programme": ["bsc-informatik"], "po_version": [2023]}
BACHELOR_OLD = {"degree": ["bachelor"], "programme": ["bsc-informatik"], "po_version": [2015]}
MASTER = {"degree": ["master"], "programme": ["msc-informatik"], "po_version": [2023]}
GENERAL_PAGE = {"degree": None, "programme": None, "po_version": None}


def scope(question: str, standalone: str = "") -> Scope:
    return question_scope(GROUPS, question, standalone)


def lines(
    scope_=Scope(GENERAL, frozenset(), False),
    asked=(),
    top=(),
    shown=None,
    compare=True,
    differ=True,
    version_table=False,
    conj="sowie",
):
    shown = top if shown is None else shown
    return scope_lines(
        GROUPS, scope_, asked, top, shown, TEXTS, compare=compare, differ=differ, version_table=version_table, conj=conj
    )


def test_master_question_reads_master_group():
    # the group word inside a compound, a generic "student" beside it, and the source side's typed degree agree
    assert scope("Ich bin Masterstudent, wie viele Credits brauche ich?") == Scope(SELF, frozenset({"master"}), False)
    assert scope("Als Masterstudierende: bis wann muss ich mich zurückmelden?").groups == {"master"}
    assert scope("Wie viele Credits hat die Master-Thesis?").kind == SUBJECT
    assert scope("Wann ist die Rückmeldefrist?") == Scope(GENERAL, frozenset(), False)
    assert source_group(MASTER, GROUPS) == "master"
    assert source_programme(MASTER, GROUPS) == "msc-informatik"


def test_source_group_reads_only_a_single_known_column_value():
    assert source_group({"degree": ["bachelor", "master"]}, GROUPS) is None  # several values: applies to several
    assert source_group({"degree": ["doctor"]}, GROUPS) is None  # not a group of the table
    assert source_group({"degree": "promotion"}, GROUPS) == "promotion"  # a bare value counts as one
    assert source_group(GENERAL_PAGE, GROUPS) is None  # NULL: applies to all
    assert source_group(None, GROUPS) is None
    assert source_programme({"programme": ["msc-informatik", "msc-computer-science"]}, GROUPS) is None
    assert source_versions({"po_version": [2023, 2015]}) == (2015, 2023)
    assert source_versions({"po_version": ["2023", True, None, 2015]}) == (2015,)
    assert source_versions(GENERAL_PAGE) == ()


@pytest.mark.parametrize(
    ("question", "kind", "groups"),
    [
        ("As a bachelor student, how many credits do I need?", SELF, {"bachelor"}),
        ("I'm a PhD student, who supervises my thesis?", SELF, {"promotion"}),
        ("I am a master student and need the deadline.", SELF, {"master"}),
        ("We are master students. When is the registration?", SELF, {"master"}),
        ("What is the difference between bachelor and master?", COMPARE, {"bachelor", "master"}),
        ("Does this also apply to master students?", COMPARE, {"master"}),
        ("Do the rules differ for doctoral candidates and master students?", COMPARE, {"promotion", "master"}),
        ("Both bachelor and master students must register.", COMPARE, {"bachelor", "master"}),
        ("Which rules apply to bachelors?", SUBJECT, {"bachelor"}),
        ("This is not for bachelors. How long is the thesis?", GENERAL, set()),
        ("How long is a non-master thesis?", GENERAL, set()),
        ("How long does the registration take?", GENERAL, set()),
    ],
)
def test_english_lead_ins(question, kind, groups):
    read = scope(question)
    assert (read.kind, read.groups) == (kind, frozenset(groups))


def test_english_follow_ups_and_contact_questions():
    assert scope("and for masters?").kind == SELF
    assert scope("What about PhD students?").groups == {"promotion"}
    assert scope("How about the master?", "Who must register for the thesis?").groups == {"master"}
    assert scope("Who is responsible for the exam registration?").contact
    assert scope("Wer ist zuständig für die Anmeldung?").contact
    assert not scope("How many credits does the thesis give?").contact


def test_german_lead_ins_and_the_standalone_question():
    assert scope("Gilt das auch für Master-Studierende?").kind == COMPARE
    assert scope("Was ist der Unterschied zwischen Bachelor und Master?").groups == {"bachelor", "master"}
    assert scope("Sowohl Bachelor als auch Master müssen sich anmelden.").kind == COMPARE
    assert scope("Wir sind Promovierende, wo melden wir uns an?").groups == {"promotion"}
    assert scope("Und für Promovierende?").kind == SELF
    # a group named only by the standalone question counts as the subject; a comparison in either text compares
    assert scope("Wie lange dauert das?", "Wie lange dauert die Master-Thesis?") == Scope(
        SUBJECT, frozenset({"master"}), False
    )
    assert scope("Wie lange dauert das?", "Unterschied Bachelor Master Thesis").kind == COMPARE
    assert scope("Ich bin kein Bachelor-Student, wie lange dauert die Thesis?").groups == frozenset()
    assert question_scope({}, "Ich bin Master", "") == Scope(GENERAL, frozenset(), False)


def test_two_sides_in_one_question_compare():
    read = scope("Gilt das für Promovierende und für Bachelor-Studierende?")
    assert (read.kind, read.groups) == (COMPARE, {"promotion", "bachelor"})
    assert named_sides("Für Master gilt X, für Promovierende Y.", GROUPS) == {"studium", "promotion"}


def test_compare_labels_name_the_asked_groups_only():
    assert compare_labels(GROUPS, {"bachelor", "master"}) == "Bachelor-Studierende sowie Master-Studierende"
    assert compare_labels(GROUPS, {"master", "promotion"}, "and") == "Master-Studierende and Promovierende"
    # one asked group is contrasted with the top group of the other question-side (not a programme)
    assert compare_labels(GROUPS, {"promotion"}) == "Promovierende sowie Studierende"
    assert label_of_side(GROUPS, "studium") == "Studierende"
    assert label_of_side(GROUPS, "nowhere") == "nowhere"
    assert join_labels(["a", "b", "c"], "and") == "a, b and c"


def test_sources_differ_per_axis():
    assert sources_differ([BACHELOR, MASTER], GROUPS)
    assert not sources_differ([BACHELOR, BACHELOR_OLD], GROUPS)  # one programme, two versions: the version axis
    assert sources_differ([MASTER, {"programme": ["msc-computer-science"], "po_version": [2023]}], GROUPS)
    assert not sources_differ([BACHELOR, BACHELOR, GENERAL_PAGE, None], GROUPS)  # NULL rows never count
    # a plain master source beside an M.Sc. Informatik source is no difference: the programme axis has one value
    assert not sources_differ([MASTER, {"degree": ["master"], "po_version": [2023]}], GROUPS)
    # a source valid for several degrees is one source, not a difference
    assert not sources_differ([{"degree": ["bachelor", "master"]}, BACHELOR], GROUPS)
    assert versions_of([BACHELOR_OLD, MASTER, GENERAL_PAGE]) == [2023, 2015]


def test_compare_line_for_two_versions():
    # the question names two versions: the compare line names them, newest first
    assert lines(asked=(2015, 2023)) == ["versions: PO 2023 sowie PO 2015"]
    assert lines(asked=(2015, 2023), compare=False) == []
    assert lines(asked=(2023,)) == []
    # a comparison of groups gets its labels; naming two versions as well does not add a second line
    both = lines(Scope(COMPARE, frozenset({"bachelor", "master"}), False), asked=(2015, 2023))
    assert both == ["compare: Bachelor-Studierende sowie Master-Studierende"]
    assert lines(Scope(COMPARE, frozenset(), False)) == []
    # a question naming no version whose best sources carry two names each version
    assert lines(top=[BACHELOR, BACHELOR_OLD]) == ["versions: PO 2023 sowie PO 2015"]
    assert lines(Scope(SELF, frozenset({"bachelor"}), False), top=[BACHELOR, BACHELOR_OLD]) == [
        "versions: PO 2023 sowie PO 2015"
    ]
    assert lines(top=[BACHELOR, BACHELOR_OLD], differ=False) == []
    # sources of two degrees: the groups line, once; both axes: both lines
    assert lines(top=[BACHELOR, MASTER]) == ["differ"]
    assert lines(top=[BACHELOR_OLD, MASTER]) == ["differ", "versions: PO 2023 sowie PO 2015"]
    # a contact question, a source-free answer and sources that agree get none
    assert lines(Scope(GENERAL, frozenset(), True), top=[BACHELOR, MASTER]) == []
    assert lines() == []
    assert lines(top=[BACHELOR, BACHELOR]) == []
    # the groups line is for general questions; a question about the asker's own group is not told to separate groups
    assert lines(Scope(SELF, frozenset({"master"}), False), top=[BACHELOR, MASTER]) == []
    assert lines(asked=(2015, 2023), conj="and") == ["versions: PO 2023 and PO 2015"]


def test_version_table_only_when_gated():
    two = [BACHELOR, BACHELOR_OLD]
    assert lines(shown=two, differ=False, version_table=False) == []
    assert lines(shown=two, differ=False, version_table=True) == ["table"]
    assert lines(shown=[BACHELOR, GENERAL_PAGE], version_table=True) == []  # one version cited: no table
    assert lines(shown=[], version_table=True) == []
    # the table follows the context's versions, whatever the best sources are; it comes after the other lines
    assert lines(Scope(SELF, frozenset(), False), top=two, shown=two, version_table=True) == [
        "versions: PO 2023 sowie PO 2015",
        "table",
    ]
    assert (
        scope_lines(
            GROUPS,
            Scope(GENERAL, frozenset(), False),
            (),
            (),
            two,
            Wording("", "", "", ""),
            compare=True,
            differ=True,
            version_table=True,
        )
        == []
    )  # a profile without the texts gets no line


def test_note_for_names_the_group_the_documents_are_for():
    asked = Scope(SELF, frozenset({"promotion"}), False)
    assert note_for(GROUPS, asked, ["bachelor", "bachelor"], "Antwort") == Note(
        "other", "Bachelor-Studierende", "Promovierende"
    )
    assert note_for(GROUPS, Scope(SELF, frozenset({"master"}), False), ["bachelor"], "Antwort") is None  # a subgroup
    assert note_for(GROUPS, asked, ["bachelor", None], "Antwort") is None  # a general source among them
    assert note_for(GROUPS, asked, [], "Antwort") is None
    assert note_for(GROUPS, Scope(SELF, frozenset({"promotion"}), True), ["bachelor"], "Antwort") is None  # contact
    assert note_for(GROUPS, Scope(COMPARE, frozenset({"promotion"}), False), ["bachelor"], "Antwort") is None
    general = Scope(GENERAL, frozenset(), False)
    assert note_for(GROUPS, general, ["studierende"], "Antwort") == Note("one", "Studierende", "Promovierende")
    assert note_for(GROUPS, general, ["studierende"], "Auch Promovierende melden sich an.") is None
    assert note_for(GROUPS, general, ["promotion", "studierende"], "Antwort") is None  # two sides cited
