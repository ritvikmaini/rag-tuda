"""Who asks and what they may read. Every user may read every document, so there is one principal, anonymous, and a
filter that allows everything; a deployment with audiences replaces the filter (and adds its SQL predicate)."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Principal:
    name: str = "anonymous"


ANONYMOUS = Principal()


class _Filter(Protocol):
    def allows(self, principal: Principal, section: Any) -> bool: ...


class AllowAll:
    def allows(self, principal: Principal, section: object) -> bool:
        return True


def allowed(sections: Mapping[str, T], principal: Principal, access: _Filter) -> dict[str, T]:
    """The sections this principal may read (with AllowAll: every one)."""
    return {sid: s for sid, s in sections.items() if access.allows(principal, s)}
