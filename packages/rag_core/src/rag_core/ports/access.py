"""The access filter port."""

from typing import Protocol

from rag_core.domain.access import Principal
from rag_core.domain.retrieval import Section


class AccessFilter(Protocol):
    def allows(self, principal: Principal, section: Section) -> bool: ...
