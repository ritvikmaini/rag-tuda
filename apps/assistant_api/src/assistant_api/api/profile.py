"""GET /api/profile: what the page's chips and footer need: the degrees, the programmes with their PO years (roles.yaml) and
the model that answers, with its provider and whether that provider is remote."""

from typing import Any

from fastapi import APIRouter, Depends, Request
from rag_core.domain.filters import degrees
from rag_core.settings.card import REMOTE_PROVIDERS

from assistant_api.api.deps import verify_token

router = APIRouter()


@router.get("/api/profile", dependencies=[Depends(verify_token)])
async def profile(request: Request) -> dict[str, Any]:
    settings = request.app.state.runtime.settings
    roles = settings.roles
    programmes = [
        {
            "id": name,
            "label_de": programme.label_de,
            "label_en": programme.label_en,
            "degree": programme.degree,
            "po_versions": [version.year for version in programme.po_versions],
        }
        for name, programme in (roles.programmes.items() if roles else ())
    ]
    answer = settings.card.answer
    return {
        "degrees": sorted(degrees(roles.model_dump())) if roles else [],
        "programmes": programmes,
        "po_versions": sorted({year for p in programmes for year in p["po_versions"]}, reverse=True),
        "answer": {"model": answer.model, "provider": answer.provider, "remote": answer.provider in REMOTE_PROVIDERS},
    }
