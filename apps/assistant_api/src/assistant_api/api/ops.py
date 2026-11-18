"""Operations routes: /healthz and /readyz."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from assistant_api import runtime as runtime_module

router = APIRouter()


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    report = await runtime_module.readiness(request.app.state.runtime)
    return JSONResponse(report, status_code=200 if report["ready"] else 503)
