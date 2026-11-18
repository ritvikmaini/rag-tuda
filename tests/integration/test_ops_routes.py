"""The assistant's operations routes through the harness: /healthz and /readyz (there is no /health or
/v1/admin/reload). Black box: no product import."""

import httpx
from rag_testing import endpoints, harness


def test_healthz_answers_and_the_removed_routes_are_gone(tokens: dict[str, str]) -> None:
    auth = {"Authorization": "Bearer " + tokens["BACKEND_REFRESH_TOKEN"]}
    with httpx.Client(base_url=harness.BACKEND, timeout=10) as client:
        assert client.get(endpoints.HEALTHZ).json() == {"status": "ok"}
        gone = (
            client.get("/health"),
            client.post("/v1/admin/reload", headers=auth),
        )
        assert [reply.status_code for reply in gone] == [404, 404]


def test_readyz_reports_the_model_servers(tokens: dict[str, str]) -> None:
    with httpx.Client(base_url=harness.BACKEND, timeout=10) as client:
        ready = client.get(endpoints.READYZ)
        assert ready.status_code == (200 if ready.json()["ready"] else 503)
        assert set(ready.json()["reports"]) == {"embedding", "reranker", "llm"}
