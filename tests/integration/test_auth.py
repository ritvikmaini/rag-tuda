"""Access to the assistant: every token route refuses a request without a token
(401, FastAPI's HTTPBearer since 0.122) and with a wrong one (401); the API docs are off; /healthz needs no token."""

import httpx
import pytest
from rag_testing import endpoints, harness


@pytest.mark.parametrize(("method", "path"), endpoints.TOKEN_ROUTES)
def test_a_token_route_refuses_a_missing_and_a_wrong_token(method: str, path: str) -> None:
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        missing = client.request(method, path)
        wrong = client.request(method, path, headers={"Authorization": "Bearer not-the-token"})
    assert (missing.status_code, missing.json()) == endpoints.MISSING_TOKEN
    assert (wrong.status_code, wrong.json()) == endpoints.WRONG_TOKEN


def test_the_api_docs_are_off_and_health_needs_no_token() -> None:
    with httpx.Client(base_url=harness.BACKEND, timeout=30) as client:
        assert [client.get(path).status_code for path in endpoints.DOCS] == [404, 404, 404]
        assert client.get(endpoints.HEALTHZ).json() == {"status": "ok"}
