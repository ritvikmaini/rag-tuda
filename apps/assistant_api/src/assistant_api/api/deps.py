"""Token check and rate limit."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from assistant_api.api.token_auth import token_matches

oauth2_scheme = HTTPBearer()


def _rate_limit_key(request: Request) -> str:
    """Per conversation (the page's conversation_id; the client address only as a fallback), so a frontend that relays all
    users through one address cannot throttle everyone."""
    return request.query_params.get("conversation_id") or get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)


def _expected(request: Request) -> str | None:
    """The secret file backend_token (the only source); no
    default token. Outside verify_token, which only calls token_matches (test_runtime_hardening: no comparison inside it)."""
    secret = request.app.state.runtime.settings.overlay.server.token
    return secret.get_secret_value() if secret is not None else None


async def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> None:
    if not token_matches(credentials.credentials, _expected(request)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
