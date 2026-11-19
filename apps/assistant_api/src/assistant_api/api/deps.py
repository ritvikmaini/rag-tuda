"""Token check and rate limit."""

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address

from assistant_api.api.token_auth import token_matches

oauth2_scheme = HTTPBearer()


def _rate_limit_key(request: Request) -> str:
    """The client address: a caller cannot win a fresh allowance by inventing conversation ids."""
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key)
_chat_rate = "20/minute"


def set_chat_rate(rate: str) -> None:
    """The questions one address may ask per period (settings: server.rate_limit), set by the lifespan once the settings are read."""
    global _chat_rate
    _chat_rate = rate


def chat_rate() -> str:
    return _chat_rate


def _expected(request: Request) -> str | None:
    """The secret file backend_token (the only source); no
    default token. Outside verify_token, which only calls token_matches (test_runtime_hardening: no comparison inside it)."""
    secret = request.app.state.runtime.settings.overlay.server.token
    return secret.get_secret_value() if secret is not None else None


async def verify_token(request: Request, credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)) -> None:
    if not token_matches(credentials.credentials, _expected(request)):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing token")
