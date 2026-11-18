# pyright: standard
"""
Shared async HTTP client with connection pooling: one httpx.AsyncClient per event loop, created lazily and reused for all
requests in that loop, so keep-alive connections are pooled. Close it with `await close_http_client()` at shutdown.
"""

import asyncio
import httpx
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Store clients per event loop (using loop id as key)
# Clients are explicitly closed in lifespan shutdown via close_http_client()
_http_clients: Dict[int, httpx.AsyncClient] = {}


def _get_client_key() -> int:
    """Get a unique key for the current event loop."""
    try:
        loop = asyncio.get_running_loop()
        return id(loop)
    except RuntimeError:
        # No running loop - this shouldn't happen in async context
        return 0


def get_http_client() -> httpx.AsyncClient:
    """The async HTTP client of the current event loop (one per loop, so no client is shared across loops).
    Defaults: 100 connections, 20 kept alive for 30 s, 30 s timeout, 10 s to connect."""
    key = _get_client_key()

    if key not in _http_clients or _http_clients[key] is None:
        _http_clients[key] = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=30.0),
            timeout=httpx.Timeout(30.0, connect=10.0),
        )
        logger.info(f"Created httpx.AsyncClient for event loop {key}")

    return _http_clients[key]


async def close_http_client():
    """Close the HTTP client of the current event loop; called in the application's shutdown."""
    key = _get_client_key()

    if key in _http_clients and _http_clients[key] is not None:
        await _http_clients[key].aclose()
        del _http_clients[key]
        logger.info(f"Closed httpx.AsyncClient for event loop {key}")
