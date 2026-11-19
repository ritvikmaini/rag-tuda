"""Security headers for every response, and CORS headers for the one domain the settings name (none by default)."""

from urllib.parse import urlparse

from fastapi import Request


async def security_headers_and_cors(request: Request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # CORS is off: the page is served by the backend, so it is same-origin. Only the settings' cors_domain (and its subdomains),
    # when set, gets headers.
    origin = request.headers.get("origin")
    runtime = getattr(request.app.state, "runtime", None) if origin else None
    allowed = runtime.settings.overlay.http.cors_domain if runtime is not None else ""
    if origin and allowed:
        domain = urlparse(origin).hostname
        if domain and (domain == allowed or domain.endswith(f".{allowed}")):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"

    return response
