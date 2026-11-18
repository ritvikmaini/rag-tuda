"""Security headers and CORS for every response; the allowed domain comes from the settings."""

from fastapi import Request


async def security_headers_and_cors(request: Request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # CORS — allow configured BASE_URL domain, localhost, and common local dev origins
    origin = request.headers.get("origin")
    if origin:
        from urllib.parse import urlparse

        parsed = urlparse(origin)
        domain = parsed.hostname

        # Build the allowed-domain list from the settings
        runtime = getattr(request.app.state, "runtime", None)
        allowed_domains = {"localhost", "127.0.0.1"}
        if runtime is not None:
            allowed_domains.add(runtime.settings.overlay.http.cors_domain)

        if domain and any(domain == d or domain.endswith(f".{d}") for d in allowed_domains):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type"

    return response
