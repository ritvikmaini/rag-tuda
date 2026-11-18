"""The apps' HTTP paths as the harness calls them. The one file a change of an HTTP
contract edits."""

# the assistant (compose service backend, port 8000 in its container)
ASK = "/api/chat"
HEALTHZ = "/healthz"
READYZ = "/readyz"
# every route that needs the assistant's token, as (method, path)
TOKEN_ROUTES = (("GET", ASK + "?question=Hallo&conversation_id=harness-auth"),)
# FastAPI ≥ 0.122 answers 401 for a missing bearer token
MISSING_TOKEN = (401, {"detail": "Not authenticated"})
WRONG_TOKEN = (401, {"detail": "Invalid or missing token"})
DOCS = ("/docs", "/redoc", "/openapi.json")
