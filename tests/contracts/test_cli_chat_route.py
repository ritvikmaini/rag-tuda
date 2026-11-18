"""The command line and the backend agree on the chat route (the CLI's own tests install nothing from the workspace)."""

import urllib.parse

from fastapi.routing import APIRoute

from rag_tuda import ask


def test_the_backend_chat_route_accepts_what_the_cli_sends():
    from assistant_api.app import create_app

    route = next(r for r in create_app().routes if isinstance(r, APIRoute) and r.path == "/api/chat")
    accepted = {p.name: p.field_info.is_required() for p in route.dependant.query_params}
    sent = urllib.parse.urlsplit(ask.build_request("http://x", "t", "Frage", "cli-1", formats="v1").full_url)
    assert sent.path == route.path
    assert (
        {n for n, required in accepted.items() if required} <= set(urllib.parse.parse_qs(sent.query)) <= set(accepted)
    )
