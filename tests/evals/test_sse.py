"""Probe runner stream reading: no frame is lost when a read boundary falls anywhere in the stream, also inside a `data:`
marker or a multi-byte UTF-8 character."""

import json

import httpx
from evals import runner as rp
from hypothesis import given, settings
from hypothesis import strategies as st

TRICKY = ["data:", "data: ", "da", "ta: {", "}", "{", "\\", '"', "\n\n", "ä", "€", "😀", "日本", ' data: {"x": 1}']
TYPES = ["chunk", "status", "error"]
FRAMES = [
    ("status", "Suche"),
    ("chunk", "Die Gebühr beträgt 25 € 😀"),
    ("chunk", "x data: y"),
    ("error", "Fehler"),
    ("chunk", "Ende"),
]


def encode(frames, ascii_only=True, terminator=""):
    """The backend's frames: 'data: <json>' per frame, followed by the terminator (none by default)."""
    return "".join(
        "data: " + json.dumps({"type": t, "content": c}, ensure_ascii=ascii_only) + terminator for t, c in frames
    ).encode("utf-8")


def expected(frames):
    return (
        "".join(c for t, c in frames if t == "chunk"),
        [c for t, c in frames if t == "error"],
        [t for t, _ in frames],
    )


def read_through_ask(body, cuts):
    """Serve body in pieces split at the byte offsets in cuts and read it with the runner's own ask()."""
    offsets = sorted({c for c in cuts if 0 < c < len(body)})
    pieces = [body[a:b] for a, b in zip([0] + offsets, offsets + [len(body)])]

    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream; charset=utf-8"}, content=iter(pieces))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        answer, _docs, errors, _first, _total, types = rp.ask(client, "http://backend", "token", "Frage?", "user", 10)
    finally:
        client.close()
    return answer, errors, types


def test_a_single_read_boundary_anywhere_loses_no_frame():
    for ascii_only in (True, False):
        body = encode(FRAMES, ascii_only)
        for cut in range(1, len(body)):
            assert read_through_ask(body, [cut]) == expected(FRAMES), (
                "frame lost with a read boundary at byte %d (ascii_only=%s)" % (cut, ascii_only)
            )


CONTENT = st.one_of(st.text(max_size=30), st.lists(st.sampled_from(TRICKY), max_size=6).map("".join))


@settings(max_examples=300, deadline=None)
@given(
    frames=st.lists(st.tuples(st.sampled_from(TYPES), CONTENT), min_size=1, max_size=10),
    ascii_only=st.booleans(),
    terminator=st.sampled_from(["", "\n\n"]),
    data=st.data(),
)
def test_random_read_sizes_lose_no_frame(frames, ascii_only, terminator, data):
    body = encode(frames, ascii_only, terminator)
    cuts = data.draw(st.lists(st.integers(min_value=1, max_value=max(1, len(body) - 1)), max_size=40))
    assert read_through_ask(body, cuts) == expected(frames)
