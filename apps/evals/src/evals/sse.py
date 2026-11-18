"""The backend's stream frames (parse_sse)."""

import json


def parse_sse(buffer):
    """Split a backend SSE buffer into JSON events.

    Frames are 'data: {json}', with or without a blank line after each one; either way each frame is found by decoding
    one JSON object after a 'data:' marker, so both forms read alike.
    Returns (events, rest); rest is an incomplete tail to prepend to the next read:
    an unfinished frame, or up to four characters that may begin a marker split by the read.
    """
    decoder = json.JSONDecoder()
    events, pos = [], 0
    while True:
        start = buffer.find("data:", pos)
        if start < 0:
            return events, buffer[max(pos, len(buffer) - 4) :]
        i = start + 5
        while i < len(buffer) and buffer[i] in " \r\n":
            i += 1
        try:
            obj, end = decoder.raw_decode(buffer, i)
        except json.JSONDecodeError:
            return events, buffer[start:]
        events.append(obj)
        pos = end
