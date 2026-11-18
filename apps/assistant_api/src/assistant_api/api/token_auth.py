"""Bearer token check: constant-time, and a missing or empty expected token never matches."""

import hmac


def token_matches(given, expected):
    """True only when expected is set and given equals it. Compared as UTF-8 bytes in constant time: hmac.compare_digest
    raises TypeError on non-ASCII str, which would turn a stray header into a 500 instead of a 401."""
    if not expected or given is None:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))
