"""The crawl cache: `<crawl_dir>/<host>/` holds, per URL, the raw body and an entry beside it with the fetch date, the
`ETag` and `Last-Modified` the server sent, the content type, the URL the body came from and the SHA-256 of the body.

The body file is named after its hash (`<key>-<hash>.body`, `.pdf` for a PDF), so an entry always points at one complete
file: a body is written to a `.part` file and renamed, then the entry is replaced the same way, then the previous body is
removed. A crash between the steps leaves the previous entry and its file in use. An entry whose file is missing, or whose
JSON cannot be read, is a miss."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["Cache", "Entry", "write_atomic"]


def write_atomic(path: Path, data: bytes) -> None:
    """Write `data` to a `.part` file beside `path` and rename it over `path`; no `.part` file is left behind."""
    temp = path.with_name(path.name + ".part")
    try:
        temp.write_bytes(data)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Entry:
    url: str
    final_url: str
    content_type: str
    content_hash: str
    fetched: str  # the date (ISO) of the last request that confirmed or replaced the body
    etag: str | None
    last_modified: str | None
    file: str  # the body's file name in the host folder


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _stem(self, url: str) -> Path:
        return self.root / (urlsplit(url).hostname or "unknown").lower() / hashlib.sha256(url.encode()).hexdigest()[:20]

    def entry(self, url: str) -> Entry | None:
        stem = self._stem(url)
        try:
            entry = Entry(**json.loads(stem.with_suffix(".json").read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        return entry if entry.url == url and self.file(entry).is_file() else None

    def file(self, entry: Entry) -> Path:
        return self._stem(entry.url).parent / entry.file

    def body(self, entry: Entry) -> bytes:
        return self.file(entry).read_bytes()

    def store(
        self,
        url: str,
        final_url: str,
        body: bytes,
        content_type: str,
        etag: str | None,
        last_modified: str | None,
        fetched: str,
    ) -> Entry:
        """Keep `body` as the content of `url`, replacing the previous one."""
        digest = hashlib.sha256(body).hexdigest()
        stem = self._stem(url)
        stem.parent.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if body.startswith(b"%PDF-") else ".body"
        entry = Entry(
            url, final_url, content_type, digest, fetched, etag, last_modified, f"{stem.name}-{digest[:16]}{suffix}"
        )
        if not self.file(entry).is_file():
            write_atomic(self.file(entry), body)
        self._write(entry)
        self._sweep(entry)
        return entry

    def touch(self, entry: Entry, fetched: str, etag: str | None, last_modified: str | None) -> Entry:
        """The same body, confirmed by a request on `fetched` (the server's validators replace the old ones when sent)."""
        entry = Entry(
            **{
                **asdict(entry),
                "fetched": fetched,
                "etag": etag or entry.etag,
                "last_modified": last_modified or entry.last_modified,
            }
        )
        self._write(entry)
        return entry

    def drop(self, url: str) -> None:
        stem = self._stem(url)
        for path in stem.parent.glob(f"{stem.name}*"):
            path.unlink(missing_ok=True)

    def _write(self, entry: Entry) -> None:
        write_atomic(self._stem(entry.url).with_suffix(".json"), json.dumps(asdict(entry), indent=1).encode("utf-8"))

    def _sweep(self, entry: Entry) -> None:
        stem = self._stem(entry.url)
        for path in stem.parent.glob(f"{stem.name}-*"):
            if path.name != entry.file:
                path.unlink(missing_ok=True)
