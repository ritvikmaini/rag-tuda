"""The TU profile as the harness's indexer reads it: `python -m rag_testing.tu_profile <profile folder> <target folder>`
writes the three files the indexer reads (profile.yaml, sources.yaml, roles.yaml) for a crawl of the fixture mini-site.

The URLs stay the real hosts' (the overlay's `crawl.routes` sends those two hosts to the mini-site's services), but only the
two hosts the mini-site stands for, the faculty's and the university's, stay in sources.yaml's allow-list: the crawl can
fetch nothing else. profile.yaml keeps its settings but loses the statute source (the mini-site serves no law XML) and has
its web source's minimum document count lowered to what the mini-site yields. The real files are not touched."""

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

HOSTS = {  # the profile's host: the origin of the mini-site service that answers for it (the overlay's routes)
    "www.informatik.tu-darmstadt.de": "http://minisite:8091",
    "www.intern.tu-darmstadt.de": "http://minisite-intern:8092",
}
MIN_DOCUMENTS = (
    2  # the gate counts a source's PDFs that produced documents, and the mini-site has two; the profile asks for 30
)
WEB_SOURCE, LAW_SOURCE = "tu_web", "laws"


def sources(text: str) -> dict[str, Any]:
    """sources.yaml with the mini-site's hosts only and no statutes."""
    data = yaml.safe_load(text)
    if HOSTS.keys() - data["hosts"].keys():
        raise ValueError("sources.yaml lacks a host the mini-site stands for")
    data["hosts"] = {host: data["hosts"][host] for host in HOSTS}
    data.pop("laws")
    return data


def profile(text: str) -> dict[str, Any]:
    """profile.yaml without the statute source and with the lowered minimum."""
    data = yaml.safe_load(text)
    ingest = data["ingest"]["sources"]
    del ingest[LAW_SOURCE]
    ingest[WEB_SOURCE]["min_documents"] = MIN_DOCUMENTS
    return data


def write(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name, change in (("profile.yaml", profile), ("sources.yaml", sources)):
        data = change((source / name).read_text(encoding="utf-8"))
        (target / name).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    shutil.copyfile(source / "roles.yaml", target / "roles.yaml")


if __name__ == "__main__":
    write(Path(sys.argv[1]), Path(sys.argv[2]))
