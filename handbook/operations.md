# Operations

## First start

```sh
uv run rag doctor     # the machine, the cards it can run
uv run rag up         # tu-informatik by default; crawls the listed hosts
```

`rag up --yes` confirms model downloads; without it nothing is downloaded. For every profile but `example` the confirmation also
names the docling layout and table models and the EasyOCR weights, which the indexer fetches itself at the first PDF. A card
that is already on the machine is not downloaded again, and a second `up` changes nothing. `--profile` defaults to the profile
the folder was last started with, else `tu-informatik`; `--profile example` indexes the bundled labour laws with no crawl. A
profile without a `profiles/<name>/profile.yaml` is refused before any step, and the message lists the profiles that exist.

The index counts as built when the indexer's `corpus_build_stats.json` exists and names the profile asked for, and the database
volume exists and is older than the stats file; otherwise `up` plans a build. So switching profiles builds the other
profile's index, and a removed or reset volume is rebuilt.

State lives in `.rag/` (git-ignored): the API token, the database role passwords, one secret folder per native process, pid
files, logs, `state.json`, the crawl cache `crawl/` and the manifests `snapshots/`. Everything private is created with mode
0600 in a 0700 folder and is never rotated once it exists, except the files compose bind-mounts into containers that do not
run as root (the database role passwords in `.rag/secrets/roles/`, the compose secrets in `secrets/`): they are 0644 inside
their 0700 folder, since a 0600 file of the host user is unreadable to the container user on Linux.

## Reindex and rollback

```sh
uv run rag reindex            # re-crawl the core source set, build a new generation, swap, restart the backend
uv run rag reindex --full     # also the full source set (about 42 minutes in two builds, see below)
uv run rag reindex --resume   # continue an interrupted crawl
uv run rag rollback           # the previous generation is live again
```

Both need Docker and the stack's Postgres to be up, which a first `rag up` does. The two builds that brought the full set's index to 6972 sections took about 42 minutes on an M2 Pro
(30 + 12) on top of a core index whose chunks were cached; a full build from nothing was not timed, and the
"takes hours" of `rag reindex --help` and the crawl sizes that `rag up`, `rag reindex` and `rag doctor` print are design estimates. The crawl is polite (about one request per second per host), so schedule `rag reindex` rarely; the README has one
cron line and one launchd example. `RAG__CRAWL__CONTACT` in the environment adds a contact to the crawler's user agent.

The indexer refuses a build that shrank against the last build, and compares only with a build of the same profile and source
set (core or full, plus registered folders); a build of another profile or source set is not compared and says so. It also
refuses a build in which a page of the live generation (as its snapshot manifest lists it) yields no main content, because the
site changed under the host's selector; a page that was not in the live generation is only named in the summary. Only a build of the
same source set as the last one is checked for this. To accept either refusal on purpose, delete `corpus_build_stats.json` in the
state folder: the next build has no reference (the index then counts as not built, see above). `rag reindex --full` also gets
past them when the last build was of the core set, since a build of another source set is not compared; a second `--full`
is checked again. A snapshot manifest the build cannot read (`.rag/snapshots/<generation>.json`) fails the build; delete that file
to go on. `rag rollback`
re-records the stats of the generation it makes live, and it stops when the index was built for another profile than the stack
runs. After a profile switch, go back with `rag up --profile <name>`, not with `rag rollback`.

## Retention

The retention job removes conversations older than 90 days (and the legacy checkpoint threads) and Langfuse traces older than 30
days. Without `--apply` it only reports. It runs with the backend's settings, so inside the backend container in compose mode:

```sh
docker compose --env-file .env -f deploy/compose.yaml exec -T backend python -m assistant_api.retention --apply
```

and natively with the environment `rag up` gives the backend (the profile, the card and the secrets folder):
`uv run --frozen --package assistant-api python -m assistant_api.retention --apply`. A step that fails prints one line and the
others still run; the trace step does not apply when tracing is off. The exit status is 0 when every step that applies ran.

## Your own documents

`rag add <folder> [--name N]` registers a folder of Markdown, text or PDF files as a source in `.rag/sources.json`; the next
`rag reindex` builds it with the `folder` connector. This works in native mode only: compose mode prints a note on reindex and
rollback when folders are registered, because mounting arbitrary host folders into the indexer container is not supported.

## Models

`rag up --card <name>` picks a card; without it the class that fits the machine decides. `--model` swaps the Ollama model a
card derives from, `--llm-url` points at an OpenAI-compatible server you run yourself. A hosted card needs `GROQ_API_KEY` in
the environment of `rag up`; the key reaches the backend process only and is not stored. `rag up --mode compose` pulls the
card's model itself after your yes; on the by-hand compose path (see the README) the pull is a manual step:
`docker compose --env-file .env -f deploy/compose.yaml exec ollama ollama pull <tag>`.

## Stopping and purging

`rag down` stops everything and is safe to repeat; data stays. `rag down --purge --yes` also deletes `.rag/crawl` and
`.rag/snapshots`, the crawled content. When files stay behind (a compose container wrote them as root), the command fails and
names the path and the remedy: `sudo rm -rf <path>`, or a one-off container that removes it.

## Development checks

`just ci` is the gate; `just pre-push` is the part without Docker. A single harness leg: `just harness example` or
`just harness tu` (the TU leg crawls a fixture mini-site, never the real hosts). `just ci` takes about 20 minutes on an M1 Max (an estimate) and needs
Docker (on macOS `colima start --cpu 6 --memory 12`), Node and a first-run download of Chromium.

## Troubleshooting

- `rag eval` or `rag reindex` says nothing was started from this folder: run `rag up` first.
- The backend answers 401 on every `/api` request in compose mode: the compose command lacked `--env-file .env`.
- `/readyz` stays unhealthy: no index is published yet; build the first one.
- Reranking or the semantic search times out on a 16 GB Mac: start Ollama with `LLAMA_ARG_CACHE_RAM=1024` (the macOS app: `launchctl setenv LLAMA_ARG_CACHE_RAM 1024`, then restart it) and see "A 16 GB Mac" in `handbook/models.md`.
- A slow Mac times out: raise the card's `idle_s`, `llm_read_s` and `deadline_s`; the shipped values leave room for the retrieval's own waits on top of the model times measured on an M2 Pro with 16 GB (`handbook/models.md`).
