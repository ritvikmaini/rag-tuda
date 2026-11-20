# Operations

## First start

```sh
uv run rag doctor     # the machine, the cards it can run
uv run rag up         # tu-informatik by default; crawls the listed hosts
```

`rag up --yes` confirms model downloads; without it nothing is downloaded. A card that is already on the machine is not
downloaded again, and a second `up` changes nothing. `--profile` defaults to the profile the folder was last started with,
else `tu-informatik`; `--profile example` indexes the bundled labour laws with no crawl.

State lives in `.rag/` (git-ignored): the API token, the database role passwords, one secret folder per native process, pid
files, logs, `state.json`, the crawl cache `crawl/` and the manifests `snapshots/`. Everything private is created with mode
0600 in a 0700 folder and is never rotated once it exists.

## Reindex and rollback

```sh
uv run rag reindex            # re-crawl the core source set, build a new generation, swap, restart the backend
uv run rag reindex --full     # also the full source set (takes hours)
uv run rag reindex --resume   # continue an interrupted crawl
uv run rag rollback           # the previous generation is live again
```

Both need Docker and the stack's Postgres to be up, which a first `rag up` does. The crawl is polite (about one request per second per host), so schedule `rag reindex` rarely; the README has one
cron line and one launchd example. `RAG__CRAWL__CONTACT` in the environment adds a contact to the crawler's user agent.

## Your own documents

`rag add <folder> [--name N]` registers a folder of Markdown, text or PDF files as a source in `.rag/sources.json`; the next
`rag reindex` builds it with the `folder` connector. This works in native mode only: compose mode prints a note on reindex and
rollback when folders are registered, because mounting arbitrary host folders into the indexer container is not supported.

## Models

`rag up --card <name>` picks a card; without it the class that fits the machine decides. `--model` swaps the Ollama model a
card derives from, `--llm-url` points at an OpenAI-compatible server you run yourself. A hosted card needs `GROQ_API_KEY` in
the environment of `rag up`; the key reaches the backend process only and is not stored. In compose mode, model pulls are a
manual step: `docker compose --env-file .env -f deploy/compose.yaml exec ollama ollama pull <tag>`.

## Stopping and purging

`rag down` stops everything and is safe to repeat; data stays. `rag down --purge --yes` also deletes `.rag/crawl` and
`.rag/snapshots`, the crawled content.

## Development checks

`just ci` is the gate; `just pre-push` is the part without Docker. A single harness leg: `just harness example` or
`just harness tu` (the TU leg crawls a fixture mini-site, never the real hosts). `just ci` takes about 20 minutes and needs
Docker (on macOS `colima start --cpu 6 --memory 12`), Node and a first-run download of Chromium.

## Troubleshooting

- `rag eval` or `rag reindex` says nothing was started from this folder: run `rag up` first.
- The backend answers 401 on every `/api` request in compose mode: the compose command lacked `--env-file .env`.
- `/readyz` stays unhealthy: no index is published yet; build the first one.
- A slow Mac times out: raise the card's `idle_s`, `llm_read_s` and `deadline_s`; the shipped values were measured on an M1 Max.
