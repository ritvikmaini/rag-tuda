# rag-tuda

Cited answers on open-weight models. The assistant answers a question from its indexed documents only, each statement with a
numbered link to its source; the language model's own knowledge serves only for conversation. It answers in German, or in the
question's language, and shapes an answer as a table, a checklist, a chart or a mind map when the question calls for one.
Everything runs on your own hardware; no cloud service sees a question.

Today the repository holds the engine, a single-page chat client and one profile, `profiles/example`: ten German federal
labour laws, indexed by the `folder` connector. A profile holds everything specific to one document collection.

## Architecture at a glance

```text
page (web/page) --GET /api/chat, server-sent events--> backend (FastAPI, one pipeline slot)
  backend: 1 query analysis ............................ vllm (Qwen3-14B-AWQ)
           2 search: the question's vector ............. embedding_service (snowflake-arctic-embed-m-v2.0)
                     sections and full text ............ pgvector (views rag_chunks, rag_sections)
           3 rerank ................................... reranker_service (bge-reranker-v2-m3)
           4 answer, streamed with citations .......... vllm
indexer (a one-off container, `indexer build`) -> a new index generation -> views repointed -> the backend reloads it on its next start
evals (a job container, never running on its own): probe runs, the gate and the smoke, over HTTP to the backend
```

The stream (format 1) sends `meta` first, then `status` (five steps), `docs`, `chunk`, `citations` and `error` frames, and
`done` last; each frame is `data: <json>` followed by a blank line. The compose file publishes one host port, the backend's
`127.0.0.1:8700`; everything else is reached on the compose network or with `docker exec`.

## Repository layout

| Folder | What it holds |
|---|---|
| `src/rag_tuda/` | the command line (`doctor`, `up`, `down`, `ask`, `add`) and the `.rag/` state (the token, the database role passwords, one secret folder per native process, pid files, logs); standard library only |
| `apps/` | `assistant_api` (the backend: FastAPI app and runtime, settings, the LangGraph graph, its own image); `indexer` (settings, connectors found by entry point, generations, the one-shot `indexer build` command and its status, housekeep, rollback and compare commands); `evals` (probe runner, gate, stats, compare, rescore, smoke; its own job image) |
| `packages/` | `rag_core` (shared code: settings, domain, ports, services, adapters) and `rag_testing` (the harness fakes; never in a service image), workspace members |
| `profiles/` | `example/`: `profile.yaml`, `messages.de.yaml`, `prompts/`, `gate.yaml` with the smoke probes in `probes/`, and the law texts in `documents/` |
| `model_servers/` | the two model servers, `embedding/` and `reranker/`: each its own uv project and lock, outside the workspace |
| `deploy/` | `compose.yaml` (the stack, project `rag-tuda`), `compose.native.yaml` (Postgres alone on `127.0.0.1:55432`, for the native stack), `pgvector/` (settings, client authentication, init scripts, role names), `env/` (the overlays), `sql/`, the harness (`compose.ci.yaml`, `ci/`, `fakes/`) |
| `web/page/` | the chat client: static files, the vendored libraries, its unit and browser tests |
| `tests/` | unit, contract (`contracts/`), harness (`integration/`) and repository tests |
| `secrets/` | credential files, never in git |
| `data/` | runtime data, not in git |

## Setup

`uv run rag doctor` checks the machine and lists the model cards it can run. On a Mac, `uv run rag up --profile example --card
s-qwen3-4b-instruct-2507 --yes` starts Postgres in Docker on `127.0.0.1:55432`, migrates, starts the embedding (8701) and
reranker (8702) servers and the backend (8700) as processes of yours (pid files in `.rag/pids`, logs in `.rag/logs`), builds the
index when there is none, creates the Ollama model `rag-<card>` with the card's window, sends a first request to load it and prints
the link. Ollama must be running (`ollama serve`). A model that is already on the machine (the pinned snapshots in the Hugging Face
cache, the Ollama tag) is not downloaded again; a second `up` changes nothing, and `uv run rag down` stops everything (safe to
repeat, data stays; `--purge` also deletes `.rag/crawl` and `.rag/snapshots`). On Linux or WSL2, `--mode compose` runs the compose
files below with the card, its window, the profile and the secrets taken from `.rag/` (build, then migrate, then up, then a
first request to the answer model). A hosted card (`l-groq-...`) needs `GROQ_API_KEY` in the environment of `rag up`; the key goes to
the backend process only and is never stored.

The stack needs Docker with compose on Linux or WSL2. `deploy/compose.yaml` runs on the CPU with Ollama; add
`-f deploy/compose.nvidia.yaml` for NVIDIA GPUs on the model servers, and `-f deploy/compose.vllm.yaml` (named last) to serve
the language model with vLLM instead of Ollama. The compose file names the project `rag-tuda`, so containers are
`rag-tuda-<service>-1` and volumes `rag-tuda_<name>`.

```sh
git clone <this repository> rag-tuda; cd rag-tuda
cp .env_example .env    # then set BACKEND_REFRESH_TOKEN
PYTHONPATH=src python -m rag_tuda.state init    # the secret files under secrets/
docker compose --env-file .env -f deploy/compose.yaml up -d pgvector
docker compose --env-file .env -f deploy/compose.yaml --profile migrate run --rm migrate
docker compose --env-file .env -f deploy/compose.yaml up -d
docker compose --env-file .env -f deploy/compose.yaml exec ollama ollama pull qwen3:4b-instruct-2507-q4_K_M
```

Ollama starts without a model, so the last command downloads the default card's model (about 2.5 GB) once.

Then build the first index: `docker compose --env-file .env -f deploy/compose.yaml --profile build run --rm indexer build`.
The backend reports healthy (`/readyz`) only once an index is published.

Every compose command names the file and the root `.env`; a missing `--env-file` leaves `BACKEND_REFRESH_TOKEN` unset, and the
backend then refuses every `/api` request.

For development (macOS or Linux): install `uv`, `just`, Docker with compose and buildx (on macOS through Colima: `colima start
--cpu 6 --memory 12`), gitleaks and pre-commit; then `uv sync --frozen --all-packages`, `pre-commit install --hook-type
pre-commit --hook-type pre-push` and `just ci`.

## Checks: just ci

`just ci` runs, identically on a development machine and in CI: `lint` (ruff, deptry, gitleaks, the three locks), `types`
(basedpyright), `imports` (import-linter), `unit` (pytest, the coverage ratchet), `db-tests` (the tests that need a migrated
pgvector), `contracts` (shared fixtures, the stream schema, the profile's data), `model-servers` (each model server's tests in
its own environment, with fake models), `images` and `harness` (the real backend and indexer images against fake model
servers). `just pre-push` (the part without Docker) runs before every push.

`just harness example` runs the example profile's leg: the ten law texts of `profiles/example` are indexed and queried through
the real images.

## Profiles and sources

A profile holds what is specific to one deployment.

- `profile.yaml` holds the values that decide the index (chunking, extraction, the sources and their minimums, the embedding
  model) and the assistant's answer-affecting values (identity, ranking, history, guard, generation); it is baked into the
  indexer and assistant images, so an image tag fixes them. `messages.de.yaml` holds the assistant's German texts, under `labels`
  the origin labels of the sources, and `prompts/` its four prompts, read by the backend from the image.
- `gate.yaml` holds the probe sets and thresholds `evals` measures a change against, and the fixed probes of `evals smoke`.
- `deploy/env/<env>.yaml` holds the technical values of one environment (endpoints, hosts, paths); the assistant's own technical
  values are `RAG__…` entries in the backend's compose environment (`deploy/compose.yaml`).
- A source is a connector registered under the entry point group `rag_core.connectors` (today `folder`, in the indexer: Markdown,
  text and PDF files of a directory) and named in the profile with `connector:`.
- A new profile is a folder beside `profiles/example` under `profiles/`, built into the indexer, assistant and evals images with
  `--build-arg PROFILE=<folder>` (the other images carry no profile).
