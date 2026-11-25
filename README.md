# rag-tuda

**A study-rules assistant for TU Darmstadt's Fachbereich Informatik that answers with numbered citations, runs on
open-weight models, and fits on a 16 GB laptop.**

Ask "How many weeks do I have for the bachelor thesis?" and get an answer that rests on the right version of the examination
regulations (PO), links every statement to the page or PDF it came from, and says so when the documents hold no answer.
It covers the Bachelor, Master and doctoral rules of the department, the university-wide regulations, semester dates and
fees, and the residence permit for studying. One command crawls the sources, builds the index and opens the chat page.

> An independent student project. **Not an official service of TU Darmstadt and not legal advice.** Every answer is
> AI-generated and without guarantee: check the cited source.

## Highlights

- **A 4B model on a laptop gets 32 of 35.** The default model is Qwen3 4B (a 2.5 GB download). On an M2 Pro with 16 GB it
  passed 32 of the 35 hand-written questions of the project's own test set, with a median answer time of 26 s. No GPU server,
  no API key, no proprietary model. ([the fine print](#results))
- **Version-aware, and it fails closed.** Degree, programme, PO version and document type are typed columns that the
  crawler fills by fixed rules, never by a model. Once a question, or a chip on the page, names a version, they are hard
  filters in SQL, so a source of another version cannot be cited. A regulation without a programme and a version is not
  indexed at all, and a filtered search that finds nothing says which filter applied instead of quietly searching
  without it.
- **The safety-relevant parts are code, not prompt.** The citation list, the "Based on: PO 2023 of the B.Sc. Informatik"
  line, the residence-law disclaimer and the note that an English translation is not binding are built by code from the
  sources an answer cites.
- **A retrieval pipeline you can take apart.** Hybrid search (vectors and German full text with acronym expansion), score
  fusion, a cross-encoder that reranks whole sections, and a deterministic hop that follows references such as
  "§ 22 APB" with no model call. `rag eval --retrieval-only --ablation` measures what each stage buys.
- **One architecture, four model classes.** A model card (one TOML file) holds a model's window, budget, timeouts and
  quirks, so the same code serves a 4B model on a laptop, a 27B to 35B model on a larger Mac, a hosted open model on Groq
  or a vLLM server. Only the small class is measured so far.
- **Private by default.** Everything binds `127.0.0.1`, the API needs a token, there is no telemetry and no access log.
  A question leaves the machine only if you choose a hosted model card, and the page then names the provider.
- **A polite, configuration-driven crawler.** A host allow-list, one request per second per host, its own RFC 9309
  robots.txt matcher, `noindex` and `noai` honoured, conditional re-fetches, OCR for PDFs without a text layer.
- **Index generations with rollback.** Every build is a new generation that goes live in one atomic swap; `rag rollback`
  brings the previous one back, and a build that shrank or lost its page content is refused.
- **Answers in more than prose.** The page renders tables, checklists you can tick, numbered steps, charts and mind maps,
  and turns the last answer into a shorter, longer, plain-language or English version, with the citations kept.
- **Engineered like a product.** 2,370 Python tests, import contracts that keep the domain code free of frameworks, pinned
  and hash-checked dependencies, and a CI run that drives the real container images against fake model servers and a
  fixture copy of the university site.

## What you can ask

These are questions from the test set (`profiles/tu-informatik/smoke/smoke_v1.json`), in German or English:

- *Wie viele Wochen habe ich Zeit für die Bachelorarbeit?*
- *Was hat sich bei der Bachelorarbeit im B.Sc. Informatik zwischen der PO 2015 und der PO 2023 geändert?*
- *Kann ich mich rückwirkend für ein Semester beurlauben lassen, das schon vorbei ist?*
- *Which English level do I need for the Master's programme Computer Science?*
- *How soon after I arrive in Germany do I have to register my address?*
- *Wie viele Arbeitstage im Jahr darf ich mit einer Aufenthaltserlaubnis zum Studium höchstens arbeiten?*

Chips on the page set your degree, programme and PO version once, so "How long is the thesis?" is answered for *your*
regulations; explicit text in a question ("nach der PO 2015") outranks the chip. German answers use "Sie", an English
question gets an English answer, and where a German and an English version of a regulation both exist the binding German
text wins. Ask about the canteen menu or another department's degree and the assistant says it found nothing, because it
answers from its indexed documents only.

## Quick start

On an Apple-silicon Mac with 16 GB or more you need Docker (Colima, OrbStack or Docker Desktop),
[`uv`](https://docs.astral.sh/uv/) and Ollama running (`ollama serve`; on a 16 GB Mac start it as
`LLAMA_ARG_CACHE_RAM=1024 ollama serve`, see [`handbook/models.md`](handbook/models.md)).

```sh
git clone https://github.com/ritvikmaini/rag-tuda.git && cd rag-tuda
uv run rag doctor     # checks the machine and lists the model cards it can run
uv run rag up         # crawls the listed TU hosts (about 8 minutes), builds the index, prints the link to the chat page
uv run rag ask "Wie viele Wochen habe ich Zeit für die Bachelorarbeit?" --degree bachelor
```

No crawl, just a look? The bundled example profile indexes ten German labour laws from files in the repository:

```sh
uv run rag up --profile example --yes
```

On x86_64 Linux or WSL2 the whole stack runs in Docker: `uv run rag up --mode compose`
(see [Compose mode](#compose-mode-x86_64-linux-or-wsl2)).

`rag up` starts Postgres in Docker, migrates, starts the embedding server, the reranker server and the backend, builds the
index when there is none, creates the Ollama model for the chosen card and prints a link that carries the access token.
Nothing is downloaded without your yes, a model already on the machine is not downloaded again, and a second `rag up`
changes nothing.

## How it works

```mermaid
flowchart LR
  crawl[crawl<br/>allow-listed hosts] --> gen[index generations<br/>pgvector, atomic swap]
  q([question]) --> an[query analysis<br/>filters, language]
  an --> ret[hybrid retrieval<br/>vectors + German full text]
  gen --> ret
  ret --> fuse[fusion cut]
  fuse --> rr[rerank<br/>bge-reranker-v2-m3]
  rr --> par[whole parents]
  par --> hop[reference hop]
  hop --> ans[answer<br/>numbered citations<br/>appended lines]
```

1. **Analysis.** One call to the model returns JSON: the language, a standalone question, search queries, the intent, and
   the degree, programme, PO version and document type. Deterministic readers of the question text and the page chips
   outrank the model's values, and if the JSON is invalid the plain question is searched and the readers still run.
2. **Hybrid retrieval.** The question's vector against the embeddings, and Postgres full text in German (plus English for
   English rows), both under the hard filters. Acronyms expand into OR groups ("PO" or "Prüfungsordnung").
3. **Fusion cut.** Scores are normalised per leg and combined; only the best candidates go on, which is what keeps
   retrieval at about 2 s.
4. **Rerank.** A cross-encoder scores whole parent sections against the question. Below the threshold a source is dropped;
   with none left the assistant declines and names the filter.
5. **Reference hop.** "§ 22 APB" in a kept section is resolved by pattern to the section it names. No model call.
6. **Iterative round** (larger model classes only): one extra search for questions that need more than one document.
7. **Answer.** The prompt is fitted to the card's context window and the answer streams with `[n]` markers. Code builds the
   citations and appends the PO-version line and the disclaimers.

```text
page (web/page) --GET /api/chat, server-sent events--> backend (FastAPI + LangGraph, one pipeline slot)
  backend: 1 query analysis ............. the card's OpenAI-compatible endpoint (Ollama natively)
           2 search: the question's vector ... embedding server (snowflake-arctic-embed-m-v2.0)
                     sections and full text .. Postgres + pgvector (views rag_chunks, rag_sections)
           3 rerank ..................... reranker server (bge-reranker-v2-m3)
           4 answer, streamed with citations . the card's OpenAI-compatible endpoint
indexer (a one-off job, `indexer build`) -> a new index generation -> views repointed -> the backend reloads it
evals (a job, never running on its own): probe runs, the gate and the smoke, over HTTP to the backend
```

The stream (format 1) sends `meta` first, then `status` (five steps the page shows while it works), `docs`, `chunk`,
`citations` and `error` frames, and `done` last. There is no approximate vector index: an exact scan over a few thousand
sections is fast and always returns `k` rows under a filter. More in [`handbook/architecture.md`](handbook/architecture.md)
and [`handbook/decisions.md`](handbook/decisions.md).

**The stack:** Python 3.12, FastAPI, LangGraph, Postgres 16 with pgvector, sentence-transformer model servers on PyTorch
(CUDA, MPS or CPU), Ollama, vLLM or Groq behind one OpenAI-compatible adapter, docling and EasyOCR for PDFs, a static
page with no build step (vanilla JavaScript; Showdown, DOMPurify, Chart.js and Markmap vendored and integrity-checked),
Docker Compose, `uv` workspaces and `just`.

## Model classes

| Class | Cards | Where it runs |
|---|---|---|
| S | `s-qwen3-4b-instruct-2507` (default, 2.5 GB), `s-qwen3.5-4b` (4.0 GB), `s-gemma4-e4b` (9.5 GB) | Ollama on the machine; an 8K window |
| M | `m-qwen3.6-35b-a3b` (22.6 GB), `m-qwen3.8-27b` (17.7 GB), `m-gemma4-31b` (20.4 GB) | Ollama on a machine with 24 GB or more usable; a 16K window, the iterative round |
| L | `l-groq-gpt-oss-120b`, `l-groq-gpt-oss-20b`, `l-groq-qwen3.8-27b` | Groq's hosted endpoint for the answer; analysis, embedding and reranking stay local |
| XL | `xl-vllm-example` | a vLLM server on NVIDIA GPUs; shipped as an example to copy |

`rag doctor` picks the class that fits the machine; `rag up --card <name>` chooses a card. The embedder
(`Snowflake/snowflake-arctic-embed-m-v2.0`) and the reranker (`BAAI/bge-reranker-v2-m3`) are the same for every class and
pinned by revision. Every model is open-weight. A hosted card needs `GROQ_API_KEY` in the environment of `rag up`; only the
answer call goes to the provider, and you should enter no personal matters while one is active (Groq's free-tier data
terms have not been read in full). Cards, quirks and memory settings: [`handbook/models.md`](handbook/models.md).

## Results

Everything here was measured on one machine: an Apple M2 Pro with 16 GB, Ollama 0.40.17, the default card, one run per
number. The question set is the project's own 35 hand-written questions (32 with a gold source, 3 with no answer in the
corpus by design).

| What | Result |
|---|---|
| Answers passing their checks (`rag eval --smoke`) | **32 of 35** (2 failed, 1 refused as out of scope) |
| Whole answer, median | 26 s, mostly reading the 8K-token prompt; the first token after about 21 s at a full window |
| Retrieval, median | about 2 s |
| Gold source in the top 5 after reranking | 0.88 (0.72 without the filters and the fusion cut) |
| Memory | the three model processes at 9.8 GB of the 10.4 GB a 16 GB Mac leaves usable (read in a retrieval-only run, not during answers) |
| Core index | about 270 pages and 50 PDFs, 3,666 sections, 15,472 chunks; crawl 8 min, embedding and build 3 min 14 s |
| Full index (`rag reindex --full`) | 1,487 pages, about 177 PDFs and 2 laws, 6,799 sections, 29,794 chunks |

What each retrieval stage adds (`rag eval --retrieval-only --ablation`, full index, 32 golds):

| Stage | hit@1 | hit@5 | MRR | Median latency |
|---|---|---|---|---|
| no filters | 0.50 | 0.72 | 0.608 | 17.28 s |
| + filters | 0.56 | 0.81 | 0.661 | 14.36 s |
| + acronyms | 0.56 | 0.81 | 0.659 | 13.78 s |
| + fusion cut | 0.59 | 0.88 | 0.702 | 2.04 s |
| + whole parents | 0.59 | 0.88 | 0.694 | 2.28 s |
| + reference hop (the product) | 0.59 | 0.88 | 0.694 | 2.28 s |

How far these numbers carry:

- One machine, one run per cell, 35 questions: one question is 0.03, a difference under 0.05 is noise.
- The questions were revised after a first run on the same set, so it is not a blind test.
- The filters and the fusion cut carry the gain. Acronyms, whole parents and the reference hop show none *on this set*
  (the hop never added a section); they are design choices the set is too small to judge.
- The M, L and XL cards have no measurements of any kind: they do not fit this machine or need an account. The XL card has
  never been run.
- The first three ablation rows rerank about 117 sections and ran with a longer reranker limit; their latencies describe
  the ablation, not the product.

The bake-offs behind the pinned embedder and reranker, the per-card speed and quality tables and every caveat are in
[`handbook/evaluation.md`](handbook/evaluation.md) and [`handbook/models.md`](handbook/models.md).

<details>
<summary>Index sizes in full</summary>

The default profile's core set (the department's study pages and regulations, the university-wide regulations, the residence
pages and two laws) came to about 270 pages, about 50 PDFs, 3666 sections and 15472 chunks. On an M2 Pro with 16 GB the crawl and
the PDF downloads took about 8 minutes and the embedding and build 3 minutes 14 seconds. (A re-crawl of the same set gave 3637
sections and 15373 chunks, the index of the embedder bake-off.) The full set (`rag reindex --full`) adds the TU-wide study
rules and the Landkreis pages: 6799 sections and 29794 chunks in the published index (the build's own `children` count is 29810)
from 1487 pages (1213 on `www.tu-darmstadt.de`, 265 on the faculty host, 5 on `www.ladadi.de`, 3 on
`digitales-rathaus.darmstadt.de`, 1 on `www.intern.tu-darmstadt.de`), about 177 PDFs and 2 laws, as they stand after
a further build that applies the `exclude` list (about two minutes, everything from the cache). The two builds before it, which
brought the index to 6972 sections, took about 42 minutes (30 + 12; the first sent 1750 requests) on top of a core index whose
chunks were cached; a full build from nothing was not timed. The crawl sizes that `rag up`, `rag reindex` and `rag doctor` print and
the "takes hours" of `rag reindex --help` are design estimates, not these measurements.

</details>

## What it does not do

- **It is a single-user local tool, not a hosted service.** Every port binds `127.0.0.1`, access is one shared token, and
  a local card answers one question at a time (a second one waits, then gets a "busy" message). There are no user
  accounts, no sign-on and no TLS. A department that wants to serve many students needs a reverse proxy with TLS and
  sign-on in front, and a model server that batches requests (the vLLM path, which is untested here).
- **Only the small class is proven.** See [Results](#results).
- **It knows three programmes.** B.Sc. Informatik, M.Sc. Informatik and M.Sc. Computer Science, the doctorate, the
  university-wide rules and the residence pages. The department's other programmes are left out of the crawl.
- **It ships no index.** TU Darmstadt's content may not be redistributed, so every installation crawls for itself.
- **It is not fast on a laptop.** About 26 s per answer on a 16 GB Mac.
- **It gets things wrong.** 3 of 35 on its own set, and fees and dates go stale between re-crawls. Treat an answer as a
  pointer to the source.
- **Platforms.** Native mode needs an Apple-silicon Mac; compose mode needs x86_64 Linux or WSL2 (the model servers' images
  do not build on arm64 Linux). Your own folders (`rag add`) work in native mode only.

## Background

This project is inspired by work I did during an internship at a German municipality. With their permission, this
repository is an abstracted version with a similar implementation, tailored to a public document collection: the study
rules of TU Darmstadt's Fachbereich Informatik. The aim is that a student, or the department itself, can run it and ask
questions about the university's and the department's rules and documents, and that the whole thing stays on open-weight
models and on hardware you own.

It is also meant to be read. The parts I would point a reviewer to: the fail-closed metadata and filters
(`apps/indexer/src/indexer/crawl/metadata.py`, `packages/rag_core/src/rag_core/domain/filters.py`), the retrieval
sequence over ports (`packages/rag_core/src/rag_core/services/retrieval.py`), index generations
(`apps/indexer/src/indexer/index/generations.py`), the request guard that ends every request in bounded time
(`apps/assistant_api/src/assistant_api/chat/guard.py`) and the retrieval eval (`apps/assistant_api/src/assistant_api/retrieval_eval.py`).

## Commands

| Command | What it does |
|---|---|
| `uv run rag doctor` | checks the machine and picks a class; `--card` shows one card's download size and whether it fits |
| `uv run rag up` | brings the stack up; `--card`, `--profile` (only a profile that exists), `--mode native\|compose`, `--model`, `--llm-url`, `--yes` |
| `uv run rag ask "<question>"` | asks one question and streams the cited answer; `--degree`, `--programme`, `--po`, `--lang` |
| `uv run rag reindex` | builds a new index generation from a re-crawl, swaps it in and restarts the backend; `--full`, `--resume` |
| `uv run rag rollback` | makes the previous index generation live again and restarts the backend; refused when the index was built for another profile than the stack runs (to go back to a profile, use `rag up --profile <name>`) |
| `uv run rag eval` | the deploy smoke (stream structure) against the running stack; `--smoke` plays the profile's smoke set with the answer checks; `--retrieval-only [--ablation]` scores retrieval alone, native mode only |
| `uv run rag add <folder>` | registers a folder of your own documents (Markdown, text, PDF) as a source; then `rag reindex` |
| `uv run rag down` | stops everything (safe to repeat; data stays); `--purge --yes` also deletes `.rag/crawl` and `.rag/snapshots`, and fails naming the path and the remedy when files stay behind (written as root by a container) |

`rag up` runs Postgres on `127.0.0.1:55432`, the embedding server on 8701, the reranker on 8702 and the backend on 8700 as
processes of yours (pid files in `.rag/pids`, logs in `.rag/logs`). `--profile` defaults to the profile this folder was
last started with, else `tu-informatik`; a name without a `profiles/<name>/profile.yaml` is refused before any step. For
every profile but `example` the download prompt also names the docling layout and table models and the EasyOCR weights
that the indexer fetches at the first PDF. The index counts as built for a profile when the indexer's stats name that
profile and the database volume is older than the stats; otherwise `up` builds it, so switching profiles builds the other
profile's index and a removed volume is rebuilt.

## Make it yours: profiles and sources

A profile is a folder under `profiles/` that holds everything specific to one document collection, so pointing the
assistant at another department is configuration, not code.

- `profile.yaml` holds the values that decide the index (chunking, extraction, the sources and their minimums, the embedding
  model) and the assistant's answer-affecting values (identity, ranking, history, guard, generation); it is baked into the
  indexer and assistant images, so an image tag fixes them. `messages.de.yaml` and `messages.en.yaml` hold the assistant's
  texts and the origin labels of the sources, and `prompts/` its prompts.
- `sources.yaml` is the crawler's host allow-list, `roles.yaml` names the programmes with their PO versions, and
  `acronyms.yaml` the term groups.
- `gate.yaml` holds the probe sets and thresholds `evals` measures a change against, and the fixed probes of the deploy smoke.
- `deploy/env/<env>.yaml` holds the technical values of one environment (endpoints, hosts, paths); the assistant's own technical
  values are `RAG__…` entries in the backend's compose environment (`deploy/compose.yaml`).
- A source is a connector registered under the entry point group `rag_core.connectors` and named in the profile with
  `connector:`. There are three: `folder` (Markdown, text and PDF files of a directory), `tuda_web` (the polite crawler over a
  host allow-list) and `law_xml` (statutes from the official XML of gesetze-im-internet.de).
- A new profile is a folder beside the existing ones, built into the indexer, assistant and evals images with
  `--build-arg PROFILE=<folder>`.

Two profiles ship: `tu-informatik` (the default) and `example`, ten German federal labour laws read by the `folder`
connector, which CI uses and which needs no crawl. Your own documents: `rag add <folder>`, then `rag reindex`.

## Sources and rights

- The crawled pages and PDFs live only in `.rag/crawl/` on your machine. They are never committed, never redistributed; the
  repository holds configuration and, in the smoke set, short quotes. The index is local and not to be redistributed: TU
  Darmstadt's imprint says its content may not be reproduced in other publications without consent.
- Answers quote short passages and link to the source.
- The crawler's host allow-list is `profiles/tu-informatik/sources.yaml`: `www.informatik.tu-darmstadt.de`,
  `www.intern.tu-darmstadt.de`, `www.tu-darmstadt.de`, `digitales-rathaus.darmstadt.de`, `www.gesetze-im-internet.de` (the
  statutes' official XML) and, in a full build, `www.ladadi.de`. `www.darmstadt.de` and `h-da.de` are never fetched. The
  crawler honours robots.txt (RFC 9309) and `<meta name="robots">`, and sends the user agent
  `rag-tuda/<version> (+<contact>)`.
- Set `RAG__CRAWL__CONTACT` in your environment to put a contact into the user agent; it is optional and is never read from
  a file in the repository. `rag reindex` prints a note when it is not set.
- README screenshots must come from the fixture mini-site, never from crawled pages; none is committed yet.

More in [`handbook/sources.md`](handbook/sources.md).

## Warnings

- **Not legal advice, not an official TU service.** The page footer says so too.
- **Prompt injection through crawled pages.** Text hidden in a page's markup (`hidden`, `aria-hidden="true"`, inline `display:none` or `visibility:hidden`) is dropped when
  a page is extracted; text hidden by a CSS class and the text of PDFs are not covered, and crawled text still reaches
  the model's prompt; treat an answer as a pointer to the source.
- **Remote cards.** Questions leave the machine only to the answer endpoint of a hosted card, and the page shows it. Groq's
  free-tier data terms are not fully read: no personal matters while a remote card is active.
- **No telemetry.** Every host port binds `127.0.0.1`, `/api/*` needs a Bearer token, and the backend's HTTP access log is off
  (the question is a GET parameter). Tracing to a Langfuse server of your own is off unless you configure it.

## Operations

Keep the index current with `rag reindex`, on a schedule. It needs Docker and the stack's Postgres running (a first `rag up`
done), and it crawls again, so schedule it rarely. `rag reindex --resume` continues an interrupted crawl; `rag reindex --full`
adds the full source set. The indexer's size check compares a build only with the last build of the same profile and source
set (core or full, plus your folders). `rag rollback` makes the previous generation live again, and it stops when the index
was built for another profile than the stack runs: after switching profiles, go back with `rag up --profile <name>`, not with
`rag rollback`. `uv run rag down` stops the stack and keeps the data. More in
[`handbook/operations.md`](handbook/operations.md).

<details>
<summary>Scheduling the reindex (cron, launchd) and the retention job</summary>

A scheduled job has a bare environment. It needs `docker` and `uv` on its `PATH`, and with a hosted (`l-groq-...`) card the
backend restart fails unless `GROQ_API_KEY` is in the job's environment too; `RAG__CRAWL__CONTACT` is optional (`rag reindex`
only prints a note when it is unset).

One cron line, with a `PATH` (use the full path of `uv`, from `command -v uv`, and of this folder; add `GROQ_API_KEY=...` and
`RAG__CRAWL__CONTACT=...` lines beside `PATH` when you need them):

```text
PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
17 3 * * 0  cd /path/to/rag-tuda && /path/to/uv run rag reindex >> .rag/logs/reindex.log 2>&1
```

One launchd example for the Mac, `~/Library/LaunchAgents/rag-tuda.reindex.plist`, loaded with
`launchctl load ~/Library/LaunchAgents/rag-tuda.reindex.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>rag-tuda.reindex</string>
  <key>WorkingDirectory</key><string>/path/to/rag-tuda</string>
  <key>ProgramArguments</key><array>
    <string>/path/to/uv</string><string>run</string><string>rag</string><string>reindex</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    <!-- with a hosted card also: <key>GROQ_API_KEY</key><string>...</string> -->
  </dict>
  <key>StartCalendarInterval</key><dict>
    <key>Weekday</key><integer>0</integer><key>Hour</key><integer>3</integer><key>Minute</key><integer>17</integer>
  </dict>
  <key>StandardOutPath</key><string>/path/to/rag-tuda/.rag/logs/reindex.log</string>
  <key>StandardErrorPath</key><string>/path/to/rag-tuda/.rag/logs/reindex.log</string>
</dict></plist>
```

The retention job removes conversations older than 90 days and Langfuse traces older than 30 days; without `--apply` it
only reports. In compose mode, from the repository root:

```sh
docker compose --env-file .env -f deploy/compose.yaml exec -T backend python -m assistant_api.retention --apply
```

Natively, with the environment `rag up` gives the backend (the profile, the card and the secrets folder):
`uv run --frozen --package assistant-api python -m assistant_api.retention --apply`. A run without tracing configured skips
the trace step and still exits 0 when the other steps worked.

</details>

## Compose mode (x86_64 Linux or WSL2)

`uv run rag up --mode compose` runs the compose files below with the card, its window, the profile and the secrets taken from
`.rag/` (download the two model snapshots after your yes, build, then migrate, then up, pull the card's model after your yes,
then a first request to the answer model). The model servers' locks cover x86_64 Linux and Apple silicon only, so their images
do not build on an arm64 Linux host.

The stack needs Docker with compose. `deploy/compose.yaml` runs on the CPU with Ollama; add `-f deploy/compose.nvidia.yaml` for
NVIDIA GPUs on the model servers, and `-f deploy/compose.vllm.yaml` (named last) to serve the language model with vLLM
instead of Ollama. The compose file publishes one host port, the backend's `127.0.0.1:8700`, and names the project
`rag-tuda`, so containers are `rag-tuda-<service>-1` and volumes `rag-tuda_<name>`.

<details>
<summary>The same steps by hand, and vLLM</summary>

A fresh clone downloads the two model snapshots, builds, then migrates:

```sh
git clone https://github.com/ritvikmaini/rag-tuda.git rag-tuda; cd rag-tuda
mkdir -p .rag; echo '{}' > .rag/indexer-overlay.yaml    # the TU profile has no folder sources; the default overlay names the example's law texts
cp .env_example .env    # then set BACKEND_REFRESH_TOKEN, RAG_PROFILE=tu-informatik (the default there is example) and RAG_INDEXER_OVERLAY=../.rag/indexer-overlay.yaml
uv run python -m rag_tuda.state init    # the secret files under secrets/ (0644 inside a 0700 folder: the containers do not run as root)
HF_HUB_DISABLE_XET=1 uv tool run --from huggingface_hub==0.36.2 huggingface-cli download Snowflake/snowflake-arctic-embed-m-v2.0 --revision 95c2741480856aa9666782eb4afe11959938017f
HF_HUB_DISABLE_XET=1 uv tool run --from huggingface_hub==0.36.2 huggingface-cli download BAAI/bge-reranker-v2-m3 --revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
docker compose --env-file .env -f deploy/compose.yaml --profile build --profile evals build
docker compose --env-file .env -f deploy/compose.yaml up -d --wait pgvector
docker compose --env-file .env -f deploy/compose.yaml --profile migrate run --rm migrate
docker compose --env-file .env -f deploy/compose.yaml up -d
docker compose --env-file .env -f deploy/compose.yaml exec ollama ollama pull qwen3:4b-instruct-2507-q4_K_M
```

The two downloads go to `~/.cache/huggingface`, which compose mounts (set `RAG_HF_CACHE` in `.env` for another home); the model
servers load only from that cache and never download. Ollama starts without a model; the last command downloads the default
card's model (about 2.5 GB) once, and you run it after deciding to download. Pulling by hand is the manual path only:
`rag up --mode compose` pulls the card's model itself after your yes.

Then build the first index: `docker compose --env-file .env -f deploy/compose.yaml --profile build run --rm indexer build`.
The backend reports healthy (`/readyz`) only once an index is published. Every compose command names the file and the root
`.env`; a missing `--env-file` leaves `BACKEND_REFRESH_TOKEN` unset, and the backend then refuses every `/api` request.

**vLLM (XL).** `deploy/compose.vllm.yaml` serves the card's model at a pinned revision and loads the weights offline from the
host's Hugging Face cache (`~/.cache/huggingface`). Download that revision there first (the revision is in the `--revision`
flag of `compose.vllm.yaml`, for example with `hf download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --revision <revision>`); the
container starts with `HF_HUB_OFFLINE=1` and downloads nothing itself. To serve another model, copy the card and change both
together. This path is documented but has not been run.

</details>

## Repository layout

| Folder | What it holds |
|---|---|
| `src/rag_tuda/` | the command line (`doctor`, `up`, `down`, `reindex`, `rollback`, `eval`, `ask`, `add`) and the `.rag/` state (the token, the database role passwords, one secret folder per native process, pid files, logs); standard library only |
| `apps/assistant_api/` | the backend: the FastAPI app and runtime, the LangGraph graph, the request guard, the turn store and its migrations, the stream's frames, the retrieval eval, the retention job |
| `apps/indexer/` | the connectors (found by entry point), the crawler, PDF extraction and OCR, structure-aware chunking, index generations, the one-shot `indexer build` command with its status, housekeep, rollback and compare commands |
| `apps/evals/` | the probe runner, the answer checks, the gate, stats, compare, rescore and the deploy smoke; its own job image |
| `packages/rag_core/` | the shared code in layers: `domain` (pure functions: filters, fusion, ranking, citations, references, the context budget), `ports`, `services` (analysis, retrieval, prompts), `adapters` (Postgres, the model servers, the chat model, Langfuse) and `settings` (profile, model card) |
| `packages/rag_testing/` | the harness fakes (LLM, embedder, reranker) and the fixture mini-site that stands in for the university's hosts; never in a service image |
| `model_servers/` | the embedding and reranker servers: each its own uv project and lock, outside the workspace |
| `models/cards/` | one TOML file per model card |
| `profiles/` | `tu-informatik/` (profile, messages, prompts, roles, acronyms, sources, gate and smoke set) and `example/` (the same, with its law texts in `documents/`) |
| `deploy/` | `compose.yaml` (the stack), `compose.native.yaml` (Postgres alone, for the native stack), the NVIDIA and vLLM overlays, `pgvector/` (settings, client authentication, init scripts, roles), `env/` (the overlays), the harness (`compose.ci*.yaml`, `fakes/`) |
| `web/page/` | the chat client: static files, the vendored libraries, its unit and browser tests |
| `handbook/` | architecture, operations, evaluation, sources, models and design decisions |
| `tests/` | unit, contract (`contracts/`), harness (`integration/`) and repository tests |

`.rag/`, `secrets/` and `data/` hold runtime state and credentials and are never in git.

## Checks: just ci

For development (macOS or Linux): install `uv`, `just`, Docker with compose and buildx (on macOS through Colima: `colima start
--cpu 6 --memory 12`), gitleaks and pre-commit; then `uv sync --frozen --all-packages`, `pre-commit install --hook-type
pre-commit --hook-type pre-push` and `just ci`.

`just ci` runs, identically on a development machine and in CI: `lint` (ruff, deptry, gitleaks, the three locks), `types`
(basedpyright), `imports` (import-linter), `unit` (pytest, the coverage ratchet), `cli` (the command line's tests in an
environment holding the root project alone), `db-tests` (the tests that need a migrated pgvector), `contracts` (shared
fixtures, the stream schema, the profile's data), `model-servers` (each model server's tests in its own environment, with fake
models), `images`, `harness` (the real backend and indexer images against fake model servers) and `page` (the page's Node
tests, then its browser suite in Chromium against streams recorded from the TU leg; it needs Node and downloads Chromium on
the first run). `just pre-push` (the part without Docker) runs before every push.

`just harness example` runs the example profile's leg: the ten law texts of `profiles/example` are indexed and queried through
the real images.

`just harness tu` runs the TU profile's leg: the crawler reads the fixture mini-site (`packages/rag_testing`, two services of
the harness network that stand in for the faculty's and the university's hosts) under the profile's own host names, the
indexer's overlay routing those two hosts to the services, and the questions are asked of the index it builds. Nothing is
fetched from the internet.

## Licence

MIT; see [`LICENSE`](LICENSE). Third-party models, vendored libraries and data sources keep their own licences and terms and
are listed in [`NOTICE`](NOTICE). The crawled content of the university's and the authorities' sites is not part of this
repository and not covered by its licence; the short quotes from those sites in the profile and its smoke set remain their
publishers'.
