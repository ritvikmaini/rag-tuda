# Architecture

rag-tuda answers questions from an index of documents, with numbered citations, on open-weight models. This page follows a
document from the crawl to an answer. [`../README.md`](../README.md) has the diagram.

## Parts

| Part | Where | Role |
|---|---|---|
| command line | `src/rag_tuda/` | `rag doctor`, `up`, `down`, `reindex`, `rollback`, `eval`, `ask`, `add`; standard library only, reads the model cards with `tomllib` |
| backend | `apps/assistant_api/` | FastAPI and LangGraph; serves the page and `/api/*`; one pipeline slot per process |
| indexer | `apps/indexer/` | the crawler, connectors, chunking, index generations; run as a one-shot job (`indexer build`) |
| evals | `apps/evals/` | probe runner, gate, smoke; its own job image |
| embedding server | `model_servers/embedding/` | `snowflake-arctic-embed-m-v2.0`, port 8701 |
| reranker server | `model_servers/reranker/` | `bge-reranker-v2-m3`, port 8702 |
| database | `deploy/pgvector/` | Postgres with pgvector; native mode publishes `127.0.0.1:55432` |
| page | `web/page/` | the chat client (static files, vendored libraries) |

Every host port binds `127.0.0.1`. `/api/*` needs a Bearer token. The backend's HTTP access log is off, because the question
travels as a GET parameter. There is no telemetry.

## Runtime modes

- **Native** (Apple-silicon Mac): Postgres alone in Docker; the embedding server, reranker server, indexer and backend are
  processes of the user, with pid files in `.rag/pids` and logs in `.rag/logs`; Ollama runs on the host.
- **Compose** (x86_64 Linux or WSL2): the whole stack in `deploy/compose.yaml`, with Ollama as a service; `compose.nvidia.yaml`
  gives the model servers a GPU and `compose.vllm.yaml` serves the language model with vLLM.

The backend may call a remote OpenAI-compatible endpoint (Groq) for the answer. That is the only outbound call at question
time, and the page shows which model and provider answer. At build time the crawler is the outbound traffic, and the indexer
also downloads docling's layout and table models and the EasyOCR weights at the first PDF.

## Index

The indexer writes a new **generation** of the index (tables plus the metadata of each document) and repoints the views
`rag_chunks` and `rag_sections` to it in one swap. The backend reloads the new generation on its next start, which
`rag reindex` triggers. `rag rollback` repoints the views to the previous generation. Every build also writes a snapshot
manifest in `.rag/snapshots/`: per document the URL, content hash, fetch date and metadata.

Each document and its sections carry typed columns: `regulation` (`po`, `mhb`, `spp`, `apb`, `eso`, `promotion`, `residence`,
`info`, `form`), `degree`, `programme`, `po_version`, `archived`, `valid_from`, `lang` and `twin_key`. The crawler fills them
from the URL folder, the section heading, the link title and the file name; the language model never writes them. There is no
vector index: an exact scan over thousands of sections is fast and always returns `k` rows under a filter.

Chunking keeps the document structure: parents of about 2000 characters hold children of about 450.

## From question to answer

1. **Analysis.** One non-streamed call to the card's model returns a JSON object (language, standalone question, search
   queries in German and, for English questions, English, intent, degree, programme, PO version, document type, whether the
   question needs more than one document). Explicit text in the question ("PO 2015", "im Master"), the page chips and a
   term-to-PO parser outrank the model's values. If the JSON fails the schema, the analysis falls back to the plain question
   and the deterministic readers still run.
2. **Hybrid retrieval.** The question's vector against the embeddings and full text (`german`, plus `english` for English
   rows), with the hard filters applied in SQL. Acronyms from `acronyms.yaml` expand into OR groups inside the main query.
3. **Fusion cut.** Scores are normalised per leg and combined; hits merge across queries by their best fused score and only
   the top `pre_rerank_k` of the card go on.
4. **Rerank.** The reranker scores whole parents against the question; documents below the threshold are dropped, and with
   none left the "nothing found" answer names the filter that applied.
5. **Reference hop.** References such as "§ 22 APB" in the kept parents are resolved by pattern to the sections they name and
   added if the reranker keeps them. No model call.
6. **Iterative round** (M, L and XL cards only): for a question that needs more than one document, one extra round of
   follow-up queries.
7. **Answer.** The prompt is built within the card's window; the answer streams with `[n]` markers. The citations frame is
   built by code from the sources alone. Code appends the PO-version line, the residence disclaimer and the note that an
   English translation is not legally binding, whenever the cited sources call for them.

German answers use "Sie"; the answer follows the question's language, and where a German and an English version of a
regulation both exist, the binding German text wins.

## The stream

`GET /api/chat` answers with server-sent events (format 1): `meta` first (with the card's `deadline_s` and `idle_s`), then
`status` (five steps), `docs`, `chunk`, `citations` and `error` frames, and `done` last. The JSON schema is
`apps/assistant_api/stream_v1.schema.json`; `just stream-schema` regenerates it.

## Page

Static files served by the backend. Chips for degree, programme and PO version are sticky, need no login, and are sent with
each question; the "understood" step shows the filter that applied. Answers render as Markdown, tables, charts (Chart.js) or
mind maps (Markmap). The footer says the page is a student project, not an official service of TU Darmstadt, and not legal
advice.
