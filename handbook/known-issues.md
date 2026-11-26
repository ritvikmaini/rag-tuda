# Known issues

What is known not to work, and what has never been run, as of November 2026. Everything measured is described in
[`evaluation.md`](evaluation.md) and [`models.md`](models.md), each with its machine.

## Not tested yet

- **The M cards** (`m-qwen3.6-35b-a3b`, `m-qwen3.8-27b`, `m-gemma4-31b`): written, never run. Their 16K window and the extra
  search round for questions that need several documents have run only against fake models.
- **The L cards** (Groq): never run (no API key was available). Groq's free-tier data terms have not been read in full.
- **The XL card** (vLLM on NVIDIA GPUs): an example to copy, documented but never run.
- **`s-gemma4-e4b`**: no valid run on a 16 GB Mac (the reranker timed out under swap in both runs).
- **The larger embedder and reranker candidates**, `Qwen3-Embedding-4B` and `Qwen3-Reranker-4B`: they did not fit a 16 GB Mac.
- **Compose mode with real models** on x86_64 Linux or with NVIDIA GPUs: CI runs compose only against fake model servers.
  `rag down --purge` against files a container left as root has not been checked on its own.
- **Memory while answers are generated.** The 9.8 GB reading is from a retrieval-only run; during the 35-question run the Mac
  stood at 6.8 to 8.3 GB of swap.
- **A full build from nothing** has not been timed (the 42 minutes are two builds on a cached core index).
- **Real Safari.** The page's browser checks passed in WebKit (Playwright) and Chromium.
- **A larger question set.** The TU set has 35 questions, written for this project and checked against the source text by
  an AI reviewer; no person has reviewed them yet.

## Known defects

Answers:

- On the TU set, the default card fails S26 and S30 and wrongly declines S35; S17 passes in some runs and fails in others.
- On a question that spans two topics, the small class may find sources for only one and answer the other from the model's
  own knowledge, at times citing the source it did find. The larger classes' extra search round is meant for this case.
- The "shorter" version of one answer (S05) once contradicted the original.
- The small model's prose can misname a document's structure (for example "Absätze 4 und 5" for §§ 4 and 5) even when the
  cited content is right.
- Some sources contradict each other (the ISS checklist and the Landkreis page, the Erasmus rates, the bank-account advice,
  an old 2021 block on one FAQ page); the assistant may quote either.

Index and sources:

- The full index includes about 500 news pages and some pages off the study topic (prospective students, campus management,
  the student union).
- The department's knowledge-base articles redirect to hosts outside the allow-list and are not indexed.
- The snapshot manifest lists 4 pages fewer and 5 PDFs more than the database counts for the same generation; not yet
  investigated.
- `rag doctor`, `rag up` and `rag reindex` print design estimates for the crawl (about 300 pages and 120 to 150 PDFs for the
  core set) rather than the measured sizes in [`sources.md`](sources.md#index-sizes).
- Model downloads take more disk than announced, because whole model repositories are fetched.
- Text hidden by a CSS class and the text of PDFs are not filtered for prompt injection.

The page:

- The browser's own print command (Cmd+P or Ctrl+P) is not print-styled; use the page's own print action.
- A source's tooltip can show table markup.
- Tables of credit points ("10 CP") are never offered as charts.

The command line:

- `rag rollback` right after a profile switch is refused; go back with `rag up --profile <name>`.
- After `rag rollback` or `indexer publish` of a full-set generation, the next core build is refused as a shrink, because
  those records name no source set; run `rag reindex --full`, or delete `corpus_build_stats.json` to accept it.
- The S cards' timeouts were measured on an M1 Max and an M2 Pro; a base-chip Mac may need `idle_s`, `llm_read_s` and
  `deadline_s` raised (see [`operations.md`](operations.md#troubleshooting)).
