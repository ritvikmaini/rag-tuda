# Design decisions

The choices where the design could have gone another way.

- **Acronyms and variants.** `acronyms.yaml` expands a matched term into an OR group inside the main full-text query
  ("PO" or "Prüfungsordnung"). A Postgres thesaurus file would need server-side files before the tables exist, so it is not
  used.
- **Tests.** Tests run against the example profile and the fixture mini-site; no test depends on data outside the
  repository.
- **Chips.** The page has chips (buttons) for degree, programme and PO version. They are sticky, need no login, go with each
  question and show in the "understood" step. Explicit text in a question outranks a chip for that question.
- **Retrieval eval.** `rag eval --retrieval-only` and `--ablation` are part of the minimal eval: the bake-offs need a
  retrieval score, and the ablation table is the evidence for the design.
- **Reference hop.** A deterministic hop resolves "§ n APB"-style references with no model call, on every class. The
  model-driven extra round is limited to the M, L and XL classes.
- **Official copies.** The per-programme PDFs are indexed; the Satzungsbeilage issues are linked as the official publication
  and not indexed; a department copy of a university-wide regulation yields to the original.
- **Build sets.** `rag up` builds the core source set; `rag reindex --full` adds the full set; the crawl resumes after an
  interruption.
- **The M-class default.** The mixture-of-experts `qwen3.6:35b-a3b` is the M-class default because it should answer fastest
  of the M cards; the M bake-off that would confirm it has not run yet.
- **Filters and NULL.** For `po`, `mhb` and `spp` documents, programme and PO version are required, or the document is
  not indexed and the build summary lists it. For every other type NULL means "applies to all". An empty filtered result gives
  a "nothing found" answer naming the filter, never a silent unfiltered retry.

## Other choices

- **No vector index.** The corpus is thousands of sections, so an exact scan is fast, and a filtered scan always returns `k`
  rows (a filtered approximate index may not).
- **Fail closed on versions.** A wrong-version or untagged regulation cannot be cited, because the hard filters and the
  indexing rule keep it out.
- **Quality floor in code.** The citations frame, the PO-version line, the residence disclaimer and the translation note are
  built in code, not left to the prompt; a decline happens when no source passes the reranker threshold.
- **One adapter, cards for the rest.** Model-specific workarounds live in the card's quirks, not in the code.
- **Open-weight only.** No proprietary model API; every runtime model is open-weight.
- **Crawled content stays local.** It lives in `.rag/crawl/`, never in git.
