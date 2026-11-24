# Evaluation

Three things are measured against the running stack: that a deploy works (the smoke), whether answers hold up (the probe
runs), and where the right source ranks (the retrieval eval). `rag eval` reads `<profile>/gate.yaml`: `rag eval` runs its fixed
probes (the deploy smoke), `rag eval --smoke` plays its smoke set (the set `smoke`) with the answer checks.

## The TU question set

The TU question set is `profiles/tu-informatik/smoke/smoke_v1.json`: hand-written questions in the evals runner's probe format
(`{"version", "conversations": [{"id", "set", "kind", "turns": [{"q", "chips", "gold", "expect"}]}]}`). `expect` holds the checks
of the expected facts. `gold` is the source: `{url, heading | page, quote, po_version}`, the quote one sentence of one section,
whitespace-normalised, copied from the extracted text for a PDF; a gold of several sources adds `also` (more sources of the same
shape) and `rule`, `any` (twins of one page) or `all` (several documents). `chips` are the page's `degree`, `programme` and `po`,
sent with the question. A question with no answer in the corpus has no gold. The fees and dates are listed as `retrieval.volatile` in
`gate.yaml`. A gold whose quote is no longer found in any section of its url is reported **stale**, not failed, by
`rag eval --retrieval-only` (and a failed volatile answer is `stale` in `rag eval --smoke` when only content checks failed). The test `test_gold_present_in_latest_snapshot`
(marker `local_crawl`) checks the quotes against a local crawl. The questions on the full source set (TU-wide study pages, the
Landkreis pages) are added after the full build. The example profile has its own probes in `profiles/example/probes/`.

## The deploy smoke

`rag eval` runs the fixed probes of `gate.yaml` (five in the example profile: about six requests, about two minutes) and checks structure only, so a
correct deploy never fails on answer variance: no error frames, the time to first token, the sequence of status steps, and the
`docs`, `chunk` and `citations` frames.

## Checks on answers

The answer checks run in the gate and probe runs of `apps/evals`, as `expect` keys of a probe: every `[n]` maps to a source; the
answer's language equals the question's; the PO-version line is present when a version-bound source is cited; "Sie" in German
answers; the residence disclaimer when a residence source is cited.

`evals run` (and so `rag eval --smoke`) exits 0 whatever the verdicts, so a person reads its `RESULT` line: `n ok, n FAIL, n stale of n
turns`. A volatile turn is `stale` only when content checks (`contains_any`, `digits_any`, `source_url_any`) failed; an error frame,
an empty answer, a wrong language, a missing "Sie" or a broken citation is `FAIL` for every turn.

## Retrieval eval

`rag eval --retrieval-only` reads the probe sets of `gate.yaml`, runs against the native stack only, and reports, after fusion and after reranking, hit@1,
hit@3, hit@5, hit@10 and MRR. Only the analysis call uses the language model, and it is cached per run. The code lives in
`apps/assistant_api/src/assistant_api/retrieval_eval.py` because the import contract bars `evals` from importing the
assistant. `--sets` (comma-separated set names) and `--limit` (the first N conversations of each set) are options of
`python -m assistant_api.retrieval_eval`, the module `rag eval --retrieval-only` runs; `rag eval` itself has only
`--retrieval-only` and `--ablation`.

`--ablation` runs one row per stage turned on, cumulatively: no filters, + filters, + acronyms, + fusion cut, + whole parents,
+ reference hop. It shows what each part of the design buys.

The one run available today is on the example profile, four questions with a gold source: after fusion hit@1 0.25 and MRR
0.625, after reranking hit@1 1.00 and MRR 1.000. It shows the eval runs end to end and says nothing about TU's documents. The
TU ablation table is pending the TU smoke set and index.

## Bake-offs (not run yet)

Planned order, each on one index generation: (1) the embedder on retrieval-only hit@k and index time; (2) the reranker on
post-rerank hit@k and latency, then the refit of `threshold`, `relevance_floor` and `min_docs`; (3) answer models per class on
the smoke set: check pass rate, citation validity, version correctness, decline correctness, time to first token, tokens per
second, tokens per answer, with reasoning on and off where a model has both. Candidates: embedders `snowflake-arctic-embed-m-v2.0`
(today), `snowflake-arctic-embed-l-v2.0` and `Qwen3-Embedding-4B`; rerankers `bge-reranker-v2-m3` (today),
`Qwen3-Reranker-0.6B` and `Qwen3-Reranker-4B`. Winners go into the cards and the profile. A small smoke set detects only large
differences: ties go to the smaller or faster option.

## Hosted-model runs

For a Groq card, analysis, embedding and reranking stay local and only the generation call goes to Groq. A run paces itself
to the card's `limits.tpm` and the account's daily cap, which is checked on the account's limits page first.

## Deferred

A larger question set, the full panels, a noise-floor gate, a cross-model comparison table, and citation verification as a metric.
