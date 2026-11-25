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
(marker `local_crawl`, run on request and not by default: `uv run --frozen --all-packages pytest -m local_crawl tests/profiles`; another `-m` expression on the command line replaces the default one, so `pytest -m <something>` can run it too) checks the quotes against the crawl of the live generation of a local build; it needs a full-set build for S31 to S35. The set holds 35 questions: S01 to S30 on the core set and S31 to S35 on the full
set's sources (the TU-wide study pages, the Landkreis pages); 32 have a gold source and S18 to S20 have none. The example profile has its own probes in `profiles/example/probes/`.

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

After the table, one block per row lists the probes, a line per conversation (set/id) with fixed columns: `probe kind returned fused
rank gold_score top_score status`, `-` for no value. `kind` is `gold` or `none`: a conversation without a gold is a no-answer
probe, retrieved like the others and left out of the table. `returned` is the length of the final list the retrieval hands to
the answer, and 0 is a decline. `fused` and `rank` are the gold's rank after fusion and after reranking. `gold_score` is the
gold's `cross_encoder_score` (the best section that is the gold) and `top_score` the best score among all candidates the
reranker scored, before the threshold, the relevance floor and the cut: the numbers to read `threshold`, `relevance_floor` and
`min_docs` off. The abstain of the ranking compares `relevance_floor` with the best score after the recency and date bonuses, so
it can differ from `top_score` when a profile sets a bonus. `status` is `ok`, `rerank_failed` (the reranker did not answer; its
0.0 scores are no evidence, so both score columns show `-`) or `error` (the retrieval failed).

`--ablation` runs one row per stage turned on, cumulatively: no filters, + filters, + acronyms, + fusion cut, + whole parents,
+ reference hop. It shows what each part of the design buys.

The ablation of the TU profile, on the M2 Pro with 16 GB (macOS 26, AC power, Ollama 0.40.17 with `LLAMA_ARG_CACHE_RAM=1024`),
full index (generation 6), arctic-embed-m-v2.0 and bge-reranker-v2-m3, default card, 32 golds, 210 retrievals without a failed rerank:

| Stage | hit@1 | hit@3 | hit@5 | hit@10 | MRR | Median latency |
|---|---|---|---|---|---|---|
| no filters | 0.50 | 0.66 | 0.72 | 0.81 | 0.608 | 17.28 s* |
| + filters | 0.56 | 0.75 | 0.81 | 0.81 | 0.661 | 14.36 s* |
| + acronyms | 0.56 | 0.72 | 0.81 | 0.81 | 0.659 | 13.78 s* |
| + fusion cut | 0.59 | 0.81 | 0.88 | 0.88 | 0.702 | 2.04 s |
| + whole parents | 0.59 | 0.78 | 0.88 | 0.88 | 0.694 | 2.28 s |
| + reference hop | 0.59 | 0.78 | 0.88 | 0.88 | 0.694 | 2.28 s |

\* These rows rerank about 117 sections: their retrieval takes a median 13.8 to 17.3 s (max 28.9 s); the first attempt, at the product's
15 s reranker limit, lost 53 reranks, so the run used a 60 s limit. Their latencies describe the ablation, not the product, whose retrieval is the last row. The run used a relevance floor of 0.1, the product's
is 0.0; the floor changed no gold's rank, only the final list of two questions.

The filters and the fusion cut each raise hit@5 by 0.07 to 0.09 and hit@1 by only 0.06 and 0.03, below the 0.10 called wide below. Whole parents, acronyms and the reference hop show no measurable gain on
this set, and the hop never added a section. Nothing reaches 1.00 (four golds, S05, S22, S26 and S30, are missing from the final list
from the fusion cut on), so the set is not too easy; the example profile's four-question run (measured on an M1 Max with 64 GB) does reach 1.00 after reranking and only
shows that the eval runs. Reading the numbers: one machine, one run per cell; 35 questions with 32 golds, so one question is 0.03 to
0.04, a difference under 0.05 is noise, 0.10 is wide, and no two rows differ by 0.10 except the first against the last three; the questions
were revised after a first run on the same set, so it is not a blind test; reranking lowers hit@10 against the fused list (0.91 to 0.84); the reranker table in
`handbook/models.md` (hit@5 0.84, MRR 0.662) comes from a set in which two golds name no German twin, unlike the ablation's, and must not be
put in one column with the last row (0.88, 0.694).

## Bake-offs

A bake-off runs each candidate on one index generation, in this order: (1) the embedder on retrieval-only hit@k and index time;
(2) the reranker on post-rerank hit@k and latency, then the refit of `threshold`, `relevance_floor` and `min_docs`; (3) answer
models per class on the smoke set: check pass rate, citation validity, version correctness, decline correctness, time to first
token, tokens per second, tokens per answer, with reasoning on and off where a model has both. The candidates are the embedders
`snowflake-arctic-embed-m-v2.0` (the pinned one), `snowflake-arctic-embed-l-v2.0` and `Qwen3-Embedding-4B`, and the rerankers
`bge-reranker-v2-m3` (the pinned one), `Qwen3-Reranker-0.6B` and `Qwen3-Reranker-4B`. The winners go into the cards and the
profile. A small smoke set detects only large differences: ties go to the smaller or faster option. The results (embedder
`snowflake-arctic-embed-m-v2.0`, reranker `bge-reranker-v2-m3`, default S card `s-qwen3-4b-instruct-2507`) and what is not measured (M, L and XL
cards, `Qwen3-Embedding-4B`, `Qwen3-Reranker-4B`, Groq, `s-gemma4-e4b` validly, a larger question set) are in `handbook/models.md` and the README,
each with its machine.

## Hosted-model runs

For a Groq card, analysis, embedding and reranking stay local and only the generation call goes to Groq. A run paces itself
to the card's `limits.tpm` and the account's daily cap, which is checked on the account's limits page first.

## Deferred

A larger question set, the full panels, a noise-floor gate, a cross-model comparison table, and citation verification as a metric.
