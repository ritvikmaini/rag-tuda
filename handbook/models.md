# Models

All models are open-weight and reached through one OpenAI-compatible adapter: Ollama on the machine, Groq's hosted endpoint,
or a vLLM server. No proprietary model API is used.

## Cards

A card (`models/cards/<name>.toml`, TOML so that the standard-library-only command line reads it) holds a model's facts and
the system's behaviour for it: provider and URL, the context window, the token budget (`window_tokens`, `answer_tokens`,
`final_k`, `full_sources`, history), the timeouts (`deadline_s`, `idle_s`, `llm_read_s`, `analysis_s`, `retries`), the
reasoning setting, the sampling values and the structured-output mode. The window is the smallest of `budget.window_tokens`,
the model's `context_window` and, for a hosted card, the free tier's per-minute token limit.

| Class | Cards | Download | Window tokens |
|---|---|---|---|
| S | `s-qwen3-4b-instruct-2507` (default) | 2.5 GB | 8000 |
| S | `s-qwen3.5-4b` | 4.0 GB | 8000 |
| S | `s-gemma4-e4b` | 9.5 GB | 8000 |
| M | `m-qwen3.6-35b-a3b` | 22.6 GB | 16000 |
| M | `m-qwen3.8-27b` | 17.7 GB | 16000 |
| M | `m-gemma4-31b` | 20.4 GB | 16000 |
| L | `l-groq-gpt-oss-120b`, `l-groq-gpt-oss-20b`, `l-groq-qwen3.8-27b` | hosted | 8000 (the free tier's `limits.tpm`) |
| XL | `xl-vllm-example` | a vLLM server | 64000 |

A card whose answer provider is remote must name its own analysis block, so that analysis stays on a local model; `rag up`
refuses otherwise. The XL card is shipped as an example to copy and has not been run.

`rag doctor` lists the cards that fit the machine; `rag up` without `--card` takes the default card of the class that fits.

## What the S cards were measured at

On an M2 Pro with 16 GB (10.4 GB usable), Ollama 0.40.17, AC power, with a prompt of the card's full 8000-token window (7.0K to
7.5K tokens) and a 700-token answer:

| Card | Prefill | Decode | First token | 700-token answer | Cold load |
|---|---|---|---|---|---|
| `s-qwen3-4b-instruct-2507` | 365 tokens/s | 36 tokens/s | 21 s | 41 s | 1 s |
| `s-qwen3.5-4b` | 375 tokens/s | 42 tokens/s | 20 s | 37 s | 1.5 s |
| `s-gemma4-e4b` | 570 tokens/s | 41 tokens/s | 18 s | 35 s | 5 s |

The S cards' timeouts (`deadline_s` 150, `idle_s` 60, `llm_read_s` 60, `analysis_s` 40, `retries` 1) are well above these cases: `idle_s`
also bounds the silence of one `get_docs` call, whose stage timeouts add up to about 57 s (retrieval 10 + 2 s, parent lookup 5 s,
reranker 15 s, then the reference hop's 5 + 5 + 15 s), so the limits leave room for the stack's own waits. `window_tokens` is the
largest budget whose first token stays under half of `idle_s`: the full 8000-token window's first token is at most 21 s, a 16K
prompt takes 29 to 61 s. `s-gemma4-e4b` stays at 8000 for memory too: the model takes 9.5 GB of the 10.4 GB usable, and its 16K prompt's
first token (29 s) is just under half of `idle_s`. Ollama 0.40.17 answers HTTP 501 ("structured
output is unavailable") to a `response_format` for `qwen3.5:4b` and `gemma4:e4b`, whose cards use `structured_output = "prompt"`
(10 of 10 analysis replies were valid JSON of the schema). On an M1 Max with 64 GB
(Ollama 0.40.12) the default card reached the first token of a 6K-token prompt after 16.4 s and decoded at 25 to 33 tokens/s.
The M, L and XL cards have no measurements.

## What the S cards scored

On the M2 Pro with 16 GB (Ollama 0.40.17, `LLAMA_ARG_CACHE_RAM=1024`, AC power unless stated), `rag eval --smoke` on the 35 questions of
`profiles/tu-informatik` against the full index (generation 6), one run per card:

| Card | ok | fail | stale | Median answer time |
|---|---|---|---|---|
| `s-qwen3-4b-instruct-2507` (default) | 32 | 2 | 1 | 26.3 s |
| `s-qwen3.5-4b`, on battery, earlier code state | 26 | 7 | 2 | 28.9 s |
| `s-gemma4-e4b`: not measured validly | (24) | (10) | (1) | (34.6 s, 35.8 s) |

The default card fails S26 and S30 and refuses S35 (printed `stale`, it stays in the denominator). Its runs: 27 of 30 (core index,
the 30-question version of the set) and 31 of 35 (full index, on battery) failed S17, S26 and S30; 32 of 35 (full index, the run in the table)
passed S17 with the same sources, so S17 flips between runs: the model's noise is one question. The `s-qwen3.5-4b` row was taken
on battery and on an earlier code state, before the fix that reads grouped citations such as "[1, 2]" (the relevance floor was 0.0,
as it is now; qwen3 models write no grouped citations, so the fix changes nothing for that card). `s-qwen3.5-4b` is six questions behind (an invalid citation in S16, a wrong date in S14, S27 wrongly "not listed"); the
default card is also the smallest download and not slower, so it is the class's default. `s-gemma4-e4b` has no valid run: in both
runs the reranker timed out on one to two retrievals (swap 7 to 10 GB; the model takes 9.5 GB of the 10.4 GB usable), and the
brackets only say it does not beat 32 of 35. A whole answer takes 25 to 26 s at the median on this Mac on AC (25.0, 25.9 and 26.3 s in three runs) and about 27 s in two runs on battery, mostly the prompt's
prefill (a median answer of 86 tokens decodes in 2 to 3 s). One machine, one run per cell; one question is 0.03.

## Embedder and reranker

Fixed across classes and pinned by revision in the profile: embedder `Snowflake/snowflake-arctic-embed-m-v2.0`, reranker
`BAAI/bge-reranker-v2-m3`. They are served by `model_servers/`, each its own uv project, on the device that exists (CUDA,
else MPS, else CPU). Both won their bake-off on the M2 Pro with 16 GB (AC power unless stated; default card, 27 or 32 golds, one run
per cell; a difference under 0.05 is noise and goes to the smaller or faster model):

| Embedder (core index, 27 golds) | Stage | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| arctic-embed-m-v2.0 (768, the pinned one) | fusion | 0.63 | 0.78 | 0.81 | 0.93 | 0.715 |
| | reranked | 0.59 | 0.81 | 0.81 | 0.81 | 0.679 |
| arctic-embed-l-v2.0 (1024) | fusion | 0.67 | 0.78 | 0.85 | 0.89 | 0.747 |
| | reranked | 0.56 | 0.85 | 0.85 | 0.85 | 0.679 |

| Reranker (full index, 32 golds, on battery) | hit@1 | hit@3 | hit@5 | hit@10 | MRR | Median retrieval |
|---|---|---|---|---|---|---|
| none (the fused list) | 0.53 | 0.75 | 0.78 | 0.91 | 0.657 | |
| bge-reranker-v2-m3 (the pinned one) | 0.56 | 0.75 | 0.84 | 0.84 | 0.662 | 3.16 s |
| Qwen3-Reranker-0.6B | 0.56 | 0.66 | 0.81 | 0.88 | 0.646 | 8.92 s |

After reranking `l` is 0.04 ahead in hit@5 and equal in MRR, so the smaller `m` stays (median retrieval 3.25 s against 3.50 s; its
build took 3 min 14 s for 3666 sections and 15472 chunks (14481 embedded) against 5 min 46 s for 3637 sections and 15373 chunks
(14382 embedded): corpora of different size). The Qwen reranker is 2.8 times slower per retrieval (8.92 s against 3.16 s) and 3.2 times
slower for the reranker call alone (8.0 s against 2.5 s), peaks at 7.2 to 7.9 GB against 5.9 to 6.5 GB, and its scores sit near 1.0 on questions without an answer, so no cut can use them. `threshold` 0.50 and `min_docs` 3 stay. `Qwen3-Embedding-4B` (the load alone grew swap by
more than 4 GB) and `Qwen3-Reranker-4B` (aborted by the same guard) did not fit this Mac and are not measured. Reranking lowers
hit@10 against the fused list (0.91 to 0.84), and the embedder was decided on the core index only.

## A 16 GB Mac

The S stack (Postgres in the Colima VM, the embedder, the reranker, Ollama with the default card at the full 8000-token window)
shares the 10.4 GB a 16 GB Mac leaves usable. Measured on an M2 Pro (macOS 26), footprints from `footprint -p <pid>`; the reranker-alone rows and the default-Ollama `llama-server` row were taken on battery, the embedder-alone row's power source was not recorded, the other rows are on AC:

| Part | Alone | In the stack, 27 retrievals (retrieval only; Ollama nearly idle) |
|---|---|---|
| Embedder | 2.65 GB idle; 30 query embeds 0.1 s each; a batch of 128 chunks 1.6 s, 2.9 GB | 2.7 GB |
| Reranker, 37 documents, batch 4 (the MPS default) | 3.0 GB idle; peak 5.9 GB; 2.5 s a request | 4.6 GB (peak 5.1) |
| Reranker, `RERANKER_PREDICT_BATCH_SIZE=32` | peak 11.2 GB; 5.5 s a request | |
| `llama-server`, default Ollama | grows to 10.2 GB (peak 11.2) over 20 requests | |
| `llama-server`, `LLAMA_ARG_CACHE_RAM=1024` | 2.4 to 3.1 GB | 2.4 GB (peak 3.1), as Ollama answers nothing in this run |
| Colima VM (the default 4 CPUs, 6 GiB) | 1.1 GB fresh, 1.5 GB with Postgres in use | 1.5 GB |

- **Reranker.** On MPS the server scores 4 pairs per batch (`RERANKER_PREDICT_BATCH_SIZE` overrides it in native mode, the compose files pass none; CUDA and the CPU keep 32).
  Scores differ from batch 32 by at most 0.001 and the order of the best eight is the same (measured with bge-reranker-v2-m3 in
  cross-encoder mode only). Emptying the MPS cache after a
  request does not lower the peak.
- **Ollama.** Its prompt cache (`LLAMA_ARG_CACHE_RAM`, default 8192 MiB) keeps the state of every earlier prompt; with 7K-token
  answer prompts it fills the Mac's memory and sends it into swap, which times out the reranker and the semantic search. Start Ollama
  with `LLAMA_ARG_CACHE_RAM=1024 ollama serve`. The cost: the analysis prompt's shared prefix is then kept only part of the time, so an
  analysis takes 1 to 3 s when it is found and about 12 s when it is not (on 20 synthetic prompts, 17 of them answered: mean 6.9 s with
  1024, 3.2 s with the default (a run on battery; the other two on AC), 10.2 s with 0, which holds `llama-server` at 1.4 GB; the other 3 were over the 8000-token window and
  got HTTP 400). The answer's time does not change (median 20.4 s, about 20 s for these prompts). `rag up` does not start or
  configure Ollama, so the setting belongs to whoever starts it. With the macOS app, run `launchctl setenv LLAMA_ARG_CACHE_RAM 1024`
  and restart the app.
- **Colima.** Postgres with this index used 1.5 GB of the VM's footprint, which follows what Postgres touches (1.1 GB fresh) and does not fall
  again until `colima stop` and `colima start`. A fresh start before a long run keeps it small.
- **Result.** With these settings `rag eval --retrieval-only` ran 27 golds in 4 minutes with no `RerankError` and no
  "semantic search failed", swap at most 4.8 GB (20.6 GB with the default Ollama and batch 32). The three model processes together
  were at 9.8 GB, under the 10.4 GB, without the VM's 1.5 GB (with it about 11.2 GB); the sum of their peaks, 3.1 + 5.1 + 2.7, is
  10.9 GB. That reading is from a retrieval-only run with Ollama nearly idle, not from answers. A stack that has run several builds
  is larger until it is restarted (the reranker keeps 5.2 GB after use against 3.0 GB fresh). The reranker's batch sizes 2 and 1
  were not measured. On the healthy stack a 30-question version of the smoke set, with answers, ran with no degraded retrieval and no
  timeout (3 of the 30 failed their content checks only); the 35-question run
  of the default card, also with no degraded retrieval, stood at 6.8 to 8.3 GB.
  Watch it with `sysctl vm.swapusage`; `rerank_failed: True` in `.rag/logs/backend.log` means the answers came from unranked documents.

## Reasoning and streaming

One card value (`off`, `low`, `medium`, `high`) is mapped per provider. A model that cannot switch reasoning off uses `low`.
Any chunk from the model, reasoning included, resets the idle timer; reasoning text is never shown or stored.

## Hosted cards

A `l-groq-...` card sends the answer to Groq with the key from `GROQ_API_KEY`. Questions leave the machine only for that call,
and the page shows it. The free tier's data terms have not been read in full: no personal matters while a remote card is
active.

## vLLM

`deploy/compose.vllm.yaml` serves the XL card's model at a pinned revision and loads the weights offline from the host's
Hugging Face cache (`HF_HUB_OFFLINE=1`). Download the revision named by `--revision` in that file first. To serve another
model, copy the card and change both together.
