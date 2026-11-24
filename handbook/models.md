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
(10 of 10 analysis replies were valid JSON of the schema). An earlier measurement of the default card on an M1 Max with 64 GB
(Ollama 0.40.12): a 6K-token prompt reached its first token after 16.4 s and decoding ran at 25 to 33 tokens/s. The M, L and XL
cards have no measurements yet.

## Embedder and reranker

Fixed across classes and pinned by revision in the profile: embedder `Snowflake/snowflake-arctic-embed-m-v2.0`, reranker
`BAAI/bge-reranker-v2-m3`. They are served by `model_servers/`, each its own uv project, on the device that exists (CUDA,
else MPS, else CPU). The bake-off that may replace them has not run.

## A 16 GB Mac

The S stack (Postgres in the Colima VM, the embedder, the reranker, Ollama with the default card at the full 8000-token window)
shares the 10.4 GB a 16 GB Mac leaves usable. Measured on an M2 Pro (macOS 26, AC power), footprints from `footprint -p <pid>`:

| Part | Alone | In the stack, 27 retrievals (retrieval only; Ollama nearly idle) |
|---|---|---|
| Embedder | 2.65 GB idle; 30 query embeds 0.1 s each; a batch of 128 chunks 1.6 s, 2.9 GB | 2.7 GB |
| Reranker, 37 documents, batch 4 (the MPS default) | 3.0 GB idle; peak 5.9 GB; 2.5 s a request | 4.6 GB (peak 5.1) |
| Reranker, `RERANKER_PREDICT_BATCH_SIZE=32` | peak 11.2 GB; 5.5 s a request | |
| `llama-server`, default Ollama | grows to 10.2 GB (peak 11.2) over 20 requests | |
| `llama-server`, `LLAMA_ARG_CACHE_RAM=1024` | 2.4 to 3.1 GB | 2.4 GB (peak 3.1), as Ollama answers nothing in this run |
| Colima VM (the default 4 CPUs, 6 GiB) | 1.1 GB fresh, 1.5 GB with Postgres in use | 1.5 GB |

- **Reranker.** On MPS the server scores 4 pairs per batch (`RERANKER_PREDICT_BATCH_SIZE` overrides it; CUDA and the CPU keep 32).
  Scores differ from batch 32 by at most 0.001 and the order of the best eight is the same (measured with bge-reranker-v2-m3 in
  cross-encoder mode only). Emptying the MPS cache after a
  request does not lower the peak.
- **Ollama.** Its prompt cache (`LLAMA_ARG_CACHE_RAM`, default 8192 MiB) keeps the state of every earlier prompt; with 7K-token
  answer prompts it fills the Mac's memory and sends it into swap, which times out the reranker and the semantic search. Start Ollama
  with `LLAMA_ARG_CACHE_RAM=1024 ollama serve`. The cost: the analysis prompt's shared prefix is then kept only part of the time, so an
  analysis takes 1 to 3 s when it is found and about 12 s when it is not (on 20 synthetic prompts, 17 of them answered: mean 6.9 s with
  1024, 3.2 s with the default, 10.2 s with 0, which holds `llama-server` at 1.4 GB; the other 3 were over the 8000-token window and
  got HTTP 400). The answer's time does not change (median 20.4 s, about 20 s for these prompts). `rag up` does not start or
  configure Ollama, so the setting belongs to whoever starts it. With the macOS app, run `launchctl setenv LLAMA_ARG_CACHE_RAM 1024`
  and restart the app.
- **Colima.** Postgres with this index used 1.5 GB of the VM's footprint, which follows what Postgres touches (1.1 GB fresh) and does not fall
  again until `colima stop` and `colima start`. A fresh start before a long run keeps it small.
- **Result.** With these settings `rag eval --retrieval-only` ran 27 golds in 4 minutes with no `RerankError` and no
  "semantic search failed", swap at most 4.8 GB (20.6 GB with the default Ollama and batch 32). The three model processes together
  were at 9.8 GB, under the 10.4 GB, without the VM's 1.5 GB (with it about 11.2 GB); the sum of their peaks, 3.1 + 5.1 + 2.7, is
  10.9 GB. On the healthy stack the smoke set's 30 questions with answers ran with no degraded retrieval and no timeout, swap was
  about 5.3 GB at the end of the run (3 of the 30 failed their content checks only).
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
