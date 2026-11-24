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
