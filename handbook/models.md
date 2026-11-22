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
| S | `s-qwen3.5-4b` | 3.3 GB | 8000 |
| S | `s-gemma4-e4b` | 6.6 GB | 8000 |
| M | `m-qwen3.6-35b-a3b` | 22.6 GB | 16000 |
| M | `m-qwen3.8-27b` | 17.7 GB | 16000 |
| M | `m-gemma4-31b` | 20.4 GB | 16000 |
| L | `l-groq-gpt-oss-120b`, `l-groq-gpt-oss-20b`, `l-groq-qwen3.8-27b` | hosted | 8000 (the free tier's `limits.tpm`) |
| XL | `xl-vllm-example` | a vLLM server | 64000 |

A card whose answer provider is remote must name its own analysis block, so that analysis stays on a local model; `rag up`
refuses otherwise. The XL card is shipped as an example to copy and has not been run.

`rag doctor` lists the cards that fit the machine; `rag up` without `--card` takes the default card of the class that fits.

## What the S card was measured at

On one M1 Max with 64 GB and Ollama 0.40.12, `s-qwen3-4b-instruct-2507`: a prompt of about 6K tokens reaches its first token
after 16.4 s (about 400 tokens/s prefill); decoding runs at 25 to 33 tokens/s. The card's comment gives the full 8K window as
a first token after about 18 s and a 700-token answer ending after about 36 s. The card's timeouts are 1.5 to 2 times the slowest
case, for slower Macs. The M, L and XL cards have no measurements yet.

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
