import os
from logging.config import dictConfig

# Gunicorn configuration file

# Server socket
bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8000")

# Worker processes: one. The GPU pipeline slot (MAX_CONCURRENT_RAG_REQUESTS) is an asyncio semaphore per process, so a
# second worker would be a second GPU consumer and freeze the card. gunicorn's master imports this file before it forks:
# more workers than the slot stop the start with the message below.
MAX_WORKERS = 1
workers = int(os.environ.get("UVICORN_WORKERS", "1"))
if workers > MAX_WORKERS:
    raise SystemExit(
        f"UVICORN_WORKERS={workers}: the backend runs with at most {MAX_WORKERS} worker (one GPU, one pipeline slot)"
    )
worker_class = os.environ.get("GUNICORN_WORKER_CLASS", "uvicorn.workers.UvicornWorker")

# Server mechanics
keepalive = int(os.environ.get("GUNICORN_KEEPALIVE", "5"))
# Increased timeout to handle long RAG pipelines (retrieval + reranking + LLM generation)
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "600"))

# Centralized logging configuration. RefFilter sits on the one handler every logger reaches, so every
# line carries `ref=<first 8 characters of the turn id>` of its request (`ref=-` outside one), rag_core's too, without an import
# (assistant_api.logref). The access log is off: its line holds the question (GET /api/chat?question=...). UvicornWorker
# copies gunicorn.access's handlers and level onto uvicorn.access, so no handler and level WARNING silence the line there too.
# The per-request line is the assistant's `[Turn] outcome=... code=... intent=... ttft_ms=... total_ms=...` (chat/turns_store.py).
LOG_LEVEL = os.environ.get("GUNICORN_LOGLEVEL", "INFO").upper()
logconfig_dict = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"ref": {"()": "assistant_api.logref.RefFilter"}},
    "formatters": {
        "default": {
            "()": "assistant_api.logref.ClassOnlyFormatter",  # frames and class, never an exception message
            "fmt": "%(levelprefix)s %(asctime)s - %(name)s - ref=%(ref)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "filters": ["ref"],
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": LOG_LEVEL,
    },
    # Ensure logs from libraries also use the root logger settings
    "loggers": {
        "gunicorn.error": {
            "level": LOG_LEVEL,
            "handlers": ["default"],
            "propagate": False,
        },
        "gunicorn.access": {
            "level": "WARNING",
            "handlers": [],
            "propagate": False,
        },
    },
}

# Apply the logging configuration
dictConfig(logconfig_dict)
