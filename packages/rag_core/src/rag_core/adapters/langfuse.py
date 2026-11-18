"""Langfuse for the assistant: the client, built when the runtime starts. The LangChain callback handler belongs to the graph
(assistant_api.graph.tracing)."""

import logging
import os
import time
from pathlib import Path
from typing import Any

from langfuse import Langfuse


def init_langfuse(
    url: str, secrets_dir: str | Path = "/run/secrets", max_retries: int = 5, retry_delay: float = 2.0
) -> tuple[Any | None, bool]:
    """(client, authenticated). A client whose auth check failed is still returned: it traces once Langfuse answers.
    None only when the setup failed (no key files)."""
    client: Any | None = None
    try:
        public_key = (Path(secrets_dir) / "langfuse_public_key").read_text().strip()
        secret_key = (Path(secrets_dir) / "langfuse_secret_key").read_text().strip()
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key  # the SDK's handler reads the client it finds through these
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = url
        for attempt in range(max_retries):
            try:
                client = Langfuse(public_key=public_key, secret_key=secret_key, host=url, timeout=5)
                if client.auth_check():
                    logging.info("Langfuse client is authenticated and ready!")
                    return client, True
                logging.warning(f"Langfuse auth check failed (attempt {attempt + 1}/{max_retries})")
            except Exception as e:
                logging.warning(f"Langfuse connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
        logging.warning("Langfuse authentication failed after all retries. Running without Langfuse.")
        return client, False
    except Exception as e:
        logging.warning(f"Langfuse setup failed: {e}")
        return client, False
