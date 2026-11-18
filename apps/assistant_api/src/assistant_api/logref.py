"""The error reference on every log line: the chat route stores the first 8 characters of the turn id in a ContextVar;
RefFilter, set on the one handler every logger reaches (gunicorn_conf.py), adds it as `ref=` to each record (`ref=-` outside
a request). ClassOnlyFormatter, the handler's formatter, prints a traceback without any exception message.
asyncio tasks and asyncio.to_thread copy the context; langchain_core's run_in_executor wraps the call in
copy_context().run. A raw loop.run_in_executor, ThreadPoolExecutor or threading.Thread loses it (tests/assistant_api/
test_logref.py keeps the product free of them)."""

import logging
import traceback
from contextvars import ContextVar

from uvicorn.logging import DefaultFormatter

REF: ContextVar[str] = ContextVar("ref", default="-")


def set_ref(turn_id: str) -> None:
    REF.set(turn_id[:8].lower())


class RefFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ref = REF.get()
        return True


class ClassOnlyFormatter(DefaultFormatter):
    """A logged exception prints its frames and its class, never a message: messages carry input values (pydantic input_value=,
    psycopg DETAIL, the tsquery of a search), and the line must not hold a question. Chained exceptions are not followed."""

    def formatException(self, ei) -> str:  # type: ignore[override]
        cls, _, tb = ei
        return "".join(traceback.format_tb(tb)) + (cls.__name__ if cls else "")
