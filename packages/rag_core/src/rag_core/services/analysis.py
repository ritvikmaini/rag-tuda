"""The query analysis use case: render the analysis prompt, ask the model within the time limit, parse. The graph's analyse
node calls it."""

import asyncio
import logging
from datetime import datetime

from rag_core.domain.dates import format_date
from rag_core.domain.query_analysis import QueryAnalysis, bounded, parse_query_analysis, retrieval_queries
from rag_core.ports.chat import ChatModel
from rag_core.services.prompts import PromptService
from rag_core.settings.profile import Identity

logger = logging.getLogger(__name__)


async def analyse(
    *,
    question: str,
    history: str,
    now: datetime,
    identity: Identity,
    prompts: PromptService,
    model: ChatModel,
    timeout_s: float,
    max_queries: int,
    first_turn: bool,
) -> tuple[QueryAnalysis, list[str]]:
    """Standalone question, intent, search queries and date range in one call; on any failure the question as typed."""
    variables = {
        "question": question,
        "history": history,
        "current_date": format_date(now, identity.date_format, identity.day_names),
        "today_iso": now.strftime("%Y-%m-%d"),
        "bot_name": identity.bot_name,
    }

    async def call() -> str:
        # the prompt fetch may wait for Langfuse: off the event loop and inside the limit
        messages = await asyncio.to_thread(prompts.render, "question-rewriting", variables)
        return await model.complete(messages)

    try:
        content, error = await bounded(call(), timeout_s)
        if error:
            raise RuntimeError(error)
    except Exception as e:
        logger.error(
            f"[QueryAnalysis] rewriter call failed, using the question as is: {str(e).partition(':')[0]}"
        )  # the reason is "<Class>: <message>"
        content = ""
    analysis = parse_query_analysis(content, question)
    queries = retrieval_queries(analysis, question, first_turn=first_turn)[:max_queries]
    logger.info(
        # counts and codes only: a user's question and its rewrites never enter a log
        f"[QueryAnalysis] parsed={analysis.parsed} intent={analysis.intent} queries={len(queries)} "
        f"dates={analysis.date_from}..{analysis.date_to} shape={analysis.answer_shape}"
    )
    return analysis, queries
