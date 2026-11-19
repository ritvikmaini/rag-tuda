"""The iterative round's model call: render the follow-up prompt, ask the analysis model within the time limit, parse."""

import asyncio
import logging
from collections.abc import Sequence

from rag_core.domain.follow_up import parse_follow_up
from rag_core.domain.query_analysis import bounded
from rag_core.ports.chat import ChatModel
from rag_core.services.prompts import FOLLOW_UP_PROMPT, PromptService
from rag_core.settings.profile import Identity

logger = logging.getLogger(__name__)


async def follow_up_queries(
    *,
    question: str,
    searched: Sequence[str],
    sections: str,
    identity: Identity,
    prompts: PromptService,
    model: ChatModel,
    timeout_s: float,
) -> list[str]:
    """Up to two further search queries for what the kept sections (their digest) leave open; none when the model is done,
    answers late or answers badly (round one's result then stands)."""
    variables = {
        "question": question,
        "searched": "\n".join(f"- {query}" for query in searched),
        "sections": sections,
        "bot_name": identity.bot_name,
    }

    async def call() -> str:
        messages = await asyncio.to_thread(prompts.render, FOLLOW_UP_PROMPT, variables)
        return await model.complete(messages)

    try:
        content, error = await bounded(call(), timeout_s)
        if error:
            raise RuntimeError(error)
    except Exception as e:
        logger.error(f"[FollowUp] call failed, keeping the first search: {str(e).partition(':')[0]}")
        return []
    queries = parse_follow_up(content, searched)
    logger.info(f"[FollowUp] queries={len(queries)}")  # a count only: a question and its rewrites never enter a log
    return queries
