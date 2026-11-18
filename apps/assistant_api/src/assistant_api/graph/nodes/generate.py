"""The answer node: the response generator over the prompt service and the context budget; the model is awaited here, inside
the node, so LangChain's callbacks stream the answer. A documents answer to a question about the next or a coming event is
held back and replaced by one sentence when it writes a past date as coming. With the profile's status_groups switches the
node also tells the status groups apart: a question about the asker's own group drops the other sides' sources, a
comparison and a general question whose best sources differ by group get a prompt line, and a German answer cited only
from one side's sources is closed with a note on whom it concerns."""

import asyncio
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from rag_core.adapters.chat_openai import BUSY, TIMEOUT, TOO_LONG, accepts_extras, ainvoke_waiting, failure_of

from rag_core.domain.answer_shapes import (
    answer_language,
    format_instruction,
    language_reminder,
    shape_of,
    shape_reminder,
)
from rag_core.domain.context_budget import ContextBudget, fit_context, rank_score
from rag_core.domain.dates import asks_for_event, format_date, shown_as_coming, written_dates
from rag_core.domain.history import context_turns, conversation_state, select_turns
from rag_core.domain.query_analysis import RETRIEVAL_INTENTS
from rag_core.domain.status_groups import (
    SUBJECT,
    COMPARE,
    GENERAL,
    SELF,
    Scope,
    compare_labels,
    note_for,
    question_scope,
    source_group,
)

from assistant_api.graph.history import as_messages, insert_history, to_messages
from assistant_api.graph.state import AgentState, GraphDeps

logger = logging.getLogger(__name__)
CITATION = re.compile(r"\[(\d+)\]")


def with_reminder(messages, reminder):
    """A new list in which the last user message ends with the reminder (after a blank line). No reminder or no user
    message: the list is returned unchanged."""
    if not reminder:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            changed = HumanMessage(content=str(messages[i].content) + "\n\n" + reminder)
            return list(messages[:i]) + [changed] + list(messages[i + 1 :])
    return messages


def with_instruction(messages, instruction):
    """A new list in which the first SystemMessage ends with the format instruction (on a new line). The code appends it
    instead of a prompt variable, so every image works with every version of the prompt. No SystemMessage or an
    empty instruction: the list is returned unchanged."""
    if not instruction:
        return messages
    for i, message in enumerate(messages):
        if isinstance(message, SystemMessage):
            changed = SystemMessage(content=str(message.content) + "\n" + instruction)
            return list(messages[:i]) + [changed] + list(messages[i + 1 :])
    return messages


def holds_answer(state: Any) -> bool:
    """True when generate takes its documents branch for a question (the user's or the standalone one) about the next or a
    coming event and answers in German (the past words are German; an English variant streams): such an answer is not
    streamed but checked and sent whole."""
    current = str(state["messages"][-1].content)
    documents = (state.get("intent") or "lookup") in RETRIEVAL_INTENTS or state.get("scope_fallback")
    asked = asks_for_event(state.get("standalone_question") or current) or asks_for_event(current)
    return bool(documents and state.get("raw_docs") and asked and answer_language(state) == "Deutsch")


def past_event_reply(answer: str, context: str, today: date, texts: Any) -> str | None:
    """The sentence that replaces an answer writing a past date as coming, None for every other answer: each such
    date once, in the answer's wording and order; the first flagged sentence's citation, else the answer's first; "no newer
    date" only when no date in the context lies on or after today."""
    found = shown_as_coming(answer, today)
    if not found:
        return None
    wordings: dict[date, str] = {}
    for day, wording, _ in found:
        wordings.setdefault(day, wording)
    dates = list(wordings.values())
    if len(dates) == 1:
        text = texts.past_event.format(dates=dates[0])
    else:
        text = texts.past_events.format(dates=", ".join(dates[:-1]) + " und " + dates[-1])
    if not any(day >= today for _, _, day in written_dates(context, today)):
        text += texts.no_newer_event
    cites = CITATION.findall(found[0][2]) or CITATION.findall(answer)
    return text + (" [%s]" % cites[0] if cites else "") + "."


def usage_of(response: Any) -> dict[str, int]:
    """The tokens the provider counted for the call (prompt and completion), empty when it returned no usage."""
    usage = getattr(response, "usage_metadata", None) or {}
    if "input_tokens" not in usage or "output_tokens" not in usage:
        return {}
    return {"prompt_tokens": int(usage["input_tokens"]), "completion_tokens": int(usage["output_tokens"])}


def with_decoding_limits(llm: Any, quirks: Any, provider: str) -> Any:
    """The answer model with the card's logit bias and bad words bound to its requests where the provider accepts them (the
    model's own extra_body is kept, since a bound extra_body replaces it)."""
    if not accepts_extras(provider):
        return llm
    options: dict[str, Any] = {}
    if quirks.logit_bias:
        options["logit_bias"] = {str(token): bias for token, bias in quirks.logit_bias.items()}
    if quirks.bad_words:
        options["extra_body"] = dict(getattr(llm, "extra_body", None) or {}, bad_words=list(quirks.bad_words))
    return llm.bind(**options) if options and hasattr(llm, "bind") else llm


def make_generate(deps: GraphDeps):
    profile = deps.profile
    budget = ContextBudget(
        reserve=deps.card.budget.answer_tokens,
        margin=profile.guard.safety_margin,
        window_chars=profile.guard.source_window_chars,
        full_sources=deps.card.budget.full_sources,
    )
    labels = deps.messages.labels.source_labels()

    def answer_prompt(name: str, input_messages: list[Any], variables: dict[str, Any], position: str) -> list[Any]:
        """The messages of an answer prompt: the rag-generation form instruction and reminders added, the history inserted."""
        formatted = to_messages(deps.prompts.render(name, variables))
        reminders = [variables.get("language_reminder", "")]
        if name == "rag-generation":
            formatted = with_instruction(formatted, variables.get("format_instruction", ""))
            # the form line (with a length variant's line), then the English reminder: the last thing the model reads
            reminders = [variables.get("scope_reminder", ""), variables.get("shape_reminder", ""), *reminders]
        formatted = with_reminder(formatted, "\n\n".join(r for r in reminders if r))
        return insert_history(formatted, list(input_messages), position)

    def direct_prompt(variables: dict[str, Any], history: list[Any]) -> list[Any]:
        """The direct-reply messages, the history right after the system message."""
        formatted = with_reminder(
            to_messages(deps.prompts.render("direct-reply", variables)), variables["language_reminder"]
        )
        return insert_history(formatted, history, "after_system")

    async def generate(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        current = state["messages"][-1]
        intent = state.get("intent") or "lookup"
        position = profile.history.position
        prior = context_turns(
            state["messages"], deps.card.budget.history_turns, profile.history.max_age_hours, datetime.now(timezone.utc)
        )
        chosen = await select_turns(prior, deps.card.budget.history_tokens, deps.counter.count_text)
        history = as_messages(chosen)
        logger.info(
            f"[History] completed_fresh={len(prior)} kept={len(chosen)} stored_messages={len(state['messages'])}"
        )
        variables = {
            "question": state.get("standalone_question") or current.content,
            "context": state.get("formatted_docs", ""),
            "language": answer_language(state),
            "current_date": format_date(datetime.now(), profile.identity.date_format, profile.identity.day_names),
            "bot_name": profile.identity.bot_name,
            "intent": intent,
            "format_instruction": format_instruction(state),
            "language_reminder": language_reminder(state),
            "shape_reminder": shape_reminder(state),
        }
        sources: list[Any] = []
        groups = profile.status_groups
        table = groups.groups
        status_on = bool(table) and (
            groups.filter or groups.compare or groups.reminder != "off" or groups.note != "off"
        )
        scope = Scope(GENERAL, frozenset(), False)
        seen: set[str] = set()  # the sides of the best-ranked sources
        group_of: dict[Any, str | None] = {}  # context source number -> its status group
        if intent not in RETRIEVAL_INTENTS and not state.get("scope_fallback"):
            # About the bot, conversation requests, out of scope, urgent: no retrieval (unless the search found a strong hit
            # for an out-of-scope question, which takes the documents branch below).
            direct = dict(variables, question=current.content, conversation_state=conversation_state(len(chosen)))
            prompt = await asyncio.to_thread(direct_prompt, direct, history)
        elif state.get("raw_docs"):
            candidates: Any = state.get("grouped_raw_docs") or state["raw_docs"]
            if status_on:
                try:
                    scope = question_scope(table, str(current.content), state.get("standalone_question") or "")
                    if groups.filter and scope.kind in (SELF, SUBJECT) and len(scope.groups) == 1:
                        (own,) = scope.groups
                        keep = {None, own, table[own].part_of}  # its own, its parent group's and the general sources
                        kept = [d for d in candidates if source_group(d.metadata, table, groups.overrides) in keep]
                        logger.info(f"[StatusGroups] filter={own} dropped={len(candidates) - len(kept)}")
                        candidates = kept or candidates
                    if groups.compare and scope.kind == COMPARE:
                        variables["scope_reminder"] = deps.messages.answer.scope_compare.format(
                            groups=compare_labels(table, scope.groups)
                        )
                except Exception as exc:  # the groups never cost an answer: all sources stay
                    logger.warning(f"[StatusGroups] scope_failed={type(exc).__name__}")
            fixed_prompt = await asyncio.to_thread(
                answer_prompt, "rag-generation", [], dict(variables, context=""), position
            )
            fitted = await fit_context(budget, deps.counter, fixed_prompt, history, candidates, labels=labels)
            variables["context"] = fitted["context"]
            sources = fitted.get("sources", [])
            if status_on:
                try:
                    by_section = {
                        (d.metadata or {}).get("section_id"): source_group(d.metadata, table, groups.overrides)
                        for d in candidates
                    }
                    group_of = {s["n"]: by_section.get(s.get("section_id")) for s in sources}
                    # the best-ranked sources in the context (its numbers follow the documents, not the ranking)
                    shown = {s.get("section_id") for s in sources}
                    best = sorted(
                        (d for d in candidates if (d.metadata or {}).get("section_id") in shown), key=rank_score
                    )
                    top = best[::-1][: groups.reminder_sources]
                    seen = {
                        table[g].side for g in (by_section.get((d.metadata or {}).get("section_id")) for d in top) if g
                    }
                    if (
                        groups.reminder == "differ"
                        and scope.kind == GENERAL
                        and not scope.contact
                        and shape_of(state.get("answer_shape")) == "text"
                        and len(seen) > 1
                    ):
                        variables["scope_reminder"] = deps.messages.answer.scope_differ
                except Exception as exc:
                    logger.warning(f"[StatusGroups] reminder_failed={type(exc).__name__}")
            prompt = await asyncio.to_thread(answer_prompt, "rag-generation", fitted["history"], variables, position)
        else:
            prompt = await asyncio.to_thread(answer_prompt, "no-docs-found", history, variables, position)
        try:
            llm = deps.answer_llm
            if answer_language(state) == "Deutsch":
                llm = with_decoding_limits(llm, deps.card.quirks, deps.card.answer.provider)
            response = await ainvoke_waiting(
                llm,
                prompt,
                retries=deps.card.timeouts.retries,
                deadline_at=(config.get("configurable") or {}).get("deadline_at"),
                max_wait_s=deps.card.timeouts.idle_s,
            )
            usage = usage_of(response)
            if holds_answer(state):
                try:
                    reply = past_event_reply(
                        str(response.content), variables["context"], datetime.now().date(), deps.messages.answer
                    )
                except Exception as exc:  # the check never costs an answer: the model's is sent
                    logger.warning(f"[PastEvent] held check_failed={type(exc).__name__}")
                    reply = None
                else:
                    logger.info(f"[PastEvent] held replaced={reply is not None}")
                if reply is not None:
                    response = AIMessage(content=reply)
            elif (
                groups.note == "classic"
                and table
                and state.get("raw_docs")
                and sources
                and answer_language(state) == "Deutsch"
                and shape_of(state.get("answer_shape")) == "text"
            ):
                try:
                    answer = str(response.content)
                    numbers = {int(n) for n in CITATION.findall(answer)}
                    cited = [g for n, g in group_of.items() if n in numbers]
                    note = note_for(table, scope, cited, answer)
                    logger.info(
                        f"[StatusGroups] kind={scope.kind} asked={sorted(scope.groups)} seen={sorted(seen)} "
                        f"contact={scope.contact} note={note.kind if note else None}"
                    )
                    if note:
                        texts = deps.messages.answer
                        text = (
                            texts.scope_note_other.format(group=note.group, asked=note.other)
                            if note.kind == "other"
                            else texts.scope_note_one.format(group=note.group, other=note.other)
                        )
                        response = AIMessage(content=answer + "\n\n" + text)
                except Exception as exc:  # the note never costs an answer: the model's is sent
                    logger.warning(f"[StatusGroups] note_failed={type(exc).__name__}")
            return {"messages": [response], "generation": response.content, "context_sources": sources, "usage": usage}
        except Exception as e:
            status = getattr(e, "status_code", None)
            logger.error(f"Error during LLM invocation: {type(e).__name__}" + (f" status={status}" if status else ""))
            failure = failure_of(e, deps.card.answer.provider)
            texts = {
                TOO_LONG: deps.messages.answer.context_too_long,  # the trim path: the user is asked to shorten the question
                BUSY: deps.messages.guard.busy,
                TIMEOUT: deps.messages.guard.timeout,
            }
            if failure is None:
                raise
            if failure == TOO_LONG:
                # every 400 of a local server reads as too long: the provider's own code and type tell what it really was
                logger.error(
                    f"[ContextBudget] request rejected as too long: code={getattr(e, 'code', None)} "
                    f"type={getattr(e, 'type', None)}"
                )
            return {"messages": [], "generation": texts[failure]}

    return generate
