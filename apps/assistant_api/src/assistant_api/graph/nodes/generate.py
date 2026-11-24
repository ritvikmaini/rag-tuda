"""The answer node: the response generator over the prompt service and the context budget; the model is awaited here, inside
the node, so LangChain's callbacks stream the answer. A documents answer to a question about the next or a coming event is
held back and replaced by one sentence when it writes a past date as coming. With the profile's status_groups switches the
node also tells the status groups and PO versions apart: a comparison, and a question whose best sources are for different
groups or versions, get a prompt line; with the card's version table an answer citing several versions gets the side-by-side
format; a German answer cited only from one side's sources is closed with a note on whom it concerns. Lines taken from the
cited sources' metadata end the answer: the PO versions it rests on, the disclaimer on a residence source, the note on an
English information translation. An urgent reply missing a phone number of the profile's contact line gets the line, and the
fallback text of a failed model call carries it too. Which sources are searched is the question's hard filter, not this node's."""

import asyncio
import logging
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
from rag_core.domain.language import PROMPT_NAMES
from rag_core.domain.notes import appended_lines, urgent_line
from rag_core.domain.query_analysis import RETRIEVAL_INTENTS
from rag_core.domain.source_meta import cited_numbers
from rag_core.domain.status_groups import GENERAL, Scope, note_for, question_scope, scope_lines, source_group
from rag_core.settings.profile import StatusGroup

from assistant_api.graph.filter_line import no_match_line
from assistant_api.graph.history import as_messages, insert_history, line_texts, to_messages, without_lines
from assistant_api.graph.state import AgentState, GraphDeps, messages_for

logger = logging.getLogger(__name__)


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
    coming event and answers in German (the date grammar and the past words are German; an English answer streams): such an answer is not
    streamed but checked and sent whole."""
    current = str(state["messages"][-1].content)
    documents = (state.get("intent") or "lookup") in RETRIEVAL_INTENTS or state.get("scope_fallback")
    asked = asks_for_event(state.get("standalone_question") or current) or asks_for_event(current)
    return bool(documents and state.get("raw_docs") and asked and answer_language(state) == "de")


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
    cites = cited_numbers(found[0][2]) or cited_numbers(answer)
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
    answer_texts = line_texts(deps)
    table = {key: StatusGroup.model_validate(group) for key, group in ((deps.roles or {}).get("groups") or {}).items()}

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
        prior = without_lines(
            context_turns(
                state["messages"],
                deps.card.budget.history_turns,
                profile.history.max_age_hours,
                datetime.now(timezone.utc),
            ),
            answer_texts,
        )
        chosen = await select_turns(prior, deps.card.budget.history_tokens, deps.counter.count_text)
        history = as_messages(chosen)
        logger.info(
            f"[History] completed_fresh={len(prior)} kept={len(chosen)} stored_messages={len(state['messages'])}"
        )
        language = answer_language(state)
        catalog = messages_for(deps, language)  # the texts the code adds are in the answer's language
        labels = catalog.labels.source_labels()
        shapes = catalog.shapes.model_dump()
        variables = {
            "question": state.get("standalone_question") or current.content,
            "context": state.get("formatted_docs", ""),
            "language": PROMPT_NAMES[language],
            "current_date": format_date(datetime.now(), profile.identity.date_format, profile.identity.day_names),
            "bot_name": profile.identity.bot_name,
            "intent": intent,
            "format_instruction": format_instruction(state, shapes),
            "language_reminder": language_reminder(state, shapes),
            "shape_reminder": shape_reminder(state, shapes),
        }
        english = language == "en"
        sources: list[Any] = []
        shown: list[Any] = []  # the metadata of each source in the context
        no_match = ""  # the closing line of an answer to an empty search under a filter
        groups = profile.status_groups
        status_on = bool(table) and (
            groups.compare or groups.reminder != "off" or groups.note != "off" or deps.card.gated.version_table
        )
        scope = Scope(GENERAL, frozenset(), False)
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
                except Exception as exc:  # the groups never cost an answer
                    logger.warning(f"[StatusGroups] scope_failed={type(exc).__name__}")
            fixed_prompt = await asyncio.to_thread(
                answer_prompt, "rag-generation", [], dict(variables, context=""), position
            )
            fitted = await fit_context(budget, deps.counter, fixed_prompt, history, candidates, labels=labels)
            variables["context"] = fitted["context"]
            sources = fitted.get("sources", [])
            try:
                by_section = {(d.metadata or {}).get("section_id"): d.metadata for d in candidates}
                shown = [by_section.get(s.get("section_id")) for s in sources]
            except Exception as exc:  # the lines and the groups lose their sources' metadata; the answer goes on
                logger.warning(f"[Notes] metadata_failed={type(exc).__name__}")
                shown = [None] * len(sources)
            if status_on:
                try:
                    group_of = {s["n"]: source_group(m, table) for s, m in zip(sources, shown)}
                    # the best-ranked sources in the context (its numbers follow the documents, not the ranking)
                    ids = {s.get("section_id") for s in sources}
                    best = sorted(
                        (d for d in candidates if (d.metadata or {}).get("section_id") in ids), key=rank_score
                    )
                    top = [d.metadata for d in best[::-1][: groups.reminder_sources]]
                    applied = state.get("filter")
                    programmes = (deps.roles or {}).get("programmes") or {}
                    degrees = (
                        {
                            *(applied.degree or ()),
                            *(programmes[p]["degree"] for p in (applied.programme or ()) if p in programmes),
                        }
                        if applied
                        else set()
                    )
                    lines = scope_lines(
                        table,
                        scope,
                        (applied.po_version or ()) if applied else (),
                        top,
                        shown,
                        catalog.answer,
                        compare=groups.compare,
                        differ=groups.reminder == "differ",
                        version_table=deps.card.gated.version_table,
                        plain=shape_of(state.get("answer_shape")) == "text",
                        degrees=degrees,
                        english=english,
                    )
                    if lines:
                        variables["scope_reminder"] = "\n".join(lines)
                except Exception as exc:
                    logger.warning(f"[StatusGroups] reminder_failed={type(exc).__name__}")
            prompt = await asyncio.to_thread(answer_prompt, "rag-generation", fitted["history"], variables, position)
        else:
            # the search found nothing: the answer will say under which filter
            no_match = no_match_line(
                state.get("filter"),
                deps.roles,
                catalog.answer,
                english,
            )
            prompt = await asyncio.to_thread(answer_prompt, "no-docs-found", history, variables, position)
        try:
            llm = deps.answer_llm
            if language == "de":  # the card's token biases and bans are for German text
                llm = with_decoding_limits(llm, deps.card.quirks, deps.card.answer.provider)
            response = await ainvoke_waiting(
                llm,
                prompt,
                retries=deps.card.timeouts.retries,
                deadline_at=(config.get("configurable") or {}).get("deadline_at"),
                max_wait_s=deps.card.timeouts.idle_s,
            )
            usage = usage_of(response)
            if no_match:
                response = AIMessage(content=str(response.content) + "\n\n" + no_match)
            elif holds_answer(state):
                try:
                    reply = past_event_reply(
                        str(response.content), variables["context"], datetime.now().date(), catalog.answer
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
                and language == "de"
                and shape_of(state.get("answer_shape")) == "text"
            ):
                try:
                    answer = str(response.content)
                    numbers = set(cited_numbers(answer))
                    cited = [g for n, g in group_of.items() if n in numbers]
                    note = note_for(table, scope, cited, answer)
                    logger.info(
                        f"[StatusGroups] kind={scope.kind} asked={sorted(scope.groups)} "
                        f"contact={scope.contact} note={note.kind if note else None}"
                    )
                    if note:
                        texts = catalog.answer
                        text = (
                            texts.scope_note_other.format(group=note.group, asked=note.other)
                            if note.kind == "other"
                            else texts.scope_note_one.format(group=note.group, other=note.other)
                        )
                        response = AIMessage(content=answer + "\n\n" + text)
                except Exception as exc:  # the note never costs an answer: the model's is sent
                    logger.warning(f"[StatusGroups] note_failed={type(exc).__name__}")
            if intent == "urgent":
                try:  # no cap on the line's words: the numbers are what the reply is for
                    answer = str(response.content)
                    line = urgent_line(answer, catalog.answer.urgent_contacts)
                    if line:
                        # the answer stays as streamed (the page sends only the text after it); an empty one gives way to the line
                        response = AIMessage(content=answer + "\n\n" + line if answer.strip() else line)
                except Exception as exc:  # the line never costs an answer: the model's is sent
                    logger.warning(f"[Urgent] line_failed={type(exc).__name__}")
            if sources:
                try:
                    answer = str(response.content)
                    numbers = set(cited_numbers(answer))
                    applied = state.get("filter")
                    lines = appended_lines(
                        [m for s, m in zip(sources, shown) if s["n"] in numbers],
                        (deps.roles or {}).get("programmes") or {},
                        catalog.answer,
                        english=english,
                        term=bool(applied and applied.po_source == "term"),
                        answer=answer,
                    )
                    if lines:
                        response = AIMessage(content="\n\n".join([answer, *lines]))
                except Exception as exc:  # the lines never cost an answer: the model's is sent
                    logger.warning(f"[Notes] lines_failed={type(exc).__name__}")
            return {"messages": [response], "generation": response.content, "context_sources": sources, "usage": usage}
        except Exception as e:
            status = getattr(e, "status_code", None)
            logger.error(f"Error during LLM invocation: {type(e).__name__}" + (f" status={status}" if status else ""))
            failure = failure_of(e, deps.card.answer.provider)
            texts = {
                TOO_LONG: catalog.answer.context_too_long,  # the trim path: the user is asked to shorten the question
                BUSY: catalog.guard.busy,
                TIMEOUT: catalog.guard.timeout,
            }
            if failure is None:
                raise
            if failure == TOO_LONG:
                # every 400 of a local server reads as too long: the provider's own code and type tell what it really was
                logger.error(
                    f"[ContextBudget] request rejected as too long: code={getattr(e, 'code', None)} "
                    f"type={getattr(e, 'type', None)}"
                )
            text = texts[failure]
            if intent == "urgent" and catalog.answer.urgent_contacts:  # the numbers are what the person needs most
                text += "\n\n" + catalog.answer.urgent_contacts
            return {"messages": [], "generation": text}

    return generate
