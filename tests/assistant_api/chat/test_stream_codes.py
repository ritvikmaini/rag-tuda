"""The codes and outcomes of the service's frames."""

import uuid

from assistant_api.graph import store, tracing
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from rag_core.domain.answer_shapes import turn_variant_problem
from rag_core.domain.history import LastTurn

TURN = str(uuid.uuid4())


def test_the_trace_id_is_the_turns_seed_for_a_question_and_none_without_tracing():
    assert tracing.trace_id(None, TURN, None) is None
    assert tracing.trace_id(object(), TURN, None) == Langfuse.create_trace_id(seed=TURN)
    other = tracing.trace_id(object(), TURN, "table")
    assert other is not None and len(other) == 32 and other != Langfuse.create_trace_id(seed=TURN)


def test_a_traced_run_gets_its_own_handler_carrying_the_trace_id_meta_announced():
    shared = object()  # Runtime.handler: only its being set matters to run_config
    announced = tracing.trace_id(shared, TURN, None)  # what the route puts in meta.trace_id
    config = tracing.run_config(shared, "u", "example", trace_id=announced)
    (handler,) = config["callbacks"]
    assert isinstance(handler, CallbackHandler) and handler is not shared
    assert handler._trace_context == {"trace_id": announced}  # pyright: ignore[reportAttributeAccessIssue]
    assert config["metadata"]["langfuse_tags"] == ["example_assistant"]
    assert tracing.run_config(shared, "u", "example")["callbacks"] == [shared]  # no trace id: the shared handler
    assert "callbacks" not in tracing.run_config(None, "u", "example", trace_id=announced)  # untraced run


def test_every_turn_variant_reason_has_a_code():
    t = TURN
    reasons = {
        turn_variant_problem("poster", t, None),
        turn_variant_problem("table", None, None),
        turn_variant_problem("table", t, None),
        turn_variant_problem("table", t, LastTurn("other", True, True)),
        turn_variant_problem("table", t, LastTurn(t, True, False)),
    }
    assert reasons | {"formats off", "state unavailable"} == set(store.VARIANT_CODES)
