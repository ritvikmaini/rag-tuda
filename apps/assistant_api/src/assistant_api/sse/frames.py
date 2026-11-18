"""The answer stream's frames, format 1.

One pydantic model per type; every frame has `type` and `content`, `code` sits beside them on `error` and on a
`variant` with ok false. `encode` writes "data: " + JSON + a blank line: Python's separators, ASCII, keys in model order, NaN and
±Infinity as null (a NaN would stop the page's parser for the rest of the answer). `meta` is the first frame of every
response and `done` the last one the client stays for; both come from the guard; `done` also names the turn's index generation
and prompt versions when known (left out otherwise). `Declared` is the pipeline's outcome for the
guard, never sent. `python -m assistant_api.sse.frames` prints the JSON schema (`just stream-schema`)."""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

Shape = Literal["text", "table", "checklist", "steps", "mindmap"]
Outcome = Literal["completed", "failed", "too_long", "busy", "refused"]
ErrorCode = Literal["busy", "not_ready", "pipeline_error", "timeout", "empty_answer", "exception"]
VariantCode = Literal[
    "unknown_variant",
    "no_turn_id",
    "no_answer",
    "not_last_question",
    "no_sources",
    "formats_off",
    "state_unavailable",
    "no_version",
    "failed",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MetaContent(_Model):
    format: Literal[1] = 1
    turn_id: str
    trace_id: str | None


class Meta(_Model):
    type: Literal["meta"] = "meta"
    content: MetaContent


class Status(_Model):
    type: Literal["status"] = "status"
    content: dict[str, Any]


class ShapeFrame(_Model):
    type: Literal["shape"] = "shape"
    content: Shape


class Docs(_Model):
    type: Literal["docs"] = "docs"
    content: list[dict[str, Any]]


class Chunk(_Model):
    type: Literal["chunk"] = "chunk"
    content: str


class Citation(_Model):
    n: int
    url: str
    page: int | None
    title: str
    type: str
    origin: str
    date: str
    quote: str


class Citations(_Model):
    type: Literal["citations"] = "citations"
    content: list[Citation]


class VariantContent(_Model):
    name: str = Field(max_length=20)
    ok: bool


class Variant(_Model):
    type: Literal["variant"] = "variant"
    content: VariantContent
    code: VariantCode | None = None


class Error(_Model):
    type: Literal["error"] = "error"
    content: str
    code: ErrorCode


class DoneContent(_Model):
    outcome: Outcome
    # the index generation and the prompt versions the turn ran on, so the evals gate can tell a pair that straddled a build
    # or a publish; omitted (not null) when unknown, so the frame stays unchanged for clients that do not know the field
    generation: int | None = Field(default=None, exclude_if=lambda v: v is None)
    prompts: dict[str, int | None] | None = Field(default=None, exclude_if=lambda v: v is None)
    # true on a turn whose out-of-scope question was answered from the documents; omitted on every other turn, so their
    # frames stay unchanged
    scope_fallback: bool | None = Field(default=None, exclude_if=lambda v: v is None)


class Done(_Model):
    type: Literal["done"] = "done"
    content: DoneContent


Frame = Annotated[
    Meta | Status | ShapeFrame | Docs | Chunk | Citations | Variant | Error | Done, Field(discriminator="type")
]


@dataclass(frozen=True)
class Declared:
    """The pipeline's outcome for the guard's done frame (too long, busy, refused, a failed variant); never sent."""

    outcome: Outcome


def meta(turn_id: str, trace_id: str | None) -> Meta:
    return Meta(content=MetaContent(turn_id=turn_id, trace_id=trace_id))


def done(
    outcome: Outcome,
    generation: int | None = None,
    prompts: Mapping[str, int | None] | None = None,
    scope_fallback: bool | None = None,
) -> Done:
    return Done(
        content=DoneContent(
            outcome=outcome,
            generation=generation,
            prompts=None if prompts is None else dict(prompts),
            scope_fallback=scope_fallback,
        )
    )


def variant(name: Any, ok: bool, code: VariantCode | None = None) -> Variant:
    """The name is cut to 20 characters; a code only when ok is false."""
    return Variant(content=VariantContent(name=str(name or "")[:20], ok=bool(ok)), code=None if ok else code)


def finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    return value


def wire(frame: BaseModel) -> dict[str, Any]:
    data = frame.model_dump()
    if data.get("code", "") is None:
        del data["code"]
    return finite(data)


def encode(frame: BaseModel) -> str:
    return "data: " + json.dumps(wire(frame), ensure_ascii=True, allow_nan=False) + "\n\n"


def schema() -> dict[str, Any]:
    return TypeAdapter(Frame).json_schema()


if __name__ == "__main__":
    print(json.dumps(schema(), indent=1, sort_keys=True))
