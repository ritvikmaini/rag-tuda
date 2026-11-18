"""The variant frame's codes."""

from assistant_api.sse import frames as fr

# Why a variant or a version switch was refused, as the variant frame's code
VARIANT_CODES: dict[str, fr.VariantCode] = {
    "unknown variant": "unknown_variant",
    "no turn id": "no_turn_id",
    "no answer to change": "no_answer",
    "not the last question": "not_last_question",
    "no sources": "no_sources",
    "formats off": "formats_off",
    "state unavailable": "state_unavailable",
}
