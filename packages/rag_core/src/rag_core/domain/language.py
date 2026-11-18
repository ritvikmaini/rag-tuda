"""The language of a question and the scripts a card's quirk drops from an answer."""

import re

# CJK ideographs, kana, Hangul and CJK/fullwidth punctuation: dropped from a streamed answer
FOREIGN = re.compile("[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]")

# Function words that are clearly one language; words both languages use ("in", "an", "was" as English, "so", "will",
# "die", "info") stay out
ENGLISH = frozenset(
    "please the what how when where which who why is are do does can could should would my your i for of to with from "
    "about on have has there this that summarize explain".split()
)
GERMAN = frozenset(
    "was wie wo wann wer welche welcher der die das den dem des ein eine einen und oder ist sind zu zum zur im mit von "
    "für auf bei nach über ich mein meine kann muss darf gibt es sie bitte bedeutet steht gilt nicht auch noch".split()
)
WORD = re.compile(r"[a-zäöüß]+")
# text in quotes is a title or a quotation, not the language of the question ('...' only around words: not "don't")
QUOTED = re.compile(r'"[^"]*"|„[^“”"]*[“”"]|“[^”]*”|»[^«]*«|«[^»]*»|(?<!\w)\'[^\']*\'(?!\w)')


def strip_foreign(text: str, enabled: bool) -> str:
    """The text without CJK, kana and Hangul characters when the card's strip_cjk quirk is on; else the text itself."""
    return FOREIGN.sub("", text) if enabled else text


def looks_english_question(text: str) -> bool:
    """True when the question, quoted text left out, holds at least two English function words and no German one."""
    words = WORD.findall(QUOTED.sub(" ", str(text or "")).casefold())
    return sum(w in ENGLISH for w in words) >= 2 and not any(w in GERMAN for w in words)
