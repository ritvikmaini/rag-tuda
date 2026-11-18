"""The language of a text (`detect`, by stopword ratio), of a question (`looks_english_question`, by function words) and the
scripts a card's quirk drops from an answer."""

import re

# CJK ideographs, kana, Hangul and CJK/fullwidth punctuation: dropped from a streamed answer
FOREIGN = re.compile("[\u3400-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af\u3000-\u303f\uff00-\uffef]")

# Stopwords for `detect`. Words that mean something in both languages (in, an, so, was, will, war, hat, man, also, die, am, ...)
# are in neither set. The sets include the umlaut-free spellings (fuer, ueber, koennen) because people type without umlauts.
DE_STOPWORDS = frozenset(
    """
der den dem des das die ein eine einen einem einer eines und oder aber nicht kein keine keinen
ist sind waren bin bist wird werden wurde wurden haben hatte hatten habe hast
ich du er sie es wir ihr mir mich dich dir uns euch ihm ihn ihnen mein meine meinen dein sein seine ihre unser
von vom zu zum zur mit bei beim nach aus auf fuer ueber unter vor durch gegen ohne um als im ins
wie wo wer wen wem wann warum wieso weshalb welche welcher welches welchen wenn dass ob
auch noch schon nur sehr mehr kann koennen muss muessen soll sollen darf duerfen gibt bitte jetzt
waehrend wegen zwischen bis dann dieser diese dieses jede jeder alle
""".split()
)
EN_STOPWORDS = frozenset(
    """
the a is are were be been being do does did have has had
i you he she it we they me him her us them my your his its our their this that these those
and or but not no of to on at by for with from about into over under after before between through during without within
as if than then what which who whom whose when where why how can could shall should would may might must
there here just only very more most some any all each other such own same too
""".split()
)

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
# a run of letters of any alphabet, for `detect`
LETTERS = re.compile(r"[^\W\d_]+")
# text in quotes is a title or a quotation, not the language of the question ('...' only around words: not "don't")
QUOTED = re.compile(r'"[^"]*"|„[^“”"]*[“”"]|“[^”]*”|»[^«]*«|«[^»]*»|(?<!\w)\'[^\']*\'(?!\w)')


def strip_foreign(text: str, enabled: bool) -> str:
    """The text without CJK, kana and Hangul characters when the card's strip_cjk quirk is on; else the text itself."""
    return FOREIGN.sub("", text) if enabled else text


def looks_english_question(text: str) -> bool:
    """True when the question, quoted text left out, holds at least two English function words and no German one."""
    words = WORD.findall(QUOTED.sub(" ", str(text or "")).casefold())
    return sum(w in ENGLISH for w in words) >= 2 and not any(w in GERMAN for w in words)


def detect(text: str) -> str:
    """'en' when English stopwords outnumber German ones, else 'de' (a tie, or a text without either, is German)."""
    words = LETTERS.findall(text.lower())
    de = sum(w in DE_STOPWORDS for w in words)
    en = sum(w in EN_STOPWORDS for w in words)
    return "en" if en > de else "de"
