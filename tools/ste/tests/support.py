"""Shared test helpers: a small invented dictionary and glossary.

The tests never read the ASD-STE100 PDF and never use its words as test data.
The vocabulary below is invented for the tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

STE_DIR = Path(__file__).resolve().parent.parent
if str(STE_DIR) not in sys.path:
    sys.path.insert(0, str(STE_DIR))

from ste_dict import Dictionary, Entry  # noqa: E402
from ste_glossary import Glossary, parse_glossary  # noqa: E402
from ste_rules import Checker, Settings  # noqa: E402


def _approved(word: str, pos: str, forms: tuple[str, ...] = ()) -> Entry:
    return Entry(word, pos, True, list(forms))


def _refused(word: str, pos: str, *alternatives: tuple[str, str]) -> Entry:
    return Entry(word, pos, False, [], list(alternatives))


ENTRIES: list[Entry] = [
    # function words
    *[_approved(w, "art") for w in ("the", "a", "an")],
    *[_approved(w, "prep") for w in ("of", "in", "on", "with", "from", "to", "at", "for", "by")],
    *[_approved(w, "conj") for w in ("and", "or", "if", "when", "because", "but", "while")],
    *[_approved(w, "pron") for w in ("it", "you", "we", "they", "this", "that", "each", "one")],
    _approved("not", "adv"),
    _approved("then", "adv"),
    _approved("slowly", "adv"),
    _approved("here", "adv"),
    _approved("now", "adv"),
    _approved("also", "adv"),
    # verbs
    _approved("be", "v", ("is", "was", "are", "were")),
    _approved("do", "v", ("does", "did", "done")),
    _approved("can", "v"),
    _approved("must", "v"),
    _approved("will", "v"),
    _approved("walk", "v", ("walks", "walked", "walked")),
    _approved("carry", "v", ("carries", "carried", "carried")),
    _approved("lift", "v", ("lifts", "lifted", "lifted")),
    _approved("paint", "v"),
    _approved("give", "v", ("gives", "gave", "given")),
    _approved("use", "v", ("uses", "used", "used")),
    _approved("read", "v", ("reads", "read", "read")),
    _approved("see", "v", ("sees", "saw", "seen")),
    _approved("remove", "v", ("removes", "removed", "removed")),
    _approved("make", "v", ("makes", "made", "made")),
    _approved("install", "v"),
    # nouns
    *[_approved(w, "n") for w in (
        "cat", "dog", "tree", "stone", "river", "lamp", "box", "door", "rope", "bucket",
        "water", "house", "wall", "bird", "step", "sentence", "number", "word", "text",
        "time", "note", "detail",
    )],
    # adjectives
    _approved("tall", "adj", ("taller", "tallest")),
    _approved("red", "adj"),
    _approved("quiet", "adj"),
    _approved("heavy", "adj", ("heavier", "heaviest")),
    # not approved, with alternatives
    _refused("jump", "v", ("walk", "v")),
    _refused("big", "adj", ("tall", "adj")),
    _refused("dwelling", "n", ("house", "n")),
    _refused("utilize", "v", ("use", "v")),
    _refused("carry", "n", ("bucket", "n")),
    _refused("graph", "n", ("graph", "TN")),
    _refused("look", "v", ("see", "v")),
    _refused("rope", "v", ("lift", "v")),
    _refused("red", "v", ("paint", "v")),
    _refused("in front of", "prep", ("at", "")),
]

GLOSSARY_TEXT = """\
| Term | Type | Meaning |
|---|---|---|
| widget | technical noun | A small box. |
| widget stand | technical noun | A wall box for a widget. |
| frobnicate | technical verb | To paint a widget. |
| Zork | technical name | The name of a bucket. |
| build | technical verb | To make a widget. |
"""

SETTINGS = Settings(
    procedure_headings=["Steps", "Install"],
    units=frozenset({"ms", "GiB", "%", "s"}),
)


def dictionary() -> Dictionary:
    return Dictionary(list(ENTRIES))


def glossary() -> Glossary:
    return parse_glossary(GLOSSARY_TEXT)


def checker(settings: Settings | None = None) -> Checker:
    return Checker(dictionary(), glossary(), settings or SETTINGS)


def rules(text: str, glossary_file: bool = False, settings: Settings | None = None) -> list[str]:
    """Return the rule codes found in ``text``, in order of position."""
    return [f.rule for f in checker(settings).check(text, "t.md", glossary_file)]
