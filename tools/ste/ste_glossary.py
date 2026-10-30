"""Read the project glossary: a Markdown table of technical nouns, verbs and names."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ste_dict import plural_forms, regular_forms
from ste_markdown import TABLE_SEP_RE, split_row

KINDS = {
    "technical noun": "noun",
    "technical verb": "verb",
    "technical name": "name",
}

# Irregular past and participle forms of verbs that a glossary can list.
IRREGULAR_VERBS: dict[str, tuple[str, ...]] = {
    "begin": ("began", "begun"),
    "bind": ("bound",),
    "build": ("built",),
    "choose": ("chose", "chosen"),
    "cut": ("cut",),
    "draw": ("drew", "drawn"),
    "drive": ("drove", "driven"),
    "feed": ("fed",),
    "find": ("found",),
    "hit": ("hit",),
    "hold": ("held",),
    "keep": ("kept",),
    "know": ("knew", "known"),
    "lay": ("laid",),
    "lead": ("led",),
    "leave": ("left",),
    "lose": ("lost",),
    "read": ("read",),
    "run": ("ran", "run"),
    "send": ("sent",),
    "set": ("set",),
    "shut": ("shut",),
    "split": ("split",),
    "spin": ("spun",),
    "take": ("took", "taken"),
    "write": ("wrote", "written"),
}


class GlossaryError(Exception):
    """The glossary file has a fault."""

    def __init__(self, message: str, line: int = 0) -> None:
        super().__init__(message)
        self.message = message
        self.line = line


@dataclass
class Term:
    term: str  # lower case
    kind: str  # noun, verb or name
    meaning: str
    line: int
    display: str = ""


def _plain(cell: str) -> str:
    text = cell.strip()
    text = re.sub(r"[`*_]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Glossary:
    def __init__(self, terms: list[Term]) -> None:
        self.terms = terms
        self.surface: dict[str, list[Term]] = {}
        self.phrases: dict[int, set[str]] = {}
        for term in terms:
            for form in self._forms(term):
                self.surface.setdefault(form, []).append(term)
                words = form.count(" ") + 1
                if words > 1:
                    self.phrases.setdefault(words, set()).add(form)

    @staticmethod
    def _forms(term: Term) -> list[str]:
        base = term.term
        forms = [base]
        if term.kind == "noun":
            forms += plural_forms(base)
        elif term.kind == "verb":
            head, _, last = base.rpartition(" ")
            prefix = head + " " if head else ""
            irregular = IRREGULAR_VERBS.get(last, ())
            regular = regular_forms(last)
            if irregular:
                regular = [form for form in regular if not form.endswith("ed")]
            forms += [prefix + form for form in regular + list(irregular)]
        return forms

    def lookup(self, word: str) -> list[Term]:
        return self.surface.get(word.lower(), [])

    def kinds_of(self, word: str) -> set[str]:
        return {term.kind for term in self.lookup(word)}


def load_glossary(path: Path) -> Glossary:
    text = path.read_text(encoding="utf-8")
    return parse_glossary(text)


def parse_glossary(text: str) -> Glossary:
    lines = text.splitlines()
    terms: list[Term] = []
    seen: dict[tuple[str, str], int] = {}
    in_table = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if not in_table:
            if "|" in line and index + 1 < len(lines) and TABLE_SEP_RE.match(lines[index + 1]):
                header = [_plain(cell).lower() for _c, cell in split_row(line)]
                if header[:3] == ["term", "type", "meaning"]:
                    in_table = True
                    index += 2
                    continue
            index += 1
            continue
        if not line.strip() or "|" not in line:
            in_table = False
            index += 1
            continue
        cells = split_row(line)
        if len(cells) < 3:
            raise GlossaryError("A glossary row needs three cells: Term, Type, Meaning.", index + 1)
        display = _plain(cells[0][1])
        kind_text = _plain(cells[1][1]).lower()
        meaning = _plain(cells[2][1])
        if not display:
            raise GlossaryError("A glossary row has an empty Term cell.", index + 1)
        if kind_text not in KINDS:
            raise GlossaryError(
                f"Unknown type '{kind_text}'. Use technical noun, technical verb or technical name.",
                index + 1,
            )
        if not meaning:
            raise GlossaryError(f"The term '{display}' has no meaning.", index + 1)
        key = (display.lower(), KINDS[kind_text])
        if key in seen:
            raise GlossaryError(
                f"The term '{display}' has two {kind_text} rows (first on line {seen[key]}).",
                index + 1,
            )
        seen[key] = index + 1
        terms.append(Term(display.lower(), KINDS[kind_text], meaning, index + 1, display))
        index += 1
    return Glossary(terms)
