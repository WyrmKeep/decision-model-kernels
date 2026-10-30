"""Tokens, parenthesized groups and sentences of cleaned Markdown text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ste_markdown import ATOM, GROUP

# Token kinds
WORD = "WORD"
NUM = "NUM"
ATOM_KIND = "ATOM"
PUNCT = "PUNCT"
GROUP_KIND = "GROUP"

ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.")
ATOM_ABBREVIATIONS = ("a.m.", "p.m.")

_LEAD = "([{\"'“‘<"
_TRAIL = ".,;:!?)]}\"'”’>…"
_NUM_RE = re.compile(
    r"^(?:[+-]?\d[\d,]*(?:\.\d+)?%?"
    r"|\d+(?:[.,]\d+)?[-–]\d+(?:[.,]\d+)?%?"
    r"|\d+(?:\.\d+){2,}"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d{4}-\d{2}-\d{2})$"
)
_FLAG_RE = re.compile(r"^--?[A-Za-z][\w-]*(?:=.*)?$")
_LETTERS = r"[^\W\d_]+"
_WORD_RE = re.compile(
    _LETTERS + r"(?:['’]" + _LETTERS + r")*(?:-" + _LETTERS + r"(?:['’]" + _LETTERS + r")*)*"
)
_DOTTED_ABBR_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")


@dataclass
class Token:
    kind: str
    text: str
    start: int
    end: int
    inner: list["Token"] = field(default_factory=list)  # for GROUP tokens

    @property
    def lower(self) -> str:
        return self.text.lower().replace("’", "'")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{self.kind}({self.text!r})"


def _is_identifier(core: str) -> bool:
    if ATOM in core:
        return True
    if "_" in core and re.search(r"[A-Za-z0-9]", core):
        return True
    if re.search(r"[a-z][A-Z]", core):
        return True
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", core):
        return True
    if re.search(r"\w\.\w", core):
        return True
    if "::" in core or "->" in core or "=>" in core or "=" in core:
        return True
    if core.startswith(("@", "#", "~", "$")) and len(core) > 1:
        return True
    if core.endswith("++") and len(core) > 2:
        return True
    return False


def _is_path(core: str) -> bool:
    if "/" not in core:
        return False
    if core.startswith(("/", "~", ".", "./", "../")):
        return True
    return bool(re.search(r"[._\d~]", core))


def tokenize(text: str, units: frozenset[str] = frozenset()) -> list[Token]:
    """Split cleaned text into tokens. A number followed by a unit is one token."""
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.extend(_chunk_tokens(text[i:j], i, units))
        i = j
    return _merge_units(tokens, units)


def _chunk_tokens(chunk: str, base: int, units: frozenset[str]) -> list[Token]:
    out: list[Token] = []
    # Abbreviations keep their final period.
    stripped = chunk.rstrip(",;:!?)]}\"'”’")
    suffix = chunk[len(stripped) :]
    if stripped.lower() in ABBREVIATIONS:
        out.append(Token(WORD, stripped, base, base + len(stripped)))
        out.extend(_punct(suffix, base + len(stripped)))
        return out
    if stripped.lower() in ATOM_ABBREVIATIONS or _DOTTED_ABBR_RE.match(stripped):
        out.append(Token(ATOM_KIND, stripped, base, base + len(stripped)))
        out.extend(_punct(suffix, base + len(stripped)))
        return out
    start = 0
    end = len(chunk)
    leading: list[Token] = []
    while start < end and chunk[start] in _LEAD:
        # A leading hyphen is a flag marker, not punctuation. A leading dollar is not in _LEAD.
        leading.append(Token(PUNCT, chunk[start], base + start, base + start + 1))
        start += 1
    trailing: list[Token] = []
    while end > start and chunk[end - 1] in _TRAIL:
        # Keep a dot that is part of an identifier, for example file.name
        trailing.append(Token(PUNCT, chunk[end - 1], base + end - 1, base + end))
        end -= 1
    trailing.reverse()
    core = chunk[start:end]
    out.extend(leading)
    out.extend(_core_tokens(core, base + start, units))
    out.extend(trailing)
    return out


def _punct(text: str, base: int) -> list[Token]:
    return [Token(PUNCT, ch, base + k, base + k + 1) for k, ch in enumerate(text)]


def _core_tokens(core: str, base: int, units: frozenset[str]) -> list[Token]:
    if not core:
        return []
    if ATOM in core:
        return _split_atoms(core, base, units)
    if core in ("-", "--", "—", "–", "&", "+", "=", "→", "×", "*", "|", "~"):
        return [Token(PUNCT, core, base, base + len(core))]
    if _NUM_RE.match(core):
        return [Token(NUM, core, base, base + len(core))]
    if core in units:
        return [Token(WORD, core, base, base + len(core))]
    if _FLAG_RE.match(core):
        return [Token(ATOM_KIND, core, base, base + len(core))]
    if re.match(r"^\$\{?\w+\}?$", core):
        return [Token(ATOM_KIND, core, base, base + len(core))]
    if "/" in core:
        if _is_path(core):
            return [Token(ATOM_KIND, core, base, base + len(core))]
        parts = core.split("/")
        if all(part and part in units for part in parts):
            return [Token(WORD, core, base, base + len(core))]
        out: list[Token] = []
        offset = 0
        for part in parts:
            out.extend(_core_tokens(part, base + offset, units))
            offset += len(part) + 1
        return out
    for dash in ("—", "–"):
        if dash in core:
            out = []
            offset = 0
            for index, part in enumerate(core.split(dash)):
                if index:
                    out.append(Token(PUNCT, dash, base + offset - 1, base + offset))
                out.extend(_core_tokens(part, base + offset, units))
                offset += len(part) + 1
            return out
    if _is_identifier(core):
        return [Token(ATOM_KIND, core, base, base + len(core))]
    out = []
    covered = 0
    for match in _WORD_RE.finditer(core):
        if match.start() > covered:
            out.extend(_punct(core[covered : match.start()].replace(" ", ""), base + covered))
        out.append(Token(WORD, match.group(0), base + match.start(), base + match.end()))
        covered = match.end()
    if covered < len(core):
        rest = core[covered:]
        out.extend(_punct(rest, base + covered))
    return out


def _split_atoms(core: str, base: int, units: frozenset[str]) -> list[Token]:
    out: list[Token] = []
    piece = ""
    piece_start = 0
    for index, char in enumerate(core):
        if char == ATOM:
            if piece:
                out.extend(_atom_neighbor(piece, base + piece_start, units, bool(out)))
                piece = ""
            out.append(Token(ATOM_KIND, ATOM, base + index, base + index + 1))
            piece_start = index + 1
        else:
            if not piece:
                piece_start = index
            piece += char
    if piece:
        out.extend(_atom_neighbor(piece, base + piece_start, units, bool(out)))
    return out


def _atom_neighbor(piece: str, base: int, units: frozenset[str], after_atom: bool) -> list[Token]:
    if after_atom and re.fullmatch(r"(?:s|es|'s|’s)", piece):
        return []  # plural or possessive ending of a code span
    if piece in ("/", "-", "–", "—", "+", "=", "&", "|"):
        return [Token(PUNCT, piece, base, base + len(piece))]
    if re.fullmatch(r"[/\-–—+=&|]+", piece):
        return [Token(PUNCT, piece, base, base + len(piece))]
    trimmed = piece.strip("/-–—+=&|")
    if not trimmed:
        return []
    offset = piece.index(trimmed)
    return _core_tokens(trimmed, base + offset, units)


def _merge_units(tokens: list[Token], units: frozenset[str]) -> list[Token]:
    out: list[Token] = []
    for token in tokens:
        if (
            out
            and out[-1].kind == NUM
            and token.kind in (WORD, ATOM_KIND)
            and token.text in units
        ):
            previous = out[-1]
            out[-1] = Token(NUM, previous.text + " " + token.text, previous.start, token.end)
            continue
        out.append(token)
    return out


# --------------------------------------------------------------------------
# Groups and sentences

_OPEN_QUOTES = ("“",)
_CLOSE_QUOTES = ("”",)


def group_parens(tokens: list[Token]) -> tuple[list[Token], list[list[Token]]]:
    """Collapse parenthesized text into GROUP tokens.

    Return the outer tokens and the token lists of all groups. Rule 8.5: text in
    parentheses counts as one word, and it forms a separate sentence.
    """
    groups: list[list[Token]] = []

    def build(items: list[Token]) -> list[Token]:
        out: list[Token] = []
        index = 0
        while index < len(items):
            token = items[index]
            if token.kind == PUNCT and token.text == "(":
                depth = 1
                scan = index + 1
                while scan < len(items) and depth:
                    if items[scan].kind == PUNCT and items[scan].text == "(":
                        depth += 1
                    elif items[scan].kind == PUNCT and items[scan].text == ")":
                        depth -= 1
                    scan += 1
                if depth == 0:
                    inner_tokens = items[index + 1 : scan - 1]
                    inner = build(inner_tokens)
                    groups.append(inner)
                    out.append(
                        Token(GROUP_KIND, GROUP, token.start, items[scan - 1].end, inner=inner)
                    )
                    index = scan
                    continue
            out.append(token)
            index += 1
        return out

    return build(tokens), groups


def split_sentences(tokens: list[Token]) -> list[list[Token]]:
    """Split tokens at a period, a question mark or an exclamation mark."""
    sentences: list[list[Token]] = []
    current: list[Token] = []
    in_quote = False
    for token in tokens:
        current.append(token)
        if token.kind == PUNCT:
            if token.text == '"' or token.text in _OPEN_QUOTES:
                in_quote = token.text != '"' or not in_quote
            elif token.text in _CLOSE_QUOTES:
                in_quote = False
            elif token.text in (".", "!", "?", "…") and not in_quote:
                if _has_content(current):
                    sentences.append(current)
                    current = []
    if _has_content(current):
        sentences.append(current)
    return sentences


def _has_content(tokens: list[Token]) -> bool:
    return any(token.kind != PUNCT for token in tokens)


def count_words(tokens: list[Token]) -> int:
    """Count words by rules 8.5 to 8.7: groups, quoted text and units count as one word."""
    count = 0
    index = 0
    n = len(tokens)
    while index < n:
        token = tokens[index]
        if token.kind == PUNCT and (token.text == '"' or token.text in _OPEN_QUOTES):
            closer = None
            for scan in range(index + 1, n):
                candidate = tokens[scan]
                if candidate.kind == PUNCT and (
                    candidate.text in _CLOSE_QUOTES if token.text in _OPEN_QUOTES else candidate.text == '"'
                ):
                    closer = scan
                    break
            if closer is not None:
                count += 1
                index = closer + 1
                continue
        if token.kind != PUNCT:
            count += 1
        index += 1
    return count
