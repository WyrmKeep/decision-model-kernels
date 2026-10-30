"""Split a Markdown document into blocks of prose with source positions.

The parser is a small line-based reader. It does not implement all of
CommonMark. It knows the parts that matter for prose checks:

* fenced code, indented code, HTML comments, front matter and reference links
  are skipped
* headings, paragraphs, list items, block quotes and table cells become blocks
* inline code, links, images, URLs, HTML tags and emphasis marks are removed
  from the text, and each code span or URL becomes one placeholder character

Every character of the cleaned text maps back to a (line, column) pair in the
source file. Lines and columns start at 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ATOM = "\x01"  # placeholder for a code span, a URL or a formula
GROUP = "\x03"  # placeholder for a parenthesized group, set by the text module

Pos = tuple[int, int]

_FENCE_RE = re.compile(r"^(\s{0,3})(`{3,}|~{3,})(.*)$")
_ATX_RE = re.compile(r"^\s{0,3}(#{1,6})(?:\s+(.*?))?(?:\s+#+)?\s*$")
_SETEXT_RE = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])(\s+)(.*)$")
_QUOTE_RE = re.compile(r"^(\s{0,3}>\s?)(.*)$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_REFDEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
_HTML_BLOCK_RE = re.compile(r"^\s{0,3}</?[A-Za-z][A-Za-z0-9-]*(\s[^>]*)?/?>")
_DIRECTIVE_RE = re.compile(r"<!--\s*ste-ignore-next-line\b(?P<rules>[^>]*?)-->")
_CHECKBOX_RE = re.compile(r"^\[[ xX]\]\s+")


@dataclass
class Block:
    """A run of prose that the rules read as one unit."""

    kind: str  # paragraph, list_item, heading, cell, quote
    text: str
    pos: list[Pos]
    atoms: list[str]
    line: int
    in_procedure: bool = False
    heading_level: int = 0
    column_name: str = ""  # for table cells: the text of the column header
    table_header: bool = False
    only_code: bool = False


@dataclass
class Document:
    blocks: list[Block] = field(default_factory=list)
    ignores: dict[int, set[str]] = field(default_factory=dict)  # line -> rules ("*" = all)
    lines: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Inline cleaning


def _code_spans(raw: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    n = len(raw)
    while i < n:
        if raw[i] == "\\" and i + 1 < n:
            i += 2
            continue
        if raw[i] != "`":
            i += 1
            continue
        j = i
        while j < n and raw[j] == "`":
            j += 1
        run = j - i
        k = j
        found = -1
        while k < n:
            if raw[k] == "`":
                m = k
                while m < n and raw[m] == "`":
                    m += 1
                if m - k == run:
                    found = k
                    break
                k = m
            else:
                k += 1
        if found < 0:
            i = j
            continue
        inner = raw[j:found]
        if len(inner) > 1 and inner.startswith(" ") and inner.endswith(" "):
            inner = inner[1:-1]
        spans.append((i, found + run, inner))
        i = found + run
    return spans


_INLINE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("drop", re.compile(r"<!--.*?-->", re.S)),
    ("atom", re.compile(r"<[A-Za-z][A-Za-z0-9+.\-]*:[^<>\s]*>")),
    ("atom", re.compile(r"<[^<>\s@]+@[^<>\s]+>")),
    ("drop", re.compile(r"</?[A-Za-z][^<>]*>")),
    ("link", re.compile(r"!?\[(?P<text>[^\]]*)\]\((?P<target>[^)]*)\)")),
    ("reflink", re.compile(r"\[(?P<text>[^\]]+)\]\[[^\]]*\]")),
    ("drop", re.compile(r"\[\^[^\]]+\]")),
    ("url", re.compile(r"(?<![\w@])(?:https?://|www\.)[^\s<>\[\]()\x02]+")),
    ("atom", re.compile(r"\$\$.+?\$\$|(?<![\w$])\$(?!\s)[^$\n]+?(?<!\s)\$(?![\w$])")),
    ("drop", re.compile(r"~~")),
    ("emph", re.compile(r"(?<!\w)(?:\*{1,3}|_{1,3})(?=\S)")),
    ("emph", re.compile(r"(?<=\S)(?:\*{1,3}|_{1,3})(?!\w)")),
    ("escape", re.compile(r"\\(?=[!-/:-@\[-`{-~])")),
]
_URL_TRAIL = ".,;:!?'\"”’"


def clean_inline(raw: str, pos: list[Pos]) -> tuple[str, list[Pos], list[str]]:
    """Remove Markdown markup from ``raw``.

    ``pos`` gives the source position of each character of ``raw``. The result
    is the cleaned text, the source position of each of its characters and the
    list of atoms, one for each ATOM character in the text.
    """
    n = len(raw)
    claimed = [False] * n
    drop = [False] * n
    atom_at: dict[int, str] = {}  # start index -> atom text
    work = list(raw)

    def claim(start: int, end: int) -> bool:
        if any(claimed[start:end]):
            return False
        for index in range(start, end):
            claimed[index] = True
        return True

    for start, end, inner in _code_spans(raw):
        claim(start, end)
        atom_at[start] = inner
        for index in range(start, end):
            work[index] = "\x02"
            drop[index] = index != start
    text = "".join(work)

    for kind, pattern in _INLINE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            if kind in ("emph", "escape"):
                if any(claimed[start:end]):
                    continue
                for index in range(start, end):
                    drop[index] = True
                continue
            if kind == "url":
                while end > start and text[end - 1] in _URL_TRAIL:
                    end -= 1
                if end - start < 5 or not claim(start, end):
                    continue
                atom_at[start] = raw[start:end]
                for index in range(start + 1, end):
                    drop[index] = True
                continue
            if kind == "atom":
                if not claim(start, end):
                    continue
                atom_at[start] = raw[start:end]
                for index in range(start + 1, end):
                    drop[index] = True
                continue
            if kind == "drop":
                if not claim(start, end):
                    continue
                for index in range(start, end):
                    drop[index] = True
                continue
            # link, reflink: keep the link text, drop the markers and the target
            text_start, text_end = match.span("text")
            if any(claimed[start:text_start]) or any(claimed[text_end:end]):
                continue
            for index in range(start, text_start):
                claimed[index] = True
                drop[index] = True
            for index in range(text_end, end):
                claimed[index] = True
                drop[index] = True

    out: list[str] = []
    out_pos: list[Pos] = []
    atoms: list[str] = []
    for index in range(n):
        if index in atom_at:
            out.append(ATOM)
            out_pos.append(pos[index])
            atoms.append(atom_at[index])
            continue
        if drop[index]:
            continue
        out.append(raw[index])
        out_pos.append(pos[index])
    return "".join(out), out_pos, atoms


def _collapse(text: str, pos: list[Pos], atoms: list[str]) -> tuple[str, list[Pos], list[str]]:
    """Collapse runs of white space to one space and strip both ends."""
    out: list[str] = []
    out_pos: list[Pos] = []
    for char, where in zip(text, pos):
        if char.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                out_pos.append(where)
            continue
        out.append(char)
        out_pos.append(where)
    while out and out[-1] == " ":
        out.pop()
        out_pos.pop()
    return "".join(out), out_pos, atoms


def make_block(
    kind: str,
    segments: list[tuple[int, int, str]],
    in_procedure: bool = False,
    heading_level: int = 0,
    column_name: str = "",
) -> Block | None:
    """Build a block from source segments (line, first column, text)."""
    raw_parts: list[str] = []
    pos: list[Pos] = []
    for index, (line, col, text) in enumerate(segments):
        if index:
            raw_parts.append(" ")
            prev_line, prev_col, prev_text = segments[index - 1]
            pos.append((prev_line, prev_col + len(prev_text)))
        raw_parts.append(text)
        pos.extend((line, col + offset) for offset in range(len(text)))
    raw = "".join(raw_parts)
    cleaned, cleaned_pos, atoms = clean_inline(raw, pos)
    text, text_pos, atoms = _collapse(cleaned, cleaned_pos, atoms)
    if not text:
        return None
    only_code = all(ch in (ATOM, " ") for ch in text)
    return Block(
        kind=kind,
        text=text,
        pos=text_pos,
        atoms=atoms,
        line=segments[0][0],
        in_procedure=in_procedure,
        heading_level=heading_level,
        column_name=column_name,
        only_code=only_code,
    )


# --------------------------------------------------------------------------
# Block parsing


def split_row(line: str) -> list[tuple[int, str]]:
    """Split a table row into cells. Return (start column, text) pairs, columns from 0."""
    masked = list(line)
    for start, end, _inner in _code_spans(line):
        for index in range(start, end):
            masked[index] = "\x02"
    cells: list[tuple[int, str]] = []
    start = 0
    for index, char in enumerate(masked):
        if char == "|" and (index == 0 or masked[index - 1] != "\\"):
            cells.append((start, line[start:index]))
            start = index + 1
    cells.append((start, line[start:]))
    if cells and not cells[0][1].strip():
        cells = cells[1:]
    if cells and not cells[-1][1].strip():
        cells = cells[:-1]
    return cells


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    first = lines[index]
    if "|" not in first:
        return False
    return bool(TABLE_SEP_RE.match(lines[index + 1])) and "|" in lines[index + 1]


def parse_document(text: str, procedure_headings: list[str]) -> Document:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    doc = Document(lines=lines)
    heading_stack: list[tuple[int, bool]] = []
    proc_patterns = [
        re.compile(r"\b" + re.escape(item.lower()) + r"\b") for item in procedure_headings
    ]

    def in_procedure(ordered: bool = False) -> bool:
        # The nearest heading decides. A numbered list also inherits from any parent heading.
        if not heading_stack:
            return False
        if heading_stack[-1][1]:
            return True
        return ordered and any(flag for _level, flag in heading_stack)

    paragraph: list[tuple[int, int, str]] = []
    para_kind = "paragraph"
    item: list[tuple[int, int, str]] | None = None
    item_ordered = False

    def flush_paragraph() -> None:
        nonlocal paragraph, para_kind
        if paragraph:
            block = make_block(para_kind, paragraph, in_procedure=False)
            if block:
                doc.blocks.append(block)
        paragraph = []
        para_kind = "paragraph"

    def flush_item() -> None:
        nonlocal item
        if item:
            block = make_block("list_item", item, in_procedure=in_procedure(item_ordered))
            if block:
                doc.blocks.append(block)
        item = None

    def flush_all() -> None:
        flush_paragraph()
        flush_item()

    i = 0
    total = len(lines)
    in_fence: tuple[str, int] | None = None
    in_comment = False
    if lines and lines[0].strip() == "---":
        for j in range(1, total):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    previous_blank = True
    while i < total:
        line = lines[i]
        number = i + 1
        directive = _DIRECTIVE_RE.search(line)
        if directive:
            rules = {
                part.strip()
                for part in re.split(r"[,\s]+", directive.group("rules"))
                if part.strip()
            }
            doc.ignores.setdefault(number + 1, set()).update(rules or {"*"})
        if in_comment:
            if "-->" in line:
                in_comment = False
            i += 1
            continue
        if in_fence is not None:
            marker, length = in_fence
            stripped = line.strip()
            if stripped.startswith(marker * length) and set(stripped) == {marker}:
                in_fence = None
            i += 1
            continue
        stripped_line = line.strip()
        if stripped_line.startswith("<!--"):
            if "-->" not in stripped_line:
                in_comment = True
                i += 1
                continue
            if stripped_line.endswith("-->") and stripped_line.count("<!--") == 1:
                i += 1
                continue
        fence = _FENCE_RE.match(line)
        if fence and not (fence.group(2)[0] == "`" and "`" in fence.group(3)):
            flush_all()
            in_fence = (fence.group(2)[0], len(fence.group(2)))
            i += 1
            previous_blank = False
            continue
        if not stripped_line:
            flush_all()
            previous_blank = True
            i += 1
            continue
        heading = _ATX_RE.match(line)
        if heading:
            flush_all()
            level = len(heading.group(1))
            content = heading.group(2) or ""
            col = line.index(content) + 1 if content else 1
            block = make_block(
                "heading",
                [(number, col, content)],
                heading_level=level,
                in_procedure=in_procedure(),
            )
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            is_proc = False
            if block:
                lowered = block.text.lower()
                is_proc = any(p.search(lowered) for p in proc_patterns)
                block.in_procedure = in_procedure() or is_proc
                doc.blocks.append(block)
            heading_stack.append((level, is_proc))
            i += 1
            previous_blank = False
            continue
        setext = _SETEXT_RE.match(line)
        if setext and paragraph and para_kind == "paragraph" and item is None:
            level = 1 if setext.group(1)[0] == "=" else 2
            block = make_block("heading", paragraph, heading_level=level)
            paragraph = []
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            is_proc = False
            if block:
                is_proc = any(p.search(block.text.lower()) for p in proc_patterns)
                block.in_procedure = is_proc
                doc.blocks.append(block)
            heading_stack.append((level, is_proc))
            i += 1
            continue
        if _RULE_RE.match(line):
            flush_all()
            i += 1
            continue
        if _REFDEF_RE.match(line) and not paragraph:
            i += 1
            continue
        if _is_table_start(lines, i):
            flush_all()
            header_cells = split_row(lines[i])
            names: list[str] = []
            for c, t in header_cells:
                block = make_block("cell", [(number, c + 1, t)])
                names.append(block.text if block else "")
                if block:
                    block.column_name = block.text
                    block.table_header = True
                    doc.blocks.append(block)
            i += 2
            while i < total and lines[i].strip() and "|" in lines[i]:
                row_number = i + 1
                for index, (c, t) in enumerate(split_row(lines[i])):
                    name = names[index] if index < len(names) else ""
                    block = make_block("cell", [(row_number, c + 1, t)], column_name=name)
                    if block:
                        doc.blocks.append(block)
                i += 1
            previous_blank = False
            continue
        quote = _QUOTE_RE.match(line)
        if quote:
            flush_item()
            if para_kind != "quote":
                flush_paragraph()
                para_kind = "quote"
            content = quote.group(2)
            if content.strip():
                paragraph.append((number, len(quote.group(1)) + 1, content))
            else:
                flush_paragraph()
            i += 1
            previous_blank = False
            continue
        list_match = _LIST_RE.match(line)
        if list_match:
            flush_all()
            indent = len(list_match.group(1)) + len(list_match.group(2)) + len(list_match.group(3))
            content = list_match.group(4)
            col = indent + 1
            checkbox = _CHECKBOX_RE.match(content)
            if checkbox:
                content = content[checkbox.end() :]
                col += checkbox.end()
            item = [(number, col, content)] if content.strip() else None
            item_ordered = list_match.group(2)[0].isdigit()
            i += 1
            previous_blank = False
            continue
        if line.startswith(("    ", "\t")) and previous_blank and item is None and not paragraph:
            i += 1  # indented code
            continue
        if _HTML_BLOCK_RE.match(line) and not paragraph and item is None:
            segments = [(number, 1, line)]
            block = make_block("paragraph", segments)
            if block:
                doc.blocks.append(block)
            i += 1
            previous_blank = False
            continue
        content = line.lstrip()
        col = len(line) - len(content) + 1
        if item is not None:
            item.append((number, col, content.rstrip()))
        else:
            paragraph.append((number, col, content.rstrip()))
        previous_blank = False
        i += 1
    flush_all()
    return doc
