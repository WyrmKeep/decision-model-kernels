"""Read the ASD-STE100 dictionary from a local copy of the official PDF.

The PDF is copyrighted by ASD. This module never writes the PDF or any word
list into the repository. It extracts the dictionary at run time and keeps the
result in a cache folder outside the repository (STE_CACHE or ~/.cache/ste100).

Layout of the dictionary pages (Issue 9, part 2):

* Four columns: word and part of speech, approved meaning or alternatives,
  STE example, non-STE example.
* A word in uppercase is approved. A word in lowercase is not approved.
* An approved verb lists its forms after the part of speech.
* An approved adjective lists its comparative and superlative forms in brackets.
* A word that is not approved lists approved alternatives in column 2.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

EXTRACTOR_VERSION = 2
PDF_URL = "https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf"
POS_CODES = ("n", "v", "adj", "adv", "prep", "conj", "pron", "art")
_POS_RE = "|".join(POS_CODES)
_HEAD_RE = re.compile(r"^(?P<word>.+?)\s*\((?P<pos>" + _POS_RE + r")\)\s*,?\s*(?P<rest>.*)$")
_ALT_RE = re.compile(
    r"(?P<word>[A-Z][A-Z0-9’'\-/.… ]*?)\s*\((?P<pos>" + _POS_RE + r"|TN|TV)\)"
)
_WORD_RE = re.compile(
    r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>'
)
_FURNITURE_RE = re.compile(r"^(Issue 9|Page\s+\S+|\d{4}-\d{2}-\d{2})$")


class DictionaryError(Exception):
    """The dictionary cannot be read."""


@dataclass
class Entry:
    word: str
    pos: str
    approved: bool
    forms: list[str] = field(default_factory=list)
    alternatives: list[tuple[str, str]] = field(default_factory=list)

    def to_json(self) -> dict:
        data: dict = {"w": self.word, "p": self.pos, "a": self.approved}
        if self.forms:
            data["f"] = self.forms
        if self.alternatives:
            data["alt"] = [list(item) for item in self.alternatives]
        return data

    @classmethod
    def from_json(cls, data: dict) -> "Entry":
        return cls(
            word=data["w"],
            pos=data["p"],
            approved=data["a"],
            forms=list(data.get("f", [])),
            alternatives=[(a[0], a[1]) for a in data.get("alt", [])],
        )


# --------------------------------------------------------------------------
# PDF access


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_pdf_path() -> Path:
    env = os.environ.get("STE_PDF")
    if env:
        return Path(env).expanduser()
    return Path("~/.local/share/asd-ste100/ASD-STE100_ISSUE9.pdf").expanduser()


def cache_dir(configured: str | None = None) -> Path:
    env = os.environ.get("STE_CACHE")
    if env:
        return Path(env).expanduser()
    if configured:
        return Path(configured).expanduser()
    return Path("~/.cache/ste100").expanduser()


def download_pdf(path: Path, expected_sha256: str, url: str = PDF_URL) -> None:
    """Download the official PDF and verify its hash before it replaces any file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ste-check"})
    with urllib.request.urlopen(request, timeout=120) as response, open(temp, "wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256_file(temp)
    if actual != expected_sha256:
        temp.unlink()
        raise DictionaryError(
            f"The downloaded PDF has sha256 {actual}. Expected {expected_sha256}."
        )
    temp.replace(path)


def ensure_pdf(path: Path, expected_sha256: str, allow_download: bool) -> Path:
    if not path.exists():
        if not allow_download:
            raise DictionaryError(
                f"PDF not found at {path}. Download it from {PDF_URL} "
                "or run with --download-pdf."
            )
        download_pdf(path, expected_sha256)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise DictionaryError(
            f"PDF {path} has sha256 {actual}. Expected {expected_sha256}."
        )
    return path


def _run_pdftotext(pdf: Path, first: int | None = None, last: int | None = None) -> str:
    binary = shutil.which("pdftotext")
    if binary is None:
        raise DictionaryError("The pdftotext program (poppler) is not installed.")
    command = [binary, "-bbox"]
    if first is not None:
        command += ["-f", str(first)]
    if last is not None:
        command += ["-l", str(last)]
    command += [str(pdf), "-"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise DictionaryError("pdftotext failed: " + result.stderr.decode("utf-8", "replace"))
    return result.stdout.decode("utf-8", "replace")


# --------------------------------------------------------------------------
# Extraction

Word = tuple[float, float, float, float, str]


def _split_pages(xml: str) -> list[list[Word]]:
    pages: list[list[Word]] = []
    for chunk in xml.split("<page ")[1:]:
        words = [
            (float(a), float(b), float(c), float(d), html.unescape(e))
            for a, b, c, d, e in _WORD_RE.findall(chunk)
        ]
        pages.append(words)
    return pages


def _header(words: list[Word]) -> dict[str, float] | None:
    found: dict[str, float] = {}
    for x0, y0, _x1, _y1, text in words:
        if y0 < 95 and text in ("Word", "Approved", "ALTERNATIVES", "STE", "Non-STE"):
            found.setdefault(text, x0)
    if all(key in found for key in ("Word", "Approved", "ALTERNATIVES", "STE", "Non-STE")):
        return found
    return None


def _lines(words: list[Word]) -> list[tuple[float, list[Word]]]:
    ordered = sorted(words, key=lambda w: (round(w[1]), w[0]))
    lines: list[tuple[float, list[Word]]] = []
    for word in ordered:
        if lines and abs(lines[-1][0] - word[1]) < 3:
            lines[-1][1].append(word)
        else:
            lines.append((word[1], [word]))
    for _y, items in lines:
        items.sort(key=lambda w: w[0])
    return lines


_POS_ANY = re.compile(r"\((?:" + _POS_RE + r")\)")


def _form_like(text: str) -> bool:
    """True when a column 1 line can hold verb or adjective forms (uppercase, no part of speech)."""
    stripped = re.sub(r"\balso\b", "", text)
    if _POS_ANY.search(stripped) or stripped.rstrip().endswith("-"):
        return False
    return not any(ch.islower() for ch in stripped)


def _continues(lines: list[str], text: str, following: str) -> bool:
    """Decide whether a column 1 line belongs to the head block that is open."""
    joined = " ".join(lines)
    last = lines[-1].rstrip()
    if not _form_like(text):
        return False
    if joined.count("(") > joined.count(")") or last.endswith(","):
        return True
    if text.startswith("(") and text.rstrip().endswith((",", ")")):
        return True
    # A form line without commas, for example a regular verb form that follows its head.
    match = _HEAD_RE.match(lines[0])
    if not match or match.group("pos") != "v" or not match.group("word").isupper():
        return False
    token = text.strip().rstrip(",")
    if " " in token or "(" in token or _POS_ANY.match(following):
        return False
    return token.lower().startswith(match.group("word").lower()[:3])


def _parse_block(lines: list[str]) -> tuple[str, str, list[str]] | None:
    """Return the word, the part of speech and the listed forms of one head block."""
    head = ""
    index = 0
    match = None
    for index, line in enumerate(lines):
        if head.endswith("-"):
            # A hyphen at the end of a line is a wrapped word, unless the part is a short prefix.
            if len(head.split()[-1]) > 4:
                head = head[:-1]
        elif head:
            head += " "
        head += line
        match = _HEAD_RE.match(head.strip())
        if match:
            break
    if not match:
        text = head.strip()
        if "(" in text or not text:
            return None
        return text, "", []
    chunks = [match.group("rest")] + lines[index + 1 :]
    forms: list[str] = []
    for chunk in chunks:
        chunk = re.sub(r"\balso\b", " ", chunk)
        for piece in re.split(r"[,()]", chunk):
            piece = piece.strip()
            if piece:
                forms.append(piece.lower())
    return match.group("word").strip(), match.group("pos"), forms


_PAREN_RE = re.compile(r"^(?P<base>[^()]+?)\s*\((?P<inner>[^()]*)\)\s*$")


def _normalize_word(word: str) -> list[str]:
    """Turn a head word with a bracket note into the words that it names.

    "case (in case of)" names the phrase "in case of". "matt (or matte)" names two words.
    """
    word = re.sub(r"\s+", " ", word.strip())
    match = _PAREN_RE.match(word)
    if not match:
        return [word]
    base = match.group("base").strip()
    inner = match.group("inner").strip()
    if inner.startswith("or "):
        return [base, inner[3:].strip()]
    if base in inner:
        return [inner]
    return [f"{base} {inner}"]


def _page_entries(words: list[Word]) -> tuple[list[dict], list[tuple[float, str]]]:
    """Return head blocks of one page and the column 2 text above the first head."""
    head = _header(words)
    if head is None:
        return [], []
    col1_start = head["Word"]
    col2_start = head["Approved"] - 6
    body = [w for w in words if 89 <= w[1] < 715]
    # The body table can start column 3 a few points left of its header, so find
    # the real left edge from the words that begin the lines of column 3.
    near = [round(w[0], 1) for w in body if head["STE"] - 14 <= w[0] <= head["STE"] + 4]
    if near:
        col3_start = max(set(near), key=near.count) - 3
    else:
        col3_start = head["STE"] - 12
    col1_words = [w for w in body if w[0] < col2_start]
    col2_words = [w for w in body if col2_start <= w[0] < col3_start]
    col1_lines: list[tuple[float, str]] = []
    for y, items in _lines(col1_words):
        if items[0][0] > col1_start + 8:
            continue
        text = " ".join(w[4] for w in items)
        if _FURNITURE_RE.match(text):
            continue
        col1_lines.append((y, text))
    blocks: list[dict] = []
    current: dict | None = None
    for index, (y, text) in enumerate(col1_lines):
        following = col1_lines[index + 1][1] if index + 1 < len(col1_lines) else ""
        if current is not None:
            joined = " ".join(current["lines"])
            if not _POS_ANY.search(joined):
                # The head has no part of speech yet. It wraps when the next line is close.
                if y - current["last_y"] <= 13.5:
                    current["lines"].append(text)
                    current["last_y"] = y
                    continue
            elif _continues(current["lines"], text, following):
                current["lines"].append(text)
                current["last_y"] = y
                continue
        current = {"y": y, "last_y": y, "lines": [text], "col2": []}
        blocks.append(current)
    col2_lines = [(y, " ".join(w[4] for w in items)) for y, items in _lines(col2_words)]
    leading: list[tuple[float, str]] = []
    for y, text in col2_lines:
        target = None
        for block in blocks:
            if y >= block["y"] - 5:
                target = block
            else:
                break
        if target is None:
            leading.append((y, text))
        else:
            target["col2"].append((y, text))
    return blocks, leading


def extract_entries(pages: list[list[Word]]) -> list[Entry]:
    entries: list[Entry] = []
    previous: Entry | None = None
    for words in pages:
        blocks, leading = _page_entries(words)
        if leading and previous is not None and not previous.approved:
            _collect_alternatives(previous, leading)
        for block in blocks:
            parsed = _parse_block(block["lines"])
            if parsed is None:
                continue
            word, pos, forms = parsed
            approved = not any(ch.islower() for ch in re.sub(r"\(.*?\)", "", word))
            for name in _normalize_word(word):
                entry = Entry(word=name.lower(), pos=pos, approved=approved, forms=list(forms))
                if not approved:
                    _collect_alternatives(entry, block["col2"])
                entries.append(entry)
                previous = entry
    return entries


def _collect_alternatives(entry: Entry, lines: list[tuple[float, str]]) -> None:
    """Read the approved alternatives of a word that is not approved.

    Column 2 holds one alternative a row. A row can wrap to a second line.
    Guidance text for the writer is in lowercase and is skipped.
    """
    rows: list[str] = []
    previous_y = -100.0
    for y, text in lines:
        text = text.strip()
        if rows and y - previous_y <= 13.5:
            rows[-1] = rows[-1] + " " + text
        else:
            rows.append(text)
        previous_y = y
    for row in rows:
        matches = list(_ALT_RE.finditer(row))
        if matches:
            for match in matches:
                _add_alternative(entry, match.group("word").strip().lower(), match.group("pos"))
        elif row and _is_upper_phrase(row):
            _add_alternative(entry, row.strip().lower(), "")


def _is_upper_phrase(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _add_alternative(entry: Entry, word: str, pos: str) -> None:
    alt = (word, pos)
    if alt not in entry.alternatives:
        entry.alternatives.append(alt)


def extract_from_pdf(pdf: Path) -> list[Entry]:
    pages = _split_pages(_run_pdftotext(pdf))
    dictionary_pages = [words for words in pages if _header(words) is not None]
    if not dictionary_pages:
        raise DictionaryError("No dictionary page found in the PDF.")
    return extract_entries(dictionary_pages)


# --------------------------------------------------------------------------
# Cache


def load_entries(pdf: Path, pdf_sha256: str, cache: Path) -> list[Entry]:
    """Return the dictionary entries. Use the cache when it matches the PDF."""
    cache.mkdir(parents=True, exist_ok=True)
    cache_file = cache / f"dictionary-{pdf_sha256[:16]}-v{EXTRACTOR_VERSION}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if data.get("pdf_sha256") == pdf_sha256 and data.get("version") == EXTRACTOR_VERSION:
                return [Entry.from_json(item) for item in data["entries"]]
        except (ValueError, KeyError):
            pass
    entries = extract_from_pdf(pdf)
    payload = {
        "version": EXTRACTOR_VERSION,
        "pdf_sha256": pdf_sha256,
        "entries": [entry.to_json() for entry in entries],
    }
    temp = cache_file.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(cache_file)
    return entries


# --------------------------------------------------------------------------
# Lookup

MODAL_VERBS = frozenset({"can", "could", "will", "must", "cannot"})
_VOWELS = "aeiou"


@dataclass
class Lookup:
    """Result of one word lookup."""

    approved_pos: set[str] = field(default_factory=set)
    unapproved: list[Entry] = field(default_factory=list)
    form_of: list[tuple[str, str]] = field(default_factory=list)  # (base, ending)
    base: str | None = None

    @property
    def status(self) -> str:
        if self.approved_pos:
            return "approved"
        if self.unapproved:
            return "unapproved"
        if self.form_of:
            return "form"
        return "unknown"

    @property
    def alternatives(self) -> list[str]:
        seen: list[str] = []
        for entry in self.unapproved:
            for word, _pos in entry.alternatives:
                if word not in seen:
                    seen.append(word)
        return seen

    @property
    def unapproved_pos(self) -> set[str]:
        return {entry.pos for entry in self.unapproved}


def candidate_bases(word: str) -> list[tuple[str, str]]:
    """Return possible (base, ending) pairs for an inflected English word."""
    out: list[tuple[str, str]] = []
    if word.endswith("ies") and len(word) > 4:
        out.append((word[:-3] + "y", "s"))
    if word.endswith("es") and len(word) > 3:
        out.append((word[:-2], "s"))
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        out.append((word[:-1], "s"))
    if word.endswith("ied") and len(word) > 4:
        out.append((word[:-3] + "y", "ed"))
    if word.endswith("ed") and len(word) > 3:
        out.append((word[:-2], "ed"))
        out.append((word[:-1], "ed"))
        if len(word) > 4 and word[-3] == word[-4]:
            out.append((word[:-3], "ed"))
    if word.endswith("ing") and len(word) > 4:
        out.append((word[:-3], "ing"))
        out.append((word[:-3] + "e", "ing"))
        if len(word) > 5 and word[-4] == word[-5]:
            out.append((word[:-4], "ing"))
    if word.endswith("er") and len(word) > 3:
        out.append((word[:-2], "er"))
        out.append((word[:-1], "er"))
    if word.endswith("est") and len(word) > 4:
        out.append((word[:-3], "est"))
    return out


def regular_forms(word: str) -> list[str]:
    """Return the regular third-person and past forms of a verb."""
    consonant_y = word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS
    if word.endswith(("s", "x", "z", "ch", "sh", "o")):
        third = word + "es"
    elif consonant_y:
        third = word[:-1] + "ies"
    else:
        third = word + "s"
    forms = [third]
    if word.endswith("e"):
        forms.append(word + "d")
    elif consonant_y:
        forms.append(word[:-1] + "ied")
    else:
        cvc = (
            len(word) > 2
            and word[-3] not in _VOWELS
            and word[-2] in _VOWELS
            and word[-1] not in _VOWELS + "wxy"
        )
        if cvc:
            forms.append(word + word[-1] + "ed")
        forms.append(word + "ed")
    return forms


def plural_forms(word: str) -> list[str]:
    """Return the regular plural forms of a noun."""
    if " " in word:
        head, _, tail = word.rpartition(" ")
        return [f"{head} {p}" for p in plural_forms(tail)]
    out = [word + "s"]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        out.append(word + "es")
    if word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        out.append(word[:-1] + "ies")
    if word.endswith("fe"):
        out.append(word[:-2] + "ves")
    elif word.endswith("f"):
        out.append(word[:-1] + "ves")
    return out


def _ending_fits(ending: str, pos: str) -> bool:
    if ending == "s":
        return pos in ("n", "v")
    if ending == "ed":
        return pos in ("v", "adj")
    if ending == "ing":
        return pos == "v"
    return pos == "adj"


class Dictionary:
    """Word lookup built from the extracted entries."""

    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries
        self.approved_forms: dict[str, set[str]] = defaultdict(set)  # form -> pos set
        self.approved_base: dict[str, set[str]] = defaultdict(set)  # base -> pos set
        self.unapproved: dict[str, list[Entry]] = defaultdict(list)
        self.approved_phrases: set[str] = set()
        self.unapproved_phrases: dict[str, list[Entry]] = {}
        self.verb_bases: set[str] = set()
        self.irregular_verbs: set[str] = set()
        for entry in entries:
            if entry.pos == "v":
                self.verb_bases.add(entry.word)
            if entry.approved:
                self._add_approved(entry)
            else:
                self.unapproved[entry.word].append(entry)
                if " " in entry.word:
                    self.unapproved_phrases[entry.word] = self.unapproved[entry.word]

    def _add_approved(self, entry: Entry) -> None:
        word = entry.word
        self.approved_base[word].add(entry.pos)
        self.approved_forms[word].add(entry.pos)
        if " " in word:
            self.approved_phrases.add(word)
        forms = list(entry.forms)
        if entry.pos == "v" and forms and regular_forms(word)[-1:] != forms[-1:]:
            self.irregular_verbs.add(word)
        if entry.pos == "v" and not forms and word not in MODAL_VERBS:
            forms = regular_forms(word)
        for form in forms:
            self.approved_forms[form].add(entry.pos)
        if entry.pos == "n":
            for plural in plural_forms(word):
                self.approved_forms[plural].add("n")

    def counts(self) -> dict[str, int]:
        approved = sum(1 for e in self.entries if e.approved)
        return {"approved": approved, "unapproved": len(self.entries) - approved}

    def lookup(self, word: str) -> Lookup:
        word = word.lower()
        result = Lookup()
        pos = self.approved_forms.get(word)
        if pos:
            result.approved_pos = set(pos)
        if word in self.unapproved:
            result.unapproved.extend(self.unapproved[word])
        for base, ending in candidate_bases(word):
            for entry in self.unapproved.get(base, ()):
                if _ending_fits(ending, entry.pos) and entry not in result.unapproved:
                    result.unapproved.append(entry)
            if not result.approved_pos and base in self.approved_base:
                if ending == "ed" and base in self.irregular_verbs:
                    continue
                if any(_ending_fits(ending, pos) for pos in self.approved_base[base]):
                    result.form_of.append((base, ending))
        if result.approved_pos or result.unapproved:
            result.form_of = []
        if result.form_of:
            result.base = result.form_of[0][0]
        elif result.unapproved:
            result.base = result.unapproved[0].word
        return result

    def phrase_alternatives(self, phrase: str) -> list[str] | None:
        entries = self.unapproved_phrases.get(phrase)
        if not entries:
            return None
        return Lookup(unapproved=list(entries)).alternatives


def load_dictionary(pdf: Path, pdf_sha256: str, cache: Path) -> Dictionary:
    return Dictionary(load_entries(pdf, pdf_sha256, cache))
