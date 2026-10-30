"""The STE rule checks for Markdown text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from ste_dict import Dictionary, Lookup
from ste_glossary import Glossary
from ste_markdown import Block, parse_document
from ste_text import (
    ATOM_KIND,
    GROUP_KIND,
    NUM,
    PUNCT,
    WORD,
    Token,
    count_words,
    group_parens,
    split_sentences,
    tokenize,
)

POS_NAMES = {
    "n": "noun",
    "v": "verb",
    "adj": "adjective",
    "adv": "adverb",
    "prep": "preposition",
    "conj": "conjunction",
    "pron": "pronoun",
    "art": "article",
}

NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve thirteen
    fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
    seventy eighty ninety hundred thousand million billion first second third fourth fifth
    sixth seventh eighth ninth tenth""".split()
)

BE_FORMS = frozenset({"is", "are", "was", "were", "be", "been", "being", "am"})
PASSIVE_ADVERBS = frozenset(
    {"not", "only", "also", "always", "never", "then", "first", "already", "still", "usually",
     "often", "just", "automatically", "all", "both", "later", "once", "next"}
)
IRREGULAR_PARTICIPLES = frozenset(
    """begun broken built chosen done driven eaten fallen found forgotten given gone held
    hidden kept known left lost made paid sent shown spent taken thrown told understood
    withdrawn won worn written drawn seen bound""".split()
)
NOT_PARTICIPLES = frozenset({"bed", "red", "shed", "sled", "fled", "bled", "hundred", "sacred",
                             "naked", "wicked", "kindred", "need", "speed", "seed", "feed"})
DETERMINERS = frozenset(
    """the a an this that these those each all any some no every both your its their our my
    his her one another either neither""".split()
)
# Words that always start a noun group. "that", "this" and "those" can also be pronouns.
NOUN_STARTERS = frozenset(
    "the a an these each every its their your our my his her no another".split()
)
MODALS = frozenset({"can", "could", "will", "must", "cannot", "do", "does", "did", "not", "to"})
SUBJECT_PRONOUNS = frozenset({"you", "we", "they"})
CONDITION_STARTERS = frozenset({"if", "when", "after", "before", "while", "until", "unless"})
AUX_STARTS = frozenset(
    """is are was were be been being am has have had does did will can must may should could
    would cannot shall might""".split()
)

_CONTRACTION_RE = re.compile(
    r"^(?:\w+n't|\w+'(?:re|ve|ll|d|m)|(?:it|that|there|here|what|who|let|he|she|where|how|when)'s)$"
)

# British spelling heuristic (rule 1.14). Code only. No dictionary text.
_ISE_EXCEPTIONS = frozenset(
    """advertise advise arise chastise circumcise comprise compromise demise despise devise
    disguise enterprise excise exercise expertise franchise improvise incise merchandise
    paradise poise praise precise premise promise raise revise rise supervise surprise
    televise treatise otherwise likewise wise concise noise cruise bruise appraise apprise
    reprise vise guise mortise tortoise porpoise turquoise sunrise upraise anise surmise chemise""".split()
)
_ISE_RE = re.compile(r"^(?P<stem>[a-z]{3,}?)(?P<suffix>ise|ised|ises|ising|isation|isations|iser|isers)$")
_OUR_RE = re.compile(
    r"^(?P<stem>colo|favo|behavio|hono|labo|neighbo|rumo|vapo|humo|harbo|flavo|odo|armo|endeavo|"
    r"savo|vigo|rigo|valo|clamo|cando|parlo|tumo|demeano|fervo|ardo|splendo)ur"
    r"(?P<suffix>s|ed|ing|ful|less|able|ite|ites|ably)?$"
)
_YSE_RE = re.compile(r"^(?P<stem>analy|paraly|cataly|dialy|electroly)s(?P<suffix>e|es|ed|ing|er|ers)$")
_TRE_RE = re.compile(r"^(?P<stem>cen|me|li|thea|spec)tre(?P<suffix>s|d)?$")
_BRE_RE = re.compile(r"^(?P<stem>fi|cali|som|sa)bre(?P<suffix>s|d)?$")
_LL_RE = re.compile(
    r"^(?P<stem>model|label|travel|cancel|level|fuel|signal|channel|marvel|counsel|panel|"
    r"tunnel|quarrel|pencil|dial|total|equal|rival|duel|libel|jewel|shovel|trowel|medal)l"
    r"(?P<suffix>ed|ing|er|ers)$"
)
_BRITISH_WORDS = {
    "grey": "gray", "programme": "program", "programmes": "programs", "licence": "license",
    "licences": "licenses", "defence": "defense", "offence": "offense", "pretence": "pretense",
    "practise": "practice", "cheque": "check", "tyre": "tire", "tyres": "tires",
    "catalogue": "catalog", "catalogues": "catalogs", "judgement": "judgment", "ageing": "aging",
    "aluminium": "aluminum", "artefact": "artifact", "artefacts": "artifacts", "whilst": "while",
    "amongst": "among", "learnt": "learned", "spelt": "spelled", "mould": "mold",
    "plough": "plow", "storey": "story", "sceptic": "skeptic", "sulphur": "sulfur",
    "enrol": "enroll", "fulfil": "fulfill", "skilful": "skillful", "instalment": "installment",
    "analogue": "analog", "jewellery": "jewelry", "speciality": "specialty", "kerb": "curb",
    "towards": "toward", "fibre": "fiber", "fibres": "fibers",
}


def _phrase_part(token: Token) -> bool:
    """A word, or a name with letters and digits only (for example DeltaNet), can be part of a phrase."""
    return token.kind == WORD or (token.kind == ATOM_KIND and token.text.isalnum())


def british_spelling(word: str) -> str | None:
    """Return the American spelling when ``word`` looks British, else None."""
    if word in _BRITISH_WORDS:
        return _BRITISH_WORDS[word]
    match = _ISE_RE.match(word)
    if match:
        base = match.group("stem") + "ise"
        if base in _ISE_EXCEPTIONS or base.endswith("wise"):
            return None
        suffix = match.group("suffix").replace("is", "iz", 1)
        return match.group("stem") + suffix
    match = _OUR_RE.match(word)
    if match:
        return match.group("stem") + "r" + (match.group("suffix") or "")
    match = _YSE_RE.match(word)
    if match:
        return match.group("stem") + "z" + match.group("suffix")
    match = _TRE_RE.match(word)
    if match:
        return match.group("stem") + "ter" + (match.group("suffix") or "")
    match = _BRE_RE.match(word)
    if match:
        return match.group("stem") + "ber" + (match.group("suffix") or "")
    match = _LL_RE.match(word)
    if match:
        return match.group("stem") + match.group("suffix")
    return None


@dataclass
class Finding:
    file: str
    line: int
    col: int
    rule: str
    severity: str  # error or warning
    message: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class Settings:
    procedure_headings: list[str] = field(default_factory=list)
    units: frozenset[str] = frozenset()
    procedure_limit: int = 20
    description_limit: int = 25
    paragraph_limit: int = 6
    cluster_limit: int = 3
    max_alternatives: int = 3


@dataclass
class Context:
    procedure: bool = False
    first_of_item: bool = False


class Checker:
    def __init__(self, dictionary: Dictionary, glossary: Glossary, settings: Settings) -> None:
        self.dictionary = dictionary
        self.glossary = glossary
        self.settings = settings
        self._unit_words = frozenset(unit.lower() for unit in settings.units)
        self._lookup = lru_cache(maxsize=None)(self._lookup_uncached)

    # ------------------------------------------------------------------
    # Public

    def check(self, text: str, filename: str, glossary_file: bool = False) -> list[Finding]:
        doc = parse_document(text, self.settings.procedure_headings)
        findings: list[Finding] = []
        for block in doc.blocks:
            if glossary_file and block.kind == "cell":
                if block.table_header or block.column_name.lower() in ("term", "type"):
                    continue
            findings.extend(self._check_block(block, filename))
        findings = [f for f in findings if not _ignored(f, doc.ignores)]
        findings.sort(key=lambda f: (f.line, f.col, f.rule))
        return findings

    def describe(self, word: str) -> str:
        """Return a one-line description of how the checker sees one word."""
        lower = word.lower()
        parts: list[str] = []
        for term in self.glossary.lookup(lower):
            parts.append(f"glossary: technical {term.kind} '{term.term}'")
        result = self.dictionary.lookup(lower)
        if result.approved_pos:
            names = ", ".join(POS_NAMES.get(p, p) for p in sorted(result.approved_pos))
            parts.append(f"dictionary: approved ({names})")
        for entry in result.unapproved:
            alts = ", ".join(w for w, _p in entry.alternatives[: self.settings.max_alternatives])
            parts.append(
                f"dictionary: not approved as {POS_NAMES.get(entry.pos, entry.pos)}"
                + (f", use: {alts}" if alts else "")
            )
        for base, ending in result.form_of:
            parts.append(f"dictionary: unlisted form (-{ending}) of approved '{base}'")
        if lower in NUMBER_WORDS:
            parts.append("number word")
        british = british_spelling(lower)
        if british:
            parts.append(f"British spelling, American: {british}")
        return f"{word}: " + ("; ".join(parts) if parts else "unknown")

    # ------------------------------------------------------------------
    # Lookup helpers

    def _lookup_uncached(self, word: str) -> Lookup:
        return self.dictionary.lookup(word)

    def _alternatives_text(self, result: Lookup) -> str:
        words: list[str] = []
        for entry in result.unapproved:
            for alt, pos in entry.alternatives:
                label = alt
                if pos in ("TN", "TV") and alt:
                    label = f"{alt} (technical {'noun' if pos == 'TN' else 'verb'})"
                if label not in words:
                    words.append(label)
        return ", ".join(words[: self.settings.max_alternatives])

    def _tags(self, word: str) -> tuple[set[str], set[str], set[str]]:
        """Return the approved, unapproved and glossary parts of speech of a word."""
        result = self._lookup(word)
        gloss = {t.kind for t in self.glossary.lookup(word)}
        return set(result.approved_pos), result.unapproved_pos, gloss

    # ------------------------------------------------------------------
    # Block checks

    def _position(self, block: Block, token_or_index: Token | int) -> tuple[int, int]:
        index = token_or_index.start if isinstance(token_or_index, Token) else token_or_index
        index = max(0, min(index, len(block.pos) - 1))
        return block.pos[index]

    def _add(self, out: list[Finding], filename: str, block: Block, where: Token | int,
             rule: str, severity: str, message: str) -> None:
        line, col = self._position(block, where)
        out.append(Finding(filename, line, col, rule, severity, message))

    def _check_block(self, block: Block, filename: str) -> list[Finding]:
        out: list[Finding] = []
        tokens = tokenize(block.text, self.settings.units)
        if block.kind == "cell" and all(t.kind in (NUM, ATOM_KIND, PUNCT) for t in tokens):
            return out
        for index, char in enumerate(block.text):
            if char == ";":
                self._add(out, filename, block, index, "8.1", "error",
                          "Semicolon. Write two sentences.")
        outer, groups = group_parens(tokens)
        outer_sentences = split_sentences(outer)
        inner_sentences: list[list[Token]] = []
        for group in groups:
            inner_sentences.extend(split_sentences(group))
        is_prose = block.kind in ("paragraph", "list_item", "quote")
        for number, sentence in enumerate(outer_sentences):
            ctx = self._context(block, sentence, number)
            self._check_sentence(out, filename, block, sentence, ctx, outer=True)
        for sentence in inner_sentences:
            self._check_sentence(out, filename, block, sentence, Context(), outer=False)
        if is_prose and len(outer_sentences) > self.settings.paragraph_limit:
            self._add(out, filename, block, outer_sentences[0][0], "6.6", "error",
                      f"Paragraph has {len(outer_sentences)} sentences. "
                      f"The limit is {self.settings.paragraph_limit}.")
        return out

    def _context(self, block: Block, sentence: list[Token], number: int) -> Context:
        ctx = Context()
        if block.kind == "list_item" and block.in_procedure and number == 0:
            ctx.procedure = True
            ctx.first_of_item = True
        elif block.kind in ("paragraph", "list_item", "quote") and self._imperative(sentence):
            ctx.procedure = True
        return ctx

    # ------------------------------------------------------------------
    # Sentence checks

    def _check_sentence(self, out: list[Finding], filename: str, block: Block,
                        sentence: list[Token], ctx: Context, outer: bool) -> None:
        words = [t for t in sentence if t.kind != PUNCT]
        if not words:
            return
        self._check_vocabulary(out, filename, block, sentence, ctx)
        if block.kind == "heading":
            return
        is_prose = block.kind in ("paragraph", "list_item", "quote")
        if is_prose and outer:
            limit = self.settings.procedure_limit if ctx.procedure else self.settings.description_limit
            total = count_words(sentence)
            if total > limit:
                rule = "5.1" if ctx.procedure else "6.3"
                kind = "Procedure" if ctx.procedure else "Description"
                self._add(out, filename, block, words[0], rule, "error",
                          f"{kind} sentence has {total} words. The limit is {limit}.")
            if ctx.first_of_item and not self._imperative(sentence):
                first = words[0]
                if first.kind == WORD:
                    self._add(out, filename, block, first, "5.3", "warning",
                              f"Procedure step does not start with an imperative verb ('{first.text}').")
        self._check_passive(out, filename, block, sentence, ctx)
        self._check_clusters(out, filename, block, sentence, ctx)

    # -- vocabulary ----------------------------------------------------

    def _check_vocabulary(self, out: list[Finding], filename: str, block: Block,
                          sentence: list[Token], ctx: Context) -> None:
        consumed: set[int] = set()
        for index in range(len(sentence)):
            if index in consumed:
                continue
            for length in (4, 3, 2):
                seq = sentence[index : index + length]
                if len(seq) < length or not all(_phrase_part(t) for t in seq):
                    continue
                phrase = " ".join(t.lower for t in seq)
                if phrase in self.glossary.surface or phrase in self.dictionary.approved_phrases:
                    consumed.update(range(index, index + length))
                    break
                alternatives = self.dictionary.phrase_alternatives(phrase)
                if alternatives is not None:
                    shown = ", ".join(alternatives[: self.settings.max_alternatives])
                    hint = f" Use: {shown}." if shown else " Use a different construction."
                    self._add(out, filename, block, seq[0], "1.1", "error",
                              f"'{phrase}' is not approved.{hint}")
                    consumed.update(range(index, index + length))
                    break
        previous: Token | None = None
        for index, token in enumerate(sentence):
            if index not in consumed and token.kind == WORD:
                self._check_word(out, filename, block, sentence, index, previous, ctx)
            if token.kind != PUNCT:
                previous = token
            else:
                previous = None if token.text in (",", ";", ":", ".", "!", "?") else previous

    def _check_word(self, out: list[Finding], filename: str, block: Block,
                    sentence: list[Token], index: int, previous: Token | None,
                    ctx: Context) -> None:
        token = sentence[index]
        word = token.lower
        if _CONTRACTION_RE.match(word):
            self._add(out, filename, block, token, "4.2", "error",
                      f"Contraction '{token.text}'. Write the words in full.")
            return
        if word.endswith("'s") and len(word) > 2:
            word = word[:-2]
        elif word.endswith("'") and len(word) > 1:
            word = word[:-1]
        if "'" in word:
            self._add(out, filename, block, token, "1.1", "error",
                      f"'{token.text}' is not an approved word.")
            return
        if "-" in word:
            if self._known_simple(word):
                return
            for part in word.split("-"):
                if part and not self._known_simple(part) and not part.isdigit():
                    self._report_unknown(out, filename, block, token, part, whole=token.text)
                    return
            return
        if self._known_simple(word):
            self._check_part_of_speech(out, filename, block, sentence, index, previous, ctx)
            return
        self._report_unknown(out, filename, block, token, word)

    def _known_simple(self, word: str) -> bool:
        if word in NUMBER_WORDS or word in self._unit_words:
            return True
        if self.glossary.lookup(word):
            return True
        return bool(self._lookup(word).approved_pos)

    def _report_unknown(self, out: list[Finding], filename: str, block: Block, token: Token,
                        word: str, whole: str | None = None) -> None:
        shown = whole or token.text
        label = f"'{word}' in '{shown}'" if whole and whole.lower() != word else f"'{shown}'"
        american = british_spelling(word)
        if american:
            self._add(out, filename, block, token, "1.14", "error",
                      f"British spelling {label}. Use '{american}'.")
            return
        result = self._lookup(word)
        if result.unapproved and any(
            alt == word and pos in ("TN", "TV")
            for entry in result.unapproved
            for alt, pos in entry.alternatives
        ):
            self._add(out, filename, block, token, "1.1", "error",
                      f"{label} is not approved. Use it only as a technical term "
                      "from the glossary.")
            return
        if result.unapproved:
            alternatives = self._alternatives_text(result)
            hint = f" Use: {alternatives}." if alternatives else " Use a different construction."
            self._add(out, filename, block, token, "1.1", "error",
                      f"{label} is not approved.{hint}")
            return
        if result.form_of:
            base, ending = result.form_of[0]
            pos = ", ".join(POS_NAMES.get(p, p) for p in sorted(self.dictionary.approved_base[base]))
            extra = " Use more or most with an adjective." if ending in ("er", "est") else ""
            self._add(out, filename, block, token, "1.4", "error",
                      f"{label} is not an approved form of '{base}' ({pos}).{extra}")
            return
        self._add(out, filename, block, token, "1.1", "error",
                  f"{label} is not in the dictionary or the glossary.")

    def _check_part_of_speech(self, out: list[Finding], filename: str, block: Block,
                              sentence: list[Token], index: int, previous: Token | None,
                              ctx: Context) -> None:
        token = sentence[index]
        word = token.lower
        approved, unapproved, gloss = self._tags(word)
        result = self._lookup(word)
        if (
            previous is not None
            and previous.kind == WORD
            and previous.lower in NOUN_STARTERS
            and "v" in approved
            and not ({"n", "adj", "adv"} & approved)
            and not ({"noun", "name"} & gloss)
        ):
            alternatives = ", ".join(
                alt
                for entry in result.unapproved
                if entry.pos == "n"
                for alt, _pos in entry.alternatives[: self.settings.max_alternatives]
            )
            hint = f" Use: {alternatives}." if alternatives else ""
            self._add(out, filename, block, token, "1.2", "warning",
                      f"'{token.text}' is approved as a verb only. The noun and adjective uses are not approved.{hint}")
            return
        if "verb" in gloss or "v" in approved or "v" not in unapproved:
            return
        verb_context = self._verb_trigger(sentence, index, previous) or (index == 0 and ctx.procedure)
        if not verb_context:
            return
        names = ", ".join(POS_NAMES.get(p, p) for p in sorted(approved)) or "technical noun"
        alternatives = ", ".join(
            alt
            for entry in result.unapproved
            if entry.pos == "v"
            for alt, _pos in entry.alternatives[: self.settings.max_alternatives]
        )
        hint = f" Use: {alternatives}." if alternatives else ""
        self._add(out, filename, block, token, "1.2", "warning",
                  f"'{token.text}' is approved as a {names} only. The verb is not approved.{hint}")

    @staticmethod
    def _verb_trigger(sentence: list[Token], index: int, previous: Token | None) -> bool:
        """True when the word before a word shows that the word is a verb."""
        if previous is None or previous.kind != WORD:
            return False
        before = previous.lower
        if before in SUBJECT_PRONOUNS:
            return True
        if before not in MODALS:
            return False
        if before == "not":
            earlier = sentence[index - 2] if index >= 2 else None
            return not (earlier is not None and earlier.kind == WORD and earlier.lower in BE_FORMS)
        return True

    # -- imperative ----------------------------------------------------

    def _imperative(self, sentence: list[Token]) -> bool:
        words = [t for t in sentence if t.kind != PUNCT]
        if not words or words[0].kind != WORD:
            return False
        if words[0].lower in CONDITION_STARTERS:
            # "If the light is red, stop." The instruction follows the first comma.
            for position, token in enumerate(sentence):
                if token.kind == PUNCT and token.text == ",":
                    return self._imperative(sentence[position + 1 :])
            return False
        first = words[0].lower
        second = words[1] if len(words) > 1 else None
        if first in AUX_STARTS and first not in ("do", "be"):
            return False
        if first == "do":
            return second is not None and second.lower == "not"
        approved, unapproved, gloss = self._tags(first)
        verb_capable = first in self.dictionary.verb_bases or any(
            t.kind == "verb" and t.term == first for t in self.glossary.lookup(first)
        )
        if not verb_capable:
            return False
        if second is not None and second.kind == WORD and second.lower in AUX_STARTS:
            return False
        noun_capable = bool(
            {"n", "adj"} & (approved | unapproved) or {"noun", "name"} & gloss
        )
        if not noun_capable:
            return True
        if second is None:
            return False
        if second.kind in (ATOM_KIND, NUM, GROUP_KIND):
            return True
        return second.kind == WORD and (
            second.lower in DETERMINERS or second.lower in SUBJECT_PRONOUNS
            or second.lower in ("it", "them", "not", "sure")
        )

    # -- passive voice -------------------------------------------------

    def _check_passive(self, out: list[Finding], filename: str, block: Block,
                       sentence: list[Token], ctx: Context) -> None:
        items = [t for t in sentence if t.kind != PUNCT]
        for index, token in enumerate(items):
            if token.kind != WORD or token.lower not in BE_FORMS:
                continue
            scan = index + 1
            skipped = 0
            while (scan < len(items) and items[scan].kind == WORD
                   and items[scan].lower in PASSIVE_ADVERBS and skipped < 2):
                scan += 1
                skipped += 1
            if scan >= len(items) or items[scan].kind != WORD:
                continue
            participle = items[scan].lower
            if not self._is_participle(participle):
                continue
            severity = "error" if ctx.procedure else "warning"
            phrase = " ".join(t.text for t in items[index : scan + 1])
            self._add(out, filename, block, token, "3.6", severity,
                      f"Passive voice ('{phrase}'). Use the active voice.")

    def _third_person_verb(self, word: str) -> bool:
        """True when a word ends like a third-person verb form of a known verb."""
        if len(word) < 4 or not word.endswith("s") or word.endswith("ss"):
            return False
        bases = [word[:-1]]
        if word.endswith("es"):
            bases.append(word[:-2])
        if word.endswith("ies"):
            bases.append(word[:-3] + "y")
        return any(
            base in self.dictionary.verb_bases
            or any(t.kind == "verb" for t in self.glossary.lookup(base))
            for base in bases
        )

    def _is_participle(self, word: str) -> bool:
        if word in IRREGULAR_PARTICIPLES:
            return True
        if not word.endswith("ed") or len(word) < 4 or word in NOT_PARTICIPLES or word.endswith("eed"):
            return False
        approved, _unapproved, gloss = self._tags(word)
        if "adj" in approved:
            return False  # participle adjective, rule 3.3
        return True

    # -- noun clusters -------------------------------------------------

    def _role(self, tokens: list[Token], index: int, ctx: Context) -> str:
        """Classify a token for the noun cluster check: N noun, A adjective only, B breaker."""
        token = tokens[index]
        if token.kind == ATOM_KIND:
            return "N"
        if token.kind != WORD:
            return "B"
        word = token.lower
        if word.endswith("'s"):
            return "B"
        if word in DETERMINERS or word in MODALS or word in BE_FORMS or word in AUX_STARTS:
            return "B"
        if word in NUMBER_WORDS or word in self._unit_words:
            return "B"
        approved, unapproved, gloss = self._tags(word)
        result = self._lookup(word)
        # A reading that the dictionary does not approve counts only when no approved reading exists.
        readings = approved or ({p for p in unapproved} if not gloss else set())
        noun = bool("n" in readings or {"noun", "name"} & gloss)
        adjective = "adj" in readings
        verb = bool("v" in readings or "verb" in gloss)
        function = bool({"prep", "conj", "pron", "art", "adv"} & readings)
        if function and not noun:
            return "B"
        if noun and index and self._third_person_verb(word) and self._role(tokens, index - 1, ctx) == "N":
            return "B"  # a noun and then a word such as "checks" is a subject and a verb
        if not readings and not gloss:
            if result.form_of:
                return "B"
            return "N"  # an unknown word is probably a technical noun
        if verb and not noun:
            if adjective and not (index and tokens[index - 1].kind == WORD
                                  and tokens[index - 1].lower in MODALS):
                return "A"
            return "B"
        if verb and noun:
            before = tokens[index - 1] if index else None
            if (before is not None and before.kind == WORD
                    and (before.lower in MODALS or before.lower in SUBJECT_PRONOUNS)):
                return "B"
            if index == 0 and ctx.procedure:
                return "B"
            return "N"
        if noun:
            return "N"
        if adjective:
            return "A"
        return "B"

    def _check_clusters(self, out: list[Finding], filename: str, block: Block,
                        sentence: list[Token], ctx: Context) -> None:
        run: list[tuple[Token, str]] = []

        def finish() -> None:
            # A cluster ends with a noun. An adjective at the end is a predicate adjective.
            while run and run[-1][1] == "A":
                run.pop()
            if len(run) > self.settings.cluster_limit:
                text = " ".join(t.text if t.kind == WORD else "<code>" for t, _r in run)
                self._add(out, filename, block, run[0][0], "2.1", "warning",
                          f"Noun cluster of {len(run)} words ('{text}'). "
                          f"The limit is {self.settings.cluster_limit}.")

        for index, token in enumerate(sentence):
            role = self._role(sentence, index, ctx)
            if role in ("N", "A"):
                run.append((token, role))
            else:
                finish()
                run = []
        finish()


def _ignored(finding: Finding, ignores: dict[int, set[str]]) -> bool:
    rules = ignores.get(finding.line)
    if not rules:
        return False
    return "*" in rules or "all" in rules or finding.rule in rules
