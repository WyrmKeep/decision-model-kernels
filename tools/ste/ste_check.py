#!/usr/bin/env python3
"""Check Markdown text against ASD-STE100 Issue 9 writing rules.

Usage: python3 tools/ste/ste_check.py [paths...] [--stdin] [--format text|json] [--strict]

The checker reads the STE dictionary at run time from a local copy of the
official PDF. The repository does not contain the PDF or a word list.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ste_dict  # noqa: E402
from ste_glossary import Glossary, GlossaryError, load_glossary  # noqa: E402
from ste_rules import POS_NAMES, Checker, Finding, Settings, british_spelling  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.toml"
MARKDOWN_SUFFIXES = (".md", ".markdown")
SKIP_DIRECTORIES = {".git", "node_modules", "__pycache__", ".venv"}


class ConfigError(Exception):
    """The configuration is not valid."""


@dataclass
class Config:
    path: Path
    root: Path
    pdf_sha256: str
    pdf_url: str
    cache_dir: str
    glossary: Path
    ignore: list[str] = field(default_factory=list)
    allow_approved: frozenset[str] = frozenset()
    procedure_headings: list[str] = field(default_factory=list)
    units: frozenset[str] = frozenset()
    procedure_limit: int = 20
    description_limit: int = 25
    paragraph_limit: int = 6
    cluster_limit: int = 3


def load_config(path: Path) -> Config:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"Cannot read the configuration {path}: {error}") from error
    try:
        dictionary = data["dictionary"]
        files = data["files"]
        root = (path.parent / files.get("root", "../..")).resolve()
        limits = data.get("limits", {})
        return Config(
            path=path,
            root=root,
            pdf_sha256=dictionary["pdf_sha256"],
            pdf_url=dictionary.get("pdf_url", ste_dict.PDF_URL),
            cache_dir=dictionary.get("cache_dir", "~/.cache/ste100"),
            glossary=(root / files["glossary"]).resolve(),
            ignore=list(files.get("ignore", [])),
            allow_approved=frozenset(data.get("glossary", {}).get("allow_approved", [])),
            procedure_headings=list(data.get("procedure", {}).get("headings", [])),
            units=frozenset(data.get("units", {}).get("names", [])),
            procedure_limit=int(limits.get("procedure_sentence_words", 20)),
            description_limit=int(limits.get("description_sentence_words", 25)),
            paragraph_limit=int(limits.get("paragraph_sentences", 6)),
            cluster_limit=int(limits.get("noun_cluster_words", 3)),
        )
    except KeyError as error:
        raise ConfigError(f"The configuration {path} has no key {error}.") from error


def settings_from(config: Config) -> Settings:
    return Settings(
        procedure_headings=config.procedure_headings,
        units=config.units,
        procedure_limit=config.procedure_limit,
        description_limit=config.description_limit,
        paragraph_limit=config.paragraph_limit,
        cluster_limit=config.cluster_limit,
    )


def load_dictionary(config: Config, allow_download: bool) -> ste_dict.Dictionary:
    pdf = ste_dict.default_pdf_path()
    allow = allow_download or os.environ.get("STE_DOWNLOAD_PDF") == "1"
    ste_dict.ensure_pdf(pdf, config.pdf_sha256, allow)
    cache = ste_dict.cache_dir(config.cache_dir).resolve()
    if cache == config.root or config.root in cache.parents:
        raise ste_dict.DictionaryError(
            f"The cache folder {cache} is inside the repository. Set STE_CACHE to another folder."
        )
    return ste_dict.load_dictionary(pdf, config.pdf_sha256, cache)


def is_ignored(path: Path, config: Config) -> bool:
    try:
        relative = path.resolve().relative_to(config.root).as_posix()
    except ValueError:
        return False
    return any(fnmatch.fnmatch(relative, pattern) for pattern in config.ignore)


def collect_files(paths: list[str], config: Config, use_ignore: bool = True) -> list[Path]:
    found: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            for current, directories, names in os.walk(path):
                directories[:] = sorted(d for d in directories if d not in SKIP_DIRECTORIES)
                for name in sorted(names):
                    if name.lower().endswith(MARKDOWN_SUFFIXES):
                        found.append(Path(current) / name)
        elif path.exists():
            found.append(path)
        else:
            raise FileNotFoundError(f"No such file or folder: {item}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen or (use_ignore and is_ignored(path, config)):
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def glossary_findings(glossary: Glossary, dictionary: ste_dict.Dictionary, filename: str,
                      allow_approved: frozenset[str] = frozenset()) -> list[Finding]:
    """Check the glossary itself: terms that are also approved words, and British spelling."""
    out: list[Finding] = []
    for term in glossary.terms:
        words = term.term.split()
        if len(words) > 3:
            out.append(Finding(filename, term.line, 1, "2.1", "warning",
                               f"The term '{term.display}' has {len(words)} words. The limit is 3."))
        for word in words:
            american = british_spelling(word)
            if american:
                out.append(Finding(filename, term.line, 1, "1.14", "error",
                                   f"British spelling '{word}' in a term. Use '{american}'."))
        if len(words) == 1 and term.kind in ("noun", "verb") and term.term not in allow_approved:
            wanted = "n" if term.kind == "noun" else "v"
            approved = dictionary.approved_base.get(term.term, set())
            if wanted in approved:
                names = POS_NAMES[wanted]
                out.append(Finding(
                    filename, term.line, 1, "1.3", "warning",
                    f"The term '{term.display}' is also an approved {names}. "
                    "Rule 1.3 allows only the approved meaning."))
    return out


def format_text(findings: list[Finding]) -> str:
    return "\n".join(
        f"{f.file}:{f.line}:{f.col} {f.rule} {f.severity}: {f.message}" for f in findings
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ste_check.py",
        description="Check Markdown against ASD-STE100 Issue 9. Exit 1 when it finds errors.",
    )
    parser.add_argument("paths", nargs="*", help="Markdown files or folders. Default: the repository.")
    parser.add_argument("--stdin", action="store_true", help="Read the text from standard input.")
    parser.add_argument("--stdin-name", default="<stdin>", help="Name to show for standard input.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Configuration file.")
    parser.add_argument("--download-pdf", action="store_true",
                        help="Download the PDF from the official URL if it is missing.")
    parser.add_argument("--no-ignore", action="store_true",
                        help="Check files even when the configuration ignores them.")
    parser.add_argument("--lookup", nargs="+", metavar="WORD",
                        help="Show how the checker classifies each word, then exit.")
    parser.add_argument("--dictionary-info", action="store_true",
                        help="Show the entry counts of the dictionary, then exit.")
    return parser


def main(argv: list[str] | None = None, dictionary: ste_dict.Dictionary | None = None) -> int:
    """Run the checker. A test can pass a dictionary instead of the PDF dictionary."""
    args = build_parser().parse_args(argv)
    try:
        config = load_config(Path(args.config))
        if dictionary is None:
            dictionary = load_dictionary(config, args.download_pdf)
    except (ConfigError, ste_dict.DictionaryError) as error:
        print(f"ste_check: {error}", file=sys.stderr)
        return 2
    if args.dictionary_info:
        counts = dictionary.counts()
        print(f"pdf_sha256: {config.pdf_sha256}")
        print(f"approved entries: {counts['approved']}")
        print(f"not approved entries: {counts['unapproved']}")
        return 0
    try:
        glossary = load_glossary(config.glossary) if config.glossary.exists() else None
    except GlossaryError as error:
        rel = _display(config.glossary, config)
        print(f"{rel}:{error.line}: glossary error: {error.message}", file=sys.stderr)
        return 2
    if glossary is None:
        print(f"ste_check: the glossary {config.glossary} does not exist.", file=sys.stderr)
        return 2
    checker = Checker(dictionary, glossary, settings_from(config))
    if args.lookup:
        for word in args.lookup:
            print(checker.describe(word))
        return 0
    findings: list[Finding] = []
    file_count = 0
    if args.stdin:
        text = sys.stdin.read()
        findings.extend(checker.check(text, args.stdin_name))
        file_count += 1
    if args.paths or not args.stdin:
        try:
            files = collect_files(args.paths or [str(config.root)], config, not args.no_ignore)
        except FileNotFoundError as error:
            print(f"ste_check: {error}", file=sys.stderr)
            return 2
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                print(f"ste_check: cannot read {path}: {error}", file=sys.stderr)
                return 2
            name = _display(path, config)
            is_glossary = path.resolve() == config.glossary
            findings.extend(checker.check(text, name, glossary_file=is_glossary))
            if is_glossary:
                findings.extend(glossary_findings(glossary, dictionary, name, config.allow_approved))
            file_count += 1
    findings.sort(key=lambda f: (f.file, f.line, f.col, f.rule))
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    if args.format == "json":
        payload = {
            "findings": [f.to_dict() for f in findings],
            "summary": {"files": file_count, "errors": errors, "warnings": warnings},
        }
        print(json.dumps(payload, indent=2))
    else:
        if findings:
            print(format_text(findings))
        print(f"{file_count} files checked: {errors} errors, {warnings} warnings.",
              file=sys.stderr)
    failed = errors > 0 or (args.strict and warnings > 0)
    return 1 if failed else 0


def _display(path: Path, config: Config) -> str:
    try:
        return path.resolve().relative_to(config.root).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
