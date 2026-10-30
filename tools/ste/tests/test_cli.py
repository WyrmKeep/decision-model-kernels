import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support
import ste_check

CONFIG = """\
[dictionary]
pdf_sha256 = "{sha}"
cache_dir = "{cache}"

[files]
root = "."
glossary = "glossary.md"
ignore = ["skip/*"]

[glossary]
allow_approved = ["time"]

[procedure]
headings = ["Steps"]

[units]
names = ["ms"]
"""


class CliTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        (self.root / "config.toml").write_text(
            CONFIG.format(sha="0" * 64, cache=str(self.root / "cache")), encoding="utf-8"
        )
        (self.root / "glossary.md").write_text(support.GLOSSARY_TEXT, encoding="utf-8")
        (self.root / "good.md").write_text("The cat walks to the river.\n", encoding="utf-8")
        (self.root / "bad.md").write_text("The cat jumps to the river.\n", encoding="utf-8")
        (self.root / "skip").mkdir()
        (self.root / "skip" / "bad.md").write_text("The cat jumps.\n", encoding="utf-8")

    def tearDown(self):
        self.folder.cleanup()

    def run_cli(self, *args, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        argv = ["--config", str(self.root / "config.toml"), *map(str, args)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with mock.patch("sys.stdin", io.StringIO(stdin or "")):
                code = ste_check.main(argv, dictionary=support.dictionary())
        return code, out.getvalue(), err.getvalue()

    def test_clean_file_exits_zero(self):
        code, out, err = self.run_cli(self.root / "good.md")
        self.assertEqual((code, out), (0, ""))
        self.assertIn("1 files checked: 0 errors, 0 warnings", err)

    def test_error_exits_one_and_prints_file_line_col_rule(self):
        code, out, _err = self.run_cli(self.root / "bad.md")
        self.assertEqual(code, 1)
        self.assertRegex(out.splitlines()[0], r"^bad\.md:1:9 1\.1 error: 'jumps' is not approved")

    def test_json_format(self):
        code, out, _err = self.run_cli(self.root / "bad.md", "--format", "json")
        data = json.loads(out)
        self.assertEqual(code, 1)
        self.assertEqual(data["summary"], {"files": 1, "errors": 1, "warnings": 0})
        self.assertEqual(
            {k for k in data["findings"][0]}, {"file", "line", "col", "rule", "severity", "message"}
        )

    def test_warning_alone_exits_zero_and_strict_exits_one(self):
        warn = self.root / "warn.md"
        warn.write_text("The box is carried by the cat.\n", encoding="utf-8")
        self.assertEqual(self.run_cli(warn)[0], 0)
        self.assertEqual(self.run_cli(warn, "--strict")[0], 1)

    def test_stdin(self):
        code, out, _err = self.run_cli("--stdin", "--stdin-name", "pr.md", stdin="The cat jumps.\n")
        self.assertEqual(code, 1)
        self.assertTrue(out.startswith("pr.md:1:9 1.1"))

    def test_folder_walk_skips_ignored_paths(self):
        code, out, err = self.run_cli(self.root)
        self.assertEqual(code, 1)
        self.assertNotIn("skip/", out)
        self.assertIn("3 files checked", err)

    def test_no_ignore_checks_ignored_files(self):
        _code, out, _err = self.run_cli(self.root, "--no-ignore")
        self.assertIn("skip/bad.md:1:9", out)

    def test_default_scans_the_repository_root(self):
        code, out, _err = self.run_cli()
        self.assertEqual(code, 1)
        self.assertIn("bad.md:1:9", out)

    def test_missing_path_exits_two(self):
        code, _out, err = self.run_cli(self.root / "missing.md")
        self.assertEqual(code, 2)
        self.assertIn("No such file", err)

    def test_invalid_glossary_exits_two(self):
        (self.root / "glossary.md").write_text(
            "| Term | Type | Meaning |\n|---|---|---|\n| a | adjective | Text. |\n", encoding="utf-8"
        )
        code, _out, err = self.run_cli(self.root / "good.md")
        self.assertEqual(code, 2)
        self.assertIn("glossary.md:3: glossary error", err)

    def test_missing_config_exits_two(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ste_check.main(["--config", str(self.root / "none.toml")], dictionary=support.dictionary())
        self.assertEqual(code, 2)

    def test_lookup(self):
        code, out, _err = self.run_cli("--lookup", "jump", "widget")
        self.assertEqual(code, 0)
        self.assertIn("jump: dictionary: not approved as verb, use: walk", out)
        self.assertIn("widget: glossary", out)

    def test_dictionary_info_shows_counts_only(self):
        code, out, _err = self.run_cli("--dictionary-info")
        self.assertEqual(code, 0)
        self.assertIn("approved entries:", out)

    def test_glossary_file_lint_warns_for_an_approved_word(self):
        (self.root / "glossary.md").write_text(
            "| Term | Type | Meaning |\n|---|---|---|\n"
            "| cat | technical noun | A red lamp. |\n"
            "| time | technical noun | A red lamp. |\n"
            "| colour | technical noun | A red lamp. |\n",
            encoding="utf-8",
        )
        code, out, _err = self.run_cli(self.root / "glossary.md")
        self.assertEqual(code, 1)
        self.assertIn("glossary.md:3:1 1.3 warning", out)
        self.assertNotIn("glossary.md:4:1 1.3", out)  # allowed in the configuration
        self.assertIn("glossary.md:5:1 1.14 error", out)

    def test_cache_folder_inside_the_repository_is_refused(self):
        config = ste_check.load_config(self.root / "config.toml")
        environment = {k: v for k, v in os.environ.items() if k != "STE_CACHE"}
        with mock.patch.dict(os.environ, environment, clear=True):
            with mock.patch.object(ste_check.ste_dict, "ensure_pdf", return_value=self.root / "x.pdf"):
                with self.assertRaises(ste_check.ste_dict.DictionaryError):
                    ste_check.load_dictionary(config, False)


class RepositoryConfigTests(unittest.TestCase):
    def test_shipped_configuration_loads(self):
        config = ste_check.load_config(ste_check.DEFAULT_CONFIG)
        self.assertEqual(len(config.pdf_sha256), 64)
        self.assertEqual(config.procedure_limit, 20)
        self.assertEqual(config.description_limit, 25)
        self.assertEqual(config.paragraph_limit, 6)
        self.assertIn("Install", config.procedure_headings)
        self.assertEqual(config.glossary.name, "glossary.md")


if __name__ == "__main__":
    unittest.main()
