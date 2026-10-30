import unittest

import support
from ste_glossary import GlossaryError, parse_glossary


class GlossaryTests(unittest.TestCase):
    def setUp(self):
        self.glossary = support.glossary()

    def test_terms_have_type_and_meaning(self):
        kinds = {t.term: t.kind for t in self.glossary.terms}
        self.assertEqual(kinds["widget"], "noun")
        self.assertEqual(kinds["frobnicate"], "verb")
        self.assertEqual(kinds["zork"], "name")

    def test_noun_plural_is_known(self):
        self.assertEqual([t.term for t in self.glossary.lookup("widgets")], ["widget"])

    def test_verb_forms_are_known(self):
        for form in ("frobnicates", "frobnicated"):
            self.assertEqual([t.term for t in self.glossary.lookup(form)], ["frobnicate"])
        self.assertEqual([t.term for t in self.glossary.lookup("built")], ["build"])

    def test_name_has_no_plural(self):
        self.assertEqual(self.glossary.lookup("zorks"), [])

    def test_multi_word_terms_are_phrases(self):
        self.assertIn("widget stand", self.glossary.phrases[2])

    def test_lookup_ignores_case(self):
        self.assertEqual(len(self.glossary.lookup("ZORK")), 1)

    def test_other_tables_are_ignored(self):
        text = "| A | B |\n|---|---|\n| x | y |\n\n" + support.GLOSSARY_TEXT
        self.assertEqual(len(parse_glossary(text).terms), 5)

    def test_same_word_can_be_noun_and_verb(self):
        text = (
            "| Term | Type | Meaning |\n|---|---|---|\n"
            "| run | technical noun | One run. |\n| run | technical verb | To run. |\n"
        )
        self.assertEqual({t.kind for t in parse_glossary(text).lookup("run")}, {"noun", "verb"})


class GlossaryErrorTests(unittest.TestCase):
    HEAD = "| Term | Type | Meaning |\n|---|---|---|\n"

    def parse(self, rows):
        return parse_glossary(self.HEAD + rows)

    def test_unknown_type(self):
        with self.assertRaises(GlossaryError) as caught:
            self.parse("| a | adjective | Meaning. |\n")
        self.assertEqual(caught.exception.line, 3)

    def test_empty_meaning(self):
        with self.assertRaises(GlossaryError):
            self.parse("| a | technical noun |  |\n")

    def test_empty_term(self):
        with self.assertRaises(GlossaryError):
            self.parse("|  | technical noun | Meaning. |\n")

    def test_duplicate_term_and_type(self):
        with self.assertRaises(GlossaryError):
            self.parse("| a | technical noun | One. |\n| A | technical noun | Two. |\n")

    def test_short_row(self):
        with self.assertRaises(GlossaryError):
            self.parse("| a | technical noun |\n")


if __name__ == "__main__":
    unittest.main()
