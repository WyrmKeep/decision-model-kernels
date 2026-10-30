import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import support
import ste_dict
from ste_dict import Entry, plural_forms, regular_forms


class LookupTests(unittest.TestCase):
    def setUp(self):
        self.dictionary = support.dictionary()

    def test_approved_word_and_listed_forms(self):
        for word in ("walk", "walks", "walked", "gave", "given"):
            self.assertEqual(self.dictionary.lookup(word).status, "approved", word)

    def test_noun_plurals_are_approved(self):
        self.assertEqual(self.dictionary.lookup("boxes").approved_pos, {"n"})
        self.assertEqual(self.dictionary.lookup("cats").approved_pos, {"n"})

    def test_adjective_forms_come_from_the_list(self):
        self.assertEqual(self.dictionary.lookup("taller").approved_pos, {"adj"})
        self.assertEqual(self.dictionary.lookup("tallish").status, "unknown")

    def test_verb_without_listed_forms_gets_regular_forms(self):
        self.assertEqual(self.dictionary.lookup("painted").approved_pos, {"v"})

    def test_unapproved_word_names_its_alternative(self):
        result = self.dictionary.lookup("jumps")
        self.assertEqual(result.status, "unapproved")
        self.assertEqual(result.alternatives, ["walk"])

    def test_unlisted_form_of_approved_verb(self):
        result = self.dictionary.lookup("walking")
        self.assertEqual(result.status, "form")
        self.assertEqual(result.form_of, [("walk", "ing")])

    def test_ed_is_not_guessed_for_irregular_verbs(self):
        self.assertEqual(self.dictionary.lookup("seed").status, "unknown")

    def test_agent_noun_is_not_a_form_of_a_verb(self):
        self.assertEqual(self.dictionary.lookup("walker").status, "unknown")

    def test_phrase_alternatives(self):
        self.assertEqual(self.dictionary.phrase_alternatives("in front of"), ["at"])
        self.assertIsNone(self.dictionary.phrase_alternatives("walk home"))

    def test_counts(self):
        counts = self.dictionary.counts()
        self.assertEqual(counts["approved"] + counts["unapproved"], len(support.ENTRIES))


class FormTests(unittest.TestCase):
    def test_regular_forms(self):
        self.assertIn("walks", regular_forms("walk"))
        self.assertIn("walked", regular_forms("walk"))
        self.assertIn("carries", regular_forms("carry"))
        self.assertIn("carried", regular_forms("carry"))
        self.assertIn("boxes", plural_forms("box"))
        self.assertIn("cities", plural_forms("city"))

    def test_entry_json_round_trip(self):
        entry = Entry("walk", "v", False, ["walks"], [("go", "v")])
        self.assertEqual(Entry.from_json(entry.to_json()), entry)


class PdfAccessTests(unittest.TestCase):
    def test_ensure_pdf_reports_a_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ste_dict.DictionaryError):
                ste_dict.ensure_pdf(Path(folder) / "missing.pdf", "0" * 64, False)

    def test_ensure_pdf_rejects_a_wrong_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = Path(folder) / "file.pdf"
            pdf.write_bytes(b"not a pdf")
            with self.assertRaises(ste_dict.DictionaryError):
                ste_dict.ensure_pdf(pdf, "0" * 64, False)

    def test_ensure_pdf_accepts_a_matching_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            pdf = Path(folder) / "file.pdf"
            pdf.write_bytes(b"not a pdf")
            digest = ste_dict.sha256_file(pdf)
            self.assertEqual(ste_dict.ensure_pdf(pdf, digest, False), pdf)

    def test_cache_folder_follows_the_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {"STE_CACHE": folder}):
                self.assertEqual(ste_dict.cache_dir("~/elsewhere"), Path(folder))


@unittest.skipUnless(ste_dict.default_pdf_path().exists(), "The ASD-STE100 PDF is not on this device.")
class RealPdfTests(unittest.TestCase):
    """Check counts and invariants only. The test does not assert dictionary content."""

    @classmethod
    def setUpClass(cls):
        cls.entries = ste_dict.extract_from_pdf(ste_dict.default_pdf_path())

    def test_entry_counts_are_close_to_the_published_counts(self):
        approved = sum(1 for e in self.entries if e.approved)
        unapproved = len(self.entries) - approved
        self.assertTrue(800 <= approved <= 950, approved)
        self.assertTrue(1200 <= unapproved <= 1400, unapproved)

    def test_no_entry_repeats(self):
        keys = [(e.word, e.pos, e.approved) for e in self.entries]
        self.assertEqual(len(keys), len(set(keys)))

    def test_word_fields_are_clean(self):
        for entry in self.entries:
            self.assertEqual(entry.word, entry.word.strip().lower())
            self.assertNotIn("  ", entry.word)
            self.assertNotIn("(", entry.word)

    def test_unapproved_entries_with_alternatives_name_words(self):
        with_alternatives = [e for e in self.entries if not e.approved and e.alternatives]
        self.assertGreater(len(with_alternatives), 900)
        for entry in with_alternatives:
            for word, _pos in entry.alternatives:
                self.assertEqual(word, word.strip())


if __name__ == "__main__":
    unittest.main()
