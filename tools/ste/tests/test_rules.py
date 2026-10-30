import unittest

import support
from ste_glossary import parse_glossary
from ste_rules import Checker, Settings, british_spelling


def findings(text, glossary_file=False, settings=None):
    return support.checker(settings).check(text, "t.md", glossary_file)


def codes(text, **kwargs):
    return [f.rule for f in findings(text, **kwargs)]


def severity(text, rule):
    return {f.rule: f.severity for f in findings(text)}[rule]


class VocabularyTests(unittest.TestCase):
    def test_approved_sentence_has_no_finding(self):
        self.assertEqual(codes("The cat walks to the river."), [])

    def test_unapproved_word_names_the_alternative(self):
        (found,) = findings("The cat jumps to the river.")
        self.assertEqual((found.rule, found.severity, found.line, found.col), ("1.1", "error", 1, 9))
        self.assertIn("walk", found.message)

    def test_unknown_word(self):
        (found,) = findings("The cat walks to the zebra.")
        self.assertEqual(found.rule, "1.1")
        self.assertIn("not in the dictionary", found.message)

    def test_unlisted_verb_form(self):
        (found,) = findings("The cat walkes.")
        self.assertEqual(found.rule, "1.4")

    def test_unapproved_phrase_is_reported_once(self):
        found = findings("The cat walks in front of the door.")
        self.assertEqual([f.rule for f in found], ["1.1"])
        self.assertIn("in front of", found[0].message)

    def test_glossary_words_and_forms_pass(self):
        self.assertEqual(codes("The cat frobnicated the widgets with the widget stand."), [])
        self.assertEqual(codes("Zork is in the box."), [])

    def test_glossary_phrase_with_a_camel_case_word_passes(self):
        text = support.GLOSSARY_TEXT + "| Blue DeltaBox | technical name | A small box. |\n"
        checker = Checker(support.dictionary(), parse_glossary(text), support.SETTINGS)
        self.assertEqual([f.rule for f in checker.check("The cat sees a Blue DeltaBox.", "t.md")], [])
        self.assertEqual([f.rule for f in checker.check("The cat sees a Blue.", "t.md")], ["1.1"])

    def test_technical_name_in_dictionary_alternative_points_to_glossary(self):
        (found,) = findings("The graph is red.")
        self.assertIn("glossary", found.message)

    def test_number_words_and_units_pass(self):
        self.assertEqual(codes("The cat walks two times in 12 ms with 4 GiB and 50%."), [])

    def test_contraction(self):
        self.assertEqual(codes("The cat can't walk."), ["4.2"])

    def test_noun_use_of_verb_only_word(self):
        (found,) = findings("The lift is red.")
        self.assertEqual((found.rule, found.severity), ("1.2", "warning"))

    def test_verb_use_of_noun_only_word(self):
        (found,) = findings("You must rope the box.")
        self.assertEqual((found.rule, found.severity), ("1.2", "warning"))
        self.assertIn("lift", found.message)

    def test_not_after_be_keeps_the_adjective_reading(self):
        self.assertEqual(codes("The box is not red."), [])
        self.assertEqual(codes("You do not red the box."), ["1.2"])

    def test_semicolon(self):
        (found,) = findings("The cat walks; the dog walks.")
        self.assertEqual((found.rule, found.severity, found.col), ("8.1", "error", 14))


class BritishSpellingTests(unittest.TestCase):
    def test_pairs(self):
        pairs = {
            "colour": "color", "colours": "colors", "organise": "organize",
            "organisation": "organization", "analyse": "analyze", "centre": "center",
            "licence": "license", "behaviour": "behavior", "modelling": "modeling",
        }
        for british, american in pairs.items():
            self.assertEqual(british_spelling(british), american, british)

    def test_american_and_exception_words_pass(self):
        for word in ("color", "organize", "analyze", "center", "license", "promise", "raise",
                     "exercise", "hour", "four", "otherwise", "advise", "tour"):
            self.assertIsNone(british_spelling(word), word)

    def test_rule_is_reported_for_text(self):
        (found,) = findings("Paint the colour box.")
        self.assertEqual(found.rule, "1.14")
        self.assertIn("color", found.message)


class SentenceLengthTests(unittest.TestCase):
    LONG = "the cat walks to the river and then the dog walks to the tree and then the bird walks"

    @staticmethod
    def sentence(count, body=("The", "cat", "walks")):
        words = list(body) + (["to", "the", "river", "and", "then"] * 10)[: count - len(body)]
        return " ".join(words) + "."

    def test_description_limit_is_25_words(self):
        self.assertEqual(codes(self.sentence(25)), [])
        found = findings(self.sentence(26))
        self.assertEqual([f.rule for f in found], ["6.3"])
        self.assertIn("26 words", found[0].message)

    def test_procedure_limit_is_20_words(self):
        self.assertEqual(codes(self.sentence(20, ("Carry", "the", "box"))), [])
        found = findings(self.sentence(21, ("Carry", "the", "box")))
        self.assertEqual([f.rule for f in found], ["5.1"])
        self.assertIn("21 words", found[0].message)

    def test_parenthesis_counts_as_one_word(self):
        text = "The cat walks to the river and then the dog walks to the tree (and then the bird walks to the house and the cat also walks) now."
        self.assertNotIn("6.3", codes(text))

    def test_paragraph_limit_is_6_sentences(self):
        six = "The cat walks. " * 6
        self.assertEqual(codes(six), [])
        self.assertEqual(codes(six + "The cat walks."), ["6.6"])

    def test_table_cell_has_no_sentence_length_check(self):
        text = f"| the | text |\n|---|---|\n| {self.LONG} to the house and the cat also walks and the dog | box |\n"
        self.assertNotIn("6.3", codes(text))

    def test_heading_has_no_sentence_length_check(self):
        text = "# " + self.LONG + " to the house and the cat also walks and the dog\n"
        self.assertEqual(codes(text), [])


class ProcedureTests(unittest.TestCase):
    def test_passive_is_a_warning_in_a_description(self):
        self.assertEqual(severity("The box is carried by the cat.", "3.6"), "warning")

    def test_passive_is_an_error_in_a_procedure_item(self):
        self.assertEqual(severity("# Steps\n\n1. The box is carried by the cat.\n", "3.6"), "error")

    def test_passive_is_an_error_in_an_imperative_sentence(self):
        found = findings("Walk to the river. The box is carried by the cat.")
        self.assertEqual([f.rule for f in found], ["3.6"])
        self.assertEqual(found[0].severity, "warning")

    def test_procedure_item_must_start_with_an_imperative_verb(self):
        found = findings("# Steps\n\n1. Carry the box.\n2. The cat walks.\n")
        self.assertEqual([(f.rule, f.severity, f.line) for f in found], [("5.3", "warning", 4)])

    def test_condition_before_the_imperative_verb_is_a_procedure_step(self):
        self.assertEqual(codes("# Steps\n\n1. If the box is red, carry the box.\n"), [])
        self.assertEqual(codes("# Steps\n\n1. If the box is red, the cat walks.\n"), ["5.3"])

    def test_list_below_other_heading_is_a_description(self):
        self.assertEqual(codes("# Notes\n\n- The cat walks.\n"), [])

    def test_item_limit_applies_only_to_procedures(self):
        long_item = "Carry the box to the river and then lift the bucket and then paint the wall and then give the dog the rope."
        self.assertEqual(codes("# Steps\n\n1. " + long_item + "\n"), ["5.1"])
        self.assertEqual(codes("# Notes\n\n- " + long_item + "\n"), ["5.1"])

    def test_nearest_heading_decides_for_bullets(self):
        text = "# Install\n\n## Details\n\n- The cat walks.\n"
        self.assertEqual(codes(text), [])
        text = "# Install\n\n- The cat walks.\n"
        self.assertEqual(codes(text), ["5.3"])


class NounClusterTests(unittest.TestCase):
    def test_cluster_of_more_than_three_words_is_a_warning(self):
        (found,) = findings("The tall red quiet heavy bucket door lamp box.")
        self.assertEqual((found.rule, found.severity), ("2.1", "warning"))

    def test_cluster_of_three_words_passes(self):
        self.assertEqual(codes("The bucket door lamp is red."), [])

    def test_verb_breaks_a_cluster(self):
        self.assertEqual(codes("The bucket door lamp walks to the stone river box."), [])

    def test_third_person_verb_after_a_noun_is_not_part_of_the_cluster(self):
        self.assertEqual(codes("The tall bucket door ropes the box."), [])

    def test_limit_comes_from_the_settings(self):
        settings = Settings(procedure_headings=[], units=frozenset(), cluster_limit=2)
        self.assertEqual(codes("The bucket door lamp is red.", settings=settings), ["2.1"])


class SkippedTextTests(unittest.TestCase):
    def test_code_urls_links_and_fences_are_not_checked(self):
        text = (
            "The `jumps` and https://example.invalid/jumps and [cat](jumps.md) here.\n\n"
            "```\nThe cat jumps;\n```\n"
        )
        self.assertEqual(codes(text), [])

    def test_table_cell_with_only_code_or_numbers_is_skipped(self):
        text = "| the | cat |\n|---|---|\n| 12 | `x` |\n| 3 ms | `jumps` |\n"
        self.assertEqual(codes(text), [])

    def test_table_text_cell_gets_the_word_check(self):
        text = "| the | cat |\n|---|---|\n| the cat | jumps |\n"
        self.assertEqual(codes(text), ["1.1"])


class SuppressionTests(unittest.TestCase):
    def test_ignore_next_line_for_one_rule(self):
        text = "<!-- ste-ignore-next-line 1.1 -->\nThe cat jumps.\nThe cat jumps.\n"
        self.assertEqual([(f.rule, f.line) for f in findings(text)], [("1.1", 3)])

    def test_ignore_applies_to_the_next_line_only(self):
        text = "The cat walks.\n\n<!-- ste-ignore-next-line 1.1 -->\n\nThe cat jumps.\n"
        self.assertEqual(codes(text), ["1.1"])

    def test_ignore_other_rule_does_not_hide_a_finding(self):
        text = "<!-- ste-ignore-next-line 8.1 -->\n\nThe cat jumps.\n"
        self.assertEqual(codes(text), ["1.1"])

    def test_ignore_star_hides_all_rules(self):
        text = "<!-- ste-ignore-next-line * -->\nThe cat jumps; the zebra walks.\n"
        self.assertEqual(codes(text), [])


class GlossaryFileTests(unittest.TestCase):
    TABLE = (
        "| Term | Type | Meaning |\n|---|---|---|\n"
        "| zebra | technical noun | The cat jumps. |\n"
    )

    def test_term_and_type_cells_are_not_word_checked(self):
        self.assertEqual(codes(self.TABLE.replace("The cat jumps.", "The cat walks."), glossary_file=True), [])

    def test_meaning_cell_is_word_checked(self):
        self.assertEqual(codes(self.TABLE, glossary_file=True), ["1.1"])

    def test_same_table_in_another_file_is_word_checked_everywhere(self):
        self.assertIn("1.1", codes(self.TABLE.replace("The cat jumps.", "The cat walks.")))


class DescribeTests(unittest.TestCase):
    def test_describe_names_status_and_alternative(self):
        text = support.checker().describe("jump")
        self.assertIn("not approved", text)
        self.assertIn("walk", text)
        self.assertIn("approved (verb)", support.checker().describe("walk"))
        self.assertIn("glossary", support.checker().describe("widget"))
        self.assertTrue(support.checker().describe("zebra").endswith("unknown"))


if __name__ == "__main__":
    unittest.main()
