import unittest

import support  # noqa: F401  (sets up sys.path)
from ste_text import ATOM_KIND, NUM, WORD, count_words, group_parens, split_sentences, tokenize

UNITS = frozenset({"ms", "GiB", "%"})


def kinds(text):
    return [(t.kind, t.text) for t in tokenize(text, UNITS) if t.kind != "PUNCT"]


def lengths(text):
    outer, groups = group_parens(tokenize(text, UNITS))
    return [count_words(s) for s in split_sentences(outer)], [
        count_words(s) for g in groups for s in split_sentences(g)
    ]


class TokenizeTests(unittest.TestCase):
    def test_hyphenated_word_is_one_word(self):
        self.assertEqual(kinds("A well-known box."), [(WORD, "A"), (WORD, "well-known"), (WORD, "box")])

    def test_number_with_unit_is_one_token(self):
        self.assertEqual(kinds("It takes 12 ms."), [(WORD, "It"), (WORD, "takes"), (NUM, "12 ms")])

    def test_percent_is_part_of_the_number(self):
        self.assertEqual(kinds("Use 50%."), [(WORD, "Use"), (NUM, "50%")])

    def test_identifiers_paths_and_flags_are_atoms(self):
        tokens = kinds("Open foo_bar or a/b/c.txt or --flag.")
        self.assertEqual([k for k, _t in tokens if k == ATOM_KIND], [ATOM_KIND] * 3)

    def test_decimal_number(self):
        self.assertEqual(kinds("Set 3.5 now."), [(WORD, "Set"), (NUM, "3.5"), (WORD, "now")])

    def test_token_offsets_point_into_the_text(self):
        text = "The cat sat."
        for token in tokenize(text, UNITS):
            if token.kind == WORD:
                self.assertEqual(text[token.start : token.start + len(token.text)], token.text)


class SentenceTests(unittest.TestCase):
    def test_sentences_split_at_period_question_and_bang(self):
        self.assertEqual(lengths("The cat sat. The dog ran! Is it red?")[0], [3, 3, 3])

    def test_abbreviation_does_not_end_a_sentence(self):
        self.assertEqual(lengths("Use a box, e.g. a red box.")[0], [7])

    def test_parenthesis_counts_as_one_word_and_is_its_own_sentence(self):
        outer, inner = lengths("One (two three) four. Five.")
        self.assertEqual(outer, [3, 1])
        self.assertEqual(inner, [2])

    def test_quoted_text_counts_as_one_word(self):
        self.assertEqual(lengths("Say \u201ca b c d\u201d.")[0], [2])

    def test_empty_text_has_no_sentence(self):
        self.assertEqual(lengths("")[0], [])


if __name__ == "__main__":
    unittest.main()
