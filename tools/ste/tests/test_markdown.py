import unittest

import support  # noqa: F401
from ste_markdown import parse_document

DOC = """\
---
title: sample
---
# Install

1. Do a thing.
- Bullet here.

## Notes

1. Numbered under child.
- Bullet under child.

```
code block text
```

| a | b |
|---|---|
| 1 | text cell |

Para with `code` and [link text](http://example.invalid/z) end.
"""


def blocks(text, headings=("Install",)):
    return parse_document(text, list(headings)).blocks


class ParseTests(unittest.TestCase):
    def test_front_matter_and_fences_are_skipped(self):
        texts = [b.text for b in blocks(DOC)]
        self.assertFalse(any("sample" in t or "code block" in t for t in texts))

    def test_kinds_and_line_numbers(self):
        found = {(b.kind, b.line) for b in blocks(DOC)}
        self.assertIn(("heading", 4), found)
        self.assertIn(("list_item", 7), found)
        self.assertIn(("cell", 20), found)
        self.assertIn(("paragraph", 22), found)

    def test_list_item_below_matching_heading_is_a_procedure(self):
        items = {b.text: b.in_procedure for b in blocks(DOC) if b.kind == "list_item"}
        self.assertTrue(items["Do a thing."])
        self.assertTrue(items["Bullet here."])

    def test_numbered_item_below_child_heading_inherits_procedure(self):
        items = {b.text: b.in_procedure for b in blocks(DOC) if b.kind == "list_item"}
        self.assertTrue(items["Numbered under child."])

    def test_bullet_below_child_heading_is_a_description(self):
        items = {b.text: b.in_procedure for b in blocks(DOC) if b.kind == "list_item"}
        self.assertFalse(items["Bullet under child."])

    def test_heading_match_ignores_case_and_uses_words(self):
        text = "## How to install it\n\n- Step one.\n"
        self.assertTrue(blocks(text, ("install",))[1].in_procedure)
        text = "## Installation\n\n- Not a match.\n"
        self.assertFalse(blocks(text, ("install",))[1].in_procedure)

    def test_table_cells_know_their_column(self):
        cells = [b for b in blocks(DOC) if b.kind == "cell" and not b.table_header]
        self.assertEqual({(c.text, c.column_name) for c in cells}, {("1", "a"), ("text cell", "b")})

    def test_code_span_becomes_a_placeholder_and_link_keeps_its_text(self):
        paragraph = [b for b in blocks(DOC) if b.kind == "paragraph"][0]
        self.assertEqual(paragraph.text, "Para with \x01 and link text end.")

    def test_ignore_directive_applies_to_the_next_line(self):
        doc = parse_document("<!-- ste-ignore-next-line 1.1, 8.1 -->\nBad line.\n", [])
        self.assertEqual(doc.ignores, {2: {"1.1", "8.1"}})

    def test_ignore_directive_without_rule_means_all_rules(self):
        doc = parse_document("<!-- ste-ignore-next-line -->\nBad line.\n", [])
        self.assertEqual(doc.ignores, {2: {"*"}})

    def test_setext_heading(self):
        found = blocks("Title text\n==========\n\nBody text.\n")
        self.assertEqual([b.kind for b in found], ["heading", "paragraph"])

    def test_block_positions_map_text_to_columns(self):
        block = blocks("A **bold** word.\n")[0]
        self.assertEqual(block.text, "A bold word.")
        self.assertEqual(block.pos[block.text.index("word")], (1, 12))


if __name__ == "__main__":
    unittest.main()
