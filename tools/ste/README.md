# STE checker

This tool checks Markdown files against the rules of ASD-STE100 `Issue 9`. It finds words that are not approved. It also finds long sentences, passive verbs, contractions, and other problems. The tool helps a person to write text. It does not replace a person who reads the text.

## Install

1. Install Python 3.11 or higher.
2. Install the tool `pdftotext`.
3. Get the PDF of ASD-STE100 `Issue 9` at `https://www.asd-ste100.org/`.
4. Put the PDF at `~/.local/share/asd-ste100/ASD-STE100_ISSUE9.pdf`. Or set `STE_PDF` to the path of the PDF.

You can also add `--download-pdf` to the first run. The tool then downloads the PDF and checks its hash. The tool uses Python and `pdftotext` only.

## Run

1. Go to the root folder of the repository.
2. Run `python3 tools/ste/ste_check.py docs/glossary.md`.
3. Read the findings. Each finding is one line: `file:line:col RULE message`.
4. Change the text and run the tool again.

If you give no path, the tool checks all Markdown files of the repository. The configuration file gives a list of files to ignore.

| Flag | Effect |
|---|---|
| `--stdin` | Check the text on `stdin`. |
| `--stdin-name NAME` | Show NAME as the file name of the text on `stdin`. |
| `--format json` | Write the findings as JSON. |
| `--strict` | Count warnings as errors. |
| `--config FILE` | Use a different configuration file. |
| `--download-pdf` | Download the PDF if it is not on the device. |
| `--no-ignore` | Check the files that the configuration file ignores. |
| `--lookup WORD` | Show the type that the tool gives to each word. |
| `--dictionary-info` | Show the number of dictionary entries. |

The exit code is 0 when the tool finds no error. It is 1 when the tool finds an error. It is 2 when the PDF, the glossary, or the configuration file has a fault.

## Environment variables

| Name | Use |
|---|---|
| `STE_PDF` | The path of the PDF. |
| `STE_CACHE` | The folder for the cache of the dictionary. |
| `STE_DOWNLOAD_PDF` | Set it to `1` to let the tool download the PDF. |

The cache folder must not be in the repository. The tool stops when the cache folder is in the repository.

## Rules

| Rule | Problem | Level |
|---|---|---|
| 1.1 to 1.4 | The word is not approved. The finding gives an alternative word. | Error |
| 1.2 | The word has an approved type: a noun only or a verb only. | Warning |
| 1.14 | The word does not have the American spelling. | Error |
| 2.1 | A noun cluster has more than 3 words. | Warning |
| 3.6 | A passive verb. | Warning in a description. Error in a procedure. |
| 4.2 | A contraction. | Error |
| 5.1 | A step has more than 20 words. | Error |
| 5.3 | A step does not start with an imperative verb. | Warning |
| 6.3 | A sentence in a description has more than 25 words. | Error |
| 6.6 | A paragraph has more than 6 sentences. | Error |
| 8.1 | A sentence has a semicolon. | Error |

The configuration file sets the limits. The tool checks a heading only for words. The tool checks a table cell for words but not for the length of a sentence.

## Procedures and descriptions

The tool finds a procedure in these items:

- A list item below a heading that has a word of the list in the configuration file. Install and Run are two examples.
- An item of a list with numbers, if a heading above it has such a word.
- A sentence that starts with an imperative verb.

All other text is a description.

## Text that the tool does not check

The tool does not check these items:

- Code between `` ``` `` marks and code between `` ` `` marks.
- Names of files and paths.
- Numbers with units.
- Table cells that have only numbers or code.

To ignore one line, write `<!-- ste-ignore-next-line RULE -->` on the line before it. Write `*` to ignore all rules.

## Limits

The tool uses lists of words. It does not know the structure of a sentence. Know these limits:

- The tool can show a warning for a correct sentence. It can also show no warning for a sentence with a problem.
- The tool finds a noun cluster from the type of each word. It can show no warning for a noun cluster of 4 words.
- The tool does not check the meaning of an approved word. The meaning can be incorrect.
- A word of the glossary always passes the word check. You must make sure that each meaning is correct.
- The tool checks only some of the rules of STE.
- A person must read each text.

## Add a glossary term

1. Open `docs/glossary.md`.
2. Add one row. Keep the rows in sequence by term.
3. Write the term, the type, and the meaning.
4. Give the term one meaning only.
5. Write the meaning with approved words and with other glossary terms.
6. Run `python3 tools/ste/ste_check.py docs/glossary.md`.

Use a word of the dictionary only with its approved meaning. If you want a different meaning, make a new term with two or more words. The tool shows a warning when a term is also an approved word.

## Tests

1. Go to the root folder of the repository.
2. Run `python3 -m unittest discover tools/ste/tests`.

The tests use new sentences and a small test dictionary. They do not use the PDF. One test uses the PDF if it is on the device.

## Copyright

ASD makes the ASD-STE100 specification. The specification has a license. This repository has no copy of the PDF, no copy of its text, and no list of words from it. The tool reads the dictionary from your copy of the PDF when it runs. It keeps a cache in a different folder from the repository. Read the license of the specification before you give the PDF to other persons.
