# Notebook

The notebook records the tests of the project. Each entry has a date. The dates show that each prediction is before its result.

## How the notebook works

- Each entry is one file. The file name is `YYYY-MM-DD-<name>.md`. The file name starts with the date of the entry.
- We write a prediction entry before a test. It gives the test, the limits, the tolerances, and the prediction. We commit it before the measurement.
- We do not change a prediction entry after the measurement starts.
- We write a results entry after the test. It gives the prediction, the result of the measurement, and the difference between them. It also gives the cause of the difference.
- A results entry has a link to its prediction entry. It has links to the data in [data/](../../data/README.md).
- An inspection entry gives facts that we read in code or in text. It has no prediction. It marks each fact that we did not measure with [INFERENCE].

## Entry types

| Type | When we write it | Contents |
|---|---|---|
| Prediction | Before the measurement | The test, the limits, the tolerances, the prediction |
| Results | After the measurement | The data, the prediction, the result, the difference, the cause |
| Inspection | Before a change | The facts from code and from text, with sources |

## Entries

| Date | Entry | Type | Condition |
|---|---|---|---|
| 2026-10-30 | [Inspection of the upstream code](2026-10-30-phase0-upstream-survey.md) | Inspection | Complete |
| 2026-10-30 | [Test of the build from source](2026-10-30-phase1-equivalence-plan.md) | Prediction | Written before the first measurement. At this time, this folder has no results entry for it. |

Source: the files in this folder.

## Related documents

- [Method](../method.md): the benchmark, how we compare builds, and the tests of the answers.
- [Change documents](../changes/README.md): one document for each change.
- [Glossary](../glossary.md): the terms of the project.
