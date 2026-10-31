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
| 2026-10-30 | [Test of the build from source](2026-10-30-phase1-equivalence-plan.md) | Prediction | Written before the first measurement. The [result entry](2026-10-30-phase1-equivalence-result.md) gives the result. |
| 2026-10-30 | [Result of the test of the build from source](2026-10-30-phase1-equivalence-result.md) | Results | Complete. Build D is in all limits of the prediction entry. |
| 2026-10-30 | [Correctness baseline of the model mix8 on build D](2026-10-30-phase2-correctness-baseline.md) | Results | Complete. It has no prediction entry. The numbers are the baseline for the next entries. |
| 2026-10-31 | [Where the time of a request goes](2026-10-31-phase3-where-the-time-goes.md) | Results | Complete. It has no prediction entry. The numbers are the measurement of phase 3. |
| 2026-10-31 | [Phase 3: caps of the parts of the GPU work (result)](2026-10-31-phase3-ablation-caps.md) | Results | Complete. It has no prediction entry. A cap is the largest time that a part can remove. It is not a prediction. The table `DMK_OPS` has no rows. |
| 2026-10-31 | [Lead 30489: GLU in the quantization of the activation (result)](2026-10-31-lead-30489.md) | Results | Complete. The notebook has no prediction entry for it. It is a lead. The project does not use it in production. |
| 2026-10-31 | [Phase 4: ranked list of targets](2026-10-31-phase4-ranked-targets.md) | Prediction | Written before a change to the code of a target. The owner selects the next step. |

Source: the files in this folder.

## Related documents

- [Method](../method.md): the benchmark, how we compare builds, and the tests of the answers.
- [Data sets](../../data/README.md): the raw files and the tables that the entries use.
- [Change documents](../changes/README.md): one document for each change.
- [Glossary](../glossary.md): the terms of the project.
