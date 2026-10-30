# Documents

This folder has the documents of the project. The [README of the repository](../README.md) gives the first information.

## Contents

| Document | Contents |
|---|---|
| [method.md](method.md) | The benchmark, how we compare builds, and the tests of correctness |
| [model.md](model.md) | The facts about the model and about the server |
| [kernels.md](kernels.md) | The kernels of llama.cpp that the project can change |
| [notebook/](notebook/README.md) | The tests of the project, each with a date and a prediction |
| [changes/](changes/README.md) | One document for each change, with the same text as its pull request |
| [glossary.md](glossary.md) | The terms of the project. Each term has one meaning |

Source: the files in this folder.

## Where to start

1. Read [method.md](method.md). It tells how we measure and how we know that an answer is correct.
2. Read [model.md](model.md). It tells the facts about the model and the server.
3. Read the [notebook](notebook/README.md) to see the tests in the sequence of their dates.
4. Read a change document when a change is complete.

## Standard for the text

All documents follow ASD-STE100 Issue 9. The [checker](../tools/ste/README.md) checks each document. The [glossary](glossary.md) holds the terms of the project.

Each table has the units in the table header and one quantity in each column. A column has the same number of decimal places in all rows. The baseline is in the first row. The source is below the table.
