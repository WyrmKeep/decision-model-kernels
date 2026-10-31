# Data

This folder holds the raw files and the derived tables of each measurement. A data set is the data of one test.

The table below lists the data sets in this folder.

The [harness](../harness/README.md) makes the raw files. The [method](../docs/method.md) gives the tests that compare builds.

## Data sets

| Data set | Date | Contents |
|---|---|---|
| [`2026-10-30-window1`](2026-10-30-window1/README.md) | 2026-10-30 | The first measurement window: build `D` against the stock image, the KL divergence, the 52-decision set, `compute-sanitizer`, and the first runs of the public benchmark |
| [`2026-10-30-test-backend-ops`](2026-10-30-test-backend-ops/README.md) | 2026-10-30 | The logs of `test-backend-ops` for the 187 operations of the graph of mix8 and for five built-in tests, on build `D` |
| [`2026-10-30-card-limits`](2026-10-30-card-limits/README.md) | 2026-10-30 | The highest rates of the `RTX 3060 Ti` for the tensor cores, the DRAM and the L2 cache, and the time of a cuBLASLt int8 GEMM for the weight shapes of the model. |
| [`2026-10-31-public-baseline`](2026-10-31-public-baseline/README.md) | 2026-10-31 | The baseline of the public benchmark: three runs of build `D` with the model mix8, the file `bench/baseline.json`, and a check run with the result `PASS`. |

Add one row for each new data set. Put a link to its folder in the first column.

## Folders of a data set

Make one folder for each data set. Name it `data/<date>-<name>/`. `<date>` is the date of the first run, in the shape `YYYY-MM-DD`. `<name>` is a short name with only `a-z`, `0-9` and `-`.

| Path in the folder | Contents |
|---|---|
| `README.md` | The source of the data, and the columns, units and meaning of each table. |
| `raw/` | The raw files, as the harness wrote them. It also has the copy of the schedule file. |
| `tables/` | The derived tables, as CSV or TSV. It also has the JSON output of `answers_compare.py`. |
| The script | One script that makes all derived tables from the files in `raw/`. The README of the data set names it and gives its command. |

## Rules

1. Keep each raw file as the harness wrote it. Do not change it.
2. Make each derived table with the script of the data set. Do not change a table after the script makes it.
3. Write each derived table as CSV or TSV. Use one type in all tables of a data set.
4. Put the unit in the name of each column or in the README of the data set.
5. Make the script write each table that a document shows from this data set.
6. Write the source below each table of a document. Put a link to the folder of the data set in it.
7. Put no host name, no IP address and no secret in a data set. Put no path of a server and no model file in it.
8. Use `tools/publish/scan-dir.sh` on the folder of the data set before you add it to the repository.
9. If a raw file has one of the items of rule 7, do not add the file. Write its name and the cause in the README.

## README of a data set

Each data set has a `README.md`. Write one section for each field of the table below. Write the fields in this sequence.

| Field | Text to write | Where you find it |
|---|---|---|
| Question | The question of the test. A link to the prediction entry and to the results entry of the notebook. | The [notebook](../docs/notebook/README.md) |
| Dates | The date of the first run and the date of the last run, in UTC. | `date_utc` in the `.meta` files |
| Project commit | The commit of this repository that you used. | `git rev-parse HEAD` |
| Build commits | The commit and the branch of the fork of llama.cpp for each arm. | `git rev-parse HEAD` in the fork |
| Build flags | The full CMake command of each build. | The section Build the programs of an arm in the [harness](../harness/README.md) document |
| Image | The image of the server, with its `@sha256:` name. | `DMK_IMAGE` in the env file |
| Model | The name of the model and the quantization. Give no path. | The [benchmark](../bench/README.md) document |
| Driver | The version of the driver. | The `.meta` files |
| GPU | The name of the GPU. | `nvidia-smi` |
| Power limit | The power limit, in W. | The `.meta` files |
| Clocks | The clocks that you set, or `default`. The harness does not set clocks. | The values that you set. The `.smi` files have the SM clock of each sample. |
| Server flags | The value of `DMK_SERVER_ARGS`, or `default`. | The env file |
| Commands | The full commands of `make_schedule.py`, `run_schedule.sh`, `ab_stats.py`, `answers_compare.py` and the script of the data set. | The commands that you used |
| Arms | The short name and the arm of each run. | The schedule file |
| Raw files | Each file in `raw/`, with its contents. | The section Files of one run in the [harness](../harness/README.md) document |
| Tables | For each derived table: the file name, and a table of its columns. | The script of the data set |
| Files not added | The raw files that you did not add. Write the cause for each file. | The scan of the folder |

The tables of `ab_stats.py` have the columns that the [harness](../harness/README.md) document gives.

## Table of the columns

Use this table in the README of a data set for each derived table. Each row is one column of the derived table.

| Column | Unit | Meaning |
|---|---|---|
| The name of the column | The unit, or `None` | The values, and the method of the script |
