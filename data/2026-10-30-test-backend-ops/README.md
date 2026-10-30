# Data set: test of the operations of the model on build D (2026-10-30)

This data set holds the logs of `test-backend-ops` for all operations of the graph of the model mix8. It also holds the files with these operations. The CUDA backend of build `D` and the CPU backend of the same build do each operation. The test compares the two results. The [correctness baseline of phase 2](../../docs/notebook/2026-10-30-phase2-correctness-baseline.md) uses these data. The [data README](../README.md) gives the rules for a data set.

## Question

Does the CUDA backend of build `D` give the same result as the CPU backend for each operation of the graph of mix8? The notebook has no prediction entry for this test. The section `test-backend-ops` of the [entry of phase 2](../../docs/notebook/2026-10-30-phase2-correctness-baseline.md) gives the result.

## Dates

All files are from 2026-10-30. This date is after the [measurement window 1](../2026-10-30-window1/README.md). The logs have no time. The log of the test of the model operations shows a run time of 64 s.

## Project commit

No script of this repository made the logs. A person ran the programs. The section "Commands" gives the commands.

## Build commits

Two builds of tag `b12004` of upstream llama.cpp ran in this test. The commit of the tag is `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. Build `D` has no change to the source. [INFERENCE: the second build has no change to the source.] No branch of a fork is in use.

| Program | Build | Backends |
|---|---|---|
| `test-export-graph-ops` | A build for the CPU only | CPU |
| `test-backend-ops` | Build `D` | CUDA and CPU |

## Build flags

The CMake flags of build `D` are in the [README of measurement window 1](../2026-10-30-window1/README.md#build-flags). The tool `test-export-graph-ops` does not load backends at run time. Its build thus has the CPU backend only. The CMake flags of that build are:

```
-DGGML_BACKEND_DL=OFF -DGGML_CUDA=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_NATIVE=OFF
```

We know no other flag of this build.

## Image

The log of `test-backend-ops` shows that it loaded the CUDA backend from `/opt/dmk`. Build `D` is thus in the same folder as in measurement window 1. The logs have no image digest. [INFERENCE: the image is the stock image of measurement window 1.]

## Model

The model is the "mix8" file `h2o-lightning-4b-rq-Q4_0-mix8.gguf`. Only `test-export-graph-ops` reads it. The program `test-backend-ops` reads the operations from a file and makes the input data.

## Driver

The logs have no driver version. [INFERENCE: it is 595.71.05, the version of measurement window 1.]

## GPU

The log shows one `RTX 3060 Ti`, with compute capability 8.6 and 7,834 MiB of memory. At the start of the test, 3,318 MB of the memory was free. The server in production was on during the test. The operation tests use a small part of the memory.

## Power limit

We know no value. The test makes no speed measurement.

## Clocks

We know no value. The test makes no speed measurement. The harness does not set clocks.

## Server flags

No server ran in this test. The flags of `test-export-graph-ops` are in the section "Commands".

## Commands

The file name `mix8.gguf` in the commands means the path of the model file. For each of the three values of `-ub`, the program `test-export-graph-ops` wrote the different operations of two graphs. One graph is for a prompt. The other graph has one token for each sequence.

```
test-export-graph-ops -m mix8.gguf -c 16384 -b 2048 -ub 2048 -np 3 -fa on
test-export-graph-ops -m mix8.gguf -c 16384 -b 2048 -ub 160 -np 3 -fa on
test-export-graph-ops -m mix8.gguf -c 16384 -b 2048 -ub 48 -np 3 -fa on
```

The three runs made the files `ops-ub2048.txt`, `ops-ub160.txt` and `ops-ub48.txt`. The file `ops-all.txt` has the 187 different operations of the three files. The script checks that it has the same lines as the three files together. Its lines are in a different sequence from the three files. We do not know the command that merged the files.

The two tests ran with build `D`:

```
test-backend-ops test -b CUDA0 --test-file ops-all.txt
test-backend-ops test -b CUDA0 -o GATED_DELTA_NET,CPY,SSM_CONV,L2_NORM,SOFTPLUS
```

The first command made `tbo-D.txt`. The second command made `tbo-D-builtin.txt`. It runs the tests that are in the source of `test-backend-ops` for the five operations. The command to make the tables is:

```
python3 data/2026-10-30-test-backend-ops/make_tables.py
```

## Arms

This data set has one arm: build `D`. In each test, the CPU backend of build `D` makes the reference result.

## Raw files

The folder `raw/` has 6 files. They are as the programs wrote them. The two logs have the color codes of the terminal. The first line of a log has the driver text that the [README of measurement window 1](../2026-10-30-window1/README.md#raw-files) gives.

| File | Written by | Contents |
|---|---|---|
| `ops-ub2048.txt` | `test-export-graph-ops` | 97 different operations, with `-ub 2048` |
| `ops-ub160.txt` | `test-export-graph-ops` | 97 different operations, with `-ub 160` |
| `ops-ub48.txt` | `test-export-graph-ops` | 91 different operations, with `-ub 48` |
| `ops-all.txt` | A person | The 187 different operations of the three files |
| `tbo-D.txt` | `test-backend-ops` | The log of the test of `ops-all.txt`: one line for each test, and the summary |
| `tbo-D-builtin.txt` | `test-backend-ops` | The log of the built-in tests of the five operations |

## Tables

The script [make_tables.py](make_tables.py) makes the tables in the folder `tables/`. It stops with an error if a number is different from the summary line of its log. The tables are TSV files. The first line has the names of the columns.

### export.tsv

The number of different operations in each file. It has 4 rows.

| Column | Unit | Meaning |
|---|---|---|
| `file` | None | The name of the file |
| `ubatch_tokens` | tokens | The value of `-ub`, or `all` for the file with the merge |
| `unique_ops` | operations | The number of lines of the file. Each line is one different operation. |

### results.tsv

The number of tests for each operation and each log. It has 24 rows.

| Column | Unit | Meaning |
|---|---|---|
| `log` | None | `model-ops` for `tbo-D.txt`, or `built-in` for `tbo-D-builtin.txt` |
| `op` | None | The name of the operation |
| `ok` | tests | The number of tests with the result `OK` |
| `failed` | tests | The number of tests with the result `FAIL` |
| `not_supported` | tests | The number of tests with the result `not supported`. The summary line does not have these tests. |

### failures.tsv

One row for each test with the result `FAIL`. It has 6 rows, and all are in `tbo-D.txt`.

| Column | Unit | Meaning |
|---|---|---|
| `log` | None | The log |
| `op` | None | The name of the operation |
| `name` | None | The name of the tensor in the graph |
| `ne` | values | The four dimensions of the output tensor |
| `diagnostic` | None | The text that `test-backend-ops` wrote before the test. It gives the error, or the position of the first value that is not a number. |

## Files not added

All files that the programs wrote are in `raw/`.
