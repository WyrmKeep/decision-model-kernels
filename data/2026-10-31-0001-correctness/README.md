# Data set: correctness tests of change 0001 (2026-10-31)

This data set holds the logs of the tests of `test-backend-ops` and `compute-sanitizer` for change 0001. Change 0001 is pull request #30489 of llama.cpp. It fuses GLU into the quantization of the activation in MMQ. The [change document](../../docs/changes/0001-glu-into-quantize.md) uses these data. The [data set of the speed test](../2026-10-31-lead-30489/README.md) has the other tests of the change. The [data README](../README.md) gives the rules for a data set.

## Question

1. Does the CUDA backend of build `L` give the same results as the CPU backend? The tests are the 310 new cases of the pull request and the 187 operations of the model graph.
2. Does `compute-sanitizer` find an error in the new code? The four tools run on the new cases and on one run of the model with 256 tokens.
3. Does build `D` give the same results as build `L` for the 187 operations of the model graph?

The notebook has no prediction entry for these tests. We did the tests after the [speed test](../2026-10-31-lead-30489/README.md). The [change document](../../docs/changes/0001-glu-into-quantize.md) gives the thresholds. A threshold comes from the source of `test-backend-ops` or from the [correctness baseline of phase 2](../../docs/notebook/2026-10-30-phase2-correctness-baseline.md).

## Dates

| Item | Value |
|---|---|
| First test while the server in production runs starts | 2026-10-31 05:03:34 UTC |
| Measurement window opens | 2026-10-31 05:28:14 UTC |
| First run of the measurement window starts | 2026-10-31 05:28:32 UTC |
| Last run of the measurement window ends | 2026-10-31 05:39:42 UTC |
| Measurement window closes | 2026-10-31 05:40:04 UTC |

The logs of `test-backend-ops` have no time. The tests with the server in production ran between 05:03 UTC and the start of the measurement window. The server in production was off in the measurement window. The script of the measurement window started it again.

Source: the start time of the first command, the `date_utc` line of each `.meta` file, and the log of the script of the measurement window.

## Project commit

No script of this repository made the logs of `test-backend-ops` and of the other tests with the server in production. A person ran the commands of the section "Commands". The measurement window ran the schedule `0001-san.txt` with `harness/run_schedule.sh`. It used a work copy of the harness. The files of the work copy are the same as the files of the folder `harness/` of this repository.

The only differences are the file `harness/README.md` and the folders `ablate/`, `build/` and `profile/`. The measurement window did not use them.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `D` | Build `D` of tag `b12004`. It is the build of the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md#build-commits). | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |
| `L` | Branch `dmk/lead-30489` of the fork of llama.cpp of this project. It is the commit of change 0001. | `6b8b65e6672f07805000d3c9683b65c72cc62303`. The commit before it is the commit of `D`. |

The folders of the two builds are the same as in the speed test. We made `D` on 2026-10-30 at 21:50 UTC and `L` on 2026-10-30 at 23:07 UTC. We did not make a new build for these tests.

## Build flags

The flags are the flags of the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md#build-flags). The two builds have the target `test-backend-ops`.

## Image

| Use | Image | Digest |
|---|---|---|
| All tests | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |

The container puts the build in `/opt/dmk`. The container has the folder with `compute-sanitizer` at `/opt/tools`. Its version is 12.8.93. It is part of CUDA 12.8 of NVIDIA.

## Model

The tests of `test-backend-ops` read no model. The four runs of `compute-sanitizer` in the measurement window read the "mix8" file `h2o-lightning-4b-rq-Q4_0-mix8.gguf`. They use one chunk of 256 tokens from a text file. The file has the states of the private benchmark.

## Driver

The version of the driver is 595.71.05. The `.meta` files show it.

## GPU

The GPU is the `RTX 3060 Ti`, with compute capability 8.6. The logs show 7,834 MiB of memory. At the start of each test with the server in production, 3,268 MiB of the memory was free.

## Power limit

The power limit was 200 W in the measurement window. The `.meta` files show it. The tests with the server in production have no value. [INFERENCE: the value was 200 W, because no person changed it.]

## Clocks

We did not lock the clocks. These tests make no speed measurement.

## Server flags

No server ran in these tests.

## Commands

The tests with the server in production ran in the stock image. The folder of the build was `/opt/dmk`. The word `<cs>` below is the path `/opt/tools/compute-sanitizer-12.8.93/bin/compute-sanitizer`. The file `ops-all.txt` is the file of the [data set `2026-10-30-test-backend-ops`](../2026-10-30-test-backend-ops/README.md). The CPU backend is the reference in all tests.

Two tests ran with build `L` and no `compute-sanitizer`:

```
test-backend-ops test -b CUDA0 -o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE
test-backend-ops test -b CUDA0 --test-file ops-all.txt
```

The first command made `tbo-L-glu.txt`. The second command made `tbo-L.txt`.

The four tools of `compute-sanitizer` ran with build `L` on the same new cases. The command for the tool `<tool>` is:

```
<cs> --tool <tool> --error-exitcode 99 test-backend-ops test -b CUDA0 \
    -o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE
```

The tool `initcheck` found errors. Three more `initcheck` commands show where they are. The first command checks only the kernels with `quantize_mmq_q8_1` in the name. The second command does not check the kernels with `mul_mat_q` in the name. The third command ran on the two builds. Its tests are the Q4_0 tests of `MUL_MAT`, which build `D` also has:

```
<cs> --tool initcheck --error-exitcode 99 --kernel-name kns=quantize_mmq_q8_1 \
    test-backend-ops test -b CUDA0 -o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE
<cs> --tool initcheck --error-exitcode 99 --kernel-name-exclude kns=mul_mat_q \
    test-backend-ops test -b CUDA0 -o GLU_MUL_MAT,SWIGLU_MUL_MAT_ID,SWIGLU_MUL_MAT_ID_REUSE
<cs> --tool initcheck --error-exitcode 99 test-backend-ops test -b CUDA0 -o MUL_MAT -p q4_0
```

The first command made `san-initcheck-quantize-L.txt`. The second command made `san-initcheck-nofixup-L.txt`. The third command made `initmm-b12004-sm86.txt` for `D` and `initmm-lead-30489.txt` for `L`.

The measurement window ran the four tools on the model. The file [raw/0001-san.txt](raw/0001-san.txt) is the schedule. The harness ran it with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

Each line of the schedule has the shape `tool@<tool> <run name> build:lead-30489 perplexity -m <mix8> -ngl 999 -fa auto -b 256 -ub 256 -c 256 --chunks 1 -f <text file>`. The harness runs `llama perplexity` in the wrapper `compute-sanitizer --tool <tool> --error-exitcode 99`. The [measurement window 1](../2026-10-30-window1/README.md) ran the same lines for build `D`.

The tables came from this command:

```
python3 data/2026-10-31-0001-correctness/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `D` | `build:b12004-sm86` | Build `D` of tag `b12004` |
| `L` | `build:lead-30489` | Build `D` and the commit of change 0001 |

Most logs are for arm `L`. Two logs are for arm `D`: the file `initmm-b12004-sm86.txt`, and the file `tbo-D.txt` of the [data set `2026-10-30-test-backend-ops`](../2026-10-30-test-backend-ops/README.md).

## Raw files

The folder `raw/` has 37 files. The tools wrote them. They have no change. The logs of `test-backend-ops` and `compute-sanitizer` have the color codes of the terminal.

Each `.err` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp.

| File | Number | Written by | Contents |
|---|---:|---|---|
| `tbo-L-glu.txt`, `tbo-L.txt` | 2 | `test-backend-ops` | The logs of the two tests with no `compute-sanitizer`: one line for each test, and the summary |
| `san-<tool>-L.txt` | 4 | `compute-sanitizer` | The logs of the four tools on the new cases. `<tool>` is `memcheck`, `initcheck`, `synccheck` or `racecheck`. |
| `san-initcheck-quantize-L.txt`, `san-initcheck-nofixup-L.txt` | 2 | `compute-sanitizer` | The logs of `initcheck` with a selection of kernels by their names |
| `initmm-b12004-sm86.txt`, `initmm-lead-30489.txt` | 2 | `compute-sanitizer` | The logs of `initcheck` on the tests of Q4_0 of `MUL_MAT`, on build `D` and on build `L` |
| `<log>.err` | 10 | The container | The standard error of each of the 10 logs above. It has 4 lines: the driver text, the device, and the two backends that `test-backend-ops` loaded. |
| `lead30489-san-<tool>-L.*` | 16 | `run_tool.sh` | The 4 runs of `compute-sanitizer` on the model, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `0001-san.txt` | 1 | A person | The schedule of the 4 runs of the measurement window |

The [harness document](../../harness/README.md) gives the meaning of the files of a run.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/`. It reads the files in `raw/`. It also reads the log of build `D` in the [data set `2026-10-30-test-backend-ops`](../2026-10-30-test-backend-ops/README.md). It stops with an error if the number of tests is different from the summary line of its log. The tables are TSV files with the names of the columns in the first line.

An empty field means that the table has no value there.

### tests.tsv

One row for each log of `test-backend-ops`. It has 10 rows.

| Column | Unit | Meaning |
|---|---|---|
| `log` | None | The name of the log |
| `build` | None | `D` or `L` |
| `sanitizer` | None | The tool of `compute-sanitizer`, and its selection of kernels if it has one. `none` if the test ran with no `compute-sanitizer`. |
| `selection` | None | The flags of `test-backend-ops` that select the tests. The file `ops-all.txt` has the 187 operations of the model graph. |
| `tests_ok` | tests | The number of tests with the result `OK` |
| `tests_run` | tests | The number of tests that ran. The summary line of the log has the same number. |
| `sanitizer_errors` | errors | The number of errors in the summary of `compute-sanitizer`. Empty if `compute-sanitizer` did not run. |

### glu-ops.tsv

The results of the new cases of the pull request, for each operation. It has 3 rows. The source is `tbo-L-glu.txt`.

| Column | Unit | Meaning |
|---|---|---|
| `op` | None | The name of the operation |
| `ok` | tests | The number of tests with the result `OK` |
| `failed` | tests | The number of tests with the result `FAIL` |
| `not_supported` | tests | The number of tests with the result `not supported` |

### compare-model-ops.tsv

The results of the 187 operations of the model graph on build `D` and on build `L`. It has 1 row. The script uses the name of the operation and its parameters to find the same test in the two logs.

| Column | Unit | Meaning |
|---|---|---|
| `tests_d`, `tests_l` | tests | The number of tests in the log of build `D` and of build `L` |
| `same_set_of_tests` | None | `True` if the two logs have the same tests |
| `tests_with_different_result` | tests | The number of tests with a different result in the two logs |
| `failed_d`, `failed_l` | tests | The number of tests with the result `FAIL` in each log |

### sanitizers-model.tsv

The results of the four tools of `compute-sanitizer` on build `L` and the model. It has 4 rows. The table `sanitizers.tsv` of the [measurement window 1](../2026-10-30-window1/README.md) has the same columns for build `D`.

| Column | Unit | Meaning |
|---|---|---|
| `tool` | None | `memcheck`, `initcheck`, `synccheck`, or `racecheck` |
| `errors` | errors | The number of errors in the summary of the tool |
| `hazards_displayed` | count | The number of problems of shared memory that `racecheck` shows. Only for `racecheck`. |
| `warnings` | warnings | The number of warnings. Only for `racecheck`. |
| `seconds_per_pass` | s | The time of one pass, as `llama perplexity` writes it |
| `ppl` | None | The perplexity of the one chunk |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0001-correctness/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files that the tests wrote. A copy of the file `ops-all.txt` was in the folder of the tests. We did not add it, because the [data set `2026-10-30-test-backend-ops`](../2026-10-30-test-backend-ops/README.md) has it. A log of `initcheck` shows the first 100 errors only. The summary line of the log has the number of all errors.

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds no item in this folder.
