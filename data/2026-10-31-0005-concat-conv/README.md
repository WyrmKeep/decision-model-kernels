# Data set: change 0005, concat folded into `ssm_conv` (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of change 0005. The change replaces four nodes of the Gated DeltaNet layers with one CUDA kernel. The [results entry](../../docs/notebook/2026-10-31-0005-concat-conv-result.md) uses these data. The [change document](../../docs/changes/0005-concat-conv.md) gives the change. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build with change 0005 (arm `C`) faster than the baseline build (arm `L`) for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: The change is in the equivalence class `exact`. Does arm `C` give the same answers as arm `L`, bit for bit?
3. Tests: Do `test-backend-ops` and `compute-sanitizer` find no error in the new kernel? Is the SASS of all other kernels the same?

We committed the [prediction](../../docs/notebook/2026-10-31-0005-concat-conv-prediction.md) on 2026-10-31 at 05:15:13 UTC, before the first run.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 06:34:33 UTC |
| Last speed run starts | 2026-10-31 06:50:07 UTC |
| Measurement window of the speed test and of the runs with `compute-sanitizer` | 06:34 to 07:05 UTC |
| Measurement window of the two control runs | 07:29 to 07:30 UTC |

The 16 speed runs used about 16 minutes. The project stopped the server in production for each measurement window. One run at a time used the GPU. The tests of `test-backend-ops` and the test program ran after the speed test. They ran in a container on the same host, one at a time.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv) and the table [sanitizer-model.tsv](tables/sanitizer-model.tsv).

## Project commit

The scripts of the measurement are the same as the files of the folder `harness/` of this repository, with 4 differences. The file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/` are not in the work copy that the measurement used. The measurement did not use them. We compared the two folders with `diff -rq` after the measurement.

The SASS compare used `harness/build/sass_compare.py` of this repository. The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `c0005-bench-*` | The private benchmark, with 100 decisions | No |
| `c0005-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |
| `c0005-san-*`, `c0005-init-*` | One chunk of `perplexity` with `compute-sanitizer` | Not applicable |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `L` | Branch `dmk/stack` of the fork of llama.cpp of this project, with change 0001. It is build `lead-30489`. | `6b8b65e6672f07805000d3c9683b65c72cc62303`. The source had 0 changes. |
| `C` | Branch `dmk/0005-concat-conv` of the same fork | `a67acc087906d2049c52d7ace2a275123656eb09`. The commit before it is the commit of `L`. The source had 0 changes. |

The commit of `C` changes 7 files, with 431 added lines and 12 removed lines. The files are `ggml-cuda.cu`, `ssm-conv.cu` and `ssm-conv.cuh` in `ggml/src/ggml-cuda/`, the models `qwen35`, `qwen35moe` and `qwen3next` in `src/models/`, and `tests/test-backend-ops.cpp`.

We made `L` on 2026-10-30 at 23:07 UTC and `C` on 2026-10-31 at 05:48 UTC. The files `raw/build/MANIFEST-lead-30489.txt` and `raw/build/MANIFEST-0005-concat-conv.txt` give the facts of the builds and the SHA-256 value of each file.

## Build flags

We made the two builds with `harness/build/build.sh`. The two manifests show the same tools and the same `cmake` command.

| Tool | Version |
|---|---|
| Image for the build | `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`, digest `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` |
| `nvcc` | 12.8.93 |
| `gcc` | 14.2.0 |
| `cmake` | 3.28.3 |

The `cmake` flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

The flags are the flags of the data set [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md).

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, `llama bench`, `perplexity` and `test-backend-ops` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

Each arm runs in the same image. The container puts the build in `/opt/dmk`.

## Model

| Use | File |
|---|---|
| All runs | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8") |
| The reference answers of the quality phase | The answers of the Q8_0 model. The `.json` files give their label: `ref-q8`. The [method](../../docs/method.md) gives their source. |

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all runs. The project did not change it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,950 MHz. The column `throttle_reasons_at_start` has the value `0x0000000000000001` for the first run of the private benchmark. The bit `0x01` is `GpuIdle` of the driver.

## Server flags

All runs of the private benchmark used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs of `llama bench` and `perplexity` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The host did other work during the schedule. The cause is not in the data.

| Runs | One-minute load average |
|---|---|
| Block 1 of the private benchmark | 6.42 to 10.23 |
| Blocks 2 to 4 of the private benchmark | 10.23 to 16.38 |
| The 8 runs of `llama bench` | 14.71 to 16.69 |

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules for the speed tests came from `make_schedule.py`. The file [raw/c0005.txt](raw/c0005.txt) has the lines. The prediction has the same two commands.

```
python3 harness/make_schedule.py --prefix c0005-bench --blocks 4 --seed 50051 \
    --arm L=build:lead-30489 --arm C=build:0005-concat-conv --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix c0005-llb --blocks 4 --seed 50052 \
    --arm L=build:lead-30489 --arm C=build:0005-concat-conv --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The files [raw/c0005-san.txt](raw/c0005-san.txt) and [raw/c0005-init.txt](raw/c0005-init.txt) have the lines of the runs with `compute-sanitizer`. A line has this shape. The tool, the arm and the number of tokens are different in each line:

```
tool@initcheck <run name> <arm> perplexity -m /models/<model> -ngl 999 -fa auto -b 300 -ub 300 -c 300 --chunks 1 -f /work/domain-medium-long.txt
```

The script of the measurement window stopped the server in production, ran the schedules, and started the server in production again. The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tests of the operations ran in a container of the same image, with the command below. The file `raw/ops/test-<op>.txt` has the output of each run. For `compute-sanitizer`, the command started with `compute-sanitizer --tool <tool> --error-exitcode 99`.

```
test-backend-ops test -o <SSM_CONV_CONCAT_CPY, SSM_CONV, SSM_CONV_BIAS_SILU or CONCAT>
```

The tables came from these commands. Arm `L` is the reference arm, and the ratio is `C` divided by `L`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix c0005-bench --a L --b C --margin 2
python3 ../../harness/analysis/ab_stats.py raw --prefix c0005-llb --a L --b C --margin 2
python3 ../../harness/analysis/answers_compare.py --a 'raw/c0005-bench-b*-L.reference.json' \
    --b 'raw/c0005-bench-b*-C.reference.json'
python3 data/2026-10-31-0005-concat-conv/make_tables.py
```

The SASS files in `raw/sass-compare/` came from this command, with the cubins of the two builds. The cubins came from `cuobjdump -arch sm_86 -xelf all` on the file `libggml-cuda.so` of each build. The file `res-usage-new-kernels.txt` has the lines of `cuobjdump -res-usage` for the 6 new kernels.

```
python3 sass_compare.py <cubins of L> <cubins of C> <output folder> "ssm_conv|concat|cpy_scalar|silu"
```

The folder `code/` has the source of the test program `bitcheck.cpp`. It builds one graph with the nodes `concat`, `cpy`, `ssm_conv` and `silu`. In the mode `fused` it runs the graph as it is. In the mode `parent` it puts a `scale` by 1.0 between the `cpy` and `ssm_conv`. This stops the fusion of the new kernel.

The program writes the output and the cache of 470 cases to a file. We built it with the libraries of build `C`. We compared the two files with `cmp`.

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `L` | `build:lead-30489` | The branch `dmk/stack` with change 0001, in the image of the stock build |
| `C` | `build:0005-concat-conv` | Arm `L` and change 0005, in the same image |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In the private benchmark, arm `L` ran first in blocks 1, 2 and 4, and arm `C` ran first in block 3. In `llama bench`, arm `L` ran first in blocks 1 to 3, and arm `C` ran first in block 4.

## Raw files

The folder `raw/` has 140 files. The harness wrote 120 of them, and they have no change. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `c0005-bench-b<N>-<L or C>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `c0005-llb-b<N>-<L or C>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `c0005-san-<tool>-C.*` | 16 | `run_tool.sh` | 4 runs of `perplexity` with 300 tokens and `compute-sanitizer` |
| `c0005-init-L300.*`, `c0005-init-C256.*` | 8 | `run_tool.sh` | 2 control runs of `initcheck` |
| `c0005.txt`, `c0005-san.txt`, `c0005-init.txt` | 3 | `make_schedule.py` and a person | The schedules, with comment lines |
| `ops/test-<op>.txt` | 4 | A person | The output of `test-backend-ops` for 4 operations |
| `ops/san-<tool>-SSM_CONV_CONCAT_CPY.txt` | 4 | A person | The output of `test-backend-ops` for the new case with 4 tools |
| `ops/memcheck-test-with-40-tokens.txt` | 1 | A person | The report of `memcheck` for the first version of the new test case |
| `bitcheck/fused.txt`, `bitcheck/parent.txt`, `bitcheck/compare.txt` | 3 | A person | The output of the test program in the two modes, and the result of `cmp` and `md5sum` for the two files |
| `sass-compare/sass-compare.json`, `sass-compare/key-kernels.tsv` | 2 | `sass_compare.py` | The summary of the SASS compare, and one row for each of 89 names |
| `sass-compare/res-usage-new-kernels.txt` | 1 | A person | The registers of the 6 new kernels |
| `build/MANIFEST-<arm>.txt` | 2 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp.

The file `ops/memcheck-test-with-40-tokens.txt` is from build commit `056f5eeea`, which has the first version of the test with 40 tokens. The report shows reads after the end of the input in `ssm_conv_long_token_f32`. The last test uses 64 tokens.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

The tables `runs.tsv`, `speed-bench.tsv`, `speed-llama-bench.tsv`, `speed-bench-blocks.tsv`, `speed-llama-bench-blocks.tsv`, `quality-bench.tsv` and `answers-compare-bench.json` have the columns of the same tables in the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). Arm `L` replaces arm `D`, and arm `C` replaces arm `L`. In the tables of the blocks, the columns `load1_l_*` are for arm `L` and the columns `load1_c_*` are for arm `C`.

### ops-tests.tsv

The tests of `test-backend-ops`. It has 8 rows.

| Column | Unit | Meaning |
|---|---|---|
| `op` | None | The name of the operation |
| `sanitizer` | None | The tool of `compute-sanitizer`, or `none` |
| `compared` | None | The test compares the CUDA result with the CPU result |
| `tests`, `passed` | tests | The number of cases, and the number of cases that pass |
| `ok` | None | `True` if the backend CUDA0 passes and the tool has 0 errors and 0 warnings |
| `sanitizer_errors`, `sanitizer_warnings` | errors | The numbers of the last summary line of the tool |

### sanitizer-model.tsv

The runs of `perplexity` with one chunk and `compute-sanitizer`. It has 6 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `tool`, `arm_tag`, `tokens` | None | The name of the run, the tool, the arm, and the number of tokens of the chunk |
| `start_utc` | UTC time | The start of the run |
| `exit_code` | None | The exit code. The value 99 means that the tool found errors. |
| `errors`, `warnings` | errors | The numbers of the summary line |
| `errors_printed` | errors | The number of errors that the tool shows. The limit is 100. |
| `kernels_of_printed_errors` | None | The names of the functions in the errors that the tool shows |
| `perplexity_chunk_1` | None | The perplexity of the chunk |

### bitcheck.tsv

The result of the test program. It has 3 rows.

| Column | Unit | Meaning |
|---|---|---|
| `mode` | None | `fused`, `parent`, or `compare` |
| `configs` | cases | The number of cases that the mode ran |
| `md5_of_output` | None | The MD5 value of the output file. For `compare`, the word `IDENTICAL` or `DIFFERENT`. |

### sass-summary.tsv

The summary of the SASS compare of `libggml-cuda.so` of build `L` (A) and of build `C` (B), for `sm_86`. It has 15 rows. The script copies the `summary` object of `raw/sass-compare/sass-compare.json`. The [harness document](../../harness/README.md) gives the meaning of each key.

### sass-names-changed.tsv

The names that are not the same in the two builds. It has 6 rows, one for each new name.

| Column | Unit | Meaning |
|---|---|---|
| `name` | None | The full name of the function, with its template and its parameter types |
| `change` | None | `removed`, `added` or `different`. All 6 rows have `added`. |
| `function` | None | The name of the function without its template |
| `d_conv` | None | The value of `d_conv` in the name |
| `registers`, `stack_bytes`, `shared_bytes`, `local_bytes` | registers, bytes | The values of `cuobjdump -res-usage` |

### sass-functions.tsv

The names of `raw/sass-compare/key-kernels.tsv` in groups. One row for each function name without its template. The names are in the two builds.

The columns are the same as in the data set `2026-10-31-lead-30489`. The file has 89 of the 6,766 names that are in the two builds. The other names are not in the file. Their SASS is in the numbers of `sass-summary.tsv`.

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0005-concat-conv/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folders of the schedules. The table gives the items that we did not add.

| Files | Cause |
|---|---|
| The results folder of the schedule `c0005-nsys.txt` | The two runs did not start. The wrapper of `nsys` must have a folder with the name `target-linux-x64`, and the tool folder of the harness has a different name. The files have no measurement. |
| The two output files of the test program, 527 MB each | They are large. The file `bitcheck/compare.txt` gives their MD5 values. |
| The cubins, the libraries and the files of the build | They are binary files or large files. The manifests give the SHA-256 value of each library. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 8 items in `raw/`. All 8 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 8 files `c0005-bench-b<N>-<L or C>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The 8 items are not in the file `tools/publish/gitleaks.ignore`. The owner of the project adds them to the file. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
