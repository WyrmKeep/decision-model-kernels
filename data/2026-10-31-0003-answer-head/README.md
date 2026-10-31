# Data set: change 0003, the output head for the answer tokens only (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of change 0003. The change calculates the output head for the answer tokens only. The [results entry](../../docs/notebook/2026-10-31-0003-answer-head-result.md) uses these data. The [change document](../../docs/changes/0003-answer-head.md) gives the change. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build with change 0003 (arm `A3`) faster than the parent build (arm `L`) for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: The change is in the equivalence class `exact`. Does arm `A3` give the same answers as arm `L`, bit for bit?
3. Tests: Do `test-mul-mat-rows`, `test-backend-ops` and `compute-sanitizer` find no error?
4. Memory of the graph: Does the log of arm `A3` show no `graph reallocation` line in a full pass of the private benchmark?
5. Branch on the stack: Does the same change on top of `dmk/stack` (arm `A3R`) give the same answers as the stack (arm `S`)? Is it faster than `S`? The test has 4 blocks of the private benchmark.
6. Last commit: Do the unit tests pass on the last commit `419b8f612` (arm `F`)? Does `F` give the same answers as `S`? Does the server give correct responses to decision requests while it gets text requests at the same time?

We committed the [prediction](../../docs/notebook/2026-10-31-0003-answer-head-prediction.md) on 2026-10-31 at 06:18:10 UTC, before the first run of the speed test. The prediction is for the questions 1 and 2. The questions 3 to 6 have no prediction.

## Dates

| Item | Value |
|---|---|
| First run of the speed test (arm `L` against arm `A3`) | 2026-10-31 10:32:12 UTC |
| Last run of the speed test | 2026-10-31 10:47:45 UTC |
| Measurement window of the speed test | 10:32 to 10:48 UTC |
| Correctness measurement windows 1 to 4 | 07:33 to 07:39, 07:59 to 08:01, 08:11 to 08:14, and 10:18 to 10:26 UTC |
| Measurement window 5 (branch on the stack, speed test, `compute-sanitizer`) | 10:48 to 11:09 UTC |
| Measurement window 6 (graph with the full head, one request at a time) | 11:34 to 11:40 UTC |
| Measurement window 7 (last commit: 2 passes of the private benchmark, 2 runs with decisions and text requests) | 20:25 to 20:30 UTC |

The project stopped the server in production for each measurement window. One run at a time used the GPU. The unit tests ran in a container on the same host, one at a time, in the time between the measurement windows.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv), [correctness-runs.tsv](tables/correctness-runs.tsv) and [runs-rebased.tsv](tables/runs-rebased.tsv).

## Project commit

The scripts of the measurement are the same as the files of the folder `harness/` of this repository, with 4 differences. The file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/` are not in the work copy that the measurement used. The measurement did not use them. We compared the two folders with `diff -rq` after the measurement.

The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository. Two scripts of the client are new: `raw/extra/conc_client.py` and `raw/extra/memcheck_client.py`.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `0003-bench-*`, `0003-rb-ab-*` | The private benchmark, with 100 decisions, for the speed test | No |
| `0003-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |
| `0003-corr-*`, `0003-r-bench-*`, `0003-lv-bench-*`, `0003-rb-bench-*` | One pass of the private benchmark, for the answers | No |
| `0003-np6-*`, `0003-fin-*`, `0003-lv-np6-*`, `0003-r-np6-*`, `0003-g-np6-*`, `0003-g-ngl32-*`, `0003-r-cpuhead-*`, `0003-rb-cpuhead-*`, `0003-*-memcheck-*` | The 48 cases of the private benchmark that `conc_client.py` and `memcheck_client.py` send | No |
| `0003-final-bench-*` | Two passes of the private benchmark with the flags of production, for the answers of `F` | No |
| `0003-final-mixed-*` | The 48 cases of the private benchmark one at a time, with 20 `POST /completion` requests and 5 `POST /v1/chat/completions` requests, that `mixed_client.py` sends | No |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `L` | Branch `dmk/stack` of the fork of llama.cpp of this project, with change 0001. It is build `lead-30489`. | `6b8b65e6672f07805000d3c9683b65c72cc62303`. The source had 0 changes. |
| `A3` | Branch `dmk/0003-answer-head` of the same fork | `f0a2033e7966e37093b0187743b0fab87720f4e2`. It is 3 commits on top of the commit of `L`. The source had 0 changes. |
| `S` | Branch `dmk/stack` of the same fork. It is build `stack-fbfd9cb`. | `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978`. The source had 0 changes. |
| `A3R` | Branch `dmk/0003-answer-head-rebased` of the same fork. The one commit of change 0003 is on top of the commit of `S`. | `ca69612da671ed8d185ce39dcf8dae0a1023a570`. The source had 0 changes. |
| `F` | Branch `dmk/0003-answer-head-rebased`, the head of the pull request. It is 2 commits on top of the commit of `A3R`: `c4175ae75` (the tests) and `419b8f612` (the server). | `419b8f612780dbe48e21ebb80e7eacc64e88995a`. The source had 0 changes. |

The correctness measurement windows 1 to 3 used two other builds of `A3`. Their commits are `c1d3d481b` (measurement window 1) and `4bd0c75b5` (measurement windows 2 and 3). The two builds had the same build folder as the last build, and the last build replaced them. Thus the data set has no manifest for them. [INFERENCE: the commit of each measurement window comes from the history of the Git branch and from the time of each build. The server logs do not show the commit.]

The 3 commits of `A3` are `c1d3d481b` (the change), `4bd0c75b5` (one log line for each path of the head) and `f0a2033e7` (the graph memory). The commit of `A3R` is one commit with the same changes as the 3 commits of `A3`. The commit changes 12 files, with 845 added lines and 18 removed lines. The files of ggml are `ggml.h`, `ggml.c` and `mmvq.cu`. The files in `src/` are `llama-graph.h`, `llama-graph.cpp`, `llama-context.h`, `llama-context.cpp` and `llama-ext.h`. The other files are `server-context.cpp` in `tools/server/`, and the tests `test-backend-ops.cpp`, `test-mul-mat-rows.cpp` and `CMakeLists.txt`.

The commit `c4175ae75` changes `test-backend-ops.cpp` only: the cases of `test_mul_mat_rows` have `n_ids` 10 or 33, and not `n_ids` 1, 10 or 33. The commit `419b8f612` changes `server-context.cpp` only (18 added lines and 11 removed lines). The server gives the list of tokens to `llama_context` only if each slot of the batch holds a decision. The build `0003-answer-head-rebased-tests` has the commit `c4175ae75`. We used it for tests and the data set has its manifest.

We made `A3` on 2026-10-31 at 08:26 UTC, and `S` at 08:41 UTC. We made `A3R` at 10:34 UTC, the tests build at 12:28 UTC and `F` at 19:46 UTC. The files `raw/build/MANIFEST-0003-answer-head.txt`, `raw/build/MANIFEST-stack-fbfd9cb.txt`, `raw/build/MANIFEST-0003-answer-head-rebased.txt`, `raw/build/MANIFEST-0003-answer-head-rebased-tests.txt` and `raw/build/MANIFEST-0003-answer-head-final.txt` give the facts of the builds and the SHA-256 value of each file. The manifest of `L` is in the data set [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md).

## Build flags

We made the builds with `harness/build/build.sh`. The manifests of `A3`, `S`, `A3R` and `F` show the same tools and the same `cmake` command. The manifest of the tests build `0003-answer-head-rebased-tests` has the same tools and the same `cmake` flags. Its build command has the target `test-backend-ops` only.

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
| Server container, `llama bench`, `test-mul-mat-rows`, `test-backend-ops` and `compute-sanitizer` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
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

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,875 MHz. The column `throttle_reasons_at_start` has the value `0x0000000000000001` for the first run of the private benchmark. The bit `0x01` is `GpuIdle` of the driver.

## Server flags

All runs of the speed test used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The correctness runs used these flags with some changes. The files `raw/0003-correctness*.txt` and `raw/0003-rebased.txt` show them: the flag `-lv 4`, the number of slots `-np 6`, the flag `-ngl 32`, or the flag `-ot`. The runs of `llama bench` used the flags in the section "Commands". The runs of the measurement window 7 (`raw/0003-final.txt`) used the flags of production, without a change.

## Host load

The host has 12 CPU threads. The host did other work during the schedule. The cause is not in the data.

| Runs | One-minute load average |
|---|---|
| Blocks 1 to 4 of the private benchmark (arm `L` against arm `A3`) | 1.80 to 15.64 |
| The 8 runs of `llama bench` | 9.84 to 15.80 |
| Blocks 1 to 4 of the speed test of the branch on the stack | 3.64 to 20.11 |
| The 2 passes of the private benchmark and the 2 mixed runs of the last commit (measurement window 7) | 1.38 to 2.92 |

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv), [runs-rebased.tsv](tables/runs-rebased.tsv) and [runs-final.tsv](tables/runs-final.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv), [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) and [speed-rebased-blocks.tsv](tables/speed-rebased-blocks.tsv) show it for each block.

## Commands

The schedules for the speed tests came from `make_schedule.py`. The files [raw/0003-answer-head.txt](raw/0003-answer-head.txt) and [raw/0003-rebased.txt](raw/0003-rebased.txt) have the lines. We wrote the correctness schedules without the script. They are the files `raw/0003-correctness*.txt` and the first part of `raw/0003-rebased.txt`. We also wrote [raw/0003-final.txt](raw/0003-final.txt) (measurement window 7) without the script. It has 2 `bench` lines and 2 `client` lines.

```
python3 harness/make_schedule.py --prefix 0003-bench --blocks 4 --seed 3001 \
    --arm L=build:lead-30489 --arm A3=build:0003-answer-head --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix 0003-llb --blocks 4 --seed 3002 \
    --arm L=build:lead-30489 --arm A3=build:0003-answer-head --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
python3 harness/make_schedule.py --prefix 0003-rb-ab --blocks 4 --seed 3003 \
    --arm S=build:stack-fbfd9cb --arm A3R=build:0003-answer-head-rebased --kind bench \
    --args "--write-reference {name}"
```

The script of the measurement window stopped the server in production, ran the schedule, and started the server in production again. The harness ran the schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tables came from these commands. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix 0003-bench --a L --b A3
python3 ../../harness/analysis/ab_stats.py raw --prefix 0003-llb --a L --b A3
python3 ../../harness/analysis/ab_stats.py raw --prefix 0003-rb-ab --a S --b A3R
python3 ../../harness/analysis/answers_compare.py --a 'raw/0003-bench-b*-L.reference.json' \
    --b 'raw/0003-bench-b*-A3.reference.json'
python3 ../../harness/analysis/answers_compare.py --a 'raw/0003-final-bench-F-*.reference.json' \
    --b 'raw/0003-rb-ab-b*-S.reference.json'
python3 data/2026-10-31-0003-answer-head/make_tables.py
```

The unit tests ran in the server container with these commands. `build/bin` is the folder of the build.

```
test-mul-mat-rows
test-mul-mat-rows --full
test-mul-mat-rows --all
test-backend-ops test -b CUDA0 -o MUL_MAT_ID
test-backend-ops test -b CUDA0 -o MUL_MAT
```

The tests of the last commit `419b8f612` also used these two commands. The flag `-o MUL_MAT_ROWS` selects no test (the logs have `0/0 tests passed`), and the flag `-p same_rows` selects the cases of `test_mul_mat_rows`. The data set has the logs of the two flags.

```
test-backend-ops test -b CUDA0 -p same_rows
test-backend-ops test -b CUDA0 -o MUL_MAT_ROWS
```

The runs `0003-final-mixed-*` ran the client `raw/extra/mixed_client.py`. The client sends the 48 cases of the private benchmark one at a time in one thread. In 4 other threads it sends 20 `POST /completion` requests (`n_predict` 16, 5 requests in each thread). In one more thread it sends 5 `POST /v1/chat/completions` requests (`max_tokens` 16). Each of these 5 threads waits 0.5 s after each request. The client gives the exit code 1 if one request does not give `HTTP 200`, or has no text.

The server of the runs with `compute-sanitizer` ran with the script `raw/extra/san-wrap-0003.sh`. The script starts `compute-sanitizer --tool memcheck` and stops the server only when it gets `TERM`. The container sends `TERM` at `docker stop`. Then `compute-sanitizer` writes its summary.

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `L` | `build:lead-30489` | The parent build: the branch `dmk/stack` with change 0001 |
| `A3` | `build:0003-answer-head` | `L` with change 0003 |
| `S` | `build:stack-fbfd9cb` | The branch `dmk/stack` at its last commit, with the changes 0001, 0002, 0004, 0005 and 0007 |
| `A3R` | `build:0003-answer-head-rebased` | `S` with change 0003 |
| `F` | `build:0003-answer-head-final` | `A3R` with the commits `c4175ae75` (the tests) and `419b8f612` (the server gives the list of tokens only to a batch of decisions) |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. The files `raw/0003-answer-head.txt` and `raw/0003-rebased.txt` show the sequence.

## Raw files

The folder `raw/` has 638 files and 3 folders. The harness wrote 630 of the files, and they have no change. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `0003-bench-b<N>-<L or A3>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark of the speed test, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `0003-llb-b<N>-<L or A3>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `0003-rb-ab-b<N>-<S or A3R>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark of the speed test of the branch on the stack, with the same 8 files for each run |
| `0003-corr-*`, `0003-lv-bench-*`, `0003-r-bench-*`, `0003-rb-bench-*` | 112 | `run_bench.sh` | 14 runs with one pass of the private benchmark, with the same 8 files for each run |
| `0003-np6-*`, `0003-fin-*`, `0003-lv-*`, `0003-memcheck-*`, `0003-r-np6-*`, `0003-r-cpuhead-*`, `0003-r-memcheck-*`, `0003-rb-cpuhead-*`, `0003-rb-memcheck-*`, `0003-g-*` | 324 | `run_client.sh` | 41 client runs. 37 runs have 8 files: `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1`, `.answers.json`. The 4 runs `0003-np6-*` have 7 files and no `.answers.json` file. |
| `0003-final-bench-F-<N>.*` | 16 | `run_bench.sh` | 2 passes of the private benchmark of the last commit, with the same 8 files for each run |
| `0003-final-mixed-<F or A3R>.*` | 18 | `run_client.sh` | 2 client runs with decisions and text requests. Each run has 9 files: `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1`, `.answers.json` (the decision answers), `.others.json` (the `HTTP` code, the text and the time of each text request, and the time of each decision request) |
| `0003-answer-head.txt`, `0003-correctness*.txt`, `0003-rebased.txt`, `0003-final.txt` | 8 | `make_schedule.py` and a person | The schedules. `0003-answer-head.txt` has the 16 lines of the speed test of `A3`. The part 4 of `0003-rebased.txt` has the 8 lines of the speed test of `A3R`. The other schedules came from a person. |
| `build/MANIFEST-<arm>.txt` | 5 | `build.sh` | The facts of the builds `A3`, `S`, `A3R`, `F` and the tests build (`c4175ae75`): the commit, the tools, the `cmake` command and the SHA-256 value of each file |
| `extra/conc_client.py`, `extra/memcheck_client.py`, `extra/mixed_client.py`, `extra/san-wrap-0003.sh` | 4 | A person | The three client scripts and the wrapper of `compute-sanitizer` |
| `unit-tests/<commit>/*` | 54 | `test-mul-mat-rows`, `test-backend-ops` | The logs of the unit tests and a `summary.txt` file for each commit. The commit `c1d3d481b` also has the log of the first build, before the repair of `mmvq.cu`. The folder `ca69612da/repeat` has 21 files: the 20 runs of the cases with `n_ids` 1 and 10 that we ran again, and a `summary.txt` file. The commit `419b8f612` has 11 files. |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and some tables are JSON files. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

### runs.tsv

One row for each run of the speed test (arm `L` against arm `A3`), in the sequence of the start time. It has 16 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark` or `llama bench` |
| `line_type` | None | The type of the schedule line: `bench` or `tool` |
| `block` | None | The number of the block |
| `arm_tag` | None | `L` or `A3` |
| `arm` | None | `build:lead-30489` or `build:0003-answer-head` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000000` means that no cause was active. The value `0x0000000000000001` is the bit `GpuIdle`. Empty for `tool` runs. |
| `exit_code` | None | The exit code of the program. Only for `tool` runs. |
| `lat_ms` | ms | The latency of the run: the mean of the latencies of the timed passes. Only for benchmark runs. |
| `short_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the short requests, the medium requests, or the long requests in all timed passes |
| `rps8` | requests/s | The number of requests each second with 8 requests at the same time |
| `pp64_ts`, `pp256_ts`, `pp3000_ts` | tokens/s | The prompt speed of `llama bench` for a prompt of 64, 256 or 3,000 tokens. Only for `llama bench` runs. |
| `sm_clock_mhz_median` | MHz | The median of the SM clock in the samples of the run |
| `power_draw_w_median` | W | The median of `power.draw` in the samples of the run |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run |

Source: the `.meta`, `.load0`, `.load1`, `.smi`, `.json` and `.out` files of each run.

### speed-bench.tsv

The ratio of arm `A3` to arm `L` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `L` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `A3` |
| `diff_pct` | % | 100 times (the mean of the ratios of `A3` to `L` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

### speed-llama-bench.tsv

The ratio of arm `A3` to arm `L` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The columns are the same as the columns of `speed-bench.tsv`. The `metric` is `pp64_ts`, `pp256_ts` or `pp3000_ts`. The unit of `a_mean` and `b_mean` is tokens/s.

### speed-bench-blocks.tsv

The difference of `A3` from `L` in each block of the private benchmark, and the host load. It has 4 rows. The script calculates 100 times (the value of `A3` divided by the value of `L`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_l_before`, `load1_l_after` | None | The one-minute load average before and after the run of `L` |
| `load1_a3_before`, `load1_a3_after` | None | The one-minute load average before and after the run of `A3` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `A3` from `L` |
| `rps8_diff_pct` | % | The difference of `rps8` of `A3` from `L` |

### speed-llama-bench-blocks.tsv

The difference of `A3` from `L` in each block of `llama bench`, and the host load. It has 4 rows. The columns `block`, `first_arm` and the four load columns are the same as in `speed-bench-blocks.tsv`. The columns `pp64_ts_diff_pct`, `pp256_ts_diff_pct` and `pp3000_ts_diff_pct` give 100 times (the prompt speed of `A3` divided by the prompt speed of `L`, minus 1).

### quality-bench.tsv

The quality numbers of the 8 runs of the private benchmark of the speed test. One row for each run.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `block`, `arm_tag` | None | The name of the run, its block, and its arm |
| `correct` | decisions | The number of decisions where the top answer is the label |
| `decisions` | decisions | The number of decisions |
| `agree` | decisions | The number of decisions where the top answer is the top answer of the reference |
| `mean_dp` | probability | The mean probability difference from the reference, for the 100 decisions |
| `max_dp` | probability | The largest probability difference from the reference for one decision |
| `top_answer_differs_from_reference` | decisions | The number of decisions with a top answer that is not the top answer of the reference |
| `errors` | requests | The number of requests with an error |
| `guards_true` | None | `True` if all guards of the benchmark are true |

### answers-compare-bench.json

The output of `harness/analysis/answers_compare.py` for the 8 answer files `0003-bench-b*-L.reference.json` and `0003-bench-b*-A3.reference.json`. It is a JSON file, not a TSV file. The [method](../../docs/method.md) gives the fields.

### answers-compare-window1.json, answers-compare-window4.json

The same output for the answer files of the correctness measurement window 1 (2 runs of each arm, `0003-corr-*`) and of the correctness measurement window 4 (2 runs of each arm, `0003-r-bench-*`).

### answers-compare-rebased-bench.json, answers-compare-rebased-ab.json

The same output for arm `S` against arm `A3R`. The first file is for the 4 runs of the measurement window 5 with the log level 4 (`0003-rb-bench-*`). The second file is for the 8 runs of the speed test of the branch on the stack (`0003-rb-ab-*`).

### answers-compare-final.json

The same output for the 2 answer files of `F` against the 4 answer files of arm `S`. The files of `F` are `0003-final-bench-F-*`, and the files of `S` are `0003-rb-ab-b*-S` (the speed test of the branch on the stack). The two groups of runs came from different measurement windows.

### correctness-runs.tsv

One row for each run of the correctness runs and of the measurement windows 5, 6 and 7. The rows are in the sequence of the start time. The table has 67 rows.

| Column | Unit | Meaning |
|---|---|---|
| `window` | None | The number of the measurement window: 1 to 4 (correctness), 5 (branch on the stack), 6 (the graph with the full head), or 7 (the last commit) |
| `run` | None | The name of the run |
| `line_type` | None | `bench` or `client` |
| `arm_tag` | None | `L`, `A3`, `S`, `A3R` or `F` |
| `arm_commit` | None | The first 9 characters of the commit of the arm in this measurement window |
| `slots` | slots | The number after `-np` |
| `log_level` | None | `4` if the flags have `-lv 4`, and `3` if they do not |
| `wrapper` | None | `memcheck` if the server ran in `compute-sanitizer`, and `none` if not |
| `tensor_override` | None | The text after `-ot`, or `none` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `result` | None | `answers saved`, or a text that shows that the run has no answers |
| `sched_reserve_lines` | lines | The number of lines `sched_reserve: reserving` in the server log |
| `graph_reallocation_lines` | lines | The number of lines `graph reallocation` in the server log. Only for runs with the log level 4. |
| `head_rows_info_lines` | lines | The number of lines of the head with the rows of the tokens (`the head reads the rows of`). Only runs with the log level 4 have them. |
| `head_full_info_lines` | lines | The number of lines of the full head (`the full head, then the values of`). Only runs with the log level 4 have them. |
| `memcheck_error_summary` | errors | The number in `ERROR SUMMARY` of `compute-sanitizer`. Only for runs with the wrapper that writes the summary. |

### answers-client-runs.tsv

The answers of two client runs, compared. One row for each pair. The table has 49 rows. Each client run sends the 48 cases of the private benchmark. The runs `0003-final-mixed-*` also send text requests. The table does not have these requests.

| Column | Unit | Meaning |
|---|---|---|
| `run_a`, `run_b` | None | The names of the two runs |
| `cases` | cases | The number of cases of `run_a` |
| `cases_identical` | cases | The number of cases with the same response in the two runs |
| `cases_different` | cases | The number of other cases |
| `max_abs_difference` | probability | The largest difference of a number in the answers of the cases |
| `requests_with_error_in_a` | requests | The number of requests of `run_a` that did not give `HTTP 200` |

### memcheck-requests.tsv

The requests of the runs with `compute-sanitizer`. One row for each request. It has 20 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `request_type` | None | `short1q`, `short2q`, `medium` or `long` |
| `case` | None | The case of the benchmark that the request sent |
| `http_status` | None | The `HTTP` code of the response |
| `memcheck_error_summary` | errors | The number in `ERROR SUMMARY`. Empty if the run has no summary. |

### unit-tests.tsv

The results of the unit tests. One row for each log. It has 27 rows.

| Column | Unit | Meaning |
|---|---|---|
| `build` | None | The commit of the build. The first row is the first build, before the repair of `mmvq.cu`. |
| `log` | None | The name of the log file in `raw/unit-tests/` |
| `command` | None | The command |
| `exit_code` | None | The exit code. `139` shows that the program stopped with `GGML_ASSERT`. |
| `cases`, `cases_with_different_values`, `cases_failed` | cases | The numbers that `test-mul-mat-rows` writes at its end. Only for this program. |
| `tests_passed` | tests | The number of tests that pass in `test-backend-ops`. Only for this program. |
| `assert_in_log` | None | `yes` if the log has the text `GGML_ASSERT` |

The `command` of a row can be `-p same_rows` or `-o MUL_MAT_ROWS`. The first selects the 121 cases of `test_mul_mat_rows` and `test-backend-ops` (the cases with the name `CONT`). The second selects no test: its rows have `0/0` in `tests_passed`.

### rows-cases.tsv

One row for each case in each log of the runs with `-p same_rows` (the 3 builds `f0a2033e7`, `ca69612da` and `419b8f612`, and the 20 runs of `ca69612da` that we ran again). It has 1,917 rows.

| Column | Unit | Meaning |
|---|---|---|
| `group` | None | The group of runs |
| `folder`, `log` | None | The folder in `raw/unit-tests/` and the name of the log file |
| `case` | None | The name of the case in `test-backend-ops` |
| `type_a` | None | The type of the weight |
| `n_rows`, `k`, `n_cols`, `n_ids` | None | The dimensions of the case: the rows of the weight, the length of a row, the columns of the other operand, and the rows of `ids` |
| `n_values` | values | The number of values of the result: `n_cols` times `n_ids` |
| `same_rows` | None | `1` if all columns use the same rows, and `0` if not |
| `result` | None | `OK` or `FAIL` |
| `nmse` | None | The NMSE of a case with the result `FAIL`. Empty for the other cases. |

### rows-by-ids.tsv

The number of cases, and of cases with the result `FAIL`, of `rows-cases.tsv` for each group and each number of `n_ids`. It has 9 rows. The columns are `group`, `n_ids`, `case_results`, `failed` and `failed_percent`.

### speed-rebased.tsv

The ratio of arm `A3R` to arm `S` for the speed of the private benchmark. It has 5 rows. The columns are the same as the columns of `speed-bench.tsv`. `a_mean` is for arm `S` and `b_mean` is for arm `A3R`.

### speed-rebased-blocks.tsv

The difference of `A3R` from `S` in each block, and the host load. It has 4 rows. The columns are the same as in `speed-bench-blocks.tsv`. The columns with `s` in the name are for arm `S`, and the columns with `a3r` are for arm `A3R`.

### runs-rebased.tsv

One row for each of the 8 runs of the speed test of the branch on the stack. The rows are in the sequence of the name of the run. The columns are `run`, `block`, `arm_tag`, `start_utc`, `load1_before`, `load1_after`, `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, `rps8`, `sm_clock_mhz_median` and `power_draw_w_median`. Their meaning is the same as in `runs.tsv`.

### runs-final.tsv

One row for each of the 2 passes of the private benchmark of the last commit (measurement window 7). The columns are `run`, `arm_tag`, `start_utc`, `load1_before`, `load1_after`, `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, `rps8`, `sm_clock_mhz_median` and `power_draw_w_median`. Their meaning is the same as in `runs.tsv`. The table also has the columns `decisions_correct`, `decisions_total` (the numbers of the quality phase), `errors` (requests with an error) and `guards_true`. The two runs are not a test of two arms. No run of `S` ran in this measurement window.

### mixed-load.tsv

The two runs `0003-final-mixed-F` and `0003-final-mixed-A3R`. It has 6 rows, 3 for each run: one row for each request type.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `arm_tag`, `arm_commit` | None | The name of the run, its arm and the commit of the arm |
| `request_type` | None | `decision` (the 48 cases, `POST /v1/systemone`), `completion` (`POST /completion`) or `chat` (`POST /v1/chat/completions`) |
| `requests` | requests | The number of requests of this type |
| `http_200` | requests | The number of requests with `HTTP 200` |
| `status_counts` | requests | The `HTTP` codes and their numbers, for example `501:20` |
| `requests_with_text` | requests | The number of requests with a text that is not empty. `not used` for the decisions. |
| `decision_requests_with_another_request_in_flight` | requests | The number of decision requests that ran at the same time as a `POST /completion` request or a `POST /v1/chat/completions` request. Only for the row of the decisions. |
| `server_starts_in_log` | starts | The number of lines `model loaded` in the server log. The value 1 shows that the server did not start again. Only for the row of the decisions. |
| `decision_cases_same_as_s_one_at_a_time` | cases | The number of decision cases with the same response as in the run `0003-rb-cpuhead-S-1` (arm `S`, one request at a time). Only for the row of the decisions. |
| `error_message` | None | The text of the error of the first request that did not give `HTTP 200` |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0003-answer-head/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Facts about the runs

### The flag `-ot`

The runs `0003-r-cpuhead-*` used the flag `-ot output.weight=CPU`. The file `raw/0003-correctness4.txt` has a comment on these runs. The comment tells that the weight of the head is in the CPU buffer. This is not correct. The server logs show that the flag moved 8 other weights to the CPU. They have 8 more CPU buffers of 10.62 MiB, and the buffer of `CUDA0` is 85 MiB less.

The log of arm `A3` has the line of the head with the rows of the tokens, thus the head stayed on the GPU. [INFERENCE: the model has one tensor for the input embedding and the head. The text after `-ot` selects 8 other weights and not the copy of this tensor on the GPU.]

The runs `0003-rb-cpuhead-*` used the flag `-ot ^output[.]weight=CPU`. The buffers in their logs are the same as the buffers of runs with no flag. The comment of `raw/0003-rebased.txt` has the same text as `raw/0003-correctness4.txt`, and it is also not correct.

The runs `0003-g-ngl32-*` keep the output layer on the CPU with the flag `-ngl 32`. The logs of these runs show the buffers.

### The runs of the measurement window 1 with 6 slots

The 4 runs `0003-np6-*` have no answers. The client script had the name `concurrent.py`, and this name is the name of a file of the Python library. The client stopped with an error. We changed the name of the script to `conc_client.py` and ran the tests again in the measurement windows 2, 3 and 4. The 4 runs have no `.answers.json` file.

### The sequence of the correctness runs

The measurement windows 1 to 4 came before the speed test. The measurement windows 5, 6 and 7 came after it. The measurement window 4 used the same commit as the speed test.

### The text requests of the measurement window 7

The model gives no text. The server answers `POST /completion` and `POST /v1/chat/completions` with `HTTP 501` and the text `This model is a decision model without text generation, use /v1/systemone`. The code is in `process_single_task` of `tools/server/server-context.cpp`: a task that samples tokens gets this error if the output of the model has no text. The output of the decision type `h2o-lightning` is `decisions` only.

The 20 `POST /completion` requests and the 5 `POST /v1/chat/completions` requests of each run `0003-final-mixed-*` all got `HTTP 501`. Their tasks did not get a slot. The 48 decision requests of each run got `HTTP 200` with the same answers as arm `S`. The server log of each run has one start of the server.

The two runs thus did not have a batch with a slot that is not a decision. They do not show the fault that the commit `419b8f612` repairs. The two builds `F` and `A3R` gave the same result.

### Names of the runs

The host keeps the folder of each run by the name of the run. Thus each new run of a test has a new name.

### Log level

The server writes the lines of the library, for example the lines `graph reallocation`, at the log level 4 only. The runs with the flag `-lv 4` have them. The runs of the speed test used the flags of the server in production, without `-lv 4`. Their logs have no such lines.

## Files not added

We added all files of the results folders of the schedules. We did not add these files.

| Files | Cause |
|---|---|
| The logs of the measurement windows | They have names of hosts and of programs of the project. They are not data of the runs. |
| The programs and the libraries of the builds | They are binary files. The manifests give the SHA-256 value of each file. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 32 items in `raw/`. All 32 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 32 files `<run>.json` of the runs of the private benchmark. They are the runs with a `.reference.json` file. | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 32 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
