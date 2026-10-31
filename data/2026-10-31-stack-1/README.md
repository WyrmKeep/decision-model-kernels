# Data set: the first test of the stack (2026-10-31)

This data set holds the raw files and the derived tables of the first test of the stack. The stack is the branch `dmk/stack` of the fork at `fbfd9cb8c`. It has the changes 0001, 0007, 0002, 0004 and 0005. The test compares the build of the stack (arm `S`) with build `D`, the baseline of the project. The [results entry](../../docs/notebook/2026-10-31-stack-1-result.md) uses these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build of the stack (arm `S`) faster than build `D` for the model "mix8"? Is each result in its range in the prediction? The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: Each change in the stack is in the equivalence class `exact`. Does arm `S` give the same answers as arm `D`, bit for bit?
3. Guards: Are all guards of the private benchmark true in all 12 runs?

We committed the [prediction](../../docs/notebook/2026-10-31-stack-1-prediction.md) on 2026-10-31 at 08:38 UTC, before the first run at 08:43 UTC.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 08:43:08 UTC |
| Last run of the private benchmark starts | 2026-10-31 09:02:45 UTC |
| Last run of `llama bench` starts | 2026-10-31 09:05:37 UTC |
| Measurement window | 08:42:52 to 09:06:05 UTC |

The 12 runs of the private benchmark used about 21 minutes. The 8 runs of `llama bench` used about 1 minute. The project stopped the server in production for the measurement window. One run at a time used the GPU.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv). The times of the measurement window are from its log. We did not add that log (see "Files not added").

## Project commit

The commit of this repository before this data set is `d4a57934cd72bbdf9a7e81c5f6680973ce59351a`. It has the prediction.

The scripts of the measurement are the same as the files of the folder `harness/` of this repository, with 4 differences. The file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/` are not in the work copy that the measurement used. The measurement did not use them. We compared the two folders with `diff -rq` after the measurement.

The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `stack1-bench-*` | The private benchmark, with 100 decisions | No |
| `stack1-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `D` | Tag `b12004` of llama.cpp, branch `master`, with no change | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |
| `S` | Branch `dmk/stack` of the fork of llama.cpp of this project | `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978`. The source had 0 changes. |

The commit of `S` is 6 commits after tag `b12004`. The table gives them in the sequence of the stack.

| Commit | Change | Title of the commit |
|---|---|---|
| `6b8b65e66` | 0001 | `ggml-cuda: fuse GLU into the MMQ Q8_1 activation quantization` |
| `c3ec5653b` | 0007 | `ggml-backend: do not copy the inputs that no computed node reads` |
| `372b3fece` | 0002 | `tests: add perf cases for the GATED_DELTA_NET shape of the MMA path` |
| `0657e8177` | 0002 | `ggml-cuda: GDN MMA: record why the whole-head path stays on a GPU with 38 SMs` |
| `5d61f3ca3` | 0004 | `Make the KV cache attention mask on the GPU` |
| `fbfd9cb8c` | 0005 | `CUDA: fuse the conv input concat, the conv state copy and ssm_conv + silu of the Gated DeltaNet layers` |

The difference of `S` from `D` is 28 files, with 1,728 added lines and 75 removed lines. We made `D` on 2026-10-30 at 21:50 UTC and `S` on 2026-10-31 at 08:41 UTC. The files `raw/build/MANIFEST-b12004-sm86.txt` and `raw/build/MANIFEST-stack-fbfd9cb.txt` give the facts of the builds and the SHA-256 value of each file.

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
| Server container and `llama bench` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
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

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,860 MHz for the private benchmark and 1,860 to 1,905 MHz for `llama bench`. The column `throttle_reasons_at_start` has the value `0x0000000000000001` for the first run of the private benchmark and `0x0000000000000000` for the 11 other runs. The bit `0x01` is `GpuIdle` of the driver.

## Server flags

All runs of the private benchmark used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The host did other work during the schedule. The cause is not in the data.

| Runs | One-minute load average |
|---|---|
| Block 1 of the private benchmark | 1.22 to 15.37 |
| Blocks 2 to 4 of the private benchmark | 10.79 to 18.08 |
| Block 5 of the private benchmark | 3.11 to 18.08 |
| Block 6 of the private benchmark | 2.03 to 3.11 |
| The 8 runs of `llama bench` | 1.71 to 1.97 |

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules came from `make_schedule.py`. The file [raw/stack1-bench.txt](raw/stack1-bench.txt) and the file [raw/stack1-llb.txt](raw/stack1-llb.txt) have the lines. The prediction has the same two commands.

```
python3 harness/make_schedule.py --prefix stack1-bench --blocks 6 --seed 61001 \
    --arm D=build:b12004-sm86 --arm S=build:stack-fbfd9cb --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix stack1-llb --blocks 4 --seed 61002 \
    --arm D=build:b12004-sm86 --arm S=build:stack-fbfd9cb --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The script of the measurement window stopped the server in production, ran the two schedules, and started the server in production again. The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tables came from these commands. Arm `D` is the reference arm, and the ratio is `S` divided by `D`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix stack1-bench --a D --b S --margin 2
python3 ../../harness/analysis/ab_stats.py raw --prefix stack1-llb --a D --b S --margin 2
python3 ../../harness/analysis/answers_compare.py --a 'raw/stack1-bench-b*-D.reference.json' \
    --b 'raw/stack1-bench-b*-S.reference.json'
python3 data/2026-10-31-stack-1/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `D` | `build:b12004-sm86` | Tag `b12004` of llama.cpp, built from source, in the image of the stock build |
| `S` | `build:stack-fbfd9cb` | The branch `dmk/stack` at `fbfd9cb8c`, in the same image |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In the private benchmark, arm `S` ran first in blocks 1, 4 and 6, and arm `D` ran first in blocks 2, 3 and 5. In `llama bench`, arm `D` ran first in blocks 1 and 3, and arm `S` ran first in blocks 2 and 4.

## Raw files

The folder `raw/` has 120 files. The harness wrote 118 of them, and they have no change. A person added the two manifests of the builds. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `stack1-bench-b<N>-<D or S>.*` | 84 | `run_bench.sh` | 12 runs of the private benchmark, with 7 files each: `.json`, `.reference.json`, `.server.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `stack1-llb-b<N>-<D or S>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `stack1-bench.txt`, `stack1-llb.txt` | 2 | `make_schedule.py` | The schedules, with a comment line |
| `build/MANIFEST-<arm>.txt` | 2 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

The tables `runs.tsv`, `speed-bench.tsv`, `speed-llama-bench.tsv`, `speed-bench-blocks.tsv`, `speed-llama-bench-blocks.tsv`, `quality-bench.tsv` and `answers-compare-bench.json` have the columns of the same tables in the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). Arm `S` replaces arm `L`. In the tables of the blocks, the columns `load1_d_*` are for arm `D` and the columns `load1_s_*` are for arm `S`. The table `runs.tsv` has 20 rows, `speed-bench-blocks.tsv` has 6 rows and `quality-bench.tsv` has 12 rows.

The script reads the tables `speed-bench.tsv` and `speed-llama-bench.tsv` of the data sets of the four changes. The table `change-intervals.tsv` lists them. The script writes the other tables only from the files in `raw/`.

### guards-bench.tsv

The guards of the 12 runs of the private benchmark. One row for each run. The [method](../../docs/method.md) gives the guards.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `block`, `arm_tag` | None | The name of the run, its block, and its arm |
| `guard_errors` | None | `True` if no request has an error |
| `guard_correct`, `guard_agree`, `guard_mean_dp` | None | `True` if the number is in the limit of the guard, with the baseline of `bench/baseline.json` |
| `guards_true` | None | `True` if the four guards are true |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run. The limit of the VRAM guard is 6,144 MiB. |

### prediction-vs-result.tsv

The result of each metric against the prediction. It has 8 rows. The script has the values of the prediction as a list. They are the values of the table "Predicted gain" of the [prediction entry](../../docs/notebook/2026-10-31-stack-1-prediction.md).

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, `rps8`, `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `predicted_pct` | % | The predicted change of the prediction entry |
| `range_min_pct`, `range_max_pct` | % | The lowest end and the highest end of the range of the prediction entry. The range is the predicted change ±1.5 points. |
| `result_pct`, `ci_lo_pct`, `ci_hi_pct` | % | The `diff_pct`, `ci_lo_pct` and `ci_hi_pct` of `speed-bench.tsv` or `speed-llama-bench.tsv` |
| `result_minus_predicted_points` | points of percent | The result minus the predicted change |
| `in_range` | None | `True` if the result is in the range |
| `position` | None | `inside`, `better` or `worse`. For the metrics in ms, a value that is less than the range is `better`. For `rps8` and the prompt speed, a value that is more than the range is `better`. |

### change-intervals.tsv

The result of each of the four changes for each metric of the prediction. It has 32 rows. The script reads them from the tables `speed-bench.tsv` and `speed-llama-bench.tsv` in the folders of the four data sets. The change 0002 has no row, because it does not change the program.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | The metric, as in `prediction-vs-result.tsv` |
| `change` | None | The number of the change |
| `data_set` | None | The folder of the data set of the change |
| `compared_against` | None | The build that the test of the change compared against |
| `blocks` | blocks | The number of blocks of the test of the change |
| `a_mean`, `b_mean` | ms, requests/s or tokens/s | The means of the two arms of the test of the change |
| `diff_pct`, `ci_lo_pct`, `ci_hi_pct` | % | The change and its 95 % interval from the test of the change |
| `half_width_points` | points of percent | (`ci_hi_pct` minus `ci_lo_pct`) divided by 2 |

### stack-estimate.tsv

The estimate of the prediction entry, and the interval of the four tests together. It has 8 rows.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | The metric |
| `product_of_ratios` | None | The four numbers (1 plus `diff_pct` divided by 100), one for each change, multiplied. The prediction entry used this number. |
| `product_diff_pct` | % | 100 times (`product_of_ratios` minus 1). The script checks that it is in 0.06 of the predicted change. |
| `combined_half_width_points` | points of percent | The value of `sqrt(h1^2 + h2^2 + h3^2 + h4^2)`, where `h1` to `h4` are the four values of `half_width_points`. The method thinks that the four tests have no effect on each other. |
| `combined_end_a_pct`, `combined_end_b_pct` | % | `product_diff_pct` minus and plus `combined_half_width_points` |
| `result_pct` | % | The `diff_pct` of the test of the stack |
| `result_minus_product_points` | points of percent | `result_pct` minus `product_diff_pct` |
| `result_in_combined_interval` | None | `True` if `result_pct` is between the two ends |

### gain-ms.tsv

The gain in ms of each change and of the stack, for the four metrics of the private benchmark in ms. It has 4 rows. The gain is the mean of arm A minus the mean of arm B in the test. The means are the `a_mean` and `b_mean` of the tables `speed-bench.tsv`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms` or `long_ms` |
| `gain_0001_ms`, `gain_0007_ms`, `gain_0004_ms`, `gain_0005_ms` | ms | The gain of the change in the test of the change |
| `sum_of_gains_ms` | ms | The sum of the four gains |
| `gain_of_stack_ms` | ms | The gain in the test of the stack |
| `stack_minus_sum_ms` | ms | `gain_of_stack_ms` minus `sum_of_gains_ms` |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-stack-1/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folders of the two schedules, with these differences.

| Files | Cause |
|---|---|
| The 12 files `stack1-bench-b<N>-<D or S>.launch.log` | Each file has one line: the SHA-256 name of a container of the host. It has no measurement. |
| The log of the measurement window | It has the names of programs of the host. |
| The cubins, the libraries and the files of the build | They are binary files or large files. The manifests give the SHA-256 value of each library. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 12 items in `raw/`. All 12 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 12 files `stack1-bench-b<N>-<D or S>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 12 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
