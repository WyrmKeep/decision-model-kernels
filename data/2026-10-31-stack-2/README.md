# Data set: the second test of the stack (2026-10-31)

This data set holds the raw files and the derived tables of the second test of the stack. The stack is the branch `dmk/stack` of the fork at `419b8f612`. It has the changes 0001, 0007, 0002, 0004, 0005 and 0003. The test compares the build of the stack (arm `S2`) with the image that the server in production uses (arm `P`). The [results entry](../../docs/notebook/2026-10-31-stack-2-result.md) uses these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build of the stack (arm `S2`) faster than the image (arm `P`) for the model "mix8"? Is each result in its range in the prediction? The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: Each change in the stack is in the equivalence class `exact`. Does arm `S2` give the same answers as arm `P`, bit for bit?
3. Guards: Are all guards of the private benchmark true in all 12 runs?

We committed the [prediction](../../docs/notebook/2026-10-31-stack-2-prediction.md) on 2026-10-31 at 20:43:36 UTC, before the first run at 20:43:55 UTC. The first line of the prediction gives the time "about 20:45 UTC". This time is after the time of the commit. We did not change the file. The time of the commit is the true time.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 20:43:55 UTC |
| Last run of the private benchmark starts | 2026-10-31 21:03:10 UTC |
| Last run of `llama bench` starts | 2026-10-31 21:05:53 UTC |
| Measurement window | 20:43:37 to 21:06:23 UTC |

The 12 runs of the private benchmark used about 21 minutes. The 8 runs of `llama bench` used about 1 minute. The project stopped the server in production for the measurement window. One run at a time used the GPU.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv). The times of the measurement window are from its log. We did not add this log (see "Files not added").

## Project commit

The commit of this repository before this data set is `346f9833f613d48ba37937bc54c3b4b4057636e9`. It has the prediction.

The scripts of the measurement are the same as the files of the folder `harness/` of this repository, with 9 differences. The file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/` are not in the work copy that the measurement used. The files `harness/analysis/soak_check.py`, `harness/remote/run_soak.sh`, `harness/remote/san_client.py` and `harness/remote/san_wrap.sh` are also not in the work copy. The measurement did not use them.

The file `harness/analysis/answers_compare.py` is the ninth difference. The work copy has the same lines without the function `main()`. The version in this repository has the lines in `main()`, with 3 more lines. The tables of this data set came from the version in this repository. We compared the two folders with `diff -rq` after the measurement.

The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `stack2-bench-*` | The private benchmark, with 100 decisions | No |
| `stack2-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `P` | The image `server-cuda-b12004` of llama.cpp, as in production. The label `org.opencontainers.image.revision` of the image is the commit of tag `b12004` of llama.cpp. | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b` |
| `S2` | Branch `dmk/stack` of the fork of llama.cpp of this project | `419b8f612780dbe48e21ebb80e7eacc64e88995a`. The source had 0 changes. |

The commit of `S2` is 9 commits after tag `b12004`. The table gives them in the sequence of the stack.

| Commit | Change | Title of the commit |
|---|---|---|
| `6b8b65e66` | 0001 | `ggml-cuda: fuse GLU into the MMQ Q8_1 activation quantization` |
| `c3ec5653b` | 0007 | `ggml-backend: do not copy the inputs that no computed node reads` |
| `372b3fece` | 0002 | `tests: add perf cases for the GATED_DELTA_NET shape of the MMA path` |
| `0657e8177` | 0002 | `ggml-cuda: GDN MMA: record why the whole-head path stays on a GPU with 38 SMs` |
| `5d61f3ca3` | 0004 | `Make the KV cache attention mask on the GPU` |
| `fbfd9cb8c` | 0005 | `CUDA: fuse the conv input concat, the conv state copy and ssm_conv + silu of the Gated DeltaNet layers` |
| `ca69612da` | 0003 | `Calculate the output head for the answer tokens only` |
| `c4175ae75` | 0003 | `tests: use 10 or more ids in the NMSE cases of test_mul_mat_rows` |
| `419b8f612` | 0003 | `server: keep the full output head for batches that hold a non-decision slot` |

The first 6 commits are the stack of the [first test of the stack](../2026-10-31-stack-1/README.md). The 3 commits of change 0003 are on top of them. The difference of `S2` from the tag `b12004` is 34 files, with 2,582 added lines and 93 removed lines.

We made `S2` on 2026-10-31 at 19:46 UTC. The build folder is `stack-419b8f6`. The build is the same build as arm `F` of the data set [`2026-10-31-0003-answer-head`](../2026-10-31-0003-answer-head/README.md). The file `raw/build/MANIFEST-stack-419b8f6.txt` is the same file as `raw/build/MANIFEST-0003-answer-head-final.txt` of that data set, byte for byte.

Its first line has the name `0003-answer-head-final`, and its line `src branch` has the branch `dmk/0003-answer-head-rebased`. The manifest gives the facts of the build and the SHA-256 value of each file. Arm `P` has no manifest, because the programs are the programs of the image.

## Build flags

We made the build with `harness/build/build.sh`. The manifest shows the same tools and the same `cmake` command as the manifests of the first test of the stack. It has one more target: `test-mul-mat-rows`.

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

Arm `P` runs the programs in the image. Arm `S2` runs in the same image. The container puts the build in `/opt/dmk`. In each `.err` file of `llama bench`, the line `load_backend` gives the folder of the CUDA backend. It is `/app` for arm `P` and `/opt/dmk` for arm `S2`.

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

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,830 to 1,875 MHz for the private benchmark and 1,845 to 1,905 MHz for `llama bench`.

The column `throttle_reasons_at_start` has the value `0x0000000000000000` for all 12 runs of the private benchmark. The 8 runs of `llama bench` have no value in this column.

## Server flags

All runs of the private benchmark used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The host load was low in all runs. The cause of the load is not in the data.

| Runs | One-minute load average |
|---|---|
| Block 1 of the private benchmark | 1.14 to 2.34 |
| Block 2 of the private benchmark | 1.96 to 2.42 |
| Block 3 of the private benchmark | 1.68 to 2.42 |
| Block 4 of the private benchmark | 1.66 to 1.76 |
| Block 5 of the private benchmark | 1.66 to 2.58 |
| Block 6 of the private benchmark | 1.98 to 2.64 |
| The 8 runs of `llama bench` | 1.86 to 2.19 |

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules of the private benchmark and of `llama bench` came from `make_schedule.py`. The file [raw/stack2-bench.txt](raw/stack2-bench.txt) and the file [raw/stack2-llb.txt](raw/stack2-llb.txt) have the lines. The prediction has the same two commands.

```
python3 harness/make_schedule.py --prefix stack2-bench --blocks 6 --seed 62001 \
    --arm P=stock --arm S2=build:stack-419b8f6 --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix stack2-llb --blocks 4 --seed 62002 \
    --arm P=stock --arm S2=build:stack-419b8f6 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

A script of the measurement window stopped the server in production, ran the two schedules, and started the server in production again. The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tables came from these commands. Arm `P` is the reference arm, and the ratio is `S2` divided by `P`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix stack2-bench --a P --b S2 --margin 2
python3 ../../harness/analysis/ab_stats.py raw --prefix stack2-llb --a P --b S2 --margin 2
python3 ../../harness/analysis/answers_compare.py --a 'raw/stack2-bench-b*-P.reference.json' \
    --b 'raw/stack2-bench-b*-S2.reference.json'
python3 data/2026-10-31-stack-2/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `P` | `stock` | The programs of the image `server-cuda-b12004`, the image that the server in production uses |
| `S2` | `build:stack-419b8f6` | The branch `dmk/stack` at `419b8f612`, in the same image |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In the private benchmark, arm `P` ran first in blocks 1 and 6, and arm `S2` ran first in blocks 2, 3, 4 and 5. In `llama bench`, arm `P` ran first in block 4, and arm `S2` ran first in blocks 1, 2 and 3.

## Raw files

The folder `raw/` has 119 files. The harness wrote 116 of them, and they have no change. A person added the manifest of the build. The 2 schedule files are the other files. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `stack2-bench-b<N>-<P or S2>.*` | 84 | `run_bench.sh` | 12 runs of the private benchmark, with 7 files each: `.json`, `.reference.json`, `.server.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `stack2-llb-b<N>-<P or S2>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `stack2-bench.txt`, `stack2-llb.txt` | 2 | `make_schedule.py` | The schedules, with a comment line |
| `build/MANIFEST-stack-419b8f6.txt` | 1 | `build.sh` | The facts of the build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

The tables `runs.tsv`, `speed-bench.tsv`, `speed-llama-bench.tsv`, `speed-bench-blocks.tsv`, `speed-llama-bench-blocks.tsv`, `quality-bench.tsv` and `answers-compare-bench.json` have the columns of the same tables in the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). Arm `P` replaces arm `D`, and arm `S2` replaces arm `L`. In the tables of the blocks, the columns `load1_p_*` are for arm `P` and the columns `load1_s2_*` are for arm `S2`. The column `within_margin` of the speed tables is `False` in all rows. The test has no equivalence margin. The table `runs.tsv` has 20 rows, `speed-bench-blocks.tsv` has 6 rows and `quality-bench.tsv` has 12 rows.

The script reads the tables of the three tests of the estimate. The table `stack-estimate.tsv` lists them. The script writes the other tables only from the files in `raw/`.

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

The result of each metric against the prediction. It has 8 rows. The script has the values of the prediction as a list. They are the values of the table "Predicted gain" of the [prediction entry](../../docs/notebook/2026-10-31-stack-2-prediction.md).

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, `rps8`, `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `predicted_pct` | % | The predicted change of the prediction entry |
| `range_min_pct`, `range_max_pct` | % | The lowest end and the highest end of the range of the prediction entry. The range is the predicted change ±2.0 points. |
| `result_pct`, `ci_lo_pct`, `ci_hi_pct` | % | The `diff_pct`, `ci_lo_pct` and `ci_hi_pct` of `speed-bench.tsv` or `speed-llama-bench.tsv` |
| `result_minus_predicted_points` | points of percent | The result minus the predicted change |
| `in_range` | None | `True` if the result is in the range |
| `position` | None | `inside`, `better` or `worse`. For the metrics in ms, a value that is less than the range is `better`. For `rps8` and the prompt speed, a value that is more than the range is `better`. |

### stack-estimate.tsv

The estimate of the prediction entry, and the interval of the three tests together. It has 8 rows. The script reads the tables of the three tests.

| Test | Data set | Arm A | Arm B | Tables that the script reads |
|---|---|---|---|---|
| Phase 1 | [`2026-10-30-window1`](../2026-10-30-window1/README.md) | The stock image | Build `D` | `speed-bench.tsv`, `speed-llama-bench.tsv` |
| The first test of the stack | [`2026-10-31-stack-1`](../2026-10-31-stack-1/README.md) | Build `D` | The stack at `fbfd9cb8c` | `speed-bench.tsv`, `speed-llama-bench.tsv` |
| Change 0003 | [`2026-10-31-0003-answer-head`](../2026-10-31-0003-answer-head/README.md) | The stack at `fbfd9cb8c` (arm `S`) | The stack with change 0003 (arm `A3R`) | `speed-rebased.tsv` |

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | The metric, as in `prediction-vs-result.tsv` |
| `phase1_diff_pct`, `stack1_diff_pct`, `c0003_diff_pct` | % | The `diff_pct` of the three tests. The three `llama bench` rows have no value for change 0003, because the prediction used the ratio 1 for them. |
| `phase1_half_width_points`, `stack1_half_width_points`, `c0003_half_width_points` | points of percent | (`ci_hi_pct` minus `ci_lo_pct`) divided by 2 of the three tests |
| `product_of_ratios` | None | The numbers (1 plus `diff_pct` divided by 100) of the three tests, multiplied. The prediction entry used this number. |
| `product_diff_pct` | % | 100 times (`product_of_ratios` minus 1). The script checks that it is in 0.06 of the predicted change. |
| `combined_half_width_points` | points of percent | The value of `sqrt(h1^2 + h2^2 + h3^2)`, where `h1` to `h3` are the three values of `half_width_points`. For `llama bench`, the script uses the first two values only. The method thinks that the tests have no effect on each other. |
| `combined_end_a_pct`, `combined_end_b_pct` | % | `product_diff_pct` minus and plus `combined_half_width_points` |
| `result_pct` | % | The `diff_pct` of the test of the stack |
| `result_minus_product_points` | points of percent | `result_pct` minus `product_diff_pct` |
| `result_in_combined_interval` | None | `True` if `result_pct` is between the two ends |

### gain-ms.tsv

The gain in ms of the three tests and of the test of the stack. It has 4 rows, for the four metrics of the private benchmark in ms. The gain is the mean of arm A minus the mean of arm B in the test. The means are the `a_mean` and `b_mean` of the speed tables. The arms of the three tests are in the table of `stack-estimate.tsv` above.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms` or `long_ms` |
| `gain_phase1_ms`, `gain_stack1_ms`, `gain_0003_ms` | ms | The gain in phase 1, in the first test of the stack and in the test of change 0003 |
| `sum_of_gains_ms` | ms | The sum of the three gains |
| `gain_of_stack2_ms` | ms | The gain in the test of the stack (arm `P` minus arm `S2`) |
| `stack2_minus_sum_ms` | ms | `gain_of_stack2_ms` minus `sum_of_gains_ms` |

### comparison-stack-1.tsv

The test of the stack against the first test of the stack, and against the test of change 0003. It has 8 rows. The script reads the `diff_pct` of the first test of the stack and the interval of the test of change 0003. The data sets are in the table of `stack-estimate.tsv`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | The metric |
| `stack1_a_mean`, `stack1_b_mean` | ms, requests/s or tokens/s | The `a_mean` and `b_mean` of the first test of the stack (build `D` and the stack at `fbfd9cb8c`) |
| `stack1_diff_pct` | % | The `diff_pct` of the first test of the stack (the stack at `fbfd9cb8c` against build `D`) |
| `stack2_a_mean`, `stack2_b_mean` | ms, requests/s or tokens/s | The `a_mean` and `b_mean` of this test (the image and the stack at `419b8f612`) |
| `stack2_diff_pct` | % | The `diff_pct` of this test (the stack at `419b8f612` against the image) |
| `stack2_over_stack1_pct` | % | 100 times ((1 plus `stack2_diff_pct` divided by 100) divided by (1 plus `stack1_diff_pct` divided by 100) minus 1). The two tests have different arms. The value has the effect of change 0003 and the difference of build `D` and the image. |
| `a_mean_stack2_over_stack1_pct`, `b_mean_stack2_over_stack1_pct` | % | 100 times (`stack2_a_mean` divided by `stack1_a_mean` minus 1), and the same for the columns `b_mean`. The two tests have different host loads. |
| `c0003_diff_pct`, `c0003_ci_lo_pct`, `c0003_ci_hi_pct` | % | The `diff_pct` and the interval of the test of change 0003 on the stack (`speed-rebased.tsv`). The three `llama bench` rows have no value. |
| `stack2_over_stack1_in_c0003_interval` | None | `True` if `stack2_over_stack1_pct` is in the interval of the test of change 0003. The three `llama bench` rows have no value. |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-stack-2/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library. The data sets of the three tests of the estimate must be in the repository.

## Files not added

We added all files of the results folders of the two schedules, with these differences.

| Files | Cause |
|---|---|
| The 12 files `stack2-bench-b<N>-<P or S2>.launch.log` | Each file has one line: the SHA-256 name of a container of the host. It has no measurement. |
| The log of the measurement window | It has the names of programs of the host. |
| The cubins, the libraries and the files of the build | They are binary files or large files. The manifest gives the SHA-256 value of each library. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 12 items in `raw/`. All 12 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 12 files `stack2-bench-b<N>-<P or S2>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 12 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
