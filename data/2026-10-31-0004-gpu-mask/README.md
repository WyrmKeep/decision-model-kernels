# Data set: change 0004, attention mask on the GPU (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of change 0004. The change makes the attention mask on the GPU. The build before the change makes it on the CPU and sends it to the GPU. The [prediction entry](../../docs/notebook/2026-10-31-0004-gpu-mask-prediction.md) and the [results entry](../../docs/notebook/2026-10-31-0004-gpu-mask-result.md) use these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build with change 0004 faster than build `D` for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: The change is in the equivalence class `exact`. Does the build with the change give the same answers as build `D`, bit for bit?
3. Correctness: Does the attention mask from the GPU have the same bytes as the attention mask from the CPU? Does `compute-sanitizer` find an error?
4. Gain: Which is the cause of the difference between the gain and the prediction?

We wrote the prediction on 2026-10-31 at 05:24 UTC. It is in commit `7dd4b953b8a041f9941833775f5e61fd14b54b16` of this repository. The commit `1211b05a0c70559449c61df4f4b055e3080efb0f` corrects the time in the text. The two commits are before the first run of the speed test.

## Dates

| Item | Value |
|---|---|
| First correctness run starts | 2026-10-31 06:10:27 UTC |
| Last correctness run starts | 2026-10-31 07:28:03 UTC |
| First run of the speed test starts | 2026-10-31 07:11:34 UTC |
| Last run of the speed test starts | 2026-10-31 07:27:41 UTC |
| Measurement window of the correctness runs | 06:10 to 06:14 UTC |
| Measurement window of the speed test | 07:11 to 07:29 UTC |

The measurement window of the speed test holds the 16 runs of the speed test and the 2 runs of `compute-sanitizer` with `llama perplexity`. The two runs of `compute-sanitizer` are after the 16 runs. The project stopped the server in production for the two measurement windows. The two runs of `test-backend-ops` are in `raw/`. They ran at about 07:50 UTC, when no measurement window was open. The server in production ran at that time.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv), and the `.meta` files.

## Project commit

The scripts of the measurement windows are the same as the files of the folder `harness/` in this repository. We compared the files `make_schedule.py`, `run_schedule.sh`, `remote/` and `analysis/` with `diff -r` after the measurement windows. They are the same. The measurement windows used a work copy of the files and not a clone of this repository.

The two schedules for the speed test came from `harness/make_schedule.py`. The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `gpumask-bench-*`, `0004-check-*` | The private benchmark, with 100 decisions | No |
| `gpumask-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `D` | Branch `dmk/lead-30489` of the fork of llama.cpp of this project. It is tag `b12004` and change 0001. The data set [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md) has the same build. | `6b8b65e6672f07805000d3c9683b65c72cc62303` |
| `L` | Branch `dmk/0004-gpu-mask` of the fork of llama.cpp of this project | `cf3913bedc87cb1e089e6cac562ca693620cb90e`. The commit before it is the commit of `D`. |

The commit of `L` has 17 changed files, with 733 added lines and 29 removed lines. The new files are `ggml/src/ggml-cuda/kq-mask.cu` and `ggml/src/ggml-cuda/kq-mask.cuh`. The other files are `ggml.h`, `ggml.c`, `ggml-rpc.h`, `ggml-backend-meta.cpp`, `ggml-cuda.cu`, `ggml-cpu.c`, `ops.cpp`, `ops.h`, `llama-context.cpp`, `llama-cparams.h`, `llama-graph.cpp`, `llama-graph.h`, `llama-kv-cache.cpp`, `llama-kv-cache.h` and `test-backend-ops.cpp`.

The files `raw/build/MANIFEST-lead-30489.txt` and `raw/build/MANIFEST-0004-gpu-mask.txt` give the facts of the builds and the SHA-256 value of each file. Build `L` ran on 2026-10-31 at 05:25 UTC from the same files, with 383 s of wall time. The manifest of `L` is from the second build of the same commit, at 05:46 UTC, with 13 s of wall time. In the second build, the source had 0 changes. The libraries that the measurement used are from the second build.

## Build flags

We made the two builds with the same script and the same flags. The two manifests show the same tools and the same `cmake` command.

| Tool | Version |
|---|---|
| Image for the build | `localhost/dmk-builder:cuda12.8.1-gcc14` |
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

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, `llama bench`, `llama perplexity` and `test-backend-ops` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
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

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,905 MHz. The column `throttle_reasons_at_start` has the value `0x0000000000000000` for the 8 runs of the private benchmark. No cause of a clock limit was active at the start.

## Server flags

The runs of the private benchmark in the speed test used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The correctness runs used the same flags with `-lv 4`. The runs of `0004-check-streams` did not have `--kv-unified`. The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The one-minute load average of the host was high in all runs. Other work ran on the host at the same time. It was not part of this test.

| Runs | One-minute load average |
|---|---|
| The 8 runs of the private benchmark | 4.02 to 20.92 |
| The 8 runs of `llama bench` | 17.58 to 22.15 |

The load of the host is high for the two arms. It is not the same in each block. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules for the speed test came from `make_schedule.py`. The files [raw/0004-ab-bench.txt](raw/0004-ab-bench.txt) and [raw/0004-ab-llb.txt](raw/0004-ab-llb.txt) have the lines.

```
python3 harness/make_schedule.py --prefix gpumask-bench --blocks 4 --seed 4001 \
    --arm D=build:lead-30489 --arm L=build:0004-gpu-mask --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix gpumask-llb --blocks 4 --seed 4002 \
    --arm D=build:lead-30489 --arm L=build:0004-gpu-mask --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The script of the measurement window stopped the server in production, ran the schedules, and started the server in production again. The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The files [raw/0004-correctness.txt](raw/0004-correctness.txt) and [raw/0004-san.txt](raw/0004-san.txt) have the lines of the correctness runs. The first two lines of `0004-correctness.txt` set the value `LLAMA_KQ_MASK_CHECK=1` for the server. It makes the program compare the attention mask from the GPU with the attention mask from the CPU after each pass. The third line runs the server in `compute-sanitizer` with the tool `memcheck`. The script [raw/memcheck_client.py](raw/memcheck_client.py) sends one request of each type to it. The two lines of `0004-san.txt` run `llama perplexity` in `compute-sanitizer` with a pass of 256 tokens and a pass of 2,048 tokens.

The two runs of `test-backend-ops` used this command in the image of the server, with the build in `/opt/dmk`:

```
test-backend-ops test -o KQ_MASK
test-backend-ops perf -o KQ_MASK
```

The tables came from these commands. Arm `D` is the reference arm, and the ratio is `L` divided by `D`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix gpumask-bench --a D --b L
python3 ../../harness/analysis/ab_stats.py raw --prefix gpumask-llb --a D --b L
python3 ../../harness/analysis/answers_compare.py --a 'raw/gpumask-bench-b*-D.reference.json' \
    --b 'raw/gpumask-bench-b*-L.reference.json'
python3 data/2026-10-31-0004-gpu-mask/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `D` | `build:lead-30489` | The build before the change. It writes the attention mask on the CPU. |
| `L` | `build:0004-gpu-mask` | Build `D` and change 0004. It makes the attention mask on the GPU. |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In blocks 1 and 3 of the private benchmark, arm `D` ran first. In blocks 2 and 4, arm `L` ran first. In blocks 1 and 2 of `llama bench`, arm `D` ran first. In blocks 3 and 4, arm `L` ran first.

## Raw files

The folder `raw/` has 137 files. The harness wrote 130 of them, and they have no change. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `gpumask-bench-b<N>-<D or L>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `gpumask-llb-b<N>-<D or L>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `0004-check-unified.*`, `0004-check-streams.*` | 16 | `run_bench.sh` | 2 runs of the private benchmark with one timed pass. The server makes the attention mask on the GPU and on the CPU, and compares them. Each run has 8 files. |
| `0004-san-memcheck.*` | 8 | `run_client.sh` | The server in `compute-sanitizer` and the requests: `.client.out`, `.answers.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `0004-san-memcheck-ppl256-L.*`, `0004-san-memcheck-ppl2048-L.*` | 8 | `run_tool.sh` | 2 runs of `llama perplexity` in `compute-sanitizer`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `0004-ab-bench.txt`, `0004-ab-llb.txt` | 2 | `make_schedule.py` | The schedules of the 16 runs of the speed test |
| `0004-correctness.txt`, `0004-san.txt` | 2 | A person | The schedules of the correctness runs |
| `memcheck_client.py` | 1 | A person | The client that sends one request of each type |
| `0004-tbo-kq-mask-test.out`, `0004-tbo-kq-mask-perf.out` | 2 | `test-backend-ops` | The output of the test and of the time of the kernel |
| `build/MANIFEST-<arm>.txt` | 2 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

The `.server.log` files of the correctness runs have the log level 4 (`-lv 4`). They have one line `KQ mask check` for each pass. The server log of `0004-san-memcheck` has no line `ERROR SUMMARY`. The tool `compute-sanitizer` writes that line when the program ends, but the server in the container did not write it. The log has no line of an error. The two runs of `llama perplexity` have the line.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

### runs.tsv

One row for each run of the speed test, in the sequence of the start time. It has 16 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark` or `llama bench` |
| `line_type` | None | The type of the schedule line: `bench` or `tool` |
| `block` | None | The number of the block |
| `arm_tag` | None | `D` or `L` |
| `arm` | None | `build:lead-30489` or `build:0004-gpu-mask` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000000` means that no cause was active. Empty for `tool` runs. |
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

The ratio of arm `L` to arm `D` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks. The test has no equivalence margin.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `D` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `L` |
| `diff_pct` | % | 100 times (the mean of the ratios of `L` to `D` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

### speed-llama-bench.tsv

The ratio of arm `L` to arm `D` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The script `harness/analysis/ab_stats.py` makes it.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | tokens/s | The mean of arm `D` |
| `b_mean` | tokens/s | The mean of arm `L` |
| `diff_pct` | % | 100 times (the mean of the ratios of `L` to `D` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

### speed-bench-blocks.tsv

The difference of `L` from `D` in each block of the private benchmark, and the host load. It has 4 rows. The script calculates 100 times (the value of `L` divided by the value of `D`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_d_before`, `load1_d_after` | None | The one-minute load average before and after the run of `D` |
| `load1_l_before`, `load1_l_after` | None | The one-minute load average before and after the run of `L` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `L` from `D` |
| `rps8_diff_pct` | % | The difference of `rps8` of `L` from `D` |

### speed-llama-bench-blocks.tsv

The difference of `L` from `D` in each block of `llama bench`, and the host load. It has 4 rows.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_d_before`, `load1_d_after` | None | The one-minute load average before and after the run of `D` |
| `load1_l_before`, `load1_l_after` | None | The one-minute load average before and after the run of `L` |
| `pp64_ts_diff_pct`, `pp256_ts_diff_pct`, `pp3000_ts_diff_pct` | % | 100 times (the prompt speed of `L` divided by the prompt speed of `D`, minus 1) |

### quality-bench.tsv

The quality numbers of the 8 runs of the private benchmark in the speed test. One row for each run.

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

The output of `harness/analysis/answers_compare.py` for the 8 answer files `gpumask-bench-b*-D.reference.json` and `gpumask-bench-b*-L.reference.json`. It is a JSON file, not a TSV file. The [method](../../docs/method.md) gives the fields.

### correctness.tsv

The result of the correctness runs. One row for each run. It has 5 rows. The script reads the server logs and the output of `compute-sanitizer`.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `mask compare`, `memcheck, server, 4 requests` or `memcheck, llama perplexity` |
| `count` | passes or requests | For `mask compare`: the number of passes that the program compared. For the server in `compute-sanitizer`: the number of requests with an answer. |
| `different_or_errors` | values or errors | For `mask compare`: the sum of the values that are different between the attention mask from the GPU and the attention mask from the CPU. For `compute-sanitizer`: the number of errors. |
| `max_n_kv`, `max_n_tokens` | cells, tokens | The largest number of KV cells and of tokens in one pass that the program checked |
| `device_mask_enabled_in_log` | None | `True` if the server log has the line `KQ mask on the device enabled` |
| `abort_in_log` | None | `True` if the server log has the word `ABORT` |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0004-gpu-mask/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folders of the schedules. We did not add these files.

| Files | Cause |
|---|---|
| The server run `0004-check1` and the 25 starts of the server that did not start | They ran when a measurement window of a different change could be open. The project made a rule after that time: no program can use the GPU while a measurement window is open. The results entry gives their times. |
| The first runs of `test-backend-ops` | The two runs in `raw/` replace them. They have the same result. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 10 items in `raw/`. All 10 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 10 files `gpumask-bench-b<N>-<D or L>.json`, `0004-check-unified.json` and `0004-check-streams.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` has no line for these 10 items. The text of the file gives the cause for the items of the data sets before this one.
