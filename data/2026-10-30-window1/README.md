# Data set: measurement window 1 (2026-10-30)

This data set holds the raw files and the derived tables of the first measurement window. In this measurement window the GPU did 40 runs in 44 minutes. The [result of phase 1](../../docs/notebook/2026-10-30-phase1-equivalence-result.md) and the [correctness baseline of phase 2](../../docs/notebook/2026-10-30-phase2-correctness-baseline.md) use these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Phase 1: Does build `D` of tag `b12004` have the same speed and the same answers as the stock image? The [prediction entry](../../docs/notebook/2026-10-30-phase1-equivalence-plan.md) gives the test, the limit of the speed difference, and the prediction.
2. Phase 2: Which are the numbers of the correctness baseline for the model "mix8" on build `D`? The numbers are the KL divergence from the Q8_0 model, the 52-decision set, and the result of `compute-sanitizer`. The measurement window also made the reference answers of the public benchmark and its first runs.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-30 22:07:26 UTC |
| Last run starts | 2026-10-30 22:40:12 UTC |
| Measurement window | 22:07 to 22:51 UTC |

The project stopped the server in production for the measurement window. One run at a time used the GPU.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv).

## Project commit

The scripts of the measurement window are the scripts of the commit `0d93d5fa29c00428858baf4ee4d221fdee712444` of this repository. One script is different: `harness/run_schedule.sh`. The version of the measurement window accepted overrides at the start of a line, for values without spaces. The version of the commit has no overrides. [INFERENCE: we compared the files after the measurement window and found no other difference.]

After the measurement window, three scripts changed: `harness/run_schedule.sh`, `harness/remote/start_server.sh` and `harness/remote/run_client.sh`. The changes add overrides with spaces, `DMK_SERVER_WRAP`, `DMK_TOOL_CAPS` and `DMK_STOP_TIMEOUT`. They also set the entry point `/app/llama-server` for arm `S`. The measurement window used none of these changes. [INFERENCE: this comes from the code. The entry point is the program that the image starts by default, and we did not run the schedule again with the new scripts.]

The speed numbers of the private benchmark come from the client of the private benchmark. Its `bench.py` measures the same quantities as `bench/bench.py`, with the same procedure. The two files have different default paths and a different path for the file of `--write-reference`. The file `bench/bench.py` also does not run the checks for a file with the text `not-measured`. The client of the private benchmark has no such code. The runs of the public benchmark and of the 52-decision set used `bench/bench.py` and `bench/dectest/run_dectest.py` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `p1-bench-*` | The private benchmark, with 100 decisions | No |
| `p2-pub-*` | The public benchmark in `bench/` | Yes |
| `p2-dectest-*` | The 52-decision set, with the cases of the private benchmark | No. `bench/dectest/` has the cases of the public benchmark. |
| `p1-kld-*`, `p2-kld-*` | Texts for the KL divergence: the WikiText-2 test file, and the states of the 16 medium requests and long requests of the private benchmark | No |

The private benchmark is the measurement reference. The public benchmark has the same token counts. The latency of the public benchmark is the same as the latency of the private benchmark in 0.2 % or less. This is true for the runs on a host with a low load.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `S` | The stock image | The label of the image gives the commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b` and the version `b12004`. |
| `D` | Build `D` of tag `b12004` | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |

Build `D` is the baseline build for the work that follows. It has no change to the code of llama.cpp. We made it on 2026-10-30 at 21:50 UTC. Before the measurement window, the `sm_86` SASS of `D` was the same as the SASS in the stock image for all 6,754 kernel names. The check used `harness/build/sass_compare.py`.

## Build flags

We made `D` with `harness/build/build.sh`. The script runs `cmake` in a container for the build. The table gives the tools.

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

The flags are the flags of `.devops/cuda.Dockerfile` at `b12004` with five differences: `-DCMAKE_CUDA_ARCHITECTURES=86-real`, `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`, `-DLLAMA_USE_PREBUILT_UI=OFF`, `-DGGML_CCACHE=OFF` and `-DLLAMA_BUILD_TESTS=ON`. The [harness document](../../harness/README.md) gives more information about the build.

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, arm `S` and arm `D` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

Arm `D` runs in the same image as arm `S`. The container puts the build in `/opt/dmk`. The runs with `compute-sanitizer` use version 12.8.93 of the tool, from a folder that the container attaches at `/tools`.

## Model

| Use | File |
|---|---|
| Model of all speed tests and of the 52-decision set | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8") |
| Reference model for the KL divergence and for the reference answers of the public benchmark | `h2o-lightning-4b-Q8_0.gguf` |

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all runs. The project did not change it. After the measurement window the power limit was 200 W.

## Clocks

We did not lock the clocks. The GPU used its default clocks. During load the GPU was in the state `P2`. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. After the measurement window the clocks were at the default values.

## Server flags

All runs of arms `S` and `D` used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

## Host load

The host has 12 CPU threads. During blocks 1 to 3 of the private benchmark, a different program used the CPU of the host. The one-minute load average was 12 to 19 in these blocks. From block 4 it was 2 to 3. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run.

## Commands

The schedules for the speed tests came from `make_schedule.py`:

```
python3 harness/make_schedule.py --prefix p1-bench --blocks 6 --seed 20261030 \
    --arm S=stock --arm D=build:b12004-sm86 --kind bench --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix p1-llb --blocks 4 --seed 20261031 \
    --arm S=stock --arm D=build:b12004-sm86 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

A person wrote all other lines of the file [raw/window1.txt](raw/window1.txt). The table gives the program and the flags of each group of lines.

| Runs | Line type | Program and flags |
|---|---|---|
| `p1-kld-*-base`, `p2-kld-*-Q8-base` | `tool` | `llama perplexity -m <model> -f <text> -ngl 999 -fa auto -b 2048 -ub 2048 -c 512 --chunks 20 --kl-divergence-base <file>` |
| Other `p1-kld-*` and `p2-kld-*` | `tool` | The same flags, with `--kl-divergence` and the base file |
| `p2-san-*` | `tool@<tool name>` | `llama perplexity -m <mix8> -ngl 999 -fa auto -b 256 -ub 256 -c 256 --chunks 1 -f <benchmark states>`, in the wrapper `timeout 1800 compute-sanitizer --tool <tool name> --error-exitcode 99` |
| `p2-dectest-*` | `client` | `run_dectest.py --url http://127.0.0.1:18080 --cases cases.json --out /runs/{name}.answers.json` |
| `p2-pub-*` | `client` | `bench/bench.py` with `--write-reference {name}`. The reference run also has `--passes 1` and the Q8_0 model. |

The base files of the KL divergence were in a `tmpfs` folder. Each run read the first 20 chunks of one text. One text is the WikiText-2 test file. The other text is the file with the states of the 16 medium requests and long requests of the private benchmark. The harness ran the schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `S` | `stock` | The programs of the stock image |
| `D` | `build:b12004-sm86` | Build `D` of tag `b12004`, in the same image |

Each run of the speed tests started a new server. In each block, a seed set the sequence of the two arms.

## Raw files

The folder `raw/` has 237 files. The harness wrote 236 of them, and they have no change. The 237th file is `window1.txt`. The section "Files that we changed" gives its change. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `p1-bench-b<N>-<S or D>.*` | 96 | `run_bench.sh` | 12 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `p1-llb-b<N>-<S or D>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `p1-kld-*`, `p2-kld-*` | 40 | `run_tool.sh` | 10 runs of `llama perplexity` for the KL divergence, with 4 files each |
| `p2-dectest-<S or D>.*` | 16 | `run_client.sh` | 2 runs of the 52-decision set, with 8 files each: `.answers.json`, `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `p2-pub-*` | 36 | `run_client.sh` | 4 runs of the public benchmark, with 9 files each: the 8 files of a benchmark run and `.client.out` |
| `p2-san-*-D.*` | 16 | `run_tool.sh` | 4 runs of `llama perplexity` in `compute-sanitizer`, with 4 files each |
| `window1.txt` | 1 | A person | The schedule of the measurement window |

The [harness document](../../harness/README.md) gives the meaning of each file type. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp or from `compute-sanitizer`, and we saw no effect from it.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files. The first line has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

### runs.tsv

One row for each run, in the sequence of the start time. It has 40 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | The test: `private benchmark`, `llama bench`, `KL divergence`, `52 decisions`, `public benchmark` or `sanitizer` |
| `line_type` | None | The type of the schedule line: `bench`, `client`, `tool`, or `tool@<wrapper>` |
| `arm` | None | `stock` or `build:b12004-sm86` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before` | None | The one-minute load average of the host before the run |
| `load1_after` | None | The one-minute load average of the host after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000001` means that the GPU did no work. Empty for `tool` runs. |
| `lat_ms` | ms | The latency of the run: the mean of the latencies of the timed passes. Only for benchmark runs. |
| `short_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the short requests, the medium requests, or the long requests in all timed passes |
| `rps8` | requests/s | The number of requests each second with 8 requests at the same time |
| `sm_clock_mhz_median` | MHz | The median of the SM clock in the samples of the run. The samples at the start and at the end are in it. |
| `power_draw_w_median` | W | The median of `power.draw` in the samples of the run |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run |

Source: the `.meta`, `.load0`, `.load1`, `.smi` and `.json` files of each run.

### speed-bench.tsv

The ratio of arm `D` to arm `S` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 6 blocks, with the flag `--margin 2`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `S` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `D` |
| `diff_pct` | % | 100 times (the mean of the ratios of `D` to `S` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |
| `within_margin` | None | `True` if the interval is in ±2 % |

### speed-llama-bench.tsv

The same as `speed-bench.tsv`, for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The metrics are `pp64_ts`, `pp256_ts` and `pp3000_ts`. The unit of `a_mean` and `b_mean` is tokens/s.

### speed-bench-blocks.tsv

The difference of `D` from `S` in each block of the private benchmark, and the host load. It has 6 rows. The script calculates 100 times (the value of `D` divided by the value of `S`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_s_before`, `load1_s_after` | None | The one-minute load average before and after the run of `S` |
| `load1_d_before`, `load1_d_after` | None | The one-minute load average before and after the run of `D` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `D` from `S` |
| `rps8_diff_pct` | % | The difference of `rps8` of `D` from `S` |

### answers-compare-bench.json

The output of `harness/analysis/answers_compare.py` for the 12 answer files `p1-bench-b*-S.reference.json` and `p1-bench-b*-D.reference.json`. It is a JSON file, not a TSV file. The [harness document](../../harness/README.md) gives the keys.

### kld.tsv

The KL divergence of the 6 runs that read a base file. One row for each run.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `text` | None | The text: the WikiText-2 test file, or the benchmark states |
| `compared_arm` | None | The arm of the run |
| `base` | None | `S base` for the base file of arm `S` with mix8, or `Q8_0` for the base file of the Q8_0 model |
| `mean_kld`, `mean_kld_error` | None | The mean KL divergence for each token, and its error |
| `max_kld` | None | The largest KL divergence of one token |
| `kld_p99_9` | None | The 99.9 % percentile of the KL divergence |
| `same_top_token_pct`, `same_top_token_error_pct` | % | The part of the tokens with the same top token, and its error |
| `ppl_compared`, `ppl_base` | None | The perplexity of the run, and of the base file |
| `ppl_ratio`, `ppl_ratio_error` | None | The ratio of the two perplexities, and its error. The error is empty when `llama perplexity` writes `-nan`. |

The base file keeps the log-probabilities as 16-bit integers with a scale for each token. A run against a base file of the same build thus gives a KL divergence that is not 0. The two `S base` rows of the same arm give this KL floor of the method.

### decisions52.tsv

The result of the 52-decision set for each arm. One row for each arm and group.

| Column | Unit | Meaning |
|---|---|---|
| `arm` | None | `S` or `D` |
| `group` | None | `all`, or the group of the cases: `triage`, `check`, `style`, or `commit` |
| `correct` | decisions | The number of correct decisions |
| `decisions` | decisions | The number of decisions |
| `misses` | decisions | The number of incorrect decisions |
| `missed_decisions` | None | The incorrect decisions, as `<case>.<question>`, with a space between them |

The two answer files `p2-dectest-S.answers.json` and `p2-dectest-D.answers.json` have the same bytes. The script writes this check on the standard output.

### sanitizers.tsv

The result of the four tools of `compute-sanitizer` on arm `D`. One row for each tool.

| Column | Unit | Meaning |
|---|---|---|
| `tool` | None | `memcheck`, `initcheck`, `synccheck`, or `racecheck` |
| `errors` | errors | The number of errors in the summary of the tool |
| `hazards_displayed` | count | The number of problems of shared memory that `racecheck` shows. Only for `racecheck`. |
| `warnings` | warnings | The number of warnings. Only for `racecheck`. |
| `seconds_per_pass` | s | The time of one pass, as `llama perplexity` writes it |
| `ppl` | None | The perplexity of the one chunk |

### public-bench.tsv

The runs of the public benchmark. One row for each run.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `role` | None | `reference` for the Q8_0 run that made the reference answers, or `baseline` for a run of mix8 |
| `model` | None | The file name of the model |
| `timed_passes` | passes | The number of timed passes |
| `lat_ms`, `short_ms`, `medium_ms`, `long_ms` | ms | The same as in `runs.tsv` |
| `rps8` | requests/s | The same as in `runs.tsv` |
| `correct` | decisions | The number of correct decisions of 100 |
| `decisions` | decisions | The number of decisions |
| `errors` | errors | The number of requests with an error |

The baseline runs ran before the reference run. They have no numbers for the metrics `agree` and `mean_dp`. The file `bench/baseline.json` comes from the next measurement window.

## Steps to make the tables again

1. Run `python3 data/2026-10-30-window1/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. The script uses these commands in the folder of the data set:

```
python3 ../../harness/analysis/ab_stats.py raw --prefix p1-bench --a S --b D --margin 2
python3 ../../harness/analysis/ab_stats.py raw --prefix p1-llb --a S --b D --margin 2
python3 ../../harness/analysis/answers_compare.py --a 'raw/p1-bench-b*-S.reference.json' \
    --b 'raw/p1-bench-b*-D.reference.json'
```

## Files not added

The results folder also had the two files `p1-bench-ab.tsv` and `p1-llb-ab.tsv`. They hold tables that `ab_stats.py` made from the raw files. We did not add them to `raw/`, because a script makes these tables from the raw files. The files `tables/speed-bench.tsv` and `tables/speed-llama-bench.tsv` have the same values.

## Change in a raw file

The file `raw/window1.txt` is the schedule of the measurement window. One line had the path of a folder with model files on the GPU host. We replaced that path with `/path/to/q8-models`. This is the only change in `raw/`.

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 13 items in `raw/`. All 13 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 12 files `p1-bench-b<N>-<S or D>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |
| `p2-san-racecheck-D.err` | `dmk-private-ipv4` | Line 7 has the time that llama.cpp writes in a log line. The rule reads it as an IP address of a local network. |

The file `tools/publish/gitleaks.ignore` lists the 13 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
