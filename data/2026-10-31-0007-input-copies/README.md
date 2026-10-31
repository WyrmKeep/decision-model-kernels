# Data set: change 0007, uploads of the scheduler (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of change 0007. The change makes the scheduler of ggml skip the uploads that no `COMPUTE` node reads. The [results entry](../../docs/notebook/2026-10-31-0007-input-copies-result.md) and the [change document](../../docs/changes/0007-input-copies.md) use these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build with the change faster than the build before it, for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: The change is in the equivalence class `exact`. Does the build give the same answers as the build before it, bit for bit?
3. Mechanism: Does a pass of the new build have 3 uploads and 8 sync calls less?
4. SASS: Does `libggml-cuda.so` of the new build have the same SASS for `sm_86`?

We wrote the prediction before the first run of the new build in the test. The [prediction entry](../../docs/notebook/2026-10-31-0007-input-copies-prediction.md) is in the notebook. Its commit is `ebe92631baed2ce220fefbef0d2d1e2045efca0b`, from 2026-10-31 05:32:15 UTC.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 05:53:42 UTC |
| Last run starts | 2026-10-31 06:09:41 UTC |
| Measurement window | 05:53 to 06:10 UTC |

The 18 runs of this data set are the runs of one schedule file. The project stopped the server in production for the measurement window. One run at a time used the GPU. The log of the measurement windows shows that no other measurement window was open at this time.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv).

## Project commit

The scripts of the measurement window are the same as the files of the folder `harness/` of this repository, in commit `31a5d54f528ee654d9fc4f7216c14e0373bbdbcc`. The only differences are `harness/README.md` and the folders `harness/ablate/`, `harness/build/`, and `harness/profile/`. The measurement window did not use them, with one difference. The client of the Nsight Systems runs is `nsys_client.py` of `harness/profile/extra/`. The SASS compare used `harness/ablate/sass_check.sh`. [INFERENCE: we compared the folders with `diff -r` after the measurement window.]

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `c0007-bench-*` | The private benchmark, with 100 decisions | No |
| `c0007-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |
| `c0007-nsys-*` | 2 warm-up requests and 10 requests of the type `short2q`, with Nsight Systems | No |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `L` | Branch `dmk/stack` of the fork of llama.cpp of this project. It has the pull request #30489. | `6b8b65e6672f07805000d3c9683b65c72cc62303`. The source had 0 changes. |
| `I` | Branch `dmk/0007-input-copies` of the same fork | `c3ec5653bda455b6d49976412ab5b7904f55fb13`. The commit before it is the commit of `L`. The source had 0 changes. |

The change is one commit. It changes one file, `ggml/src/ggml-backend.cpp`, with 23 added lines and 3 removed lines. We made `I` on 2026-10-31 at 05:45 UTC in 346 s. The build of `L` is the build of the [lead data set](../2026-10-31-lead-30489/README.md). The files `raw/build/MANIFEST-L.txt` and `raw/build/MANIFEST-I.txt` give the facts of the builds and the SHA-256 value of each file.

## Build flags

We made the two builds with `harness/build/build.sh`. The two manifests show the same tools and the same `cmake` command. The flags are the flags of the [lead data set](../2026-10-31-lead-30489/README.md). The build of `I` has the same tools as the build of `L`.

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, and `llama bench`, of the two arms | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

Each arm runs in the same image. The container puts the build in `/opt/dmk`.

## Model

All runs used the file `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8"). The reference answers of the quality phase are the answers of the Q8_0 model. The [method](../../docs/method.md) gives their source.

## Driver, GPU, power limit and clocks

The version of the driver is 595.71.05. The GPU is the `RTX 3060 Ti`. The power limit was 200 W in all runs. We did not lock the clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,710 to 1,943 MHz.

The column `throttle_reasons_at_start` has the value `0x0000000000000001` for the first run, `c0007-bench-b1-I`. That value is the bit `GpuIdle`. All other runs of the private benchmark have the value `0x0000000000000000`.

## Server flags

All runs used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. Other work on the host made a high load at the start of the measurement window. The load fell in the next blocks. The cause of the load is not in the data.

| Runs | One-minute load average |
|---|---|
| Blocks 1 and 2 of the private benchmark | 6.47 to 21.49 |
| Blocks 3 and 4 of the private benchmark | 3.15 to 6.47 |
| The 8 runs of `llama bench` | 2.81 to 3.78 |

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules came from `make_schedule.py`. The file [raw/c0007.txt](raw/c0007.txt) has the lines. We ran the two commands with the same seeds. The lines are the same as in the file. The file also has the 2 lines of the Nsight Systems runs. These 2 lines have the word `@REMOTE_DIR@` for the work folder of the GPU host.

```
python3 harness/make_schedule.py --prefix c0007-bench --blocks 4 --seed 70071 \
    --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix c0007-llb --blocks 4 --seed 70072 \
    --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The script of the measurement window stopped the server in production, ran the schedule, and started the server in production again. The harness ran the schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tables came from these commands. Arm `L` is the reference arm, and the ratio is `I` divided by `L`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix c0007-bench --a L --b I
python3 ../../harness/analysis/ab_stats.py raw --prefix c0007-llb --a L --b I
python3 ../../harness/analysis/answers_compare.py --a 'raw/c0007-bench-b*-L.reference.json' \
    --b 'raw/c0007-bench-b*-I.reference.json'
python3 data/2026-10-31-0007-input-copies/make_tables.py
```

The SASS compare came from this command, with the two files `libggml-cuda.so`:

```
bash harness/ablate/sass_check.sh <output folder> <libggml-cuda.so of L> <libggml-cuda.so of I>
```

The Nsight Systems runs used the flags `profile --trace=cuda,osrt,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`. The schedule file shows the full lines.

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `L` | `build:lead-30489` | The build before the change, in the image of the stock build |
| `I` | `build:0007-input-copies` | The build with the change, in the same image |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In blocks 1 and 2 of the private benchmark, arm `I` ran first. In blocks 3 and 4, arm `L` ran first. In blocks 1 to 3 of `llama bench`, arm `L` ran first. In block 4, arm `I` ran first.

## Raw files

The folder `raw/` has 116 files. The harness wrote 113 of them, and they have no change. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `c0007-bench-b<N>-<L or I>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `c0007-llb-b<N>-<L or I>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `c0007-nsys-<L or I>.*` | 16 | `run_client.sh` | 2 runs with Nsight Systems, with 8 files each: `.client.json`, `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `c0007.txt` | 1 | `make_schedule.py` | The schedule of the 18 runs, with 6 comment lines |
| `sass-compare/sass-compare.json` | 1 | `sass_compare.py` | The summary of the SASS compare. The lists of the different names and the names of one build are empty. |
| `build/MANIFEST-<L or I>.txt` | 2 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

The tables `mechanism.tsv` and `mechanism-pass-types.tsv` come from the Nsight Systems captures. The captures are not in this repository. The section "Steps to make the tables again" gives the steps.

### runs.tsv

One row for each run, in the sequence of the start time. It has 18 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark`, `llama bench` or `nsys` |
| `line_type` | None | The type of the schedule line: `bench`, `tool` or `client` |
| `block` | None | The number of the block |
| `arm_tag` | None | `L` or `I` |
| `arm` | None | `build:lead-30489` or `build:0007-input-copies` |
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

The ratio of arm `I` to arm `L` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `L` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `I` |
| `diff_pct` | % | 100 times (the mean of the ratios of `I` to `L` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

### speed-llama-bench.tsv

The ratio of arm `I` to arm `L` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The columns are the columns of `speed-bench.tsv`. The metric is `pp64_ts`, `pp256_ts` or `pp3000_ts`, and the unit of the means is tokens/s.

### speed-bench-blocks.tsv

The difference of `I` from `L` in each block of the private benchmark, and the host load. It has 4 rows. The script calculates 100 times (the value of `I` divided by the value of `L`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_l_before`, `load1_l_after` | None | The one-minute load average before and after the run of `L` |
| `load1_i_before`, `load1_i_after` | None | The one-minute load average before and after the run of `I` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `I` from `L` |
| `rps8_diff_pct` | % | The difference of `rps8` of `I` from `L` |

### speed-llama-bench-blocks.tsv

The difference of `I` from `L` in each block of `llama bench`, and the host load. It has 4 rows. The first six columns are the columns of `speed-bench-blocks.tsv`. The last three columns are `pp64_ts_diff_pct`, `pp256_ts_diff_pct` and `pp3000_ts_diff_pct`. They give 100 times (the prompt speed of `I` divided by the prompt speed of `L`, minus 1).

### quality-bench.tsv

The quality numbers of the 8 runs of the private benchmark. One row for each run.

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

The output of `harness/analysis/answers_compare.py` for the 8 answer files `c0007-bench-b*-L.reference.json` and `c0007-bench-b*-I.reference.json`. It is a JSON file, not a TSV file. The [method](../../docs/method.md) gives the fields.

### sass-summary.tsv

The summary of the SASS compare of `libggml-cuda.so` of build `L` (A) and of build `I` (B), for `sm_86`. It has 15 rows. The script copies the `summary` object of `raw/sass-compare/sass-compare.json`. The [harness document](../../harness/README.md) gives the meaning of each key.

| Column | Unit | Meaning |
|---|---|---|
| `item` | None | The key of the summary |
| `value` | None | The value of the key |

### mechanism.tsv

The uploads and the sync calls of a pass, from the two Nsight Systems runs. It has 4 rows. The script finds the passes: a pass is a run of 900 or more kernel launches. The upload phase of a pass starts at the sync call before the first upload. It ends at the first sync call after the last upload. Only passes with more than 3 tokens count.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | `c0007-nsys-L` or `c0007-nsys-I` |
| `tokens_in_pass` | None | The group of passes: `1 to 64 tokens` or `65 to 256 tokens`. The number of tokens is the number of bytes of the largest upload divided by 10,240. |
| `passes` | passes | The number of passes in the group |
| `uploads_per_pass` | uploads | The mean number of host to device copies in the upload phase |
| `sync_calls_in_upload_phase` | sync calls | The mean number of sync calls (`cudaStreamSynchronize`) in the upload phase |
| `uploaded_mb_per_pass` | MB | The mean of the bytes of the uploads in a pass |
| `upload_phase_ms` | ms | The mean time from the start of the first sync call to the end of the last sync call of the upload phase |

### mechanism-pass-types.tsv

The passes of the two Nsight Systems runs, in groups with the same number of uploads and sync calls. It has 4 rows. The columns are `run`, `uploads`, `sync_calls_in_upload_phase` and `passes`.

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0007-input-copies/make_tables.py` in the repository folder. The script writes all tables. It keeps `mechanism.tsv` and `mechanism-pass-types.tsv` as they are.
2. Make the two mechanism tables again: put the captures `c0007-nsys-L.nsys-rep` and `c0007-nsys-I.nsys-rep` in one folder. Set `DMK_PROFILE_DIR` to this folder and `DMK_NSYS` to the command line tool of Nsight Systems. The script uses `nsys export --type sqlite` on each capture. A folder with the two files `c0007-nsys-L.sqlite` and `c0007-nsys-I.sqlite` also works without `DMK_NSYS`.
3. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folder of the schedule, with these differences.

| Files | Cause |
|---|---|
| `c0007-nsys-L.nsys-rep` and `c0007-nsys-I.nsys-rep` | They are binary files of 2.5 MB each. The two mechanism tables have their results. |
| `key-kernels.tsv` of the SASS compare | It has 497 KB. The summary in `sass-compare.json` shows that all 6,766 names have the same SASS. |
| The log of the measurement window | It has the names of programs of the host. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 8 items in `raw/`. All 8 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 8 files `c0007-bench-b<N>-<L or I>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 8 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
