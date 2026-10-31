# Data set: baseline of the public benchmark (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of the baseline of the public benchmark. The measurement is part of the second measurement window. It has three baseline runs, one run that makes `bench/baseline.json`, and one check run. The file `bench/baseline.json` is a copy of one raw file of this data set. The rules for a data set are in the [data README](../README.md).

## Question

Which are the guard numbers of the public benchmark for the model "mix8" on build `D`? The numbers are `correct`, `agree` and `mean_dp`. The [README of the benchmark](../../bench/README.md#source-of-the-baseline) gives the values and the procedure.

This test has no prediction entry, because it measures a baseline and not a change. The notebook has no results entry for it at this time. The [correctness baseline of phase 2](../../docs/notebook/2026-10-30-phase2-correctness-baseline.md) tells that the baseline comes from this measurement.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 00:01:53 UTC |
| Last run starts | 2026-10-31 00:07:25 UTC |

The project stopped the server in production for the measurement window. One run at a time used the GPU. The measurement window also has other tests. They are not part of this data set.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv).

## Project commit

The scripts of the measurement are the scripts of the commit `e91d24b53413fb9922120760e5c8317a95a6d9c2` of this repository. We compared the files after the measurement window. The scripts of `harness/` have the same bytes as the scripts at the commit. The client files `bench.py`, `cases.json`, `reference.json` and `tools/make_baseline.py` of the measurement have the same bytes as the files in `bench/` at the commit.

At the time of the measurement, the file `baseline.json` that the client read for the runs 1 to 3 had the mark `not-measured`. Thus the client did not do the quality guards in these runs. The result files have `guard_pass` with the value `null` for them.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `D` | Build `D` of tag `b12004` | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |

This is the build `D` of the [data set of the first measurement window](../2026-10-30-window1/README.md#build-commits). We did not build it again. The file `MANIFEST.txt` of the build has the SHA-256 checksum `c6b818a5c0cb4fbf4f0cad93971542e7db65d38922d8105519e020865f019d90`. The copy of the build on the GPU host has the same checksum for this file. The time of the file is 2026-10-30 21:50 UTC.

The baseline is the build `D`, not the stock image. On the private benchmark, build `D` gives the same answers as the stock image. The [result of phase 1](../../docs/notebook/2026-10-30-phase1-equivalence-result.md) shows this. We made no run of the public benchmark with the stock image. [INFERENCE: the same answers on the private benchmark mean the same answers on the public benchmark. We made no test of this.]

## Build flags

We made `D` with `harness/build/build.sh`. The tools and the flags are the flags of the [data set of the first measurement window](../2026-10-30-window1/README.md#build-flags). The table gives the tools.

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

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, arm `D` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

The container puts build `D` in `/opt/dmk`.

## Model

| Use | File |
|---|---|
| Model of all runs of this data set | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8") |
| Model of the reference answers in `bench/reference.json` | `h2o-lightning-4b-Q8_0.gguf` |

The reference answers are not new. They come from the run `p2-pub-ref-q8` of the [data set of the first measurement window](../2026-10-30-window1/README.md). The runs of this data set did not use the Q8_0 model.

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all runs. The project did not change it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. During load the GPU was in the state `P2`. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run.

## Server flags

All runs used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

## Host load

The one-minute load average of the host was 1.06 to 1.70 in all runs. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run.

## Commands

A person wrote the schedule [raw/window2-pub.txt](raw/window2-pub.txt). The harness did not use `make_schedule.py` for it. The harness ran the schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The table gives the program and the flags of each line of the schedule.

| Runs | Line type | Program and flags |
|---|---|---|
| `p3-pub-base-1`, `p3-pub-base-2`, `p3-pub-base-3` | `client` | `bench.py --cases cases.json --reference reference.json --out /runs/{name}.json --write-reference {name} --baseline baseline.json`. The file `baseline.json` had the mark `not-measured`. |
| `p3-pub-mkbase` | `client` | `tools/make_baseline.py` with the three result files, `--note from-source-b12004-sm86-build-D_mix8_production-flags_2026-10-31` and `--out /runs/p3-pub-mkbase.baseline.json` |
| `p3-pub-base-4` | `client` | The same flags as the runs 1 to 3, with `--baseline /runs/p3-pub-mkbase.baseline.json` |

The harness did not use `ab_stats.py`, because the data set has one arm. The script of the data set used `answers_compare.py` with this command in the folder of the data set:

```
python3 ../../harness/analysis/answers_compare.py --a 'raw/p3-pub-base-[123].reference.json' \
    --b 'raw/p3-pub-base-4.reference.json'
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `D` | `build:b12004-sm86` | Build `D` of tag `b12004`, in the image of the table above |

Each run started a new server.

## Raw files

The folder `raw/` has 45 files. The harness wrote 44 of them, and they have no change. The 45th file is `window2-pub.txt`. It is the schedule of the measurement. It has no change. The table groups the files by the name of the run.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `p3-pub-base-<N>.*` | 36 | `run_client.sh` | 4 runs of the public benchmark, with 9 files each: `.json`, `.reference.json`, `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `p3-pub-mkbase.*` | 8 | `run_client.sh` | The run that makes the baseline: `.baseline.json`, `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `window2-pub.txt` | 1 | A person | The schedule of the measurement |

The [harness document](../../harness/README.md) gives the meaning of each file type. The file `p3-pub-mkbase.baseline.json` is the file that `tools/make_baseline.py` wrote. The project put it in `bench/baseline.json` and did not change it.

The run `p3-pub-mkbase` sends no request. The server of this run waits, and its sampler file shows the load of a server that waits. The result files have `startup_s` with the value `null`, because the client ran without the flag `--startup-s`.

Each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. In the TSV files the first line has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

The script also writes four checks on the standard output. Each check must give `True`, or `PASS` for the last check:

1. The values in `baseline.json` are the values that `make_baseline.py` gives for the three runs.
2. The file `bench/baseline.json` has the same bytes as `raw/p3-pub-mkbase.baseline.json`.
3. The values of the guards in `p3-pub-base-4.json` are the values that the script calculates.
4. `bench/summarize.py` gives `PASS` for `p3-pub-base-4`.

### runs.tsv

One row for each run, in the sequence of the start time. It has 5 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `step` | None | `baseline run`, `make baseline`, or `check run` |
| `line_type` | None | The type of the schedule line |
| `arm` | None | `build:b12004-sm86` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before` | None | The one-minute load average of the host before the run |
| `load1_after` | None | The one-minute load average of the host after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000000` means that no cause limits the clocks. |
| `lat_ms` | ms | The latency of the run: the median of the mean latencies of the timed passes. Empty for `p3-pub-mkbase`. |
| `short_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the short requests, the medium requests, or the long requests in all timed passes |
| `rps8` | requests/s | The number of requests each second with 8 requests at the same time |
| `sm_clock_mhz_median` | MHz | The median of the SM clock in the samples of the run. The samples at the start and at the end are in it. |
| `power_draw_w_median` | W | The median of `power.draw` in the samples of the run |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run |

Source: the `.meta`, `.load0`, `.load1`, `.smi` and `.json` files of each run.

### public-bench.tsv

The result of the four runs of the public benchmark. One row for each run.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `step` | None | `baseline run` or `check run` |
| `model` | None | The file name of the model |
| `timed_passes` | passes | The number of timed passes |
| `lat_ms` | ms | The same as in `runs.tsv` |
| `spread_pct` | % | The range of the mean latencies of the timed passes, divided by their median |
| `short_ms`, `medium_ms`, `long_ms` | ms | The same as in `runs.tsv` |
| `p95_ms` | ms | The 95th percentile of the latency of all requests in the timed passes |
| `rps8` | requests/s | The same as in `runs.tsv` |
| `decisions_per_s` | decisions/s | The number of decisions that the server finishes each second, with 8 requests at the same time |
| `correct` | decisions | The number of decisions with a top answer that is the same as the label |
| `decisions` | decisions | The number of decisions |
| `agree_decisions` | decisions | The number of decisions with a top answer that is the same as the reference |
| `agree` | 0 to 1 | `agree_decisions` divided by `decisions` |
| `mean_dp`, `max_dp` | probability | The mean and the largest probability difference to the reference |
| `flips` | None | The decisions with a top answer that is different from the reference, as `<case>.<question>`, with a space between them |
| `errors` | errors | The number of requests with an error |
| `guard_pass` | None | `True` when the run stays in all guards. Empty when the baseline file had the mark `not-measured`. |

Source: the `.json` file of each run.

### baseline-check.tsv

The values of `p3-pub-mkbase.baseline.json` and the values that the script calculates from the three baseline runs with the rules of `make_baseline.py`. One row for each field. It has 8 rows.

| Column | Unit | Meaning |
|---|---|---|
| `field` | None | The name of the field in `baseline.json` |
| `rule` | None | The rule that gives the value from the three runs |
| `value_from_runs` | The unit of the field | The value that the script calculates |
| `value_in_baseline_json` | The unit of the field | The value in the file |
| `equal` | None | `True` when the two values are the same |

### guard-check.tsv

The guards of the check run `p3-pub-base-4`. One row for each guard. It has 5 rows.

| Column | Unit | Meaning |
|---|---|---|
| `guard` | None | `errors`, `correct`, `agree`, `mean_dp`, or `vram` |
| `run_value` | The unit in the column `unit` | The value of the run. For `vram`, the largest `memory.used` in the sampler file. |
| `limit` | The unit in the column `unit` | The limit. For `correct`, `agree` and `mean_dp` it is the value in `baseline.json`, changed by a value of the guard: minus 1, minus 0.02, or plus 0.01. For `vram` it is 6,144 MiB. |
| `unit` | None | The unit of `run_value` and `limit` |
| `rule` | None | How the guard compares the value with the limit |
| `passes` | None | `True` when the value stays in the limit |
| `guard_in_run_json` | None | The value of the guard in `p3-pub-base-4.json`. Empty for `vram`, because the client does not calculate it. |

### answers-compare-baseline.json

The output of `harness/analysis/answers_compare.py` for the four answer files `p3-pub-base-<N>.reference.json`. The files of the runs 1 to 3 are arm A of the script. The file of run 4 is arm B. It is a JSON file, not a TSV file. The [harness document](../../harness/README.md) gives the keys. The four files have the same values, and the largest difference of a probability is 0.

## Steps to make the tables again

1. Run `python3 data/2026-10-31-public-baseline/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time.

## Files not added

The measurement has no file that we did not add. The results folder had these 45 files and no other file.

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 4 items in `raw/`. All 4 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 4 files `p3-pub-base-<N>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 4 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
