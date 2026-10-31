# Data set: ablation of the work of a request (2026-10-31)

This data set holds the raw files and the derived tables of the first ablation. The ablation build skips the work of one part of the model in each run. The runs give the time only. The answers are not correct. The [results entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) uses these data. The rules for a data set are in the [data README](../README.md).

## Question

Which parts of the GPU work use how much of the time of a request? The test gives the cap of each part for a short request, a medium request and a long request. The cap is the largest time that a faster part can remove. It is not a prediction.

There is no prediction entry. The [results entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) gives the caps and their limits. The [ablation README](../../harness/ablate/README.md) gives the meaning of each arm.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 00:09:15 UTC |
| Last run starts | 2026-10-31 00:47:41 UTC |
| Measurement window | 00:02 to 01:14 UTC |

The project stopped the server in production for the measurement window. One run at a time used the GPU. The table gives the runs of each group.

| Group | Runs | First run starts | Last run starts |
|---|---:|---|---|
| Smoke test | 5 | 00:09:15 | 00:10:09 |
| Stage 1 | 31 | 00:15:54 | 00:43:13 |
| Operation runs | 2 | 00:44:18 | 00:44:53 |
| `llama bench` | 18 | 00:45:05 | 00:47:41 |

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv).

## Project commit

The commit of this repository below the commit of this data set is `e91d24b53413fb9922120760e5c8317a95a6d9c2`. The scripts `run_schedule.sh`, `make_schedule.py`, `remote/*` and `harness.env.example` of the measurement window are the same as the scripts of that commit. We compared the files after the measurement window.

The ablation tools were in a folder that is not in this repository during the measurement window. The commit of this data set puts them in `harness/ablate/`:

| File | Difference from the file of the measurement window |
|---|---|
| `make_ablate_runs.py` | It reads the work folder of the GPU host from a flag or from `DMK_REMOTE_DIR`. With the same work folder, it makes the same schedules and the same files of the arms. We compared them byte by byte. |
| `extra/ablate/ablate_bench.py` | One line of the first comment is different. |
| `summarise_ablate.py` | It is a new version. It makes the tables of this data set. The numbers are the numbers of the summary that we made in the measurement window. |
| `sass_check.sh` | New. It runs `harness/build/sass_compare.py` for the two builds. |

## Benchmarks

The client of all runs of stage 1 and of the operation runs is `harness/ablate/extra/ablate/ablate_bench.py`. It sends the 48 cases of the private benchmark. The private benchmark is the measurement reference. It has 32 short requests, 8 medium requests and 8 long requests.

The file `bench/cases.json` of this repository is the public benchmark. It has the same numbers of requests and the same token counts, but the states are different. The client does not read the answers.

The `llama bench` runs use the model file and the flags of the server in production. The [ablation README](../../harness/ablate/README.md) gives the prompt lengths.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `ref` (arm `build:b12004-sm86`) | Build `D` of tag `b12004` | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. `llama bench` reports `3c809f443`. |
| All other arms (arm `build:ablate-timing`) | The branch `dmk/ablate-timing` of the fork of llama.cpp | `aaf86b0528b37f1125a397efecf169a06b51447e` (`b12004-3-gaaf86b052`). We made the build on 2026-10-30 at 22:35:51 UTC. `llama bench` reports `aaf86b052`. |

The branch has three commits above the tag `b12004`:

| Commit | Change |
|---|---|
| `c3747185f2dac0f38fe842f70fc1a1ee6bc4c0bc` | The environment variables `DMK_ABLATE`, `DMK_ABLATE_SYNC` and `DMK_ABLATE_SYNC_DUMP` |
| `5d35fda2c27c8e2f89acffaa3b671aa8b03e287c` | The lines `DMK_TIMING` for each ubatch |
| `aaf86b0528b37f1125a397efecf169a06b51447e` | The line `DMK_TIMING request` in the server, and the value 0 for an answer value that is not a number |

The branch has a fourth commit, `de091da699f7f63c739e82aea03aa2fa9f550834`. It repairs the table `DMK_OPS`. The runs of this data set did not use it. The [results entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) gives the cause.

We compared the `sm_86` SASS of the build `aaf86b052` with the SASS of the baseline build `b12004-sm86` after the measurement window. The SASS is the same for all 6,754 kernel names. We used `harness/ablate/sass_check.sh`. The file `raw/sass-compare-ablate-timing.json` and the table [sass.tsv](tables/sass.tsv) show the result.

The two files `libggml-cuda.so` that we compared have the same SHA-256 as the files on the GPU host: `b891d84884b46936e9d824987b94d7a98764a43988894e279d1457a87c14340a` for `aaf86b052` and `af227de3d569e628bcbc270768c59c245c5ec1a7c6a8975d9df0a7120901c2ff` for `b12004-sm86`.

## Build flags

We made the two builds with `harness/build/build.sh`. The script runs `cmake` in a container for the build. The table gives the tools.

| Tool | Version |
|---|---|
| Image for the build | `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`, digest `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` |
| `nvcc` | 12.8.93 |
| `gcc` | 14.2.0 |
| `cmake` | 3.28.3 |

The `cmake` flags of the two builds are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

The [data set of the first measurement window](../2026-10-30-window1/README.md) gives the same flags.

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, all arms | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

All arms run in the image of the stock build. The container puts the build in `/opt/dmk`. The `llama bench` runs use the same image and the same folder for the build.

## Model

All runs use the file `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8").

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all 56 runs. The project did not change it. The column `power_limit_w` of [runs.tsv](tables/runs.tsv) shows it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. The samples at the start and at the end of a run are in the median.

The median clock of the arm `base` is 1845 MHz. The other arms of stage 1 have 1830 to 1860 MHz. The table gives the arms that have a different clock.

| Arms | Median SM clock (MHz) | Median power (W) |
|---|---:|---:|
| `mmq`, `matmul`, `matmul_q`, `mmq_quantize` | 1950 | 141 to 152 |
| `elementwise` | 1935 | 178 |
| `quantize` | 1905 | 191 |
| `glu`, `nofusion_glu` | 1875 | 193 to 194 |
| `all`, `all_timing` | 1755, 1770 | 46 to 47 |

The arm `base` has a median power of 194.6 W. A skipped part gives less work to the GPU. The remaining kernels then run at a higher clock. The caps of these arms thus have a clock effect. The [results entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) shows how this changes the meaning of the caps.

Source: the columns `sm_clock_mhz_median` and `power_draw_w_median` of [runs.tsv](tables/runs.tsv).

## Server flags

All server arms used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The `llama bench` runs use `-ngl 999 -fa auto -b 2048 -ub 2048 -n 0 -r 5 -o json` and the prompt lengths `-p 8,32,160,1100,3200`.

## Host load

The one-minute load average of the host before and after a run is in [runs.tsv](tables/runs.tsv). It includes the load that the run makes.

| Group | Lowest value | Highest value |
|---|---:|---:|
| Smoke test | 1.47 | 1.65 |
| Stage 1 | 1.32 | 3.07 |
| Operation runs | 2.08 | 2.10 |
| `llama bench` | 1.59 | 2.26 |

## Commands

The schedules came from `make_ablate_runs.py`. The measurement window used a version of the script that was not in this repository. The first line of each schedule file records its flags.

```
python3 harness/ablate/make_ablate_runs.py --out <schedule folder> --remote-dir <work folder of the GPU host> \
    --build ablate-timing --passes 5 --blocks-core 2 --seed 20261101
```

The script writes five schedules. The measurement window ran four of them. The schedule `abl-s2.txt` was not run.

| Schedule | Runs | Line type | Program and flags |
|---|---:|---|---|
| `abl-smoke.txt` | 4 | `client` | `ablate_bench.py --cases /bench/cases.json --size all --passes 1 --out /runs/{name}.json`. The run of the arm `sync` has `--linger-s 3`. |
| `abl-smoke.txt` | 1 | `tool` | `llama bench -m <mix8> -ngl 999 -fa auto -b 2048 -ub 2048 -p 8,160 -n 0 -r 1 -o json`, in the wrapper of the arm `mmq` |
| `abl-s1.txt` | 31 | `client` | `ablate_bench.py --cases /bench/cases.json --size all --passes 5 --out /runs/{name}.json` |
| `abl-ops.txt` | 2 | `client` | The same flags, with `--passes 2 --linger-s 3` |
| `abl-llb.txt` | 18 | `tool` | `llama bench -m <mix8> -ngl 999 -fa auto -b 2048 -ub 2048 -p 8,32,160,1100,3200 -n 0 -r 5 -o json`, in the wrapper of the arm. The arm `ref` has no wrapper. |

A `client` line of an arm of the build `ablate-timing` has the override `DMK_DOCKER_EXTRA=--env-file=<path of the .env file of the arm>`. A `tool` line has the override `DMK_WRAP=/opt/dmk/abl/<arm>.sh`. The arm `ref` has no override. The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The measurement window ran the schedules in the sequence smoke test, stage 1, operation runs, `llama bench`. Each schedule has one results folder. This test did not use `ab_stats.py` or `answers_compare.py`. The script `make_tables.py` makes all tables:

```
python3 data/2026-10-31-ablation/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `ref` | `build:b12004-sm86` | The baseline build with no environment variable. Stage 1 has three runs of it: `ref-start`, `ref-mid` and `ref-end`. |
| `base` | `build:ablate-timing` | The ablation build with no environment variable |
| `timing`, `all_timing` | `build:ablate-timing` | The arms with `DMK_TIMING=1`. The arm `all_timing` also skips all nodes. |
| `all` | `build:ablate-timing` | The arm that skips all nodes |
| `mmq`, `mmvq`, `quantize`, `matmul`, `matmul_q`, `mmq_quantize`, `head`, `fattn`, `gdn`, `gdn_conv_l2`, `glu`, `norm`, `ssm_conv`, `elementwise`, `copies`, `cpy`, `set_rows`, `concat`, `get_rows`, `l2norm`, `rope` | `build:ablate-timing` | The arms that skip one part of the model |
| `nofusion`, `nofusion_glu`, `nographs` | `build:ablate-timing` | The arms with `GGML_CUDA_DISABLE_FUSION=1` or `GGML_CUDA_DISABLE_GRAPHS=1` |
| `sync`, `sync_all` | `build:ablate-timing` | The arms with `DMK_ABLATE_SYNC=1` |

The [ablation README](../../harness/ablate/README.md) gives the environment variables of each arm. A run name is `abl-<group>-b1-<arm>`. The `b1` is the number of the block. Each group has one block. The seed 20261101 set the sequence of the runs.

| Group | Arms |
|---|---|
| Smoke test | `base`, `all`, `all_timing`, `sync`, and the tool run `llb-mmq` |
| Stage 1 | `ref` (3 runs), `base`, `timing`, `all`, `all_timing`, `nofusion`, `nofusion_glu`, `nographs`, and the 21 arms of parts |
| Operation runs | `sync`, `sync_all` |
| `llama bench` | `ref`, `base`, `timing`, `all`, `all_timing`, and the 13 arms `mmq`, `mmvq`, `quantize`, `matmul`, `matmul_q`, `head`, `fattn`, `gdn`, `glu`, `norm`, `ssm_conv`, `elementwise`, `copies` |

## Raw files

The folder `raw/` has 374 files. The harness wrote 373 of them. The 374th file is `sass-compare-ablate-timing.json`, the output of `harness/build/sass_compare.py`. The table groups the files by the name of the run.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `abl-smoke-<arm>.*` | 32 | `run_client.sh` | 4 runs of the client, with 8 files each: `.json`, `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `abl-smoke-llb-mmq.*` | 4 | `run_tool.sh` | 1 run of `llama bench`, with 4 files: `.out`, `.err`, `.smi`, `.meta` |
| `abl-s1-b1-<arm>.*` | 248 | `run_client.sh` | 31 runs of the client, with 8 files each |
| `abl-ops-<arm>.*` | 16 | `run_client.sh` | 2 runs of the client, with 8 files each |
| `abl-llb-b1-<arm>.*` | 72 | `run_tool.sh` | 18 runs of `llama bench`, with 4 files each |
| `abl-llb.txt` | 1 | `run_schedule.sh` | The copy of the schedule of the `llama bench` runs. The other three schedule files are in the section Files not added. |
| `sass-compare-ablate-timing.json` | 1 | `sass_compare.py` | The output of the tool that compares the SASS of the two builds. `make_tables.py` cannot make it again from this folder. |

The [harness document](../../harness/README.md) gives the meaning of each file type. The file `.json` of a client run is the result file of `ablate_bench.py`. The file `.out` of a tool run is the JSON output of `llama bench`. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

The `.server.log` file of the arms `timing` and `all_timing` has the lines `DMK_TIMING`. The three runs with `DMK_ABLATE_SYNC=1` (`abl-smoke-sync`, `abl-ops-sync`, `abl-ops-sync_all`) have the line `DMK_ABLATE active: ... sync=1`. They have no line `DMK_OPS`. The [results entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) gives the cause.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. It runs `harness/ablate/summarise_ablate.py`, which makes seven tables. `make_tables.py` makes `runs.tsv` and `sass.tsv`. The tables are TSV files. The first line has the names of the columns. An empty field means that the table has no value there.

A cap is the value of the arm `base` minus the value of the arm. The column `<x>_cap_pct` is 100 times the cap divided by the value of the arm `base`. A cap below 0 means that the arm is slower than `base`.

### runs.tsv

One row for each run, in the sequence of the start time. It has 56 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `stage` | None | `smoke`, `stage 1`, `sync` or `llama bench` |
| `line_type` | None | The type of the schedule line: `client` or `tool` |
| `arm` | None | `build:ablate-timing` or `build:b12004-sm86` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `requests` | requests | The number of requests of the timed passes of a client run. It is 48 for each pass. |
| `errors` | requests | The number of requests with an error |
| `seq_s` | s | The time of all timed passes of a client run |
| `llama_bench_commit` | None | The value `build_commit` in the output of `llama bench`. Only for tool runs. |
| `sm_clock_mhz_median` | MHz | The median of the SM clock in the samples of the run |
| `power_draw_w_median` | W | The median of `power.draw` in the samples of the run |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run |

Source: the `.meta`, `.load0`, `.load1`, `.smi`, `.json` and `.out` files of each run.

### drift-ref.tsv

The latency of the arm `ref` at the start, in the middle and at the end of stage 1. One row for each request type. It has 3 rows.

| Column | Unit | Meaning |
|---|---|---|
| `size` | None | `short`, `medium` or `long` |
| `start_ms`, `mid_ms`, `end_ms` | ms | The mean latency of the requests of this type in the runs `ref-start`, `ref-mid` and `ref-end` |
| `spread_pct` | % | 100 times (the highest value minus the lowest value) divided by the mean of the three values |

### caps-request.tsv

The mean latency and the cap of each arm of the server, for each request type. One row for each arm. It has 29 rows. The row `base` is the first row.

| Column | Unit | Meaning |
|---|---|---|
| `arm` | None | The short name of the arm |
| `runs` | runs | The number of runs of the arm. It is 1 for each arm, and 3 for `ref`. |
| `short_mean_ms`, `medium_mean_ms`, `long_mean_ms` | ms | The mean latency of all requests of this type in the timed passes. For `ref` it is the mean of the three runs. |
| `short_cap_ms`, `medium_cap_ms`, `long_cap_ms` | ms | The cap: the value of `base` minus the value of the arm |
| `short_cap_pct`, `medium_cap_pct`, `long_cap_pct` | % | The cap in % of the value of `base` |

### pass-spread.tsv

The means of the timed passes of each run. One row for each arm and request type. It has 87 rows.

| Column | Unit | Meaning |
|---|---|---|
| `arm` | None | The short name of the arm |
| `size` | None | `short`, `medium` or `long` |
| `passes` | passes | The number of pass means. It is 5 for each run and 15 for `ref`. |
| `pass_min_ms`, `pass_max_ms` | ms | The lowest and the highest mean latency of one pass |
| `spread_pct` | % | 100 times (the highest value minus the lowest value) divided by the mean of the pass means |

### caps-llama-bench.tsv

The time of one prompt in `llama bench`, and the cap, for each arm. One row for each arm. It has 18 rows. The row `base` is the first row.

| Column | Unit | Meaning |
|---|---|---|
| `arm` | None | The short name of the arm |
| `pp<N>_ms` | ms | The time of one prompt of `<N>` tokens. The script calculates it as 1000 times `<N>` divided by the value `avg_ts` of `llama bench`. `avg_ts` is the mean of 5 tests. `<N>` is 8, 32, 160, 1100 or 3200. |
| `pp<N>_cap_ms` | ms | The cap: the value of `base` minus the value of the arm |
| `pp<N>_cap_pct` | % | The cap in % of the value of `base` |

### timing-ubatch.tsv

The medians of the lines `DMK_TIMING ubatch` of the two arms with `DMK_TIMING`. One row for each run and range of the number of tokens of a ubatch. It has 10 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | `abl-s1-b1-timing` or `abl-s1-b1-all_timing` |
| `n_tokens_class` | tokens | The number of tokens of the ubatch: `1-8`, `9-32`, `33-128`, `129-512` or `513-2048`. A range with no ubatch has no row. |
| `ubatches` | ubatches | The number of ubatches in the range |
| `apply_us`, `build_us`, `alloc_us`, `inputs_us`, `submit_us`, `total_us` | µs | The median of the host time of each step of a ubatch. `build_us` and `alloc_us` are 0 when the pass uses the graph of the previous pass again. |
| `reuse_pct` | % | The part of the ubatches in the range that used the graph of the previous pass again |

### timing-summary.tsv

The other lines `DMK_TIMING` of the same two runs. It has 2 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `ubatches`, `decodes`, `syncs`, `requests` | lines | The number of lines `DMK_TIMING ubatch`, `decode`, `sync` and `request` in the server log |
| `sync_wait_median_us`, `sync_wait_p90_us` | µs | The median and the 90 % percentile of the time that the host waits for the GPU |
| `passes_per_sync_median` | passes | The median number of passes between two waits |
| `request_total_median_us`, `request_wait_median_us`, `request_build_median_us`, `request_format_median_us` | µs | The median of the time of a request in the server: in total, for the model, for the build of the prompt, and for the output text |
| `cuda_direct`, `cuda_graph_capture`, `cuda_graph_replay` | passes | The number of times that the CUDA backend ran a graph in this mode |
| `cuda_host_median_us` | µs | The median of the time that the CUDA backend uses on the host to run a graph |

### ops-lines.tsv

The number of `DMK_OPS` lines in the server log of each run with `DMK_ABLATE_SYNC=1`. It has 3 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `sync_flag` | None | 1 if the log shows `DMK_ABLATE active` with `sync=1` |
| `ops_dumps` | tables | The number of times that the log shows the table |
| `ops_lines` | lines | The number of `DMK_OPS` lines in the log. It is 0 in all three runs. |

### sass.tsv

The summary of `sass_compare.py` for the build `aaf86b052` (side B) against the baseline build `b12004-sm86` (side A). It has 15 rows.

| Column | Unit | Meaning |
|---|---|---|
| `quantity` | None | The name of the value in the summary of `sass_compare.py` |
| `value` | None | The value. The names with `names_` count kernel names. The names with `instr_` count instructions. |

The two builds have the same SASS when `names_sass_different`, `names_encoding_different`, `names_only_A`, `names_only_B` and `names_resusage_different` are 0. They are 0 here.

Source of all tables: the files in `raw/` and [make_tables.py](make_tables.py).

## Steps to make the tables again

1. Run `python3 data/2026-10-31-ablation/make_tables.py` in the repository folder.
2. Read the lines that the script writes. They give the number of rows of each table.
3. Make sure that `git status` shows no change in `tables/`.

## Files not added

The folder `raw/` has no copy of the three schedule files `abl-smoke.txt`, `abl-s1.txt` and `abl-ops.txt`. `run_schedule.sh` wrote a copy of each in the results folders. The scan found 34 items of the type `dmk-server-path` in them: 4 in `abl-smoke.txt`, 28 in `abl-s1.txt` and 2 in `abl-ops.txt`.

The cause is the same in all of them. The override `DMK_DOCKER_EXTRA=--env-file=...` of each `client` line has the path of the work folder on the GPU host. The file `abl-llb.txt` has no such path, and it is in `raw/`. `make_tables.py` does not read the schedule files.

The [ablation README](../../harness/ablate/README.md) shows how to make the schedules again. The only difference is the path of the work folder.

## Files that the scan finds

The command is:

```
DMK_IGNORE_FILE=tools/publish/gitleaks.ignore tools/publish/scan-dir.sh data/2026-10-31-ablation
```

The scan of the first version of the folder had the three schedule files. It found the 34 items that the section Files not added gives. We did not add these files. The scan of the files in `raw/` finds no item.
