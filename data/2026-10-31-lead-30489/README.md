# Data set: lead 30489, GLU in the quantization of the activation (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of pull request #30489 of llama.cpp. The pull request fuses GLU into the quantization of the activation in MMQ. The project measures it as a lead and does not use it in production. The [results entry](../../docs/notebook/2026-10-31-lead-30489.md) uses these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is the build with the patch of #30489 faster than build `D` for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Answers: The patch is in the equivalence class `exact`. Does the build with the patch give the same answers as build `D`, bit for bit?
3. SASS: Which kernels for `sm_86` have different SASS in the build with the patch?

We wrote the prediction on 2026-10-30, before the measurement. It is in a file that is not in this repository. The notebook has no prediction entry for this test. The [results entry](../../docs/notebook/2026-10-31-lead-30489.md) gives the text of the prediction. The [inspection of the upstream code](../../docs/notebook/2026-10-30-phase0-upstream-survey.md) gives the source of this lead.

## Dates

| Item | Value |
|---|---|
| First run starts | 2026-10-31 00:47:53 UTC |
| Last run starts | 2026-10-31 01:03:49 UTC |
| Measurement window | 00:02 to 01:14 UTC |

The 16 runs of this data set are the runs of one schedule file. They used about 16 minutes. The measurement window also did other tests. The data of the other tests are not in this data set. The project stopped the server in production for the measurement window. One run at a time used the GPU.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv).

## Project commit

The scripts of the measurement window are the same as the files of the folder `harness/` in commit `e91d24b53413fb9922120760e5c8317a95a6d9c2` of this repository. The only differences are the file `harness/README.md` and the folder `harness/build/`. The measurement window did not use them. [INFERENCE: we compared the two folders with `diff -r` after the measurement window. The measurement window used a work copy of the files and not a clone of this repository.]

The two builds came from the file `build.sh` of the work copy. It is different from `harness/build/build.sh` in two comment lines only. The SASS compare used a version of `sass_compare.py` from the work copy. The file `harness/build/sass_compare.py` has a different structure and the same method. [INFERENCE: we did not run the file of this repository on the cubins.]

The speed numbers come from the client of the private benchmark. The measurement did not use the folder `bench/` of this repository.

## Benchmarks

| Runs | Benchmark | In this repository |
|---|---|---|
| `lead30489-bench-*` | The private benchmark, with 100 decisions | No |
| `lead30489-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `D` | Build `D` of tag `b12004`. It is the same build as in the data set [`2026-10-30-window1`](../2026-10-30-window1/README.md). | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |
| `L` | Branch `dmk/lead-30489` of the fork of llama.cpp of this project | `6b8b65e6672f07805000d3c9683b65c72cc62303`. The commit before it is the commit of `D`. |

The commit of `L` is a `git cherry-pick -x` of the one commit `730f0425773c0ece8fbdd5e73b80b749e6bf81b4` of pull request #30489. The commit changes 6 files, with 531 added lines and 31 removed lines. The files are `ggml-cuda.cu`, `mmq.cu`, `mmq.cuh`, `quantize.cu` and `quantize.cuh` in `ggml/src/ggml-cuda/`, and `tests/test-backend-ops.cpp`. The commit of the pull request is on top of `b12004`. The cherry-pick did not stop.

We made `D` on 2026-10-30 at 21:50 UTC and `L` on 2026-10-30 at 23:07 UTC. The files `raw/build/MANIFEST-b12004-sm86.txt` and `raw/build/MANIFEST-lead-30489.txt` give the facts of the builds and the SHA-256 value of each file.

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

The flags are the flags of `.devops/cuda.Dockerfile` at `b12004` with five differences. The five differences are in the [README](../2026-10-30-window1/README.md) of the data set `2026-10-30-window1`. The build of `L` used 577 s.

## Image

| Use | Image | Digest |
|---|---|---|
| Server container, and `llama bench`, of the two arms | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
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

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,935 MHz. The column `throttle_reasons_at_start` has the value `0x0000000000000024` for the two runs of block 4 of the private benchmark. The section "Tables" gives its meaning.

## Server flags

All runs used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The one-minute load average of the host was low in block 1 of the private benchmark and high in the next blocks. The cause is not in the data.

| Runs | One-minute load average |
|---|---|
| Block 1 of the private benchmark | 1.61 to 2.56 |
| Blocks 2 to 4 of the private benchmark | 3.00 to 18.70 |
| The 8 runs of `llama bench` | 14.61 to 18.70 |

The data set [`2026-10-30-window1`](../2026-10-30-window1/README.md) has a load of 2.13 to 2.84 in its blocks 4 to 6. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The tables [speed-bench-blocks.tsv](tables/speed-bench-blocks.tsv) and [speed-llama-bench-blocks.tsv](tables/speed-llama-bench-blocks.tsv) show it for each block.

## Commands

The schedules for the speed tests came from `make_schedule.py`. The file [raw/lead-30489.txt](raw/lead-30489.txt) has the lines. We ran the two commands again with the same seeds. The lines are the same as in the file.

```
python3 harness/make_schedule.py --prefix lead30489-bench --blocks 4 --seed 30489 \
    --arm D=build:b12004-sm86 --arm L=build:lead-30489 --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix lead30489-llb --blocks 4 --seed 30490 \
    --arm D=build:b12004-sm86 --arm L=build:lead-30489 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The script of the measurement window stopped the server in production, ran the schedule, and started the server in production again. The harness ran the schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The tables came from these commands. Arm `D` is the reference arm, and the ratio is `L` divided by `D`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix lead30489-bench --a D --b L --margin 2
python3 ../../harness/analysis/ab_stats.py raw --prefix lead30489-llb --a D --b L --margin 2
python3 ../../harness/analysis/answers_compare.py --a 'raw/lead30489-bench-b*-D.reference.json' \
    --b 'raw/lead30489-bench-b*-L.reference.json'
python3 data/2026-10-31-lead-30489/make_tables.py
```

The SASS files in `raw/sass-compare/` came from this command, with the cubins of the two builds. The cubins came from `cuobjdump -arch sm_86 -xelf all` on the file `libggml-cuda.so` of each build. The names that the command selected for `key-kernels.tsv` are not in the files.

```
python3 sass_compare.py <cubins of D> <cubins of L> <output folder> [name-regex ...]
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `D` | `build:b12004-sm86` | Build `D` of tag `b12004`, in the image of the stock build |
| `L` | `build:lead-30489` | Build `D` and the commit of pull request #30489, in the same image |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. In block 1 of the private benchmark, arm `L` ran first. In blocks 2 to 4, arm `D` ran first. In all 4 blocks of `llama bench`, arm `L` ran first. The seed gave this sequence.

## Raw files

The folder `raw/` has 106 files. The harness wrote 97 of them, and they have no change. The table groups the files by the name of the run. The `<N>` is the number of the block.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `lead30489-bench-b<N>-<D or L>.*` | 64 | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `lead30489-llb-b<N>-<D or L>.*` | 32 | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `lead-30489.txt` | 1 | `make_schedule.py` | The schedule of the 16 runs, with 4 comment lines |
| `sass-compare/sass-compare.json` | 1 | `sass_compare.py` | The summary of the SASS compare, the names with different SASS, and the names that only one build has |
| `sass-compare/key-kernels.tsv` | 1 | `sass_compare.py` | One row for each of 3,450 names. It has the number of instructions and registers of each build. It also shows if the SASS is the same. |
| `sass-compare/diff-<hash>.txt` | 4 | `sass_compare.py` | The difference of the SASS text of the 4 names with different SASS |
| `sass-compare/fused-ds4-swiglu-math.txt` | 1 | A person | The instructions that calculate SiLU in the fused kernel for the layout `DS4` and in the kernel `unary_gated_op_kernel<silu,float>` of build `D` |
| `build/MANIFEST-<arm>.txt` | 2 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The [harness document](../../harness/README.md) gives the meaning of each file type of a run. Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

The last lines of `fused-ds4-swiglu-math.txt` are a comment about the 4 `quantize_mmq_nvfp4` kernels and 2 lines of empty text. They show no instruction. The four files `diff-<hash>.txt` hold the instruction difference.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and one table is a JSON file. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

### runs.tsv

One row for each run, in the sequence of the start time. It has 16 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark` or `llama bench` |
| `line_type` | None | The type of the schedule line: `bench` or `tool` |
| `block` | None | The number of the block |
| `arm_tag` | None | `D` or `L` |
| `arm` | None | `build:b12004-sm86` or `build:lead-30489` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000000` means that no cause was active. The value `0x0000000000000024` is the sum of two bits: `0x04`, the power cap of the driver, and `0x20`, the heat limit of the driver. Empty for `tool` runs. |
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

The ratio of arm `L` to arm `D` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks, with the flag `--margin 2`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `D` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `L` |
| `diff_pct` | % | 100 times (the mean of the ratios of `L` to `D` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |
| `within_margin` | None | `True` if the interval is in ±2 %. The column is for the test with an equivalence margin. This measurement did not declare an equivalence margin. |

### speed-llama-bench.tsv

The ratio of arm `L` to arm `D` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The script `harness/analysis/ab_stats.py` makes it with the flag `--margin 2`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | tokens/s | The mean of arm `D` |
| `b_mean` | tokens/s | The mean of arm `L` |
| `diff_pct` | % | 100 times (the mean of the ratios of `L` to `D` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |
| `within_margin` | None | `True` if the interval is in ±2 %. The column is for the test with an equivalence margin. This measurement did not declare an equivalence margin. |

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

The output of `harness/analysis/answers_compare.py` for the 8 answer files `lead30489-bench-b*-D.reference.json` and `lead30489-bench-b*-L.reference.json`. It is a JSON file, not a TSV file. The [method](../../docs/method.md) gives the fields.

### sass-summary.tsv

The summary of the SASS compare of `libggml-cuda.so` of build `D` (A) and of build `L` (B), for `sm_86`. It has 15 rows. The script copies the `summary` object of `raw/sass-compare/sass-compare.json`. The [harness document](../../harness/README.md) gives the meaning of each key.

| Column | Unit | Meaning |
|---|---|---|
| `item` | None | The key of the summary |
| `value` | None | The value of the key |

### sass-names-changed.tsv

The 28 names that are not the same in the two builds. One row for each name.

| Column | Unit | Meaning |
|---|---|---|
| `name` | None | The full name of the function, with its template and its parameter types |
| `change` | None | `removed` if only build `D` has the name, `added` if only build `L` has the name, `different` if the two builds have the name with different SASS |
| `function` | None | The name of the function without its template |
| `layout` | None | The layout of the data of the Q8_1 quantization in the name: `D4`, `DS4` or `D2S6`. Only for `quantize_mmq_q8_1`. |
| `glu_op` | None | The GLU operation in the name of a kernel that only build `L` has. `none (COUNT)` is the kernel for the quantization with no fusion. Only for these kernels. |
| `instructions_a`, `instructions_b` | instructions | The number of SASS instructions in build `D` and in build `L`. Only for `different` names. |

The name has the operation as a number. The script decodes the numbers with the lists `ggml_glu_op` and `mmq_q8_1_ds_layout` in the source of the commit of `L`. The numbers 1, 2, 3 and 6 are `GEGLU`, `SWIGLU`, `SWIGLU_OAI` and `SWIGLU_CLAMP`. The number 7 is `GGML_GLU_OP_COUNT`.

### sass-functions.tsv

The names of `raw/sass-compare/key-kernels.tsv` in groups. One row for each function name without its template. It has 67 rows. The names are in the two builds.

| Column | Unit | Meaning |
|---|---|---|
| `function` | None | The name of the function without its template |
| `names` | names | The number of names in the group |
| `sass_identical`, `sass_different` | names | The number of names with the same SASS text, and with different SASS text |
| `encoding_identical` | names | The number of names with the same code of the instructions, with the control bits |
| `instructions_a`, `instructions_b` | instructions | The sum of the numbers of SASS instructions of build `D` and of build `L` |
| `max_registers_a`, `max_registers_b` | registers | The largest number of registers for each thread in the group |
| `max_static_shared_a_bytes`, `max_static_shared_b_bytes` | bytes | The largest `SHARED` value in the group, from `cuobjdump`. It does not include the shared memory that the launch sets. |

The file has 3,450 of the 6,748 names that are in the two builds. The other names are not in the file. Their SASS is in the numbers of `sass-summary.tsv`.

## Steps to make the tables again

1. Run `python3 data/2026-10-31-lead-30489/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folder of the schedule. We did not add these files.

| Files | Cause |
|---|---|
| The two tables `lead-30489-bench-ab.tsv` and `lead-30489-llb-ab.tsv` of the work copy | A script makes these tables from the raw files. The files `tables/speed-bench.tsv` and `tables/speed-llama-bench.tsv` have the same values. |
| The cubins, the libraries and the files `*-resusage.txt` of the SASS compare | They are binary files or large files. The manifests give the SHA-256 value of each library. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 8 items in `raw/`. All 8 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 8 files `lead30489-bench-b<N>-<D or L>.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` lists the 8 items. `publish.sh` reads it. For the hook or for `scan-dir.sh`, set `DMK_IGNORE_FILE` to the path of that file.
