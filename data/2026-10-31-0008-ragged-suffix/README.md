# Data set: change 0008, ragged batching of the question suffixes (2026-10-31)

This data set holds the raw files and the derived tables for change 0008 of the fork of llama.cpp. The change puts the suffixes of some questions in one pass, also when the suffixes have different lengths. The [results entry](../../docs/notebook/2026-10-31-0008-ragged-suffix-result.md) and the [change document](../../docs/changes/0008-ragged-suffix.md) use these data. The rules for a data set are in the [data README](../README.md).

## Question

1. Speed: Is build `B` (arm `build:0008-ragged-suffix`) faster than build `A` (arm `build:lead-30489`) for the model "mix8"? The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`.
2. Prediction: Are the changes of `short_ms`, `medium_ms`, `long_ms`, `lat_ms`, `rps8` and `llama bench` the same as in the [prediction entry](../../docs/notebook/2026-10-31-0008-ragged-suffix-prediction.md)? The commit time of the entry is 2026-10-31 07:02:58 UTC. This is before the first run of build `B`.
3. Correctness: The change is in the equivalence class `re-associated`. Do the tests of this equivalence class give the results that the prediction entry declared?
4. Use of the change: Do ubatches with suffixes of different lengths occur in each phase of the private benchmark?

## Dates

| Item | Value |
|---|---|
| First run with build `B` starts | 2026-10-31 07:52:10 UTC |
| Last run starts | See the section "Raw files" and the table `runs.tsv` |
| First correctness measurement window | 07:51:23 to 07:57:27 UTC |
| Second correctness measurement window | 08:05:25 to 08:08:56 UTC |
| Speed measurement window | 08:15:42 to 08:31:59 UTC |
| Measurement window of the request types | 08:34:02 to 08:42:52 UTC |
| Measurement window of the census | See the file `raw/0008-type-census.meta` |

The GPU did the work of one run at a time. The project stopped the server in production for each measurement window and started it again at the end. The logs of `test-backend-ops` have no time. Four of them ran before the first measurement window, with the server in production on. The other logs came from the first and the second measurement window.

Source: the `date_utc` line of each `.meta` file, and the logs of the measurement windows.

## Project commit

The scripts of the measurement windows are the same as the files of the folder `harness/` in commit `2ef13adc1e9211652e1be2801a8d23b863528662` of this repository. That commit is the prediction entry. The only differences are the file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/`. The measurement windows did not use them. We compared the two folders with `diff -r` after the last measurement window.

The client `latency_by_type.py` is in this folder. The measurement windows ran it as a client of the harness. The benchmark in `bench/` has no change.

## Benchmarks

| Runs | Test | In this repository |
|---|---|---|
| `ragged0008-bench-*`, `0008-check-*`, `0008-check2-B` | The private benchmark, with 100 decisions | No |
| `ragged0008-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |
| `ragged0008-type-*`, `0008-type-census` | The client `latency_by_type.py`: the phase `seq` of the private benchmark, with the latency for each request type | The script is in this folder |
| `0008-dectest-*` | The 52-decision set | `bench/dectest/` |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `A` | Branch `dmk/lead-30489` of the fork of llama.cpp of this project. It is arm `L` of the [data set `2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). | `6b8b65e6672f07805000d3c9683b65c72cc62303` |
| `B` | Branch `dmk/0008-ragged-suffix` of the fork. It has three commits on top of the commit of `A`. | `82eb3a931a7dca82388a847dfc79f77746280b4b`. The source had 0 changes. |

The three commits of the change are:

| Commit | Contents |
|---|---|
| `c175584ef1c7ef486ce2ccb1b6698daad233756d` | The change: two new operations, the new code of `split_equal`, the hybrid memory, the graph of the model, and the tests |
| `766629ebcb235a12aae73d3f8db4215342c17b16` | A log line for each ubatch with suffixes of different lengths. The test `test-ragged-suffix` gets its limits from the KL floor. |
| `82eb3a931a7dca82388a847dfc79f77746280b4b` | The test `test-ragged-suffix` writes the NMSE of the states for each recurrent layer |

The first correctness measurement window used the build of commit `766629ebc`. The four `test-backend-ops` runs before it used this build or the build of commit `c175584ef`. [INFERENCE: we wrote the first build to the host while the work waited for the GPU lock.] All other runs used the build of commit `82eb3a931`.

The builds of `766629ebc` and `82eb3a931` have the same code for the server and for the CUDA backend. The file `raw/build/MANIFEST-0008-ragged-suffix.txt` is the manifest of the last build. It gives the SHA-256 value of each file.

## Build flags

We made build `B` with the build script of the project. The script made only the files that the new commits changed. It used the builder image of build `A`. The manifest shows the tools and the `cmake` command. The targets are `llama-server llama-app llama-bench llama-perplexity llama-tokenize test-backend-ops test-batch-alloc test-ragged-suffix`.

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
| Server container, `llama bench`, `llama perplexity` and `test-backend-ops`, of the two arms | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

Each arm runs in the same image. The container puts the build in `/opt/dmk`. The tool `compute-sanitizer` has version 12.8.93. It is in the folder `/tools` of the container.

## Model

| Use | File |
|---|---|
| All runs | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8") |
| The reference answers of the quality phase | The answers of the Q8_0 model. The `.json` files give their label: `ref-q8`. |

The runs of `llama perplexity` use two texts. One is WikiText-2, and the other is the text of the states of the benchmark. Each run uses 20 chunks of 512 tokens. The test program `test-ragged-suffix` uses the second text.

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all runs. The project did not change it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run. It is 1,845 to 1,905 MHz.

## Server flags

All runs used the flags of the server in production:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

The runs `0008-check-*` and `0008-san-memcheck` added `-lv 4`. The runs `0008-check2-B` and `0008-type-census` added `-lv 5`. The level 4 does not write the log lines of the ubatches. The level 5 writes them and also the text of the prompts. The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The one-minute load average of the host was 1.65 to 17.04 in the 8 runs of the private benchmark. It was 13.33 to 17.04 in the 8 runs of `llama bench`. The cause is not in the data. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. The block tables show it for each block.

## Commands

The schedules for the speed tests came from `make_schedule.py`. The files [raw/0008-ab.txt](raw/0008-ab.txt) and [raw/0008-type.txt](raw/0008-type.txt) have the lines. In the file `0008-type.txt` we changed the first word of each line from `tool` to `client` after `make_schedule.py` wrote it.

```
python3 harness/make_schedule.py --prefix ragged0008-bench --blocks 4 --seed 80080 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix ragged0008-llb --blocks 4 --seed 80081 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
python3 harness/make_schedule.py --prefix ragged0008-type --blocks 4 --seed 80082 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind tool \
    --args "/extra/0008/latency_by_type.py 18080 5"
```

The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The files [raw/0008-correctness.txt](raw/0008-correctness.txt), [raw/0008-correctness2.txt](raw/0008-correctness2.txt) and [raw/0008-census.txt](raw/0008-census.txt) are the schedules of the correctness runs and of the census. The commands below are not in a schedule file. Each ran in the container of the harness, in a measurement window, with the build in `/opt/dmk`.

```
test-backend-ops test -b CUDA0 -o SSM_CONV_RAGGED
test-backend-ops test -b CUDA0 -o GATED_DELTA_NET_RAGGED
test-backend-ops test -b CUDA0 -p "cache=1,lens="
test-backend-ops test -b CUDA0 -o GATED_DELTA_NET,CPY,SSM_CONV,L2_NORM,SOFTPLUS
test-backend-ops test -b CUDA0 --test-file <file of the 187 operations of the graph>
test-ragged-suffix -m <mix8 file> --text <text of the states> --lens 43,47,73 --prefix 1000
test-ragged-suffix -m <mix8 file> --text <text of the states> --lens 51,120 --prefix 60
test-ragged-suffix -m <mix8 file> --text <text of the states> --lens 2,33,2,64,31 --prefix 300
test-ragged-suffix -m <mix8 file> --text <text of the states> --lens 43,47,73 --prefix 600 --all-logits
test-batch-alloc
```

The second `compute-sanitizer` run started the server with `--tool memcheck --error-exitcode 99`. It sent one request of each type. A script then sent `SIGTERM` to the server and waited for the end of `compute-sanitizer`. The first `compute-sanitizer` run (`0008-san-memcheck`) used the harness. The harness stops the container with `SIGTERM`, and `compute-sanitizer` did not write its summary. We ran it again to get the summary.

The tables came from these commands. Arm `A` is the reference arm, and the ratio is `B` divided by `A`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix ragged0008-bench --a A --b B --margin 1
python3 ../../harness/analysis/ab_stats.py raw --prefix ragged0008-llb --a A --b B --margin 1
python3 ../../harness/analysis/answers_compare.py --a 'raw/ragged0008-bench-b*-A.reference.json' \
    --b 'raw/ragged0008-bench-b*-B.reference.json'
python3 data/2026-10-31-0008-ragged-suffix/make_tables.py
```

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `A` | `build:lead-30489` | The build of pull request #30489 on tag `b12004`. It is the build before this change. |
| `B` | `build:0008-ragged-suffix` | Build `A` and the three commits of this change |

Each run of the private benchmark started a new server. In each block, a seed set the sequence of the two arms. The column `first_arm` of the block tables shows it.

## Raw files

The harness wrote the files with the scripts `run_bench.sh`, `run_tool.sh` and `run_client.sh`. We did not change them. The test programs wrote the files in `raw/tests/` to their output, and a script kept them.

| Files | Written by | Contents |
|---|---|---|
| `ragged0008-bench-b<N>-<A or B>.*` | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `ragged0008-llb-b<N>-<A or B>.*` | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `ragged0008-type-b<N>-<A or B>.*` | `run_client.sh` | 8 runs of the client `latency_by_type.py`, with 7 files each: `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `0008-check-A.*`, `0008-check-B.*` | `run_bench.sh` | One pass of the private benchmark in the first correctness measurement window, with `-lv 4`. The files are the files of a benchmark run. |
| `0008-check2-B.*` | `run_bench.sh` | One pass of the private benchmark with `-lv 5` in the second correctness measurement window. It has no `.server.log`: see "Files not added". |
| `0008-check2-B.server-ubatch-lines.txt` | `grep` | The lines of the server log of `0008-check2-B` with `split_equal: sets with different lengths` or `all slots are idle`. 277 lines. |
| `0008-type-census.*` | `run_client.sh` | One pass of the client with `-lv 5`. It has no `.server.log`. It has a `.server-ubatch-lines.txt` file, made with the same `grep`. |
| `0008-kld-<wiki or domain>-A-base.*` | `run_tool.sh` | `llama perplexity` of arm `A` that writes the base file with the logits. 4 files each: `.out`, `.err`, `.smi`, `.meta`. |
| `0008-kld-<wiki or domain>-A-self.*` | `run_tool.sh` | `llama perplexity` of arm `A` against the base file of arm `A`. This gives the KL floor. |
| `0008-kld-<wiki or domain>-B-vsA.*` | `run_tool.sh` | `llama perplexity` of arm `B` against the base file of arm `A` |
| `0008-dectest-<A or B>.*` | `run_client.sh` | The 52-decision set on arm `A` or arm `B`. The file `.answers.json` has the answers. The file `.client.out` has the number of correct decisions. |
| `0008-san-memcheck.*` | `run_client.sh` | The first `compute-sanitizer` run, with `-lv 4`. The client sent 4 requests. The harness did not keep the summary of `compute-sanitizer`. |
| `0008-correctness.txt`, `0008-correctness2.txt`, `0008-census.txt` | A person | The schedules of the correctness measurement windows and of the census. They have comment lines. |
| `0008-ab.txt`, `0008-type.txt` | `make_schedule.py` | The schedules of the speed test and of the test of the request types |
| `tests/conv.log`, `tests/gdn.log`, `tests/tbo-cache.log`, `tests/builtin.log`, `tests/graph.log` | `test-backend-ops` | The logs of the 5 runs of `test-backend-ops` |
| `tests/inproc-<case>.log` | `test-ragged-suffix` | The logs of the 4 runs of the test program in the second correctness measurement window |
| `tests/test-batch-alloc.log` | `test-batch-alloc` | The log of the unit test of `split_equal` |
| `tests/memcheck2.out`, `tests/0008-san-memcheck2.*` | A script | The output of the second `compute-sanitizer` run, and its client files `.answers.json` and `.launch.log` |
| `tests/summary.txt`, `tests/inproc-summary.txt`, `tests/c2-summary.txt` | A script | The exit codes of the commands of the two correctness measurement windows |
| `build/MANIFEST-<arm>.txt` | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

The first correctness measurement window ran the four test programs `test-ragged-suffix` too. The second measurement window wrote the same file names again, and the first logs are not in the folder. The first measurement window gave the same numbers for the cases that it ran. The file `tests/inproc-summary.txt` shows the exit codes of that first run. The exit code 1 shows that a check of the test had the result `FAIL`.

Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and three tables are JSON files. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

### runs.tsv

One row for each run of the speed schedule, in the sequence of the start time. It has 16 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark` or `llama bench` |
| `line_type` | None | The type of the schedule line: `bench` or `tool` |
| `block` | None | The number of the block |
| `arm_tag` | None | `A` or `B` |
| `arm` | None | `build:lead-30489` or `build:0008-ragged-suffix` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `driver` | None | The version of the driver |
| `power_limit_w` | W | The power limit of the GPU |
| `throttle_reasons_at_start` | None | The value of `clocks_throttle_reasons.active` at the start. The value `0x0000000000000000` means that no cause was active. The value `0x0000000000000001` means that the GPU did no work. Empty for `tool` runs. |
| `exit_code` | None | The exit code of the program. Only for `tool` runs. |
| `lat_ms` | ms | The latency of the run: the median of the means of the timed passes. Only for benchmark runs. |
| `short_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the short requests, the medium requests, or the long requests in all timed passes |
| `rps8` | requests/s | The number of requests each second with 8 requests at the same time |
| `pp64_ts`, `pp256_ts`, `pp3000_ts` | tokens/s | The prompt speed of `llama bench` for a prompt of 64, 256 or 3,000 tokens. Only for `llama bench` runs. |
| `sm_clock_mhz_median` | MHz | The median of the SM clock in the samples of the run |
| `power_draw_w_median` | W | The median of `power.draw` in the samples of the run |
| `memory_used_mib_max` | MiB | The largest `memory.used` in the samples of the run |

Source: the `.meta`, `.load0`, `.load1`, `.smi`, `.json` and `.out` files of each run.

### speed-bench.tsv

The ratio of arm `B` to arm `A` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks, with the flag `--margin 1`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `A` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |
| `within_margin` | None | `True` if the interval is in ±1 %. The prediction declared this equivalence margin for `llama bench` only. For the private benchmark, the column has no meaning. |

### speed-llama-bench.tsv

The ratio of arm `B` to arm `A` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The script makes it with the flag `--margin 1`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | tokens/s | The mean of arm `A` |
| `b_mean` | tokens/s | The mean of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |
| `within_margin` | None | `True` if the interval is in ±1 %. The prediction declared this equivalence margin. |

### speed-bench-blocks.tsv

The difference of `B` from `A` in each block of the private benchmark, and the host load. It has 4 rows. The script calculates 100 times (the value of `B` divided by the value of `A`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_a_before`, `load1_a_after` | None | The one-minute load average before and after the run of `A` |
| `load1_b_before`, `load1_b_after` | None | The one-minute load average before and after the run of `B` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `B` from `A` |
| `rps8_diff_pct` | % | The difference of `rps8` of `B` from `A` |

### speed-llama-bench-blocks.tsv

The difference of `B` from `A` in each block of `llama bench`, and the host load. It has 4 rows.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_a_before`, `load1_a_after` | None | The one-minute load average before and after the run of `A` |
| `load1_b_before`, `load1_b_after` | None | The one-minute load average before and after the run of `B` |
| `pp64_ts_diff_pct`, `pp256_ts_diff_pct`, `pp3000_ts_diff_pct` | % | The difference of the prompt speed of `B` from `A` |

### quality-bench.tsv

The quality numbers of each run of the private benchmark. It has 8 rows. The guards are the limits of the benchmark on the answers, the errors and the speed.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `block`, `arm_tag` | None | The number of the block and the arm |
| `correct`, `decisions` | decisions | The number of correct decisions and the number of decisions |
| `agree` | decisions | The number of decisions with the same top answer as the reference answers |
| `mean_dp`, `max_dp` | probability | The mean and the largest difference of the probability from the reference answers |
| `top_answer_differs_from_reference` | decisions | The number of decisions with a different top answer than the reference |
| `errors` | None | The number of errors of the run |
| `guards_true` | None | `True` if all guards of the benchmark are true |

### answers-compare-bench.json

The output of `harness/analysis/answers_compare.py` for the 4 runs of arm `A` against the 4 runs of arm `B`. It has the fields `self_repeat_a` and `self_repeat_b`. It also has `cross_identical_all`, `cross_max_abs_dp` (the largest probability difference between the arms) and `cross_top_flips` (the decisions with a different top answer).

### type-runs.tsv

The mean latency of each request type in each run of the client `latency_by_type.py`. It has 8 rows. The types are short requests with one question (60 requests in a run), short requests with two questions (100), medium requests (40) and long requests (40).

| Column | Unit | Meaning |
|---|---|---|
| `run`, `block`, `arm_tag`, `arm`, `start_utc` | None | The name, block, short name of the arm, arm, and start time of the run |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `short1q_ms`, `short2q_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the requests of the type in the 5 timed passes |

### speed-type.tsv

The ratio of arm `B` to arm `A` for each request type. It has 4 rows. The script `make_tables.py` calculates it with the method of `ab_stats.py`. This is the mean of the ratios of the 4 blocks, with a bootstrap of 10,000 samples and seed 12345. The script checks that the method gives the numbers of `ab_stats.py` for `lat_ms`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `short1q_ms`, `short2q_ms`, `medium_ms` or `long_ms` |
| `blocks` | blocks | The number of blocks |
| `a_mean`, `b_mean` | ms | The mean of arm `A` and of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval |
| `block1_diff_pct` to `block4_diff_pct` | % | The difference of `B` from `A` in each block |

### check-runs.tsv

The three passes of the private benchmark of the correctness measurement windows, with the guards. It has 3 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `arm_tag` | None | The name of the run and the arm |
| `server_log_verbosity` | None | The value of `-lv` |
| `correct`, `decisions`, `agree`, `mean_dp`, `max_dp`, `top_answer_differs_from_reference`, `errors`, `guards_true` | None | The same as in `quality-bench.tsv` |
| `lat_ms`, `rps8` | ms, requests/s | The latency and the number of requests each second of the run. The log lines of the ubatches made the runs slower. Do not use them for the speed. |

### answers-compare-check.json

The output of `answers_compare.py` for the run `0008-check-A` against the run `0008-check-B`. It has the same fields as `answers-compare-bench.json`.

### decisions-52.tsv

The 52-decision set on the two arms. It has 2 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `arm_tag` | None | The name of the run and the arm |
| `correct`, `decisions` | decisions | The number of correct decisions and the number of decisions |
| `wrong_decisions` | None | The names of the incorrect decisions |

### answers-compare-decisions-52.json

The output of `answers_compare.py` for the answers of the two runs of the 52-decision set. The script first puts the answers into the file shape of the benchmark. The file has the same fields as `answers-compare-bench.json`.

### ops-tests.tsv

The result of each run of `test-backend-ops`. It has 5 rows.

| Column | Unit | Meaning |
|---|---|---|
| `test` | None | The name of the test |
| `filter` | None | The selection of cases that the command used |
| `log` | None | The log in `raw/` |
| `passed`, `cases` | cases | The number of cases with the result `OK` and the number of cases |
| `failing_operations` | None | The operations of the cases with the result `FAIL`, with the number of cases |

The CUDA output of each case must be the same as the CPU output, with an NMSE of 5e-4 or less. The 6 cases with `FAIL` in the graph file are the 6 cases that build `A` has.

### unit-tests.tsv

The result of `test-batch-alloc`. It has 1 row with the columns `test`, `tests`, `assertions`, `failures`, `exceptions` and `skipped`. All are numbers.

### suffix-compare.tsv

The results of the test program `test-ragged-suffix`. It has 20 rows: 5 results for each of the 4 cases. A case is one run of the program.

| Column | Unit | Meaning |
|---|---|---|
| `case` | None | `medium`, `short2q`, `five` or `alllogits` |
| `suffix_lengths` | tokens | The lengths of the suffixes, one for each sequence |
| `prefix_tokens` | tokens | The number of tokens of the shared prefix |
| `rows` | rows | The number of rows of logits that the test compares: one for each sequence, or one for each token with `--all-logits` |
| `comparison` | None | The two decodes that the row compares. `floor (equal again)` is the equal split against the equal split of a second run. |
| `kl_mean`, `kl_max` | `nats` | The mean and the largest KL divergence of the rows |
| `max_abs_logit` | logit | The largest difference of a logit, without its sign |
| `same_top_token`, `rows_compared` | rows | The number of rows with the same top token, and the number of rows |
| `conv_max_abs`, `ssm_max_abs` | None | The largest difference of the conv state and of the SSM state, without its sign |
| `conv_nmse`, `ssm_nmse` | None | The largest NMSE of one layer state |
| `conv_identical`, `conv_vectors`, `ssm_identical`, `ssm_vectors` | layer states | The number of layer states that are the same bit for bit, and the number of layer states |

### suffix-passes.tsv

The number of passes that each decode used for the suffixes. It has 20 rows with the columns `case`, `suffix_lengths`, `decode` and `passes_of_the_suffixes`.

### suffix-checks.tsv

The checks that the test program wrote, with the result `ok` or `FAIL`. It has 32 rows. The row with the result `info` gives the KL floor and the limits that the test used. The limits are the limits of the prediction entry.

### suffix-layers.tsv

The NMSE of the conv state and of the SSM state for each recurrent layer. Each row has 6 values: the NMSE of the two states for each of three pairs of decodes. The pairs are `ragged_vs_equal`, `equal_vs_one_at_a_time` and `ragged_vs_one_at_a_time`. The NMSE of a layer is the largest value of its sequences.

### kl-perplexity.tsv

The KL divergence of `llama perplexity` on the two texts. It has 4 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `text_set`, `text`, `comparison` | None | The run, the text set, the text, and the two runs that the row compares |
| `mean_kld`, `max_kld`, `kld_percentile_99_9` | `nats` | The mean, the largest, and the 99.9 percentile of the KL divergence |
| `same_top_p_pct` | % | The part of the positions with the same top token |
| `rms_dp_pct` | % | The `RMS` of the difference of the probability of the top token |
| `mean_ppl_q`, `mean_ppl_base` | None | The mean perplexity of the run and of the base file |

### ragged-ubatches.tsv

The ubatches with suffixes of different lengths in the pass of the private benchmark with `-lv 5` (`0008-check2-B`), for each phase of the benchmark. It has 4 rows. The script gives each line to a phase with the number of `all slots are idle` lines before it. The first of these lines comes when the server starts. Each request of the phases `warmup`, `quality` and `seq` ends with one of them.

| Column | Unit | Meaning |
|---|---|---|
| `phase` | None | `warmup`, `quality`, `seq` or `tput` |
| `requests` | requests | The number of requests of the phase |
| `ubatches_with_different_lengths` | ubatches | The number of ubatches where the suffixes of the sequences have different lengths |
| `ragged_taken`, `ragged_not_taken` | ubatches | The number of these ubatches that the program put in one pass, and the number that it did not |
| `tokens_in_taken`, `tokens_in_not_taken` | tokens | The sum of the tokens of these ubatches |
| `largest_step_taken`, `largest_step_not_taken` | tokens | The largest step between two lengths in these ubatches. The limit is 128. |
| `sequences_per_ubatch` | None | The number of ubatches for each number of sequences |

### census-type.tsv

The same table for the pass of the client `latency_by_type.py` with `-lv 5` (`0008-type-census`). It has the phases `warmup` and `seq`, with the same columns.

### memcheck.tsv

The result of the second `compute-sanitizer` run. The rows are the `ERROR SUMMARY`, the exit code of the server, the exit code of the client, and the four requests with their latency.

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0008-ragged-suffix/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library.

## Files not added

We added all files of the results folders of the schedules. We did not add these files.

| Files | Cause |
|---|---|
| `0008-check2-B.server.log` and `tests/0008-san-memcheck2.server.log` | The server wrote them with `-lv 5`. They have the full text of the prompts of the private benchmark. The text has names and paths that are not for a public repository. The file `0008-check2-B.server-ubatch-lines.txt` has the lines that the table uses. |
| `0008-type-census.server.log` | The same cause. The file `0008-type-census.server-ubatch-lines.txt` has the lines that the table uses. |
| The scripts of the measurement windows that ran the tests | They have paths of the hosts. The section "Commands" gives the commands. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 11 items in `raw/`. All 11 are false positives.

| Files | Rule | Cause |
|---|---|---|
| The 8 files `ragged0008-bench-b<N>-<A or B>.json` and the 3 files `0008-check-A.json`, `0008-check-B.json` and `0008-check2-B.json` | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` must have the 11 items before the scan gives no finding. The lines are below. The scan gives no finding with these lines. The cause is the same as in the other data sets.

```
data/2026-10-31-0008-ragged-suffix/raw/0008-check-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/0008-check-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/0008-check2-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b1-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b1-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b2-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b2-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b3-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b3-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b4-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix/raw/ragged0008-bench-b4-B.json:generic-api-key:46
```
