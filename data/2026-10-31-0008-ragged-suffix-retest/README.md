# Data set: change 0008, second test of ragged batching of the question suffixes (2026-10-31)

This data set holds the raw files and the derived tables for the second test of change 0008 of the fork of llama.cpp. The change puts the suffixes of some questions in one pass, also when the suffixes have different lengths. The [data set of the first test](../2026-10-31-0008-ragged-suffix/README.md) has the data of the first test. The [results entry](../../docs/notebook/2026-10-31-0008-ragged-suffix-retest-result.md) of the second test uses these data. The rules for a data set are in the [data README](../README.md).

## Question

The [prediction entry](../../docs/notebook/2026-10-31-0008-ragged-suffix-retest-prediction.md) gives the limits for the decision. It uses a decode of one sequence at a time as the reference. The data set answers these questions.

1. Logits and states: Is the result of the test program `test-ragged-suffix` for the 8 new inputs in the limits for the decision? The limits use the same reference for ragged batching and for the equal split.
2. Answers: Are the answers of build `B` with `-np 3` in the limits for the decision? The reference is build `A` with `-np 1`. The questions of a request run one at a time with `-np 1`.
3. The 52-decision set: Do build `A` and build `B` give the same number of correct decisions?
4. Operations and memory: Do the tests of `test-backend-ops`, `test-batch-alloc` and `compute-sanitizer` give the results that the prediction entry declared?
5. Speed: Is build `B` (arm `build:0008-ragged-rebased`) faster than build `A` (arm `build:stack-fbfd9cb`) for the model "mix8"? The test has 4 blocks of the private benchmark, 4 blocks of `llama bench` and 4 blocks of the client `latency_by_type.py`.
6. Use of the change: Do ubatches with suffixes of different lengths occur in each phase of the private benchmark?

The prediction entry declares no equivalence margin. The script runs `ab_stats.py` without the flag `--margin`.

## Dates

| Item | Value |
|---|---|
| Build of arm `A` | 2026-10-31 08:41:40 UTC |
| Commit of the prediction entry | 2026-10-31 09:19:03 UTC |
| Build of arm `B` | 2026-10-31 09:21:20 UTC |
| Correctness measurement window | 09:22:50 to 09:36:02 UTC |
| First run in a `.meta` file | 2026-10-31 09:26:26 UTC (`0008r-A3a`) |
| Last run of the correctness measurement window | 2026-10-31 09:35:34 UTC (`0008r-dectest-B`) |
| Speed measurement window | 09:36:02 to 09:59:32 UTC |
| Run of the graph file with the new operation numbers | 10:01 UTC (arm `A`) and 10:03 UTC (arm `B`), after the speed measurement window |

The GPU did the work of one run at a time. The project stopped the server in production for each measurement window and started it again at the end. The logs of `test-backend-ops`, `test-batch-alloc` and `test-ragged-suffix` have no time. They came from the first part of the correctness measurement window, before the run `0008r-A3a`. The commit of the prediction entry is before the build of arm `B` and before each run.

The runs of `test-backend-ops` with the new operation numbers (`graph2-A.log` and `graph2-B.log`) started after the speed measurement window closed. The GPU lock was free then. The times in the table are the times of the last write to the two logs. These runs were in no measurement window, and they do not measure speed.

| Result | Measurement window |
|---|---|
| Logits and states of the 8 inputs, `test-backend-ops` with the file of the first test, `test-batch-alloc`, `compute-sanitizer` | Correctness measurement window |
| Answers of the private benchmark with `-np 3` and `-np 1`, the 52-decision set, the census of the ubatches | Correctness measurement window |
| Speed of the private benchmark, `llama bench` and the client `latency_by_type.py` | Speed measurement window |
| `test-backend-ops` with the new operation numbers | None |

Source: the `date_utc` line of each `.meta` file, the logs of the measurement windows, the manifests of the builds and `git log`.

## Project commit

The scripts of the measurement windows are the same as the files of the folder `harness/` in commit `1f1a5cd7afb3fc7f0d08518f5530dcfd04e82f1b` of this repository. That commit is the prediction entry. The only differences are the file `harness/README.md` and the folders `harness/ablate/`, `harness/build/` and `harness/profile/`. The measurement windows did not use them. We compared the two folders with `diff -r` after the correctness measurement window.

The client `latency_by_type.py` is in this folder. It is the same file as in the data set of the first test. The measurement windows ran it as a client of the harness. The folders `bench/` and `harness/` have no change since the commit of the first prediction entry.

## Benchmarks

| Runs | Test | In this repository |
|---|---|---|
| `ragged0008r-bench-*`, `0008r-A3a`, `0008r-A3b`, `0008r-B3a`, `0008r-B3b`, `0008r-A1`, `0008r-B1`, `0008r-B3-lv5` | The private benchmark, with 100 decisions | No |
| `ragged0008r-llb-*` | `llama bench`, with prompts of 64, 256 and 3,000 tokens | Not applicable |
| `ragged0008r-type-*` | The client `latency_by_type.py`: the phase `seq` of the private benchmark, with the latency for each request type | The script is in this folder |
| `0008r-dectest-*` | The 52-decision set | `bench/dectest/` |
| `0008r-san-memcheck` | One request of each type in the server that `compute-sanitizer` runs | No |

The [method](../../docs/method.md) gives the private benchmark, its phases and its metrics.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `A` | Branch `dmk/stack` of the fork of llama.cpp of this project. It has the tag `b12004` and the changes 0001, 0007, 0002 (tests), 0004 and 0005. | `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978`. The source had 0 changes. |
| `B` | Branch `dmk/0008-ragged-suffix` of the fork. It has four commits on top of the commit of `A`. | `7b3698d452ca8a219b581974a32f3ae6c7456c54`. The source had 0 changes. |

The four commits of the change are:

| Commit | Contents |
|---|---|
| `f210825e6ffb9114710b1ec32d01c2f755f8b491` | The change: two new operations, the new code of `split_equal`, the hybrid memory, the graph of the model, and the tests |
| `93d51f9e24a7e0fc4073b777ba61862a14982739` | A log line for each ubatch with suffixes of different lengths. The test `test-ragged-suffix` gets its limits from the KL floor. |
| `e34f9dedef9b0784a85b827dbb989da978177dcd` | The test `test-ragged-suffix` writes the NMSE of the states for each recurrent layer |
| `7b3698d452ca8a219b581974a32f3ae6c7456c54` | The test `test-ragged-suffix` compares ragged batching and the equal split with a decode of one sequence at a time. It gives the limits for each input and the data for the sums. |

The code of the change is the code of the first test, put on the branch `dmk/stack`. The commits of the first test are on the branch `dmk/0008-ragged-suffix-v1` of the fork. Build `B` is the only build of this data set that has the test programs `test-batch-alloc` and `test-ragged-suffix`. The files `raw/build/MANIFEST-stack-fbfd9cb.txt` and `raw/build/MANIFEST-0008-ragged-rebased.txt` are the manifests of the two builds. They give the SHA-256 value of each file.

## Build flags

We made the two builds with the build script of the project in the builder image. The manifests show the tools and the `cmake` command. The targets of build `B` are `llama-server llama-app llama-bench llama-perplexity llama-tokenize test-backend-ops test-batch-alloc test-ragged-suffix`. The targets of build `A` are the same, without `test-batch-alloc` and `test-ragged-suffix`.

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
| Server container, `llama bench` and `test-backend-ops`, of the two arms | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Client container | `python:3.12-slim` | `sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159` |

Each arm runs in the same image. The container puts the build in `/opt/dmk`. The tool `compute-sanitizer` has version 12.8.93. It is in the folder `/tools` of the container.

## Model

| Use | File |
|---|---|
| All runs | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` ("mix8") |
| The reference answers of the quality phase | The answers of the Q8_0 model. The `.json` files give their label: `ref-q8`. |

The test program `test-ragged-suffix` uses two texts. One is WikiText-2, and the other is the text of the states of the benchmark. The flag `--kl-budget` gives the KL budget of the text. The budget is 0.027830 for WikiText-2 and 0.057587 for the text of the states. They are the KL divergence of mix8 from the Q8_0 model.

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`.

## Power limit

The power limit was 200 W in all runs. The project did not change it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_median` of [runs.tsv](tables/runs.tsv) shows the median SM clock of each run of the speed measurement window. It is 1,845 to 1,920 MHz. The median SM clock of each of the 7 runs of the private benchmark of the correctness measurement window is 1,845 to 1,860 MHz. The `.smi` files of the runs show it.

## Server flags

The runs of the server used the flags of the server in production. The list below shows the runs with different flags.

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

- The runs `0008r-A1` and `0008r-B1` used `-np 1`. The questions of a request run one at a time with `-np 1`.
- The run `0008r-B3-lv5` and the run `0008r-san-memcheck` added `-lv 5`. The level 5 writes the log line of each ubatch with suffixes of different lengths. It also writes the text of the prompts.
- The runs of `llama bench` used the flags in the section "Commands".

## Host load

The host has 12 CPU threads. The one-minute load average of the host was 1.91 to 18.34 in the 8 runs of the private benchmark of the speed test. It was 10.80 to 13.17 in the 8 runs of `llama bench`, and 3.38 to 14.43 in the 8 runs of the client `latency_by_type.py`. It was 1.39 to 2.15 in the 7 runs of the private benchmark of the correctness measurement window. The cause is not in the data.

The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) and [type-runs.tsv](tables/type-runs.tsv) show the load of each run. The block tables show it for each block.

## Commands

The schedules for the speed tests came from `make_schedule.py`. The files [raw/0008r-ab.txt](raw/0008r-ab.txt) and [raw/0008r-type.txt](raw/0008r-type.txt) have the lines. In the file `0008r-type.txt` we changed the first word of each line from `tool` to `client` after `make_schedule.py` wrote it.

```
python3 harness/make_schedule.py --prefix ragged0008r-bench --blocks 4 --seed 80090 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix ragged0008r-llb --blocks 4 --seed 80091 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
python3 harness/make_schedule.py --prefix ragged0008r-type --blocks 4 --seed 80092 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind tool \
    --args "/extra/0008/latency_by_type.py 18080 5"
```

The harness ran each schedule with this command:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

The file [raw/0008r-correctness.txt](raw/0008r-correctness.txt) is the schedule of the correctness runs of the private benchmark and of the 52-decision set. It has comment lines. The commands below are not in a schedule file. Each ran in the container of the harness, in the correctness measurement window, with the build in `/opt/dmk`. The measurement window ran them in this sequence, before the schedule.

```
test-backend-ops test -b CUDA0 -o SSM_CONV_RAGGED
test-backend-ops test -b CUDA0 -o GATED_DELTA_NET_RAGGED
test-backend-ops test -b CUDA0 -p "cache=1,lens="
test-backend-ops test -b CUDA0 -o GATED_DELTA_NET,CPY,SSM_CONV,KQ_MASK,CONCAT
test-backend-ops test -b CUDA0 --test-file <file of the 187 operations of the graph, first test>
test-batch-alloc
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 38,66,102 --prefix 800
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 29,31,90 --prefix 1500
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 77,130 --prefix 40
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 52,118 --prefix 120
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 4,21,9,55,26 --prefix 500
test-ragged-suffix -m <mix8 file> --text <text of the states> --kl-budget 0.057587 --lens 35,60,111 --prefix 1800 --all-logits
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 33,36,81 --prefix 250 --all-logits
test-ragged-suffix -m <mix8 file> --text <WikiText-2 file> --kl-budget 0.027830 --lens 61,64,99 --prefix 3000
```

The command with `-o GATED_DELTA_NET,CPY,SSM_CONV,KQ_MASK,CONCAT` and the command with the file of the 187 operations ran on the two arms. The other commands ran on arm `B`. The two operations `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED` are not in build `A`.

The command with the file of the 187 operations ran a second time on the two arms. It used the file `ops-all-stack.txt` and ran after the speed measurement window:

```
test-backend-ops test -b CUDA0 --test-file <ops-all-stack.txt>
```

The `compute-sanitizer` run started the server of arm `B` with `--tool memcheck --error-exitcode 99` and `-lv 5`. A client of the project sent one request of each type. A script then sent `SIGTERM` to the server and waited until `compute-sanitizer` stopped. The script wrote the exit code of the server and the line `ERROR SUMMARY` of the log of the server to the file `tests/memcheck.out`.

The tables came from these commands. Arm `A` is the reference arm, and the ratio is `B` divided by `A`. The script `make_tables.py` runs them in the folder of the data set.

```
python3 ../../harness/analysis/ab_stats.py raw --prefix ragged0008r-bench --a A --b B
python3 ../../harness/analysis/ab_stats.py raw --prefix ragged0008r-llb --a A --b B
python3 ../../harness/analysis/answers_compare.py --a 'raw/ragged0008r-bench-b*-A.reference.json' \
    --b 'raw/ragged0008r-bench-b*-B.reference.json'
python3 ../../harness/analysis/answers_compare.py --a 'raw/0008r-B3[ab].reference.json' \
    --b raw/0008r-A1.reference.json
python3 data/2026-10-31-0008-ragged-suffix-retest/make_tables.py
```

The script runs `answers_compare.py` for four groups of the correctness runs. The fourth command above is one of them. The file `tables/answers-compare.json` has the four outputs.

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `A` | `build:stack-fbfd9cb` | The build of the branch `dmk/stack`. It is the build before this change. |
| `B` | `build:0008-ragged-rebased` | Build `A` and the four commits of this change |

Each run of the private benchmark started a new server. In each block of the speed test, a seed set the sequence of the two arms. The column `first_arm` of the block tables shows it.

## Raw files

The harness wrote the files with the scripts `run_bench.sh`, `run_tool.sh` and `run_client.sh`. We did not change them. The test programs wrote the files in `raw/tests/` to their output, and a script kept them.

| Files | Written by | Contents |
|---|---|---|
| `ragged0008r-bench-b<N>-<A or B>.*` | `run_bench.sh` | 8 runs of the private benchmark, with 8 files each: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `ragged0008r-llb-b<N>-<A or B>.*` | `run_tool.sh` | 8 runs of `llama bench`, with 4 files each: `.out`, `.err`, `.smi`, `.meta` |
| `ragged0008r-type-b<N>-<A or B>.*` | `run_client.sh` | 8 runs of the client `latency_by_type.py`, with 7 files each: `.client.out`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1` |
| `0008r-<A3a, A3b, B3a, B3b, A1, B1, B3-lv5>.*` | `run_bench.sh` | One pass of the private benchmark in the correctness measurement window. Each run has 8 files: `.json`, `.reference.json`, `.server.log`, `.launch.log`, `.smi`, `.meta`, `.load0`, `.load1`. The runs `A3a`, `A3b`, `B3a` and `B3b` used `-np 3`. The runs `A1` and `B1` used `-np 1`. The run `B3-lv5` used `-np 3` and `-lv 5`. It has no `.server.log`: see "Files not added". |
| `0008r-B3-lv5.server-ubatch-lines.txt` | `grep` | The lines of the server log of `0008r-B3-lv5` with `split_equal: sets with different lengths` or `all slots are idle`. 270 lines. |
| `0008r-dectest-<A or B>.*` | `run_client.sh` | The 52-decision set on arm `A` or arm `B`. The file `.answers.json` has the answers. The file `.client.out` has the number of correct decisions. |
| `0008r-correctness.txt`, `0008r-ab.txt`, `0008r-type.txt` | A person, and `make_schedule.py` | The schedule of the correctness runs, and the schedules of the speed test and of the test of the request types |
| `ops-all-stack.txt` | The project | The file of the 187 operations of the graph with the new operation numbers. See the section "Graph file of the operations". |
| `tests/conv-B.log`, `tests/gdn-B.log`, `tests/fusion-B.log`, `tests/builtin-<A or B>.log`, `tests/graph-<A or B>.log`, `tests/graph2-<A or B>.log` | `test-backend-ops` | The logs of the runs of `test-backend-ops` |
| `tests/input<1 to 8>.log` | `test-ragged-suffix` | The logs of the 8 runs of the test program. The number of the input is the number of the row of the table of inputs in the prediction entry. |
| `tests/batchalloc.log` | `test-batch-alloc` | The log of the unit test of `split_equal` |
| `tests/memcheck.out`, `tests/0008r-san-memcheck.*` | A script | The output of the `compute-sanitizer` run, and its client files `.answers.json` and `.launch.log` |
| `tests/summary.txt`, `tests/summary-graph.txt` | A script | The exit codes of the commands of the correctness measurement window, and of the two runs with the new file |
| `build/MANIFEST-<arm>.txt` | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |

Each `.err` file and each `.server.log` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. It does not come from llama.cpp, and we saw no effect from it.

The logs of `test-ragged-suffix` have lines of the llama.cpp program that start with `~llama_context:`. Some of these lines are in the middle of the table of results of the log. The script removes them before it reads a log.

## Graph file of the operations

The runs of `test-backend-ops` with the file of the 187 operations of the graph (`graph-A.log` and `graph-B.log`) stopped with exit code 139 on the two arms. The code 139 is the code of a memory fault. The logs have no result line.

The file is `data/2026-10-30-test-backend-ops/raw/ops-all.txt`. Its operation numbers are from before change 0004. Change 0004 added the operation `KQ_MASK` at position 92 of the list of operations. In the branch `dmk/stack`, each operation after position 91 has a number that is 1 more than in the file.

The file `ops-all-stack.txt` is the same file with 1 added to each operation number of 92 or more. The number is 92 or 101 in 26 lines of the file. The logs `graph2-A.log` and `graph2-B.log` are the runs with the new file. We kept the logs of the first runs as data.

The runs with the new file gave 181 cases with the result `OK` of 187 on the two arms. The 6 cases with the result `FAIL` are the same on the two arms. They are 5 cases of the operation `CPY` and 1 case of the operation `GATED_DELTA_NET`. The data set [`2026-10-30-test-backend-ops`](../2026-10-30-test-backend-ops/README.md) has 6 cases with the result `FAIL` with the same names and shapes. The data set of the first test of change 0008 has the same count: 181 cases with the result `OK` of 187.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files, and three tables are JSON files. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script is the only source of the tables.

Each table of limits has a column `result`. The value `pass` means that the limit of the prediction entry is true. The value `FAIL` means that it is not true. The value `info` means that the prediction entry does not use the row for the decision.

### Logits and states of the 8 inputs

The program `test-ragged-suffix` makes four decodes of the suffixes of one input. The four decodes use the equal split, ragged batching, one sequence at a time, and the equal split again. The program compares the results. The prediction entry uses the decode of one sequence at a time as the reference.

In the tables, `seq` is the reference and `eq` is the equal split. `d(ragged)` is the value of ragged batching against the reference. `d(equal)` is the value of the equal split against the reference.

#### logits-inputs.tsv

The five quantities of the 8 inputs, from the table `QUANTITY` at the end of each log. It has 40 rows.

| Column | Unit | Meaning |
|---|---|---|
| `input` | None | The number of the input, 1 to 8 |
| `text` | None | `WikiText-2` or `text of the states`. The script reads it from the KL budget in the log. |
| `suffix_lengths`, `prefix_tokens` | tokens | The lengths of the suffixes, one for each sequence, and the number of tokens of the shared prefix |
| `rows` | rows | The number of rows of logits that the test compares: one for each sequence, or one for each token with `--all-logits` |
| `logits` | None | `last token of each suffix` or `all <N> tokens` |
| `quantity` | None | `kl_mean` (mean KL divergence), `kl_max` (largest KL divergence), `top_miss` (rows with a different top token), `conv_nmse` (largest NMSE of the conv state in the layers and the sequences), or `ssm_nmse` (the same for the SSM state) |
| `ragged_vs_seq` | `nats`, rows, or NMSE | `d(ragged)`: ragged batching against the reference. The value is in the unit of the quantity. |
| `equal_vs_seq` | `nats`, rows, or NMSE | `d(equal)`: the equal split against the reference |
| `ragged_vs_eq` | `nats`, rows, or NMSE | Ragged batching against the equal split |
| `ratio_ragged_to_equal` | None | `ragged_vs_seq` divided by `equal_vs_seq`. Empty when `equal_vs_seq` is 0. |
| `x3_input_log` | None | The result of the rule "each input, 3 times" in the log of the test program: `ok` or `FAIL` |
| `x3_input` | None | The same rule, calculated by the script: `d(ragged)` is 3 times `d(equal)` or less. When `d(equal)` is 0, the limit is 0. The script stops if the value is not the same as `x3_input_log`. |
| `info_class_log` | None | The level for information only, from the log: 1 if `ragged_vs_eq` is `d(equal)` or less, 2 if it is 2 times `d(equal)` or less, and 3 in all other cases |
| `info_class` | None | The same level, calculated by the script. The script stops if the value is not the same as `info_class_log`. |

The script also checks two facts. Each value of the table is the same as the value of the line that the log gives above it. The difference is less than 0.2 %. The value `top_miss` is the number of rows less the number of rows with the same top token.

#### logits-verdict.tsv

The limits of the prediction entry for the logits and the states. It has 28 rows.

| Column | Unit | Meaning |
|---|---|---|
| `rule` | None | The limit. `Sum of the 8 inputs`: the sum of `d(ragged)` is 1.5 times the sum of `d(equal)` or less. `Each input, 3 times`: `d(ragged)` of each input is 3 times `d(equal)` or less. `10 % of the KL budget`: the mean KL divergence of ragged batching is 10 % or less of the KL budget. Three rows give the checks of the test program. The rows `Information only` do not give the decision. |
| `quantity` | None | The quantity, the input, or the check |
| `value_ragged`, `value_equal` | `nats`, rows, or NMSE | For the sum, the sum of `d(ragged)` and the sum of `d(equal)`. For the other rules, `value_ragged` has the number of inputs or the value that the row names. |
| `ratio` | None | `value_ragged` divided by `value_equal` |
| `limit` | None | The limit of the rule |
| `result` | None | `pass`, `FAIL` or `info` |
| `note` | None | For the rule "each input, 3 times", the inputs above the limit with their values and ratios. For the rows `Information only`, the inputs of each level. |

#### logits-checks.tsv

The checks that the test program wrote, with the result `ok` or `FAIL`, and the budget line of the log (result `info`). It has 40 rows: 5 for each input.

| Column | Unit | Meaning |
|---|---|---|
| `input`, `suffix_lengths` | None | The number of the input and the lengths of its suffixes |
| `check` | None | The text of the check or of the budget line |
| `result` | None | `ok`, `FAIL` or `info` |

#### logits-compare.tsv

The lines of each log that compare two decodes. It has 40 rows: 5 for each input. The row `floor (equal again)` compares the equal split with the equal split of a second run. The row `ragged again vs ragged` compares ragged batching with a second run of ragged batching.

| Column | Unit | Meaning |
|---|---|---|
| `input`, `suffix_lengths`, `prefix_tokens`, `rows` | None | As in `logits-inputs.tsv` |
| `comparison` | None | The two decodes that the row compares |
| `kl_mean`, `kl_max` | `nats` | The mean and the largest KL divergence of the rows |
| `max_abs_logit` | logit | The largest difference of a logit, without its sign |
| `same_top_token`, `rows_compared` | rows | The number of rows with the same top token, and the number of rows |
| `conv_max_abs`, `ssm_max_abs` | None | The largest difference of the conv state and of the SSM state, without its sign |
| `conv_nmse`, `ssm_nmse` | None | The largest NMSE of one layer state |
| `conv_identical`, `conv_vectors`, `ssm_identical`, `ssm_vectors` | layer states | The number of layer states that are the same bit for bit, and the number of layer states |

#### logits-passes.tsv

The number of passes that each decode used for the suffixes. It has 40 rows with the columns `input`, `suffix_lengths`, `decode` and `passes_of_the_suffixes`.

#### logits-layers.tsv

The NMSE of the conv state and of the SSM state for each recurrent layer. It has 192 rows: 24 layers for each input. Each row has 6 values: the NMSE of the two states for each of three pairs of decodes. The pairs are `ragged_vs_equal`, `equal_vs_one_at_a_time` and `ragged_vs_one_at_a_time`. The NMSE of a layer is the largest value of its sequences. The columns are `input`, `layer`, and the six values.

### Answers of the private benchmark

#### correctness-runs.tsv

The quality numbers of each run of the private benchmark of the correctness measurement window, with the guards. It has 7 rows. The guards are the limits of the benchmark on the answers, the errors and the speed.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `arm_tag`, `arm` | None | The name of the run, the short name of the arm, and the arm |
| `slots` | None | The value of `-np` |
| `server_log_verbosity` | None | The value of `-lv`, or `default` |
| `start_utc` | UTC time | The start of the run, from the `.meta` file |
| `correct`, `decisions` | decisions | The number of correct decisions and the number of decisions |
| `agree` | decisions | The number of decisions with the same top answer as the reference answers of the Q8_0 model |
| `mean_dp`, `max_dp` | probability | The mean and the largest difference of the probability from the reference answers of the Q8_0 model |
| `top_answer_differs_from_reference` | decisions | The number of decisions with a different top answer than the reference of the Q8_0 model |
| `errors` | None | The number of errors of the run |
| `guards_true` | None | `True` if all guards of the benchmark are true |
| `lat_ms`, `rps8` | ms, requests/s | The latency and the number of requests each second of the run. A run has one pass. Do not use them for the speed. |

#### answers-compare.tsv

The two runs of each pair, and the difference of their answers. It has 10 rows. The answers of a run are in its `.reference.json` file. A decision is one question of one case. There are 100 decisions.

| Column | Unit | Meaning |
|---|---|---|
| `pair` | None | The name of the pair |
| `run_a`, `run_b` | None | The two runs. Side `a` is `run_a`. |
| `meaning` | None | The meaning of the pair |
| `identical_bit_for_bit` | None | `True` if all values of the two runs are the same |
| `decisions` | decisions | The number of decisions |
| `max_abs_dp` | probability | The largest value of `dp` in the 100 decisions. `dp` of a decision is the largest difference of the probability in the answer keys of the decision. The script uses the method of the functions `probs()` and `compare()` of `answers_compare.py`. |
| `mean_abs_dp` | probability | The mean value of `dp` in the 100 decisions |
| `decisions_with_dp_above_0` | decisions | The number of decisions with a `dp` of more than 0 |
| `top_answer_flips` | decisions | The number of decisions with a different top answer |
| `top_answer_flip_list` | None | Each decision with a different top answer. The text gives the decision and the top answer of side `a` and of side `b`. |

The script checks that `max_abs_dp` and the decisions with a different top answer are the values that `answers_compare.py` gives for the same files.

#### answers-compare.json

The outputs of `harness/analysis/answers_compare.py` for four groups of files: `np3_A_vs_B` (the runs `A3a` and `A3b` against `B3a` and `B3b`), `A_np3_vs_A_np1`, `B_np3_vs_A_np1` and `B_np1_vs_A_np1`. Each output has the fields `self_repeat_a`, `self_repeat_b`, `cross_identical_all`, `cross_max_abs_dp` (the largest probability difference of all pairs) and `cross_top_flips` (the decisions with a different top answer).

#### answers-verdict.tsv

The limits of the prediction entry for the answers, the 52-decision set and the guards. It has 8 rows.

| Column | Unit | Meaning |
|---|---|---|
| `rule` | None | The limit |
| `measure` | None | The value that the row gives |
| `build_b`, `build_a` | probability, decisions, or `True`/`False` | The value of build `B` and of build `A`. For the two rules with a ratio, each value is the larger value of the two runs of the arm, with `-np 3` against build `A` with `-np 1`. |
| `ratio_b_to_a` | None | `build_b` divided by `build_a` |
| `limit` | None | The limit of the rule |
| `result` | None | `pass` or `FAIL` |
| `note` | None | The value of each run, and each decision with a different top answer |

#### dectest.tsv

The 52-decision set on the two arms. It has 2 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run`, `arm_tag` | None | The name of the run and the arm |
| `correct`, `decisions` | decisions | The number of correct decisions and the number of decisions |
| `wrong_decisions` | None | The names of the incorrect decisions |

#### dectest-items.tsv

The answer of each item of the 52-decision set on the two arms. It has 52 rows.

| Column | Unit | Meaning |
|---|---|---|
| `item` | None | The case and the question |
| `top_answer_a`, `top_answer_b` | None | The top answer of arm `A` and of arm `B` |
| `same_top_answer` | None | `True` if the two top answers are the same |
| `correct_a`, `correct_b` | None | `True` if the item is not in the list `wrong_decisions` of the arm |
| `max_abs_dp` | probability | The largest difference of the probability in the answer keys of the item |
| `identical` | None | `True` if all values of the item are the same on the two arms |

#### answers-compare-decisions-52.json

The output of `answers_compare.py` for the answers of the two runs of the 52-decision set. The script first puts the answers into the file shape of the benchmark. The file has the same fields as `answers-compare.json`.

### Operations, unit test and memory check

#### ops.tsv

The result of each run of `test-backend-ops`. It has 9 rows.

| Column | Unit | Meaning |
|---|---|---|
| `test` | None | The name of the test |
| `arm_tag` | None | The arm of the run |
| `filter` | None | The selection of cases that the command used |
| `log` | None | The log in `raw/` |
| `exit_code` | None | The exit code of the command, from `tests/summary.txt` and `tests/summary-graph.txt` |
| `passed`, `cases` | cases | The number of cases with the result `OK` and the number of cases. Empty when the log has no result line. |
| `failing_operations` | None | The operations of the cases with the result `FAIL`, with the number of cases |
| `result` | None | `all cases OK`, the number of cases with `FAIL`, or the text `no result` with the exit code of a program that gave no result |

The CUDA output of each case must be the same as the CPU output, with an NMSE of 5e-4 or less.

#### ops-compare.tsv

The cases of the two arms for the tests that ran on the two arms with a result. It has 2 rows. The script compares the pairs of the text of a case and its result. Some cases have the same text.

| Column | Unit | Meaning |
|---|---|---|
| `test` | None | The name of the test |
| `cases_a`, `cases_b` | cases | The number of cases of arm `A` and of arm `B` |
| `cases_with_the_same_result_in_a_and_b` | cases | The number of cases with the same text and the same result on the two arms |
| `cases_with_FAIL_a`, `cases_with_FAIL_b` | cases | The number of cases with the result `FAIL` |
| `cases_of_a_without_the_same_result_in_b`, `cases_of_b_without_the_same_result_in_a` | cases | The number of cases that the other arm does not have with the same result |
| `cases_of_a_without_the_same_result_in_b_list`, `cases_of_b_without_the_same_result_in_a_list` | None | The text and the result of each of these cases |

#### unit-tests.tsv

The result of `test-batch-alloc`. It has 1 row with the columns `test`, `tests`, `assertions`, `failures`, `exceptions` and `skipped`. All are numbers.

#### memcheck.tsv

The result of the `compute-sanitizer` run. The rows are the `ERROR SUMMARY`, the exit code of the server, the exit code of the client, and the four requests with their latency.

### Use of the change

#### ubatches.tsv

The ubatches with suffixes of different lengths in the pass of the private benchmark with `-lv 5` (`0008r-B3-lv5`), for each phase of the benchmark. It has 4 rows. The script gives each line to a phase with the number of `all slots are idle` lines before it. The first of these lines comes when the server starts. Each request of the phases `warmup`, `quality` and `seq` ends with one of them.

| Column | Unit | Meaning |
|---|---|---|
| `phase` | None | `warmup`, `quality`, `seq` or `tput` |
| `requests` | requests | The number of requests of the phase |
| `ubatches_with_different_lengths` | ubatches | The number of ubatches where the suffixes of the sequences have different lengths |
| `ragged_taken`, `ragged_not_taken` | ubatches | The number of these ubatches that the program put in one pass, and the number that it did not |
| `tokens_in_taken`, `tokens_in_not_taken` | tokens | The sum of the tokens of these ubatches |
| `largest_step_taken`, `largest_step_not_taken` | tokens | The largest step between two lengths in these ubatches. The limit is 128. |
| `sequences_per_ubatch` | None | The number of ubatches for each number of sequences |

### Speed

#### runs.tsv

One row for each run of the speed schedule, in the sequence of the start time. It has 16 rows.

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The name of the run |
| `test` | None | `private benchmark` or `llama bench` |
| `line_type` | None | The type of the schedule line: `bench` or `tool` |
| `block` | None | The number of the block |
| `arm_tag` | None | `A` or `B` |
| `arm` | None | `build:stack-fbfd9cb` or `build:0008-ragged-rebased` |
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

#### speed-bench.tsv

The ratio of arm `B` to arm `A` for the speed of the private benchmark. It has 5 rows. The script `harness/analysis/ab_stats.py` makes it from the 4 blocks, without the flag `--margin`. The prediction entry declares no equivalence margin.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, or `rps8` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | ms, or requests/s for `rps8` | The mean of arm `A` |
| `b_mean` | ms, or requests/s for `rps8` | The mean of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

#### speed-llama-bench.tsv

The ratio of arm `B` to arm `A` for the prompt speed of `llama bench` in 4 blocks. It has 3 rows. The script makes it without the flag `--margin`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `pp64_ts`, `pp256_ts` or `pp3000_ts` |
| `blocks` | blocks | The number of blocks |
| `a_mean` | tokens/s | The mean of arm `A` |
| `b_mean` | tokens/s | The mean of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval, from a bootstrap of the blocks with 10,000 samples |

#### speed-bench-blocks.tsv

The difference of `B` from `A` in each block of the private benchmark, and the host load. It has 4 rows. The script calculates 100 times (the value of `B` divided by the value of `A`, minus 1) for each block.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_a_before`, `load1_a_after` | None | The one-minute load average before and after the run of `A` |
| `load1_b_before`, `load1_b_after` | None | The one-minute load average before and after the run of `B` |
| `lat_ms_diff_pct`, `short_ms_diff_pct`, `medium_ms_diff_pct`, `long_ms_diff_pct` | % | The difference of the latency of `B` from `A` |
| `rps8_diff_pct` | % | The difference of `rps8` of `B` from `A` |

#### speed-llama-bench-blocks.tsv

The difference of `B` from `A` in each block of `llama bench`, and the host load. It has 4 rows.

| Column | Unit | Meaning |
|---|---|---|
| `block` | None | The number of the block |
| `first_arm` | None | The arm that ran first in the block |
| `load1_a_before`, `load1_a_after` | None | The one-minute load average before and after the run of `A` |
| `load1_b_before`, `load1_b_after` | None | The one-minute load average before and after the run of `B` |
| `pp64_ts_diff_pct`, `pp256_ts_diff_pct`, `pp3000_ts_diff_pct` | % | The difference of the prompt speed of `B` from `A` |

#### quality-bench.tsv

The quality numbers of each run of the private benchmark of the speed test. It has 8 rows. The guards are the limits of the benchmark on the answers, the errors and the speed.

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

#### answers-compare-bench.json

The output of `harness/analysis/answers_compare.py` for the 4 runs of arm `A` against the 4 runs of arm `B` of the speed test. It has the fields `self_repeat_a` and `self_repeat_b`. It also has `cross_identical_all`, `cross_max_abs_dp` (the largest probability difference between the arms) and `cross_top_flips` (the decisions with a different top answer).

#### type-runs.tsv

The mean latency of each request type in each run of the client `latency_by_type.py`. It has 8 rows. The types are short requests with one question (60 requests in a run), short requests with two questions (100), medium requests (40) and long requests (40).

| Column | Unit | Meaning |
|---|---|---|
| `run`, `block`, `arm_tag`, `arm`, `start_utc` | None | The name, block, short name of the arm, arm, and start time of the run |
| `load1_before`, `load1_after` | None | The one-minute load average of the host before and after the run |
| `short1q_ms`, `short2q_ms`, `medium_ms`, `long_ms` | ms | The mean latency of the requests of the type in the 5 timed passes |

#### speed-type.tsv

The ratio of arm `B` to arm `A` for each request type. It has 4 rows. The script `make_tables.py` calculates it with the method of `ab_stats.py`. This is the mean of the ratios of the 4 blocks, with a bootstrap of 10,000 samples and seed 12345. The script checks that the method gives the numbers of `ab_stats.py` for `lat_ms`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | `short1q_ms`, `short2q_ms`, `medium_ms` or `long_ms` |
| `blocks` | blocks | The number of blocks |
| `a_mean`, `b_mean` | ms | The mean of arm `A` and of arm `B` |
| `diff_pct` | % | 100 times (the mean of the ratios of `B` to `A` minus 1) |
| `ci_lo_pct`, `ci_hi_pct` | % | The two ends of the 95 % confidence interval |
| `block1_diff_pct` to `block4_diff_pct` | % | The difference of `B` from `A` in each block |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0008-ragged-suffix-retest/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time. It uses Python 3 and no other library. It stops if one of the checks in it is not true.

## Files not added

We added all files of the results folders of the schedules. We did not add the files in this table.

| Files | Cause |
|---|---|
| `0008r-B3-lv5.server.log` and `tests/0008r-san-memcheck.server.log` | The server wrote them with `-lv 5`. They have the full text of the prompts of the private benchmark. The text has names and paths that are not for a public repository. The file `0008r-B3-lv5.server-ubatch-lines.txt` has the lines that the table uses. |
| The scripts of the measurement windows that ran the tests, and the logs of the measurement windows | They have paths and names of the hosts. The section "Commands" gives the commands. The section "Dates" gives the times. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds 15 items in `raw/`. All 15 are false positives.

| Files | Rule | Cause |
|---|---|---|
| `raw/0008r-<A1, A3a, A3b, B1, B3-lv5, B3a, B3b>.json` (7 files) and `raw/ragged0008r-bench-b<1 to 4>-<A or B>.json` (8 files) | `generic-api-key` | In line 46, the list of the decisions with a different answer has the case name `style03.secret`. The rule reads the next name in the list as a key. |

The file `tools/publish/gitleaks.ignore` must have these items before the scan gives no finding. The lines are below. The scan gives no finding with these lines. The cause is the same as in the other data sets.

```
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-A1.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-A3a.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-A3b.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-B1.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-B3-lv5.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-B3a.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/0008r-B3b.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b1-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b1-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b2-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b2-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b3-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b3-B.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b4-A.json:generic-api-key:46
data/2026-10-31-0008-ragged-suffix-retest/raw/ragged0008r-bench-b4-B.json:generic-api-key:46
```
