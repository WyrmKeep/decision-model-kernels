# Method

This document gives the method of the project. It shows how we measure the speed of a change to llama.cpp. It also shows how we make sure that the answers of the model stay correct. The [glossary](glossary.md) gives the terms of the project.

## The two questions

The method answers two questions about each change.

| Question | Test | Section |
|---|---|---|
| Is the change faster? | Runs of the benchmark in blocks, with a confidence interval | [Speed test of two builds](#speed-test-of-two-builds) |
| Do the answers stay correct? | The guards, the tests of the equivalence class, and the KL divergence | [Equivalence classes](#equivalence-classes) |

We write the prediction before we measure. We also write the equivalence margin and the tolerances of each test before we measure. Then we cannot change a limit after we see the result. The notebook records each value with its date.

A result that is far from its prediction shows that we do not know where the time goes. We find the cause before the next change. This is also true when the result is faster than the prediction. We stop work on a target when the result is different from the prediction and we cannot find the cause.

We do not change the benchmark, its labels, or its guards to make a result better. If we find a fault in the benchmark, we make a new benchmark with a new baseline.

## The benchmark

The benchmark sends each case to `POST /v1/systemone` of a `llama-server`. It measures the speed and the quality of the answers. The [README of the benchmark](../bench/README.md) shows how to use it.

### The two benchmarks

The private benchmark has the names of internal hosts, networks, and paths. It is not in this repository. The speed numbers of the project come from it, and it is the measurement reference. The results table in the [README](../README.md) and the entry of phase 1 in the notebook use it.

The folder `bench/` has the public benchmark. It has one host name for all syslog lines and the domain name `example.net`. The IP addresses come from the ranges that RFC 5737 and `2001:db8::/32` give for examples. The cases hold no data from a server.

Each replacement has the same number of tokens as the text that it replaces. You can thus compare the speed numbers of the two benchmarks. The words of the two benchmarks are different. The answers can thus have small differences. The public benchmark has a different set of reference answers and a different baseline. You must not compare the quality numbers of the two benchmarks.

The files `bench/reference.json` and `bench/baseline.json` hold the mark `"status": "not-measured"` until a person measures them. While the mark is there, the client does not do the checks that use them.

### The cases

The benchmark has 48 cases with 100 decisions. Each decision has a label. The `size` of a case is `short`, `medium`, or `long`. The requests for these cases are short requests, medium requests, and long requests.

| `size` | Cases | Decisions | State length (tokens) |
|---|---:|---:|---|
| `short` | 32 | 52 | 11 to 84 |
| `medium` | 8 | 24 | 985 to 1,307 |
| `long` | 8 | 24 | 2,899 to 3,414 |
| Total | 48 | 100 | |

Source: `bench/DATA.md`. The tokenizer of the model gives the numbers of tokens.

### Phases of the client

The client `bench/bench.py` does the phases in this sequence.

| Phase | Requests | Work of the client |
|---|---:|---|
| `warmup` | 3 | It sends the first 2 cases and the last case. It does not use the times. |
| `quality` | 48 | It sends each case one time. It compares the answers with the labels and with the reference answers. |
| `seq` | 48 in each timed pass | It does the timed passes (`--passes`, default 5). It records the time of each request. |
| `tput` | 144 | It sends each case 3 times in a random sequence. The flag `--conc` sets the number of requests at the same time. The default is 8. |
| `cold`, optional | The value of `--cold` | It waits 30 s with no request. Then it sends one request of the first case. It does this the number of times that `--cold` gives. |

The requests of the `quality` phase have no `Ref:` line. Each other request starts with a line `Ref: <hash>`. The hash is different for each request. The line makes each request new for the cache of the server. A seed sets the random sequence of each timed pass. All runs thus have the same sequences.

### Metrics

The speed metrics are the primary result of a run.

| Metric | Unit | Meaning |
|---|---|---|
| `lat_ms` | ms | The primary metric. The median of the means of the 5 timed passes (the default). The mean of a timed pass is the mean latency of its 48 requests. |
| `short_ms`, `medium_ms`, `long_ms` | ms | The mean latency of all requests of the `seq` phase for the `short`, `medium`, or `long` cases. |
| `rps8` | requests/s | The number of requests of the `tput` phase, divided by the time of the phase. The default is 8 requests at the same time. |
| `spread_pct` | % | The range of the 5 means of the timed passes, divided by their median. It shows the noise of a run. |

The quality metrics compare the answers of the `quality` phase with the labels and with the reference answers. The reference answers come from the `Q8_0` model.

| Metric | Unit | Meaning |
|---|---|---|
| `correct` | decisions, of 100 | The number of decisions where the top answer is the label. |
| `agree` | 0 to 1 | The number of decisions where the top answer is the top answer of the reference, divided by 100. |
| `mean_dp` | probability, 0 to 1 | The mean, for the 100 decisions, of the difference between the probability of the answer and the probability of the reference answer. |

The top answer of a yes or no question is yes when its probability is 0.5 or more. The top answer of a `choice` question is the selection that the server gives. The probability difference of a `choice` question is the largest difference for all selections.

### Guards

A run is good when all guards are true. The baseline value is the number of the same name in `bench/baseline.json`.

| Guard | The guard is true when | Code |
|---|---|---|
| Errors | No request has an error. | `bench/bench.py` |
| `correct` | `correct` is the baseline value minus 1, or more. | `bench/bench.py` |
| `agree` | `agree` is the baseline value minus 0.02, or more. | `bench/bench.py` |
| `mean_dp` | `mean_dp` is the baseline value plus 0.01, or less. | `bench/bench.py` |
| VRAM | The largest value of `memory.used` is 6,144 MiB or less. | `bench/summarize.py` |

Source: `bench/bench.py` and `bench/summarize.py`.

The three quality guards use the baseline and the reference. If a file has the mark `not-measured`, the client does not do these guards. The script `summarize.py` then shows `n/a`, if the other guards are true.

The sampler writes 4 lines each second. The VRAM guard uses the largest value of `memory.used` in the sampler file of the run. If the run has no sampler file, the VRAM guard is false. The flag `--vram-cap-mib` or the environment variable `BENCH_VRAM_CAP_MIB` changes the limit of 6,144 MiB.

The baseline numbers come from 3 runs of the baseline build. The script `bench/tools/make_baseline.py` makes the file `baseline.json`. It stops when the quality numbers of the 3 runs are different.

### The 52-decision set

The folder `bench/dectest/` has the 32 `short` cases as a test with 52 decisions. The test is fast. It is not a benchmark. A difference of one to three answers between two builds is noise. We use it for a fast check of the quality of the answers. It is also one test of the `approximation` equivalence class.

## Speed test of two builds

We compare two builds in blocks. A block has one run of each arm, in a random sequence that a seed sets. A slow change of the host then has the same effect on the two arms. Each run of the benchmark starts a new server.

| Part | Rule | Code |
|---|---|---|
| Schedule | One line for each run. The name of a run is `<prefix>-b<block>-<tag>`. | `harness/make_schedule.py` |
| Server | Each run of the benchmark starts a new server for its arm. | `harness/remote/run_bench.sh` |
| Ratio | For each block and each metric: the value of arm B divided by the value of arm A. | `harness/analysis/ab_stats.py` |
| Estimate | The mean of the ratios of all blocks. | `harness/analysis/ab_stats.py` |
| Interval | The 95 % confidence interval from a bootstrap of the blocks, with 10,000 samples. | `harness/analysis/ab_stats.py` |
| Equivalence margin | We declare it before the first run. | The notebook entry |

Source: the scripts in `harness/`.

### Estimate and interval

The tool `ab_stats.py` uses only the blocks that have a run of the two arms. Its estimate is the mean of the ratios of these blocks. The column `diff_pct` is the estimate minus 1, in %.

The bootstrap makes 10,000 new sets of blocks from the blocks of the test. A new set has as many blocks as the test. A block can be in a new set more than one time. The tool calculates the mean of the ratios of each new set. It puts the 10,000 means in sequence. The interval goes from the 2.5 % percentile to the 97.5 % percentile of the means.

The flags `--boot` and `--seed` set the number of samples and the seed. The defaults are 10,000 and 12345. The tool writes these columns for each metric.

| Column | Meaning |
|---|---|
| `blocks` | The number of blocks. |
| `a_mean`, `b_mean` | The mean value of arm A and of arm B. |
| `diff_pct` | The estimate of the difference, in %. |
| `ci_lo_pct`, `ci_hi_pct` | The two ends of the 95 % confidence interval, in %. |
| `within_margin` | Only with the flag `--margin`. The value is true when the full interval is between minus and plus the value of the flag. |

For benchmark runs, the metrics are `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, and `rps8`. For `llama bench` runs, the tool uses the speed of the prompt in tokens/s for each prompt length, with the name `pp<N>_ts`. A time that is smaller is faster. A rate that is larger is faster.

### Equivalence margin

The equivalence margin is the largest difference that we accept as no difference. A change is equivalent when the full 95 % confidence interval is between minus and plus the equivalence margin. The flag `--margin` of `ab_stats.py` gives it in %.

The first test uses ±2 % as the equivalence margin. It has 6 blocks of benchmark runs and 4 blocks of `llama bench` runs. The [entry of phase 1](notebook/2026-10-30-phase1-equivalence-plan.md) declares them.

### Kernel changes

A change to a kernel has two tests. The first test measures the kernel only. We lock the SM clock during this test. The power limit then cannot change the clock.

The second test is a run of the benchmark. It uses the default server flags of `harness/remote/start_server.sh`, the default power limit of 200 W, and the default clocks. The default server flags are `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`. The runs are in the sequence arm A, arm B, and arm A again. The first run and the last run use the same arm. If their values are different, the host changed during the test.

## Types of work

The answer to a question is the softmax of the logits of some answer tokens at the end of the question. The server uses only these numbers. All other work is only for them. The work of a request has one of three types.

| Type | Meaning | Example of a change | Equivalence class of the example |
|---|---|---|---|
| Necessary work | The model reads the state one time and each question one time. It writes one output row for each answer token. | A faster MMQ kernel | `re-associated` |
| Work that is not necessary | Work that the answers do not use. | A change that calculates only the logits of the answer tokens | `exact` |
| Overhead | The time for the launches of kernels and the time between them. | CUDA graphs, or a fusion of kernels | `exact` or `re-associated` |

The prediction of a change gives the type of work that the change removes or makes faster.

## Equivalence classes

We select the equivalence class of a change before we measure it. The equivalence class sets the tests that the change must do.

| Equivalence class | Example | Tests |
|---|---|---|
| `exact` | A change that calculates only the logits of the answer tokens. A change to the launches or to the sequence of the kernels. | The answers are the same bit for bit as the answers of the same build without the patch. That build must first give the same answers bit for bit in all runs. If it does not, the tests of the `re-associated` equivalence class apply. |
| `re-associated` | A fusion of kernels. A different tile shape. Ragged batching. | `test-backend-ops` in its error limits, a KL divergence at noise level, and all guards. |
| `approximation` | A lower precision. A different quantization. | The guards, a limit for the KL divergence, and the 52-decision set. A person examines the result as a change of quality. |

Source: the method of the project, set before the first change.

A `re-associated` change adds the same numbers in a different sequence. The sums can then be different in the last bits. An `approximation` changes the numbers of the model, for example with a lower precision.

### Answer tests

The tool `harness/analysis/answers_compare.py` reads the answer files that `bench.py --write-reference LABEL` writes. It does two tests. The test of one arm compares all runs of that arm. The test of two arms compares each file of one arm with each file of the other arm.

| Test | Question | Fields of the output |
|---|---|---|
| Test of one arm | Do all runs of one arm give answers with the same values? | `self_repeat_a`, `self_repeat_b`, `self_a_max_dp`, `self_b_max_dp` |
| Test of two arms | Do the two arms give answers with the same values? | `cross_identical_all`, `cross_max_abs_dp`, `cross_top_flips` |

Source: `harness/analysis/answers_compare.py`.

The fields `self_a_max_dp` and `self_b_max_dp` give the largest probability difference between runs of one arm. The field `cross_max_abs_dp` gives it between the two arms. The field `cross_top_flips` lists the decisions that have a different top answer.

## KL divergence

The KL divergence shows how much the probabilities of the new build are different from the probabilities of the baseline build for the same text. We measure it with `llama perplexity`.

First, `llama perplexity` with the flag `--kl-divergence-base` runs on the baseline build. It writes the logits of the text to a file. Then `llama perplexity` with the flag `--kl-divergence` runs on the new build. It compares its logits with the logits in the file. The two runs use the same text.

The text sets do not change.

- The first 20 chunks of the WikiText-2 test file. A chunk has 512 tokens.
- One file with the 16 `medium` and `long` states of the benchmark.

The program gives three numbers for each text set. The table also gives the tolerances of the first test.

| Number | Unit | Meaning | Tolerance of the first test |
|---|---|---|---|
| Mean KL divergence for each token | None | The mean of the KL divergence of all tokens. It is 0 when the two builds give the same probabilities. | 1e-5 or less |
| Largest KL divergence for one token | None | The largest value of one token. It shows the worst token. | 1e-3 or less |
| Same top token | % | The part of the tokens where the two builds give the highest probability to the same token. It is 100 % when all tokens are the same. | 99.9 % or more |

Source: the [entry of phase 1](notebook/2026-10-30-phase1-equivalence-plan.md).

The entry declares these tolerances for the first test. The first test compares the build from source with the stock build. Each subsequent change declares its tolerances in its notebook entry before it measures. The `re-associated` equivalence class must have a KL divergence at noise level. The `approximation` equivalence class has a limit for the KL divergence, which we declare before the test.

## Tools

| Tool | Use |
|---|---|
| `compute-sanitizer` | Finds faults in a kernel. We use it on each new kernel with four tools. The table below gives them. |
| `test-backend-ops` | Does each operation on the GPU backend and on the CPU backend. It compares the two results. Its `perf` mode gives the time of each operation. |
| `llama bench` | Measures the speed of prompts. The first test uses prompts of 64, 256, and 3,000 tokens, and it measures each prompt 5 times. |
| `llama perplexity` | Measures the KL divergence of two builds. |
| Nsight Compute | Makes a profile of one kernel. It shows how close the kernel is to the limits of the GPU. It also shows the causes that make the warps wait and gives data for each line of the code. |
| Nsight Systems | Makes a profile of one run. It shows the time of each kernel and the time between two launches. |
| `cuobjdump`, `nvdisasm` | Show the SASS of a kernel. |
| `-Xptxas -v` | A compiler flag. It shows the registers and the shared memory of each kernel, and the registers that the compiler moves to memory. |

The four tools of `compute-sanitizer` find these faults.

| Tool | Fault that it finds |
|---|---|
| `memcheck` | Access to memory out of range |
| `racecheck` | Incorrect use of shared memory by threads |
| `initcheck` | Code that reads memory that has no value |
| `synccheck` | Incorrect use of the commands that make threads wait |

The harness runs a tool in a wrapper, for example `compute-sanitizer` or `ncu`, with the schedule line `tool@<w>`. The [harness](../harness/README.md) shows how.

## External checks

We also use checks that we did not make.

| Check | Result |
|---|---|
| The limit of the GPU | No result can be faster than the limit of the bandwidth and of the speed of the GPU. A result that is faster shows a measurement error. |
| The CPU backend | It does each operation in a different procedure from the GPU backend. `test-backend-ops` compares the two. |
| The labels | The 100 decisions with labels and the answers of the `Q8_0` reference show the quality of the answers. |
| A long test | A test of 24 h must show no GPU fault, no error of `compute-sanitizer`, and the same answers all the time. |

## Records and links

The [notebook](notebook/README.md) has one entry with a date for each test. We write the prediction, the equivalence margin, and the tolerances in the entry before the measurement. We write the result after the measurement.

Each change has one document in [the change documents](changes/README.md). It has the same text as the pull request of the change. It gives the equivalence class, the prediction, the tests, and the result.

The [harness](../harness/README.md) shows how to run a test. The [data folder](../data/README.md) shows where the data of each measurement goes.
