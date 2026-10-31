# How we got here

This document tells how the project went from its first question to the checkpoint of phase 4. It goes phase by phase. For each phase it gives the question, the method, the result, the problems that we found, and the corrections. You can read it with no other information. Each number has a link to its source in the repository.

All times are UTC. The mark [INFERENCE] shows a sentence that has information that we did not measure. The [README](../README.md) has the short version.

## Dates of the work

| Time (UTC) | Work | Source |
|---|---|---|
| 2026-10-30, 21:05 | We started the project. | |
| 2026-10-30 | Phase 0: we examined upstream llama.cpp. | [Entry of phase 0](notebook/2026-10-30-phase0-upstream-survey.md) |
| 2026-10-30, 21:35 | We wrote the prediction entry of phase 1. No measurement came before it. | [Prediction entry of phase 1](notebook/2026-10-30-phase1-equivalence-plan.md) |
| 2026-10-30, 21:50 | We made build `D` from source. | [Result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) |
| 2026-10-30, 22:07 to 22:51 | Measurement window 1: the tests of phase 1 and of phase 2. | [Data set of the first measurement window](../data/2026-10-30-window1/README.md) |
| 2026-10-30, 22:35 | We made the ablation build. | [Data set of the profiles](../data/2026-10-31-profiles/README.md) |
| 2026-10-30, 22:52 to 22:56 | The run for the limits of the GPU. | [Data set of the limits](../data/2026-10-30-card-limits/README.md) |
| 2026-10-31, 00:02 to 01:14 | Measurement window 2: the baseline of the public benchmark, the ablation, the profiles, and the lead #30489. | [Entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md) |
| 2026-10-31, about 02:30 | The checkpoint of phase 4: the ranked list. | [Entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md) |

Source: the dates in the notebook entries and in the README of each data set. The owner gave the start time and the time of the checkpoint. They are not in a file of the repository.

## Start: the model, the task, and the GPU

### The model

The model is H2O Lightning 4B. Its layers are the same as the layers of Qwen3.5-4B. It is a hybrid model with 32 layers in 8 groups of four. Each group has three recurrent layers (Gated DeltaNet) and then one attention layer. In all, the model has 24 Gated DeltaNet layers and 8 attention layers.

The "mix8" quantization uses Q4_0 for the weights of the feed-forward layers. It uses Q8_0 for all other weights. The quality of the answers is the cause of this mixture. On the private benchmark, Q8_0 for all weights gave 91 correct decisions of 100. Q4_0 for all weights gave 93, and mix8 gave 94. The [model document](model.md) has the facts.

### The task

The model is a decision model. A client sends one state and a list of one to three questions to `POST /v1/systemone` of `llama-server`. The server gives one probability for each possible answer. Each answer is one token. The server reads the logits of the answer tokens only. The output head continues to calculate all 248,320 rows of the vocabulary.

The benchmark has 48 cases with 100 decisions. The name of a request type shows the length of the state and the number of questions.

| Request type | Cases | Questions in a request | Passes in a request |
|---|---:|---:|---:|
| `short1q` | 12 | 1 | 1 |
| `short2q` | 20 | 2 | 3 |
| `medium3q` | 8 | 3 | 4 |
| `long3q` | 8 | 3 | 5 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md) and the [method](method.md#the-cases). The state of a short request has 11 to 84 tokens. The state of a medium request has 985 to 1,307 tokens. The state of a long request has 2,899 to 3,414 tokens.

The questions of one request have a shared prefix. The first question of a group is the parent task. It runs the model on the shared prefix. The other questions are child tasks. They wait for the parent task and then run the model on the tokens after the prefix.

The hybrid memory of llama.cpp has a rule: all slots in one ubatch have the same number of tokens. Each different question length thus adds one pass. The [model document](model.md#equal-lengths-in-a-ubatch) gives the cause. The rule is the cause of the numbers in the last column of the table.

### The GPU

The GPU is an NVIDIA GeForce `RTX 3060 Ti`. It has the architecture `sm_86`, 38 SMs, and 8 GiB of memory. Its power limit is 200 W. The driver is 595.71.05 and CUDA is 12.8.1. The [README](../README.md#the-test-device) has the table of the test device.

### Change llama.cpp, not the flags

We tuned the flags of the server before the project started: `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`. The work before the project also measured the time of a short request with one question. The time was 36.7 ms. The time to read the layer weights one time is about 6 ms [INFERENCE: we calculated it from 2.7 GB of layer weights and 448 GB/s].

The work before the project did not find where the other 30 ms go. One possible cause is the work of the server for each request. Other possible causes are the tokenizer, the copies of the layer states, and the time between launches [INFERENCE: we did not measure these causes].

The flags cannot show the cause. Thus we measure in llama.cpp and change its code: the kernels, the graph, and the batches. The [model document](model.md#where-the-time-goes) has the numbers from the work before the project.

## Phase 0: the work of upstream

The question of phase 0 was: which work is in upstream llama.cpp, and which work is new? We did not want to do the work that upstream did. We also wanted to find leads that we can measure.

The method was an inspection with read-only access on 2026-10-30. The baseline was the release `b12004` (commit `3c809f4`). The newest release at that time was `b12017`. We wrote 200 queries for ten targets, `a` to `j`. The queries gave 7,871 rows with 2,322 different items: 1,113 pull requests, 891 issues, and 318 discussions.

We read 169 items in full. We also read the source files of `b12004` that the entry names. The [entry of phase 0](notebook/2026-10-30-phase0-upstream-survey.md#6-method-and-limits) gives the method and its limits.

| Group of results | Targets | Result |
|---|---|---|
| Done in `b12004` | `h`, `c` (decode), `e` | The chunked kernel of Gated DeltaNet for prefill, CUDA graphs in chunks for decode, and the tile tables and stream-k for MMQ |
| Done in part | `d` | The MMQ fusion of `gate`, `up` and GLU is in `b12004`. It does not include Q4_0, the type of the feed-forward layers of this model. |
| In progress upstream | `a`, `i`, `j` | A smaller output head for drafts, a repair of a decode pass that is not necessary, and parts of the copy of the layer state |
| No item found | `b`, `f`, `g`, graphs for prompt batches | Ragged batching for hybrid memory, int8 matrix kernels for Ampere, a megakernel, and CUDA graphs for prompt batches |

Source: the [summary table of the entry of phase 0](notebook/2026-10-30-phase0-upstream-survey.md#2-summary-table).

Four upstream items are important for the project:

- [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) fuses GLU into the quantization of the activation in MMQ. The project can cherry-pick it. The author measured +1.0 % to +1.8 % for the prompt speed on a 4090 GPU with Q4_K models.
- [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) removes a decode pass that is not necessary. The pass occurs when a child task has a lower slot number than its parent task.
- [#28495](https://github.com/ggml-org/llama.cpp/issues/28495) reports that flash attention reads too many KV cells with `--kv-unified`. The persons who wrote in the issue measured 42 % to 54 % lower prefill speed on HIP.
- [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) adds a smaller output head for drafts. It is the nearest upstream method for an output head for the answer tokens only. It is for drafts and not for decisions.

We also found two facts about the code. MMQ has no `cp.async` code. Only the flash attention code includes `cp-async.cuh`. A pull request that adds `cp.async` to MMQ for Blackwell ([#28572](https://github.com/ggml-org/llama.cpp/pull/28572)) closed. A maintainer wrote that the maintainer wants to make a different change to the kernel. A local patch for `sm_86` can thus have an overlap with subsequent work in upstream.

The commits after `b12004` did not change the result. The branch `master` had 13 more commits. None of them changed the CUDA code, the code that divides batches, the hybrid memory, or the decision server.

Phase 0 had two limits. The result "No item found" shows only that the queries found no item. The inspection also did not run a benchmark.

## Phase 1: the build from source and its test

### The question

Each change in the project has a baseline build to compare with. The baseline must be a build that we make from source. We cannot add a change to the stock image `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004`. Thus we compare each change with a build that we make from source. We must first know that this build is equal to the stock image. The question of phase 1 was: does the build that we make give the same speed and the same answers as the stock image?

### The set-up of the repositories

Before the first measurement we made the tools for the work. We made them first, because the project will have a public repository.

- **Repositories.** The work is in a git server that only the persons of the project can read. The public repository is on GitHub. The code of llama.cpp is in a fork. Each change has one branch in the fork.
- **Publication gate.** Data goes to GitHub only through the [publication gate](../tools/publish/README.md). A hook scans each git push, and a script scans all refs again before the data goes to GitHub. The publication gate stops secrets, internal names, IP addresses of local networks, large files, and model files.
- **STE checker.** All documents follow ASD-STE100 Issue 9. The [checker](../tools/ste/README.md) reads the dictionary from a local copy of the specification. The [glossary](glossary.md) gives one meaning to each term of the project.
- **Notebook.** Each test has an entry with a date. We write the prediction entry before the measurement. The [notebook](notebook/README.md) gives the rules.

### The build from source

We made build `D` on 2026-10-30 at 21:50 UTC with `harness/build/build.sh`. The script builds llama.cpp tag `b12004` in a container. The tools are the image `nvidia/cuda:12.8.1-devel-ubuntu24.04`, `nvcc` 12.8.93, `gcc-14` 14.2.0, and CMake 3.28.3. The source had no change.

The CMake flags are the flags of `.devops/cuda.Dockerfile` of the tag `b12004`, with five differences. The build is for `sm_86` only (`-DCMAKE_CUDA_ARCHITECTURES=86-real`). The tests are on (`-DLLAMA_BUILD_TESTS=ON`). The other three differences are `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`, `-DLLAMA_USE_PREBUILT_UI=OFF`, and `-DGGML_CCACHE=OFF`. The harness puts the build into the stock image at `/opt/dmk`. Thus the two arms use the same image.

### The prediction entry

We wrote the [prediction entry](notebook/2026-10-30-phase1-equivalence-plan.md) at 21:35 UTC, before the first measurement. It had four parts.

- **Arms.** Arm S is the stock build. Arm D is build `D`.
- **Speed test.** 6 blocks of the private benchmark and 4 blocks of `llama bench`. The limit is a 95 % bootstrap interval in ±2 % for `lat_ms` and for each prompt length. We also declared that the estimate of `lat_ms` must be in ±0.5 %.
- **Answer test.** The answers of the two arms must be the same bit for bit. If they are not, the tolerances of the entry apply.
- **Prediction.** No difference in speed. The answers are the same bit for bit if the SASS of the two builds for `sm_86` is the same.

### The result

The measurement window was 22:07 to 22:51 UTC. Two checks came before it. The `sm_86` SASS of build `D` was the same as the SASS of the stock image for all 6,754 kernel names. The `MUL_MAT` tests of `test-backend-ops` gave 2,490 correct tests of 2,490.

Build `D` was in all limits of the prediction entry.

| Metric | S mean (ms) | D mean (ms) | D against S (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` | 239.2 | 238.3 | -0.397 | -0.759 | -0.091 |
| `short_ms` | 72.3 | 72.1 | -0.314 | -0.669 | -0.002 |
| `medium_ms` | 360.3 | 359.3 | -0.276 | -0.561 | -0.041 |
| `long_ms` | 786.1 | 784.5 | -0.208 | -0.460 | -0.038 |

Source: the [result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) and the table `speed-bench.tsv` of the [data set](../data/2026-10-30-window1/README.md). The estimate is the mean of 6 block ratios.

| Prompt length (tokens) | S mean (tokens/s) | D mean (tokens/s) | D against S (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,833.7 | 2,840.5 | +0.238 | -0.113 | +0.543 |
| 256 | 4,231.0 | 4,231.4 | +0.010 | -0.066 | +0.085 |
| 3,000 | 4,615.9 | 4,618.3 | +0.052 | -0.090 | +0.194 |

Source: the [result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) and the table `speed-llama-bench.tsv` of the [data set](../data/2026-10-30-window1/README.md). The estimate is the mean of 4 block ratios.

The answers were the same bit for bit. The 6 runs of arm S gave the same answers. The 6 runs of arm D gave the same answers. The two arms gave the same answers. The largest probability difference was 0.0, and no decision had a different top answer. All 12 runs gave 94 correct decisions of 100.

The KL divergence gave the same result. D gave the same numbers as a second run of S in all decimal places of the table.

| Text | Mean KL divergence | Largest KL divergence | Same top token (%) |
|---|---:|---:|---:|
| WikiText-2 test | 0.000000 | 0.000058 | 99.980 |
| Benchmark states (16 states of medium requests and long requests) | 0.000000 | 0.000055 | 99.922 |

Source: the [result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) and the table `kld.tsv` of the [data set](../data/2026-10-30-window1/README.md).

Build `D` became the baseline for all work after phase 1. The speed of build `D` is the speed of the stock image in the limits of the test. The answers do not change.

### The problems of phase 1

**The load of the host changed the result.** Other work used the host CPU in blocks 1 to 3 of the benchmark. The 95 % interval of `lat_ms` does not include 0. Blocks 1 to 3 cause this [INFERENCE: we did not calculate the interval for blocks 4 to 6 only].

| Block | Load of the host (1-minute load average, before and after the two runs) | D against S, `lat_ms` (%) |
|---:|---|---:|
| 1 | 12.57 to 19.53 | -1.162 |
| 2 | 15.04 to 19.53 | -0.261 |
| 3 | 2.91 to 17.84 | -0.756 |
| 4 | 2.13 to 2.84 | -0.148 |
| 5 | 2.13 to 2.67 | -0.004 |
| 6 | 2.58 to 2.74 | -0.049 |

Source: the section "Host load and its effect" of the [result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md#host-load-and-its-effect). We used the table of the runs in the entry to find the range of the load.

The runs at a high load were about 2 % slower [INFERENCE: no test changed only the load]. The harness records the load of the host for each run from this point on.

**The KL floor was too near the limit.** A run against the base file that the same build wrote does not give a KL divergence of 0. The base file keeps the log-probabilities in 16 bits [INFERENCE: we read this in the code of `llama perplexity`].

The same top token was 99.922 % on the benchmark states for a second run of S against the base file of S. The limit was 99.9 %. The prediction entries after phase 1 compare with the KL floor of the same text. They do not use one number for all texts.

**One line of the prediction was not correct.** The prediction entry gave that only the list of GPU architectures is different. Five CMake flags are different. The SASS is the same, and thus the result is the same. The result entry marks this row "Not fully".

## Phase 2: the baseline of correctness

### The question

A change to a kernel must not make the answers worse. Phase 2 had one question: which numbers must each change keep? This entry has no prediction entry, because it measures a baseline and not a change. The [result of phase 2](notebook/2026-10-30-phase2-correctness-baseline.md) uses the same measurement window as phase 1.

### The KL budget

The model mix8 comes from the Q8_0 model by quantization. The KL divergence of mix8 from the Q8_0 model is the error that this quantization makes. The name of this error is the KL budget. A change to a kernel must add much less error than the KL budget.

| Text | Mean KL divergence | Largest KL divergence | Same top token (%) |
|---|---:|---:|---:|
| WikiText-2 test | 0.027830 | 1.568225 | 90.843 |
| Benchmark states (16 states of medium requests and long requests) | 0.057587 | 11.218664 | 88.588 |

Source: the [result of phase 2](notebook/2026-10-30-phase2-correctness-baseline.md#kl-budget-mix8-against-the-q8_0-model) and the table `kld.tsv` of the [data set](../data/2026-10-30-window1/README.md).

The KL floor of phase 1 is much smaller than these numbers. The numbers of the table are thus the error of the quantization and not the noise of the method.

### The 52-decision set

The 52-decision set is a fast test of the quality of the answers. Arm S and arm D gave 48 correct decisions of 52 each. The four incorrect decisions are the same on the two arms: `triage00.category`, `triage07.category`, `triage07.needs_action` and `triage08.category`. A change to a kernel must keep the 48 correct decisions.

### The tests of the kernels

| Tool of `compute-sanitizer` | Errors (count) | Time of one pass (s) |
|---|---:|---:|
| `memcheck` | 0 | 4.11 |
| `initcheck` | 0 | 2.53 |
| `synccheck` | 0 | 1.15 |
| `racecheck` | 0 | 606.47 |

Source: the [result of phase 2](notebook/2026-10-30-phase2-correctness-baseline.md#test-with-compute-sanitizer) and the table `sanitizers.tsv` of the [data set](../data/2026-10-30-window1/README.md).

The four tools found no error. The test has a limit. It uses the MMQ kernels with 256 tokens and one sequence. It does not use the MMVQ kernels, a batch with more than one sequence, or the decision path of the server. A change must run the four tools on the paths that it touches.

`test-backend-ops` compares the CUDA backend with the CPU backend. It ran the 187 different operations of the graph of mix8. 181 tests gave the same result on the two backends. 6 tests gave `FAIL`. We read the code of llama.cpp at the tag `b12004` and found a cause in the test method for the 6 tests:

- 5 tests of `CPY`. The test gives `CPY` a new output tensor. The CUDA backend writes the copy into a different tensor. The test thus reads memory that has no value [INFERENCE: we did not run a test that gives `CPY` the correct output].
- 1 test of `GATED_DELTA_NET` with 683 tokens in each sequence. The test fills the `gate` input with random values from -1 to 1. The model uses values of 0 or less. After 683 tokens the values are larger than the range of `f32` [INFERENCE: we did not run the test with a correct `gate` input].

The built-in tests of the five operations gave 390 correct tests. 310 tests had the result `not supported`. A change must keep the same 6 tests as the only tests with `FAIL`.

### The public benchmark

The public benchmark in `bench/` has the same token counts as the private benchmark. Phase 2 made its reference answers with the Q8_0 model: 93 correct decisions of 100. Three runs of mix8 gave a `lat_ms` of 237.1 ms to 237.5 ms and 94 correct decisions. On a host with a low load, the public benchmark thus has the same speed as the private benchmark. The difference of the means is -0.02 %.

On 2026-10-31, the project made the baseline of the public benchmark. Three runs of build `D` gave a `lat_ms` of 236.2 ms, 236.9 ms and 237.3 ms. All three gave 94 correct decisions of 100. A fourth run gave the result `PASS`. The [README of the benchmark](../bench/README.md#source-of-the-baseline) gives the numbers.

## The limits of the GPU

The question was: which limits does the `RTX 3060 Ti` have for the work of the model? The targets of phase 4 use these limits. Three small programs ran on 2026-10-30 from 22:52 to 22:56 UTC. The [data set](../data/2026-10-30-card-limits/README.md) has the sources and the tables.

| Quantity | Value | Unit |
|---|---:|---|
| Rate of the tensor cores for int8, 2 s | 151.54 | TOPS |
| Rate of the tensor cores for int8, 20 s (the power cap is active in all samples) | 149.26 | TOPS |
| DRAM read rate (97.1 % of the nominal rate of 435.3 GB/s) | 422.5 | GB/s |
| DRAM write rate | 416.3 | GB/s |
| L2 cache read rate, buffer of 3 MiB | 1,374 | GB/s |
| cuBLASLt int8 rate for the weights of the model, 2,048 tokens | 103.0 | TOPS |

Source: the [summary of the data set](../data/2026-10-30-card-limits/summary.md). The L2 cache of this GPU holds 3 MiB.

The project uses 149.3 TOPS as the int8 peak and 422 GB/s as the DRAM limit. The int8 rate changes with the data and with the time, because the power cap changes the clock. With operands that are all zero, the rate was 152.88 TOPS at 96 W. With random operands it was 151.54 TOPS at 158 W.

The cuBLASLt test used int8 weights, with no scales. A layout, `col32`, gave incorrect results (`MISMATCH_1023_of_3072`). We did not use its times. For the weights of the model at 2,048 tokens, cuBLASLt is at about 69 % of the int8 peak. The [phase 4 entry](notebook/2026-10-31-phase4-ranked-targets.md#where-the-time-goes) gives this value. The MMQ kernels are at less, as the next phase shows.

The floors of the census use the nominal rates (448 GB/s and 144.8 TOPS). The rates that we measured are different: 422 GB/s and 149.3 TOPS. The weight-read floor of a pass with the output head is thus 7.44 ms with the nominal rate and 7.89 ms with the rate that we measured.

## Phase 3: where the time goes

### The question and the instruments

The question of phase 3 was: where does the time of one request go? The work before the project gave a time for a short request, but it did not give a cause. Phase 3 has no prediction entry, because it measures and does not predict. We used six instruments.

| Instrument | Result | Source |
|---|---|---|
| Census of the passes | The work of each pass in numbers, with no time | [Data set of the census](../data/2026-10-30-shape-census/README.md) |
| Build with `DMK_TIMING` | The time of each step of a request | [Data set of the profiles](../data/2026-10-31-profiles/README.md) |
| Nsight Systems | When the GPU runs a kernel and when the CPU waits | [Data set of the profiles](../data/2026-10-31-profiles/README.md) |
| Nsight Compute | How near each kernel is to its limit | [Data set of the profiles](../data/2026-10-31-profiles/README.md) |
| Ablation build | The cap of each part of the work | [Data set of the ablation](../data/2026-10-31-ablation/README.md) |
| Two leads | The effect of #28495 and of #30779 | [Entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#the-two-leads) |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md) and the [entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md).

The measurement window 2 started at 00:02 UTC on 2026-10-31 and stopped at 01:14 UTC. We stopped the production server for the runs. One run at a time used the GPU.

### The census of the passes

The census gives the work of one pass in numbers. It uses the shapes of the model and the rules of the code. A pass with the output head reads 3.34 GB of weights. A pass without the output head reads 2.62 GB. The time to read the weights at the nominal rate of 448 GB/s is 7.44 ms with the head and 5.85 ms without it.

The table gives the floors of four requests. The "weights or int8" floor is the larger value of the weight-read floor and the int8 floor. The roofline floor adds the other work of the model.

| Request type | Case | Passes in the request | Tokens in all passes | Weights read (GB) | Weight-read floor (ms) | Floor of weights or int8 (ms) | Roofline floor (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `short1q` | `check03` | 1 | 158 | 3.335 | 7.444 | 7.644 | 19.073 |
| `short2q` | `triage09` | 3 | 257 | 9.290 | 20.737 | 20.737 | 39.760 |
| `medium3q` | `medium01-memory` | 4 | 1,451 | 12.625 | 28.181 | 84.569 | 170.954 |
| `long3q` | `long00-storage` | 5 | 3,384 | 15.245 | 34.029 | 177.970 | 383.300 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#census-the-work-of-one-pass) and the [data set of the census](../data/2026-10-30-shape-census/README.md). The rates are the nominal rates, 448 GB/s and 144.8 TOPS.

The census has a check with data from production. Three captures of requests in production from 2026-10-30 at 19:40 UTC came before the project. The captures show the passes of 20 requests. The census gave the pass sequence of all 20 requests correctly.

It gave the KV window of flash attention correctly for 33 of 72 passes. In the other 39 passes the KV window of the captures was wider, at most 5.0 times. This fact is the start of the lead #28495 below.

### The decomposition of the request time

We made a build with the flag `DMK_TIMING` from the branch `dmk/ablate-timing`. It writes the time of each step of a request to the log of the server. This build was 0.1 % to 0.4 % faster than the reference build in the same run. Thus the code that measures the time does not slow the server.

The next table divides the wall time of 30 requests of each type. The sum of the steps before submit, submit, GPU wait and other steps is the wall time. The GPU span is the time from the start of the first pass to the end of the last GPU wait. The pass floor is the sum of the "weights or int8" floors of the passes.

| Request type | Wall time (ms) | Before submit (ms) | Submit (ms) | GPU wait (ms) | Other steps (ms) | GPU span (ms) | Pass floor (ms) | GPU span divided by pass floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | 46.8 | 1.9 | 8.7 | 35.9 | 0.4 | 45.5 | 7.7 | 5.90 |
| `short2q` | 89.8 | 4.8 | 63.3 | 20.9 | 0.9 | 87.6 | 20.7 | 4.22 |
| `medium3q` | 368.4 | 12.5 | 329.2 | 25.2 | 1.5 | 362.3 | 84.6 | 4.28 |
| `long3q` | 807.3 | 26.4 | 751.4 | 26.2 | 3.2 | 796.4 | 177.9 | 4.48 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#decomposition-of-the-request-time) and the table `summary/request-time-decomposition.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

The result was clear. The GPU span of each request type was 4.2 to 5.9 times the pass floor. The time was not near the floor of the GPU. The next table shows the steps of a `short1q` request.

| Step | Mean (ms) | Part of the wall time (%) |
|---|---:|---:|
| Weight-read floor of one pass | 7.44 | 15.9 |
| HTTP and client time | 0.44 | 0.9 |
| Prompt build and tokenizer | 0.59 | 1.3 |
| Start of the task and start of decode | 0.24 | 0.5 |
| Graph build and submit | 9.55 | 20.4 |
| End of the request after the last wait | 0.08 | 0.2 |
| GPU wait above the floor | 28.49 | 60.8 |
| Wall time | 46.85 | 100.0 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#the-request-short1q-against-the-weight-read-floor-and-the-time-before-phase-3) and the table `summary/short1q-waterfall.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

The measurement of 36.7 ms from the work before the project is between the GPU wait (35.9 ms) and the GPU span (45.5 ms) of this run. The number from the work before the project was thus correct. The weight-read floor was only 16 % of the wall time. A GPU wait of 28.5 ms above the floor was 61 % of it.

**The correction of the text `host-bound`.** The script of the decomposition wrote the text `host-bound` for `medium3q` and `long3q`. It gave the launch loop as the cause of the GPU span. This text was incorrect.

The decomposition cannot see the GPU. The next part shows that the GPU ran a kernel for 93 % of the wall time of these requests. The entry has the [correction](notebook/2026-10-31-phase3-where-the-time-goes.md#correction-of-the-text-host-bound).

**The warm path.** A second request with the same case was faster than the first request for `short1q`, `medium3q` and `long3q`. The wall time was 40.2 ms and 41.9 ms lower in the two long cases. The data do not show the cause [INFERENCE: this entry gives no cause]. The [entry](notebook/2026-10-31-phase3-where-the-time-goes.md#the-warm-path) has the table.

### Nsight Systems: the GPU runs a kernel

Nsight Systems shows when the GPU runs a kernel. It adds 0.3 % to 1.6 % to the time of a request. The next table gives the wall time of a request, divided with the time of the GPU.

| Request type | Wall time (ms) | Time before the first kernel (ms) | GPU busy time (ms) | GPU busy time (% of the wall time) |
|---|---:|---:|---:|---:|
| `short1q` | 47.1 | 3.2 | 42.4 | 90.0 |
| `short2q` | 87.1 | 3.5 | 78.6 | 90.2 |
| `medium3q` | 359.9 | 17.3 | 333.8 | 92.8 |
| `long3q` | 795.1 | 33.3 | 745.0 | 93.7 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#where-the-wall-time-goes) and the table `summary/nsys-request-time.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

The GPU ran a kernel for 90 % to 94 % of the wall time. The idle time in the passes was 0.6 % to 3.2 % of the wall time. The GPU is thus the limit of all four request types. The GPU did not run a kernel for 6.3 % to 10.0 % of the wall time [calculated: 100 % minus the last column]. A change to the host code can remove only a part of this time.

The CPU waited for the GPU in sync calls. The table gives the CUDA calls of the decode thread as the mean of one request.

| Request type | Launches in a request | Time in the launches (ms) | Sync calls in a request | Time in the sync calls (ms) |
|---|---:|---:|---:|---:|
| `short1q` | 1,296 | 5.5 | 24.0 | 35.7 |
| `short2q` | 3,829 | 15.1 | 68.0 | 56.4 |
| `medium3q` | 4,883 | 46.9 | 91.6 | 283.9 |
| `long3q` | 6,014 | 85.0 | 112.6 | 662.1 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#where-the-cpu-waits) and the table `summary/nsys-host-cuda-calls.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

A pass has about 23 sync calls. In `medium3q` and `long3q`, 86 % to 89 % of the time in CUDA calls is time in sync calls. The launch queue was full at 1,020 launches. The data do not show the source of the sync calls.

The uploads from the host to the GPU ran at 6.7 GB/s. They used 0.71 ms (4.7 MB) in a `short1q` request and 21.26 ms (142.7 MB) in a `long3q` request. This rate fits PCIe 3.0 with a link width of 8 [INFERENCE: the data set does not record the width of the link].

The kernel time of a pass changes with its number of tokens. A pass has 1,065 to 1,296 launches. MMQ uses 60 % to 70 % of the kernel time of a pass with more than 8 tokens. The output head uses 1.64 ms in each pass that has outputs. The kernel `concat_non_cont` has 24 launches in each pass.

One pass with 2,048 tokens uses 416.4 ms of kernel time. MMQ uses 263.5 ms of it. The [entry](notebook/2026-10-31-phase3-where-the-time-goes.md#kernel-time-of-each-pass-type) has the full table.

### Nsight Compute: how near each kernel is to its limit

Nsight Compute shows the limit of one kernel. It profiles only a small group of launches in each run. The 13 runs gave 7 results `ALIGNED`, 1 result `SHIFTED` and 5 results `MISALIGNED`. A run with the result `MISALIGNED` has launches that are not the launches of its list. Its rows have the kernel name and the grid, but no model floor.

**MMQ with many tokens.** With 1,173 to 2,048 tokens, the MMQ kernels were at 32.5 % to 39.2 % of the int8 peak of 149.3 TOPS. The table has four of the rows of the entry.

| Tokens | Operation | Weight type | Time of the kernel (µs) | Int8 peak (%) | IMMA pipe active (%) | Largest stall reason |
|---:|---|---|---:|---:|---:|---|
| 1,173 | `attn_qkv` | `Q8_0` | 939.4 | 35.1 | 37.9 | `math_pipe_throttle` (27 %) |
| 1,173 | `ffn_up` | `Q4_0` | 1,017.4 | 36.4 | 39.4 | `math_pipe_throttle` (29 %) |
| 2,048 | `attn_qkv` | `Q8_0` | 1,495.2 | 38.5 | 38.3 | `math_pipe_throttle` (27 %) |
| 2,048 | `ffn_up` | `Q4_0` | 1,649.5 | 39.2 | 39.2 | `math_pipe_throttle` (29 %) |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) and the table `summary/ncu-mmq-large-n.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

Each MMQ kernel used 216 or 254 registers for each thread and 57,856 B of shared memory. This limits an SM to one thread block of 8 warps, which is 16.7 % of the warp slots. No other thread block can run on the SM while the thread block waits for loads. The issue slots were active for 34 % to 46 % of the time.

The loop of the kernel is in `ggml/src/ggml-cuda/mmq.cuh`, lines 954 to 987. It loads a tile, runs `__syncthreads`, runs `vec_dot`, and runs `__syncthreads` again. The code has no `cp.async` and no second buffer. With one thread block on each SM, the loads do not run at the same time as the matrix work [INFERENCE].

**The instructions of one MMQ kernel.** The report of one `ffn_up` kernel with 2,048 tokens shows the SASS lines. For each IMMA instruction, the kernel runs 4 `I2FP`, 4 `FMUL` and 4 `FFMA` instructions. Together these three are 63 % of the instructions that the kernel runs. The IMMA instructions have 23.4 % of the stall samples.

The line `sum += C.x[l]*dA*dB` in `mmq-vec-dot.cuh` (lines 265 to 273) can cause the number four [INFERENCE: the data show the numbers and not the cause]. This 63 % is a different number from the part of MMQ in the kernel time. The entry has the [table of the opcodes](notebook/2026-10-31-phase3-where-the-time-goes.md#the-instructions-of-one-mmq-kernel).

**MMQ with a small number of tokens.** With 26 to 102 tokens, the DRAM was the limit. The kernels were at 35 % to 80 % of the DRAM limit and at 9 % to 31 % of the int8 peak. Thus a small number of tokens and a large number of tokens are two different states. The [entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-a-small-number-of-tokens) has the table.

**The other kernels.** The table has the kernels that are not MMQ.

| Kernel | Time of the kernel (µs) | Limit | Of the limit (%) |
|---|---:|---|---:|
| `mul_mat_vec_q`, grid 4,096 | 59.1 | DRAM | 94.5 |
| `mul_mat_vec_q`, grid 2,048 | 32.5 | DRAM | 90.3 |
| `gdn_single`, grid 32 | 740.3 | DRAM | 19.9 |
| `concat_non_cont`, grid 8,192 | 18.3 | launch | 6.4 |
| `flash_attn_ext_f16`, grid 16 | 11.2 | DRAM | 27.7 |

Source: the [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) and the table `summary/ncu-other-kernels.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md).

The MMVQ kernels were at 90 % to 95 % of the DRAM limit, thus at the limit. The kernel `gdn_single` runs 32 thread blocks on 38 SMs and was at 20 % of the DRAM limit. The code in `gated-delta-net-mma.cu` (lines 403 to 416) selects this path. The cause is that 2 times the number of heads times the number of sequences is 64. This is more than the 38 SMs.

The small kernels ran for 2 µs to 3 µs. The launch limits them.

Two limits of this part are important. Five of the 13 runs gave the result `MISALIGNED`, and the entry does not give a cause for them. The entry also does not give a cause for the sync calls.

### The ablation: the cap of each part

The ablation shows how much time a part of the work uses. The ablation build skips the work of one part of the model in each run. The time of the run then falls by the time of that part. The name of this fall is the cap of the part. A cap is the largest time that a faster part can remove from a request. A cap is not a prediction.

The build has 3 commits on top of `b12004`. Its SASS is the same as the SASS of build `D` for all 6,754 kernel names. The data set has 56 runs of the private benchmark and of `llama bench`. The answers of the ablation runs are not correct, because the skipped work changes them. The client does not read them.

The latency of `base` was 71.6 ms for a short request, 357.8 ms for a medium request and 784.7 ms for a long request. The table gives the caps in % of the latency of `base`.

| Arm | Skipped work | Short request (%) | Medium request (%) | Long request (%) |
|---|---|---:|---:|---:|
| `all` | All nodes | 89.3 | 93.0 | 93.2 |
| `matmul_q` | All matrix kernels and all quantize kernels | 68.6 | 67.9 | 66.7 |
| `matmul` | All matrix kernels | 65.1 | 63.2 | 61.7 |
| `mmq` | The MMQ kernels | 61.1 | 60.4 | 60.3 |
| `gdn_conv_l2` | The nodes of the recurrent layers | 9.0 | 9.4 | 9.3 |
| `glu` | The GLU nodes and the fused GLU kernels | 3.5 | 5.7 | 6.1 |
| `elementwise` | The nodes `ADD`, `MUL`, `SCALE` and the other nodes of this type | 3.2 | 5.3 | 5.5 |
| `copies` | The nodes `CPY`, `SET_ROWS`, `CONCAT` and `GET_ROWS` | 4.2 | 4.2 | 4.0 |
| `quantize` | The quantize kernels | 3.3 | 6.7 | 7.2 |
| `norm` | The nodes `RMS_NORM` and `NORM` | 2.9 | 3.5 | 3.5 |
| `concat` | The node `CONCAT` | 2.4 | 2.9 | 2.8 |
| `fattn` | The flash attention node | 1.8 | 2.6 | 4.4 |
| `head` | The output matrix multiply and its quantize kernel | 4.0 | 1.5* | 0.7* |

Source: the [entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md#caps-for-each-part-and-each-request) and the table `caps-request.tsv` of the [data set of the ablation](../data/2026-10-31-ablation/README.md). The mark * shows a cap that is smaller than the pass spread of `base`. Such a cap can be noise.

The matrix kernels have a cap of 60 % to 65 % of the request. The cap of `mmq` is 60.3 % to 61.1 %. The profile of Nsight Systems gives the same result: MMQ is 60 % to 70 % of the kernel time. The recurrent layers have a cap of about 9 %. All other parts have a cap of 7 % or less.

The arm `all` skips all nodes. Its latency was 7.6 ms, 25.2 ms and 53.2 ms. In this arm the server builds the prompt and the graph, sets the inputs and sends the answer.

The caps do not add. The sum of the caps of eleven arms is 94.1 %, 101.8 % and 102.7 %. The cap of `all` is 89.3 %, 93.0 % and 93.2 %. One possible cause is the clock of the GPU [INFERENCE: the data set cannot show how large the part of the clock is]. The arms with a large cap had a median SM clock of 1,950 MHz at 141 W to 152 W. The arm `base` had 1,845 MHz at 194.6 W.

Noise limits the caps. The three runs of the reference arm had a drift of 0.30 %, 0.37 % and 0.40 %. The pass spread of `base` was 0.70 %, 2.04 % and 2.52 %. Each arm has one run.

The ablation also found three other facts.

- **The fusions of `b12004` help.** With `GGML_CUDA_DISABLE_FUSION=1` the latency was 3.5 %, 5.6 % and 6.0 % higher.
- **CUDA graphs do not run.** The CUDA backend ran 741 passes. 739 passes ran in the mode `direct`, 2 passes captured a graph, and no pass used a graph again. The arm that turns off graphs shows no gain. Only 2.1 % of the ubatches with 1 to 8 tokens used the graph of the previous pass again. [INFERENCE: two passes in a row have different numbers of tokens, thus different node properties.]
- **The time for each node has an error.** The `DMK_OPS` table has no rows because of an error in the build. The commit `de091da69` repairs it, but we have no test of the repair. The data set gives the time of a group of nodes only.

The `llama bench` caps show where the matrix kernels change from MMVQ to MMQ. The `mmq` cap was 61.1 % to 65.0 % at 32 tokens or more. At 8 tokens, the `mmvq` cap was 78.4 %. The change point is between 8 tokens and 32 tokens.

### The lead #30489: the first measurement of a change

The lead #30489 is the pull request of phase 0 that fuses GLU into the quantization. Phase 3 measured it as a lead. The project does not use it in production. We made a build with the patch (arm `L`) and compared it with build `D`, in 4 blocks of the private benchmark with seed 30490.

The change was a gain in speed. The table gives the means of the blocks.

| Metric | D mean (ms) | L mean (ms) | L against D (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` | 241.1 | 236.5 | -1.94 | -2.38 | -1.50 |
| `short_ms` | 73.4 | 72.5 | -1.23 | -2.14 | -0.34 |
| `medium_ms` | 362.9 | 355.7 | -1.99 | -2.24 | -1.73 |
| `long_ms` | 791.4 | 773.6 | -2.25 | -2.45 | -2.04 |

Source: the [entry of the lead](notebook/2026-10-31-lead-30489.md#speed-test-with-the-private-benchmark) and the table `speed-bench.tsv` of the [data set](../data/2026-10-31-lead-30489/README.md).

| Prompt length (tokens) | D mean (tokens/s) | L mean (tokens/s) | L against D (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,814.8 | 2,842.7 | +1.00 | +0.13 | +1.86 |
| 256 | 4,220.3 | 4,310.6 | +2.14 | +1.68 | +2.60 |
| 3,000 | 4,596.7 | 4,703.0 | +2.31 | +1.98 | +2.72 |

Source: the [entry of the lead](notebook/2026-10-31-lead-30489.md#speed-test-with-llama-bench) and the table `speed-llama-bench.tsv` of the [data set](../data/2026-10-31-lead-30489/README.md).

The answers were the same bit for bit on all 100 decisions in 8 runs. The equivalence class `exact` holds for these inputs. 6,744 of the 6,748 kernel names in the two builds have the same SASS. The 4 other names are kernels for a type that the model does not use. The matrix-multiply kernels did not change.

The measurement had three problems, and the entry gives all of them:

- **The prediction entry is not in the repository.** The method has a rule: we commit a prediction entry before a measurement. We wrote no prediction entry for this test, but we wrote the prediction in an external file on 2026-10-30. The file has the time 23:12 UTC, and the first run started at 00:47 UTC. The repository cannot show the date of the file. The [entry](notebook/2026-10-31-lead-30489.md#prediction-against-result) gives its text.
- **We did not control the load of the host.** Blocks 2 to 4 ran at a load of 3.00 to 18.70. In `llama bench`, arm L ran first in all four blocks.
- **The test has four blocks.** The interval of an estimate from 4 blocks is too small [INFERENCE]. We did not run `test-backend-ops` or `compute-sanitizer` on the build with the patch.

The prediction was a ceiling of 2.0 %, 2.7 % and 2.9 % of the kernel time for a short request, a medium request and a long request. The gain was 61 %, 74 % and 77 % of the ceiling. The gain was less than the ceiling in all three types of request, thus the prediction was correct. The entry lists six steps that must come before production use of the patch.

### The two other leads

**#28495: the KV window.** With `--kv-unified`, the flash attention time of the same request on a different slot was 1.19 to 2.25 times the time of the first request. The mean was 87 % higher. Without `--kv-unified` it was 0.8 % higher.

The GPU busy time increased by 3.4 % with `--kv-unified` and by 0.6 % without it. The pass time in a build with `DMK_TIMING` changed by 4.2 % with `--kv-unified` and by 0.2 % without it. The script uses a threshold of 4.0 %. The cap of `fattn` is 1.8 % to 4.4 % of a request, thus the possible gain is small.

**#30779: a child slot with a lower number than the parent slot.** 119 of 372 requests had a child slot with a lower number than the parent slot: 53 `medium3q` and 66 `long3q`. These requests had one more use of `llama_decode`. They were not slower.

A `medium3q` request used 364.29 ms with the lower slot and 370.39 ms without it. A `long3q` request used 800.74 ms and 815.88 ms. No request had more passes than the census model gives. The lead has no effect on the time of this benchmark.

### The result of phase 3

The question of phase 3 had an answer. The GPU runs a kernel for 90 % to 94 % of the time of a request. The MMQ kernels use 60 % to 70 % of the kernel time. These kernels are at 32.5 % to 39.2 % of the int8 peak. The host code, the launches, and the copies are a small part.

The ablation gives the same result. The arm `matmul_q` removes 67 % to 69 % of the latency.

## Phase 4: the ranked list and a correction

### The question

Phase 3 showed where the time goes. Phase 4 had a different question: which changes can make the model faster, and in which sequence must the project do them? The answer is a ranked list of targets. The list is a prediction, because we wrote it before the first change to the code of a target. The [entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md) has the list.

Each target has an equivalence class, an effort, a confidence, and a gain for each request type. The confidence is the probability that the change passes the test of its equivalence class and gives 50 % or more of its predicted gain. The expected gain is the predicted gain of `lat_ms` times the confidence. It is not a measurement. The [data set of the model](../data/2026-10-31-phase4-model/README.md) has the numbers, the script, and the tables.

### The state at the checkpoint

The time model of the entry uses the four request types. Its `lat_ms` is the mean of the 48 cases that the model calculates. It is 240.9 ms. The `lat_ms` that we measured for build `D` is 238.3 ms in phase 1.

| Request type | Time of one request (ms) | Kernel time (ms) | Weight floor (ms) | Roofline floor (ms) | Time divided by roofline floor |
|---|---:|---:|---:|---:|---:|
| `short1q` | 47.1 | 42.4 | 7.7 | 19.2 | 2.45 |
| `short2q` | 87.1 | 85.1 | 20.7 | 39.8 | 2.19 |
| `medium3q` | 360.0 | 344.9 | 84.6 | 169.1 | 2.13 |
| `long3q` | 797.0 | 761.3 | 177.9 | 371.5 | 2.15 |

Source: the [entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#where-the-time-goes) and the table `floors.tsv` of the [data set of the model](../data/2026-10-31-phase4-model/README.md).

The time of a request is 2.1 to 2.5 times the roofline floor. The entry gives eight facts, and the list uses all of them.

1. The GPU runs a kernel for 90 % to 94 % of the time of a request. The host is not the limit.
2. The MMQ kernels use about 63 % of the kernel time. They run at 35 % to 39 % of the int8 rate of the GPU. cuBLASLt runs at about 69 % of that rate, but it does not apply the scales of the groups of 32 values.
3. An MMQ thread block loads its tiles, waits at a barrier, and then runs the matrix instructions. It does not load the next tile during the matrix work. Each SM holds one thread block of 8 warps. The load instructions have about 28 % of the stall samples.
4. Each IMMA instruction has 4 `I2FP`, 4 `FMUL` and 4 `FFMA` instructions after it. They apply the scales. The kernel must have them to give the same result.
5. The MMVQ kernels read the weights at 90 % to 95 % of the DRAM rate. They are at their limit.
6. The copies to the GPU run at 6.7 GB/s. This rate fits PCIe 3.0 x8 [INFERENCE: the data set does not record the width of the link]. The attention mask of a large pass is 21 MB to 28 MB. Its copy uses 3 ms to 4 ms, and the GPU waits for the copy.
7. The kernel `gdn_single` runs 32 thread blocks on 38 SMs.
8. The concat before `ssm_conv` writes a new copy of the input. It uses 12.9 ms in a pass of 2,048 tokens.

The 63 % in fact 2 is the part of MMQ in the kernel time. The 63 % of instructions in phase 3 is a different number. Fact 2 gives the range of the MMQ rate as 35 % to 39 %. The phase 3 table has 32.5 % to 39.2 %.

### The ranked list

The rank comes from the expected gain of `lat_ms`. The table gives all 15 targets.

| Rank | Target | Equivalence class | Effort | Confidence | Gain of `lat_ms` (ms) | Gain of `lat_ms` (%) | Expected gain (%) |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | `G`. MMQ as a pipeline with `cp.async` tile loads | `exact` | `L` | 0.40 | 35.7 | 14.8 | 5.9 |
| 2 | `J`. Ragged batching for the suffixes of the questions | `re-associated` | `L` | 0.45 | 16.4 | 6.8 | 3.1 |
| 3 | `N`. New MMQ kernels, after `G` | `re-associated` | `XL` | 0.20 | 24.9 | 10.4 | 2.1 |
| 4 | `B`. Attention mask on the GPU | `exact` | `M` | 0.70 | 6.9 | 2.9 | 2.0 |
| 5 | `E`. Pull request 30489: GLU fused into the quantization | `exact` | `S` | 0.95 | 4.8 | 2.0 | 1.9 |
| 6 | `C`. Concat folded into `ssm_conv` | `exact` | `M` | 0.70 | 6.4 | 2.7 | 1.9 |
| 7 | `F`. Fusions of RMS norm kernels, element-wise kernels and quantization kernels | `exact` | `M-L` | 0.50 | 8.6 | 3.6 | 1.8 |
| 8 | `H`. MMQ tile and occupancy adjustments for this GPU | `exact` or `re-associated` | `S-M` | 0.30 | 10.2 | 4.2 | 1.3 |
| 9 | `A`. Output head for the answer tokens only | `exact` | `M` | 0.85 | 3.4 | 1.4 | 1.2 |
| 10 | `D`. Value-split path of `gdn_single` | `exact` | `S` | 0.45 | 5.0 | 2.1 | 0.9 |
| 11 | `L`. Smaller KV window for flash attention (pull request 28495) | `exact` | `M` | 0.50 | 2.8 | 1.2 | 0.6 |
| 12 | `M`. One kernel for each layer (megakernel) | `re-associated` | `XL` | 0.10 | 9.9 | 4.1 | 0.4 |
| 13 | `O`. Cache of the prompt tokens | `exact` | `S-M` | 0.50 | 1.9 | 0.8 | 0.4 |
| 14 | `P`. Remove the stream synchronizations in a pass | `exact` | `S-M` | 0.40 | 2.4 | 1.0 | 0.4 |
| 15 | `K`. CUDA graphs for passes with more than one token | `exact` | `M` | 0.30 | 2.1 | 0.9 | 0.3 |

Source: the [entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#ranked-list) and the table `ranked.tsv` of the [data set of the model](../data/2026-10-31-phase4-model/README.md). The effort `S` is 1 day or less. The effort `M` is 2 to 5 days, `L` is 1 to 2 weeks, and `XL` is more than 2 weeks.

The gains do not add, because some targets change the same time.

- `G`, `H` and `N` change the same MMQ time. `N` can give a gain only after `G`.
- `J` removes 2 of the 3 output passes of the medium requests and the long requests. After `J`, `A` gives a gain for one output pass only.
- `B` and `L` touch the attention mask.
- `E` and `F` change the same quantization kernels.
- `P` and `K` remove the same idle time in the passes.

Each target has a description in the [entry](notebook/2026-10-31-phase4-ranked-targets.md#targets). The text gives the assumption for each gain. Four examples show the type of the targets:

- **`G`** loads the next tile with `cp.async` while the matrix instructions run on the tile that it loaded before. It keeps the 4-bit values of Q4_0 in the shared memory.
- **`J`** runs one pass for the three suffixes. A pass of 163 tokens replaces 3 suffix passes in a medium request or a long request.
- **`A`** calculates the logits of the label tokens only. The output head uses 1.6 ms in each pass that has outputs.
- **`D`** gives `gdn_single` a path that divides the value dimension. It uses 128 blocks. The entry thinks that the kernel is 1.6 times faster.

### The sequence from the entry

The ranked list has the sequence of the expected gain. The entry also gives a sequence that uses the effort and the risk. Targets with a small effort and a high confidence come first.

1. `E`: the pull request #30489, which we measured as a lead.
2. `D` and `H`: small changes. Each test also shows a cause.
3. `A`: the best target for short requests.
4. `B` and `C`: the two are `exact` with a high confidence.
5. `G`: the largest target. It comes after `H`, because the result of `H` can change the method.
6. `J`: the largest target for `short2q`.
7. `F`, `L`, `P`, `K` and `O`: smaller targets.
8. `N` and `M`: the entry does not recommend them at this time. The effort is very large and the confidence is low.

### The total gain and its correction

The first version of the entry added the gains of the 8 targets `A`, `B`, `C`, `D`, `E`, `F`, `G` and `J`. The sum for `lat_ms` was 87.2 ms. This gives 153.7 ms [calculated: 240.9 ms minus 87.2 ms], a change of -36.2 %. This number was incorrect, because the sum counted some kernel time two times. `J` removes passes that `A`, `C`, `F` and `G` also make faster.

The time model replaced the sum on 2026-10-31, before the measurement of a target. It applies `J` first. Then it applies `A`, `C`, `D`, `F` and `G` to the passes that stay. It scales `E` by the GLU time and quantization time that stay after `J`. It runs `F` on the time that stays after `E`. The target `B` changes only the copies and the host time of the attention mask.

The expected gain is the mean of all 256 results in which each target gives its full gain or no gain. The probability of the full gain is the confidence of the target.

| Request type | Time at the checkpoint (ms) | Time if all 8 targets pass (ms) | Change (%) | Time with the expected gain (ms) | Change (%) |
|---|---:|---:|---:|---:|---:|
| `short1q` | 47.1 | 34.4 | -27.0 | 40.3 | -14.4 |
| `short2q` | 87.1 | 53.8 | -38.3 | 69.5 | -20.2 |
| `medium3q` | 360.0 | 230.2 | -36.1 | 292.5 | -18.7 |
| `long3q` | 797.0 | 534.4 | -33.0 | 659.9 | -17.2 |
| `lat_ms` | 240.9 | 158.4 | -34.2 | 197.8 | -17.9 |

Source: the [entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#prediction-of-the-total-gain) and the table `sum.tsv` of the [data set of the model](../data/2026-10-31-phase4-model/README.md).

The model after the correction gives 158.4 ms for `lat_ms` if all 8 targets pass. This is 4.7 ms more than the first version. The time with the expected gain is 197.8 ms, a change of -17.9 %. The entry gives the total as approximate. The new pass of `J` and the assumption about the probabilities can be incorrect.

A prediction can also be incorrect for other causes. The kernel time can change when the time of a different kernel changes. The ratios for `G`, `H`, `D`, `F` and `L` come from profiles and not from tests. The PCIe speed is a fact of this host. The model does not include the time of the first request after a period with no request.

### The rule to stop

The project stops when the expected gain is less than 3 % for each request type. Three targets are above this limit for one request type.

| Target | Largest expected gain of one request type (%) | Request type |
|---|---:|---|
| `J` | 6.7 | `short2q` |
| `G` | 6.1 | `long3q` |
| `A` | 3.2 | `short2q` |

Source: the [entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#rule-to-stop-the-project) and the column `max_expected_pct` of the table `ranked.tsv`.

Thus the rule does not stop the project at the checkpoint. The budget of the project stops at the entry of phase 4. The owner selects the next step from the list.

## The next steps

No target has code at the checkpoint. The target `E` has a measurement as a lead, and the project does not use it in production. The owner selects the next target. Each target that the project does gets a prediction entry, a result entry, and a change document.

The [conditions of the README](../README.md#condition-of-each-change) show the state of each target.

The tests are different for each equivalence class. A change in the equivalence class `exact` must give the same answers bit for bit. A change in the equivalence class `re-associated` must have three results. `test-backend-ops` must be in its limits. The KL divergence must be at the level of the noise. The guards of the benchmark must be in their limits.

The [method](method.md) gives the rules.

Before the project uses #30489 in production, it must do six steps:

1. Write a prediction entry and commit it before the measurement.
2. Run `test-backend-ops` for the GLU cases of the commit.
3. Run the four tools of `compute-sanitizer` on the build with the patch.
4. Measure again at a low load, with more blocks and a different sequence of the arms in each block.
5. Make a profile of the two builds.
6. Write a change document in `docs/changes/` with the equivalence class `exact`.

Source: the section ["Limits and next work"](notebook/2026-10-31-lead-30489.md#limits-and-next-work) of the entry of the lead.

## Known and unknown facts

At the checkpoint, the project knows these facts:

- The speed and the answers of build `D` are the same as the speed and the answers of the stock image. Thus a change of the project compares with build `D`.
- The GPU is the limit of a request, and MMQ is the largest part of the kernel time.
- The MMQ kernels are at 32.5 % to 39.2 % of the int8 peak. The structure of the MMQ loop can be the cause [INFERENCE].
- Fusions of kernels give a gain. The lead #30489 gave 1.94 % for `lat_ms`.
- The host code, CUDA graphs, and the launches are not the limit.

The project does not know these facts:

- The cause of the sync calls, which are about 23 for each pass.
- The cause of the faster second request with the same case.
- The cause of the high DRAM value of two MMQ kernels.
- How much of the cap of the matrix kernels comes from the higher clock.
- The width of the PCIe link of the test device [INFERENCE: PCIe 3.0 x8 fits the rate].
- The result of a change when the load of the host is low and the number of blocks is large.

## Problems and corrections

The table lists each problem that we found and its correction. It is the list that the entries give.

| Phase | Problem | Correction | Source |
|---|---|---|---|
| 1 | The prediction gave that only the list of GPU architectures is different. Five CMake flags are different. | The result entry marks the row "Not fully". The SASS is the same. | [Result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) |
| 1 | The load of the host changed the speed of the requests. | The harness records the load of the host for each run. | [Result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md#host-load-and-its-effect) |
| 1 | The KL floor was near the limit of 99.9 %. | The subsequent entries compare with the KL floor of the same text. | [Result of phase 1](notebook/2026-10-30-phase1-equivalence-result.md) |
| 3 | The script of the decomposition wrote the text `host-bound`. | The entry marks the text as incorrect. Nsight Systems shows that the GPU runs a kernel for most of the time. | [Entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md#correction-of-the-text-host-bound) |
| 3 | The test of the lead #30489 had no prediction entry in the repository. | The entry gives this fact and six steps for the next measurement. | [Entry of the lead](notebook/2026-10-31-lead-30489.md) |
| 3 | The `DMK_OPS` table has no rows. | The commit `de091da69` repairs the build. We have no test of it. | [Entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md#the-error-in-the-table-dmk_ops) |
| 3 | The caps do not add. | The entries give a cap as a ceiling and not as a prediction. | [Entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md) |
| 4 | The first total gain added the gains of the targets. | The time model replaced it before the measurement of a target. | [Entry of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#prediction-of-the-total-gain) |

Source: the entries in the last column.

## Where to read more

The [notebook](notebook/README.md) has the entries in full. The [data sets](../data/README.md) have the raw files, the tables, and the conditions of each measurement. The [kernels document](kernels.md) has the facts of the kernels, and the [method](method.md) has the rules of the tests.
