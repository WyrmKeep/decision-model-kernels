# How we got here

This document tells how the project went from its first question to the end of the changes of phase 5. It goes phase by phase. For each phase it gives the question, the method, the result, the problems that we found, and the corrections. You can read it with no other information. Each number has a link to its source in the repository.

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
| 2026-10-31, 05:15:13 | We wrote the prediction entry of change 0005. The first run was at 06:34. | [Prediction of change 0005](notebook/2026-10-31-0005-concat-conv-prediction.md) |
| 2026-10-31, 05:24 | We wrote the prediction entry of change 0004. The first run was at 07:11. | [Prediction of change 0004](notebook/2026-10-31-0004-gpu-mask-prediction.md) |
| 2026-10-31, 05:32:15 | We wrote the prediction entry of change 0007. The first run was at 05:53:42. | [Prediction of change 0007](notebook/2026-10-31-0007-input-copies-prediction.md) |
| 2026-10-31, 05:40 to 08:11 | Four measurement windows for the sweep and the ablation of target `H`. | [Result for target `H`](notebook/2026-10-31-0006-mmq-tiles-result.md) |
| 2026-10-31, 05:53 to 07:29 | The speed tests of changes 0007, 0005 and 0004. The test of target `D` ran from 06:17 to 06:31. | [Result of change 0007](notebook/2026-10-31-0007-input-copies-result.md), [result of change 0005](notebook/2026-10-31-0005-concat-conv-result.md), [result of change 0004](notebook/2026-10-31-0004-gpu-mask-result.md), [result for target `D`](notebook/2026-10-31-0002-gdn-value-split-result.md) |
| 2026-10-31, 06:18:10 | We wrote the prediction entry of change 0003. The first run was at 10:32:12. | [Prediction of change 0003](notebook/2026-10-31-0003-answer-head-prediction.md) |
| 2026-10-31, 07:02:58 | We wrote the prediction entry of the first test of change 0008. The first test ran from 07:51 to 08:43. | [Prediction of the first test of change 0008](notebook/2026-10-31-0008-ragged-suffix-prediction.md) |
| 2026-10-31, 08:32 to 10:04 | The measurement windows for the pipeline of target `G`. | [Result for target `G`](notebook/2026-10-31-0009-mmq-pipeline-result.md) |
| 2026-10-31, 08:38:18 | We wrote the prediction entry of the first test of the stack. The first run was at 08:43:08. The first test ran from 08:42 to 09:06 and from 10:04 to 10:10. | [Prediction of the first test of the stack](notebook/2026-10-31-stack-1-prediction.md) |
| 2026-10-31, 09:19:03 | We wrote the prediction entry of the second test of change 0008. The second test ran from 09:22 to 09:59. | [Prediction of the second test of change 0008](notebook/2026-10-31-0008-ragged-suffix-retest-prediction.md) |
| 2026-10-31, 10:32 to 10:44 | The speed test of change 0003. | [Result of change 0003](notebook/2026-10-31-0003-answer-head-result.md) |
| 2026-10-31, 20:43:36 | We wrote the prediction entry of the second test of the stack. The first run was at 20:43:55. | [Prediction of the second test of the stack](notebook/2026-10-31-stack-2-prediction.md) |
| 2026-10-31, about 21:30 | We wrote the entry of the rule to stop. The soak test of 24 hours and the change of production are the next steps. | [Entry of the rule to stop](notebook/2026-10-31-phase5-rule-to-stop.md) |

Source: the dates in the notebook entries and in the README of each data set. The times of the predictions are the commit times that the notebook gives. The owner gave the start time and the time of the checkpoint. They are not in a file of the repository.

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

## Phase 5: the changes, one at a time

### The question and the method of each change

Phase 4 gave a ranked list of 15 targets. Phase 5 had a different question for each target: does the change give its predicted gain, and does it keep the answers? The owner approved the sequence of the ranked list. The owner also let the project operate the GPU when it was necessary. A change goes into the stack only when its tests are in their limits.

Each change had the same five parts.

- **The prediction entry.** We wrote a prediction entry in the notebook and committed it before the first run of the speed test. It gives the equivalence class, the arms, the schedule, the tolerances, the predicted gain, and the results that show an incorrect prediction. We did not change it after the measurement started. The commit time shows that the prediction came first.
- **The tests of correctness.** The equivalence class sets the tests. For `exact`, the answers of all 100 decisions must be the same bit for bit as the answers of the baseline. For `re-associated`, the errors of `test-backend-ops` must be in their limits. The KL divergence must be at the level of the noise, and all guards must be true. A change with a new kernel also has `compute-sanitizer` and a test of the SASS.
- **The speed test.** The test has 4 blocks of the private benchmark and 4 blocks of `llama bench`. Each block has one run of each arm, in a random sequence. The estimate is the mean of the ratios of the blocks. The interval is the 95 % bootstrap interval of the blocks.
- **The change document.** Each change has one document in `docs/changes/`, with the same text as its pull request. It has eight sections: the summary, the equivalence class, the results, the prediction, the correctness, the procedure to reproduce, the limits, and the data.
- **The branch and the stack.** Each change has one branch in the fork. The branch starts from `dmk/stack`, and we compare it with the build of `dmk/stack`. A change that is correct and gives a gain goes into the stack. Each other change stays on its branch.

A rule also applies. A gain of less than 50 % of the prediction has a cause that we must find before the next change. The [method](method.md#equivalence-classes) gives the tests of each equivalence class, and the [change documents](changes/README.md) give the shape of a document.

### The result of each target

The table gives the nine targets that had work. The sections after it give the change, the prediction, and the result of each target. The sequence is the sequence of the work. Some work ran at the same time.

| Target | Change | Equivalence class | Prediction of `lat_ms` (%) | Result of `lat_ms` (%) | Condition |
|---|---|---|---:|---:|---|
| `E` | 0001 | `exact` | -2.0 | -1.94 | In the stack |
| `P` | 0007 | `exact` | -1.3 | -0.93 | In the stack. The interval of the result includes 0. |
| `D` | 0002 | `exact` | -2.1 | No speed test | Not used. The best value-split path is 1.15 to 1.31 times as slow. |
| `B` | 0004 | `exact` | -2.8 | -1.06 | In the stack |
| `C` | 0005 | `exact` | -2.7 | -3.02 | In the stack |
| `H` | None | `exact` or `re-associated` | -4.2 | No speed test | Not used. No rule that the code can use gives a gain. |
| `J` | 0008 | `re-associated` | -3.3 | -2.46 | Not used. One limit for the error is not true. |
| `G` | None | `exact` | -14.8 | No speed test | Not used in this shape. The pipeline gives 1.042 times at 1,025 tokens or more. The limit was 1.08. |
| `A` | 0003 | `exact` | -1.5 | -0.87 | In the stack. On the stack, the result is -1.48. |

Source: the prediction entries, the result entries, and the [change documents](changes/README.md). The prediction of `E`, `D`, `H` and `G` is the value of the [ranked list of phase 4](notebook/2026-10-31-phase4-ranked-targets.md#ranked-list). The other predictions come from the prediction entry of each change. The result is the difference from the build before the change, and a number less than 0 is a faster build. For `J`, the numbers are from the second test. The first test gave -3.32 % against a prediction of -6.6 %.

### Target `E`: GLU in the quantization (change 0001)

Change 0001 is pull request #30489 of upstream llama.cpp. The project did not write it. In `b12004`, the feed-forward layer ran one kernel for SiLU of `gate` times `up`. That kernel wrote an F32 tensor, and the quantization kernel of MMQ then read it. With the change, the quantization kernel reads `gate` and `up`, calculates SiLU of `gate` times `up` in registers, and writes the Q8_1 data. The program does not write the F32 tensor and does not read it again.

Phase 3 measured the pull request as a lead, thus this target has no prediction entry that came before its measurement. The ranked list of phase 4 uses the gain of the lead as the prediction: 2.0 % of `lat_ms`, with a confidence of 0.95. The measurement is the result. `lat_ms` is 1.94 % lower (95 % interval -2.38 % to -1.50 %), and the answers are the same bit for bit.

Phase 5 added the tests that the lead did not have. `test-backend-ops` gave 310 of 310 correct new cases. The 187 operations of the graph gave the same result as build `D`. The four tools of `compute-sanitizer` found no error in the new kernel. The SASS is the same for 6,744 of 6,748 kernel names. The change applies only to a pass of more than 8 tokens, because a pass of 8 tokens or less uses MMVQ.

Change 0001 is the first change of the stack. Each subsequent change starts from it.

### Target `P`: the uploads of the scheduler (change 0007)

Phase 4 thought that the host runs `cudaStreamSynchronize` 21 to 25 times in each pass. It also thought that the GPU waits 40 µs to 50 µs after each one. It did not find the code that starts them. The prediction was 1.0 % of `lat_ms`, with a confidence of 0.40.

The [prediction entry](notebook/2026-10-31-0007-input-copies-prediction.md) of change 0007 found the code. We committed it at 05:32:15 UTC, before the first run at 05:53:42 UTC. The sync calls are in the scheduler of ggml. A pass has 12 of them before the uploads of the inputs and 9 or 10 in the uploads. Each decision has 1 more at its end.

All sync calls are before the first kernel of a pass. No sync call is between two kernels. The spaces between two kernels are 1 µs to 2 µs, thus the profiles do not show that the GPU waits 40 µs to 50 µs. A change that makes the uploads not wait gives 0.1 ms to 0.15 ms in a pass at most. The entry does not make that change.

The entry found a different gain. A pass uploads 10 inputs to the GPU. The model gets tokens only, and the graph selects the path of the tokens. The nodes of the `mixed` path have no `GGML_TENSOR_FLAG_COMPUTE` flag, and no backend calculates them.

But the scheduler uploaded 3 inputs of these nodes. They have 7 of the 22 sync calls of a pass. [INFERENCE: we did not record the names of the inputs. The bytes, the flags and the sequence agree with the graph code.]

The change makes the scheduler record, for each upload, if a node with the flag reads it. The scheduler does not make an upload that no such node reads. The change is 26 lines in `ggml/src/ggml-backend.cpp`. It adds no kernel and no operation. A pass has 7 uploads and not 10, and 14 to 18 sync calls and not 21 to 25.

The entry predicted -1.27 % for `lat_ms` (range -0.9 % to -1.7 %), -1.0 % for `short_ms`, -1.28 % for `medium_ms` and -1.37 % for `long_ms`. The total is almost the same as the total of phase 4, but the cause is different. The gain for a short request is less than in phase 4 (0.51 ms and 0.85 ms, against 0.8 ms and 2.2 ms). The gain for a medium request and a long request is more (4.55 ms and 10.58 ms, against 3.8 ms and 3.6 ms).

The answers are the same bit for bit in 8 runs. The SASS is the same for 6,766 of 6,766 names. A pass of the new build has 3 uploads and 8 sync calls less. The prediction had 7 sync calls less. `lat_ms` is 0.93 % lower (interval -2.05 % to +0.12 %), `medium_ms` is 1.01 % lower, and `long_ms` is 1.27 % lower. `short_ms` is 0.21 % lower, and its interval includes 0.

The prediction is correct for 5 of 7 metrics. The rule of the entry is that the interval must not include 0. By this rule, the prediction is incorrect for `lat_ms` and `short_ms`. The host load was 14.6 to 21.5 in blocks 1 and 2, and 3.3 to 6.5 in blocks 3 and 4. We did not run a second measurement.

### Target `D`: the value-split path of `gdn_single` (change 0002)

The kernel `gdn_single` has a whole-head path. It runs 32 thread blocks on the 38 SMs of the GPU. The code that selects the path uses it, because 32 heads × 2 is more than 38 SMs. Phase 4 thought that a value-split path with 128 thread blocks makes `gdn_single` 1.6 times faster. This is 5.0 ms (2.1 %) of `lat_ms`, with a confidence of 0.45. The ratio of 1.6 was an assumption and not a test.

We measured the kernel time of each path before we changed the code that selects the path. The test build has 8 paths. `test-backend-ops perf` ran them on 21 shapes with 1, 2 or 3 sequences and 65 to 2,048 tokens. The whole-head path is the fastest path in all 21 shapes. The best value-split path is 1.15 to 1.31 times as slow. One sequence of 2,048 tokens uses 1,290 µs with the whole-head path and 1,696 µs with `<16,64,1>`, which is the best value-split path for one sequence.

The cause is the work of each thread block. The MMA instructions of the whole-head path fill 55 % of the time at the highest rate of the tensor core. Each slice of a value-split path does again the work that is the same for all slices. The MMA instructions of one head go from 1,344 to 1,440, 1,632 and 2,016 for slices of 64, 32 and 16 columns.

The 6 SMs that the whole-head path does not use are 16 % of the SMs. They give less than the work that the slices do again. [INFERENCE: we did not measure the time of the work that a slice does again, other than the MMA instructions. We did not make a profile of a value-split path.]

All 8 paths give the same bits in 9 shapes. All 43 cases of `GATED_DELTA_NET` in `test-backend-ops` agree with the CPU backend for each path. Thus the equivalence class `exact` was correct, but the speed is not.

We do not use `D`. The path selector keeps the whole-head path for a GPU with 38 SMs. There is no speed test, and thus no prediction entry for one. Change 0002 is a set of perf cases for `test-backend-ops` and a comment at the path selector. The program does not change.

The branch `dmk/0002-gdn-split-candidate` adds the kernel `gdn_single<16,64,1>`, and we do not merge it. [INFERENCE: we calculated that this branch makes `lat_ms` 4.1 ms (1.7 %) slower. No request ran with it.] The time model of phase 4 then has no gain from `D`. The expected gain of the total goes from 17.9 % to 16.9 % of `lat_ms`.

### Target `B`: the attention mask on the GPU (change 0004)

Before the change, the host wrote the attention mask of each pass and sent it to the GPU. A large pass has an attention mask of 28 MB, and the copy uses time. The change sends one small input, `info`, with the position and the sequences of each KV cell and of each token (16 bytes for each). A new operation, `KQ_MASK`, makes the attention mask on the GPU from it. The kernel uses 72 µs for the largest pass. The change is commit `cf3913bed` on the branch `dmk/0004-gpu-mask`.

The path on the GPU is on if three conditions are true. The cells of the pass are in one group of sequences. The model has no `SWA` and no `ALiBi`. The pass has 32 sequences or less. Each other case uses the CPU path as before, and `LLAMA_KQ_MASK_DEVICE=0` goes back to the CPU path.

We committed the [prediction](notebook/2026-10-31-0004-gpu-mask-prediction.md) at 05:24 UTC, before the first run at 07:11 UTC. It used the model of phase 4. The gain is the upload time of all inputs plus the host time of the attention mask, less the new work of the kernel. It predicted -2.8 % for `lat_ms` (range -1.8 % to -3.8 %), a gain of 6.7 ms. The prediction for `short_ms` was -1.3 %. Phase 4 gave -2.9 % for `lat_ms`.

The answers are the same bit for bit. `test-backend-ops` gave 24 of 24 correct cases for `KQ_MASK`. We compared 1,285 attention masks of the model and found 0 different values. `compute-sanitizer` with `memcheck` found no error, but the log of the server has no summary line.

The [result](notebook/2026-10-31-0004-gpu-mask-result.md) is less than the prediction. `lat_ms` is 1.06 % lower (interval -1.56 % to -0.54 %), `long_ms` is 1.52 % lower, and `medium_ms` is 1.12 % lower. `short_ms` is 0.08 % higher, and its interval includes 0.

The gain is 2.51 ms for `lat_ms`, 4.02 ms for `medium_ms` and 11.76 ms for `long_ms`. This is 37 %, 40 % and 44 % of the predicted gain. No point estimate is in its predicted range.

The cause is in the model of phase 4 and not in the code. The model used the upload time of all inputs as the copy time of the attention mask. The upload time is 7 times the copy time of the attention mask for a medium request and 3.6 times for a long request. The copy time plus the host time, less the new work, gives 3.67 ms and 11.33 ms. The gain in the test is in 0.5 ms of these values.

[INFERENCE: the other parts of the upload time are the other inputs. They also include the time that the host waits for the GPU at each copy of an input. We did not measure them.]

The change goes into the stack. The rule of the method applies because the gain is less than 50 % of the prediction. The cause is in the two paragraphs above.

### Target `C`: the concat in `ssm_conv` (change 0005)

Each Gated DeltaNet layer makes the input of `ssm_conv` with a `concat`. The `concat` puts the conv state of 3 columns before the new input. The new input is the transposed output of the `QKV` projection, thus its tokens are not adjacent in memory. The `concat` then uses the slow kernel `concat_non_cont`, which uses 12.9 ms in a pass of 2,048 tokens. A second kernel copies the last 3 columns to the cache as the new conv state.

Change 0005 is one kernel that replaces the `concat`, the copy of the conv state, `ssm_conv` and `SILU`. The kernel reads the conv state and the new input where they are. It does not write the `concat` tensor. The change is commit `a67acc087906d2049c52d7ace2a275123656eb09` on the branch `dmk/0005-concat-conv`.

We committed the [prediction](notebook/2026-10-31-0005-concat-conv-prediction.md) at 05:15:13 UTC, before the first run at 06:34 UTC. It used the rule of phase 4 that the change removes 90 % of the time of the `concat` kernel. The predicted gain is 6.4 ms (2.7 %) of `lat_ms`, with limits of 3.2 ms to 10.2 ms.

The answers are the same bit for bit. A first version of the kernel did not add the value 0 at the end of each sum. `ssm_conv` adds it when it has no bias. This changes the sign of a sum that is `-0.0`. The first test found 6,222 different bytes.

We repaired the kernel, and 470 random test cases then gave the same files. `test-backend-ops` gave 122 of 122 correct cases for the new operation. The SASS is the same for 6,766 names, and 6 names are new.

`compute-sanitizer` found 0 errors in the new kernel. `initcheck` on the model with 300 tokens found 136,608 errors in `mul_mat_q`. The baseline has 136,628 errors in the same kernel, and with 256 tokens the new build has 0 errors.

The [result](notebook/2026-10-31-0005-concat-conv-result.md) is more than the prediction. `lat_ms` is 3.02 % lower (interval -3.88 % to -1.81 %), which is a gain of 7.13 ms and 1.11 times the prediction. The prompt speed of `llama bench` is +3.35 %, +3.54 % and +3.86 %, and this is more than the predicted range. The cause of the larger gain can be that the new kernel reads with adjacent threads, and `ssm_conv` ran at 67 % of the bandwidth. [INFERENCE: no profile shows this, because the `nsys` run did not start.]

The change goes into the stack. The host load was 6 in the first block and 15 to 16 in the last block.

### Target `H`: the tile adjustments of MMQ (no change document)

Phase 4 thought that other tile dimensions and other `__launch_bounds__` can put two thread blocks on one SM. The loads of one block then overlap the MMA work of the other block. The prediction was 8 % of the MMQ time at 1,025 tokens or more and 5 % at less. This is 4.2 % of `lat_ms`, with a confidence of 0.30.

We did not change the code of llama.cpp. A sweep in a micro-benchmark ran 142 tile adjustments of the MMQ kernel. It used 6 weight shapes of the model and 29 numbers of tokens from 9 to 2,048. The baseline gives 54.3 TOPS int8 for `ffn_up` at 2,048 tokens.

At 256 tokens or more, the three large shapes have no adjustment that is more than 1.05 times faster than the baseline. The noise is 2 % to 3 %. More thread blocks on one SM do not give back load time at 1,173 tokens or more. The time of two blocks of 8 warps is 1.09 to 1.27 times the time of the baseline.

At less than 256 tokens, 57 of 174 cells have an adjustment that is 10 % faster or more. In all 57 cells, the baseline runs the fix-up kernel, and the fast adjustment gives other bits. Thus these changes are `re-associated` and not `exact`.

The [result entry](notebook/2026-10-31-0006-mmq-tiles-result.md) uses the `oracle`, which is the fastest row for each cell. The `oracle` removes 0.79 % of `lat_ms`, and 0.67 % with the limit `guard3`. For 1,025 to 2,048 tokens it removes 0.54 % of the MMQ time, against the 8 % that phase 4 predicted. The fastest rows with the same bits remove 0.21 % of `lat_ms`. The rules that the code can have make `lat_ms` 2.40 % to 0.06 % slower.

We do not use `H`, and there is no change document. No rule that the code can have gives a gain. A table of about 170 rows in the code gives 0.79 % at most, and this value includes the noise.

The cells with a large gain are `re-associated`. They must have the test of KL divergence and keep a difference from upstream. The rule to stop uses a limit of 3 %. Phase 4 gave `H` 1.3 % at most, and the sweep gives 1.0 %.

### Target `J`: ragged batching of the suffixes (change 0008)

The server runs the questions of one request as sequences with the same prefix. After the prefix, the suffixes of the questions have different lengths. The equal split of the hybrid memory gives all sequences of a ubatch the same number of tokens. A request then runs more passes, and each pass reads all weights. Ragged batching puts all suffixes in one pass. A ubatch with 2 sequences removes 1 pass at most, and a ubatch with 3 sequences removes 2 passes at most.

Two new operations, `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`, have a CPU version and a CUDA version. The Gated DeltaNet kernel runs one token at a time and uses about 28 µs for each token. A step between two lengths can be 128 tokens at most. If a condition is not true, the equal split runs. `LLAMA_RAGGED_SPLIT=0` also selects the equal split.

The equivalence class is `re-associated`. A pass with more tokens uses other matrix-multiply kernels. MMQ runs for more than 8 tokens, and MMVQ runs for 8 tokens or less. The sums then occur in a different sequence, and the last bits of the results can be different.

We committed the [first prediction](notebook/2026-10-31-0008-ragged-suffix-prediction.md) at 07:02:58 UTC. It predicted -6.6 % for `lat_ms`, with a confidence of 0.45. It gave limits for the correctness, all against the equal split. The mean KL divergence must be 1e-5 or less. The largest probability difference of the answers must be 0.01 or less, and the NMSE of each layer state must be 5e-4 or less.

The [result of the first test](notebook/2026-10-31-0008-ragged-suffix-result.md) is -3.32 % for `lat_ms` (interval -4.18 % to -2.63 %), which is 50 % of the predicted gain. `rps8` is 5.96 % higher, in the predicted range. The cause of the smaller gain is the number of ubatches with suffixes of different lengths. The phase `seq` of the benchmark has 20 of them. The prediction counted 52 removed passes for one run of the benchmark, but the 20 ubatches remove 23 passes at most.

The third question of a request can wait for the first two. The log line is `waiting for parent slot to complete`. We did not change the server.

The limits were not true. The mean KL divergence is 3.9e-4 to 1.8e-3 against a limit of 1e-5. The largest probability difference of the answers is 0.0224 against a limit of 0.01. The NMSE of the layer states is more than its limit, and no decision has a different top answer. The prediction has an [INFERENCE] that the conv states are the same bit for bit. The test shows that this is not true.

A control gives the cause. The limits were against the equal split. The control compares the equal split with a decode of one sequence at a time. The distance is as large as the distance of ragged batching from the equal split.

The cause is the selection of the kernel. In the equal split, a pass of 8 tokens or less uses MMVQ. Ragged batching and a decode of one sequence use MMQ.

All other tests are in their limits. They are the tests of the two new operations, the graph of 187 operations, the guards, and the 52-decision set. They also include `compute-sanitizer` and `split_equal`.

We did not change the limits after we saw the data. The [second prediction](notebook/2026-10-31-0008-ragged-suffix-retest-prediction.md) has new limits and 8 new inputs. We committed it at 09:19:03 UTC, before the first run at 09:26 UTC. The test used the stack at `fbfd9cb8c`, and the change is commit `7b3698d45` on the branch `dmk/0008-ragged-suffix`.

The reference is a decode of one sequence at a time, because it is the correct result for a request. For each quantity, `d(equal)` is the distance of the equal split from the reference, and `d(ragged)` is the same for ragged batching. The sum of `d(ragged)` for the 8 inputs must be 1.5 times the sum of `d(equal)` or less. The value of one input must be 3 times `d(equal)` or less. The prediction for `lat_ms` was -3.3 % (range -1.8 % to -4.5 %), with a confidence of 0.65.

The [result of the second test](notebook/2026-10-31-0008-ragged-suffix-retest-result.md) shows that the speed is in the predicted range. `lat_ms` is 2.46 % lower (interval -3.80 % to -1.46 %). `short_ms` is 2.00 % lower, `medium_ms` is 4.68 % lower, and `long_ms` is 1.75 % lower. A short request with 1 question is 1.38 % slower, and we do not know the cause. `rps8` is 4.80 % higher.

The sums are in the limit of 1.5 times. The five ratios are 0.78, 1.29, 1.17, 0.92 and 1.08. The largest probability difference of the answers is 0.019074, against 0.022426 for build A with `-np 3`. The 52-decision set is in its limit.

But the limit of 3 times for one input is not true for 3 of 40 pairs of an input and a quantity. Input 7 has a largest KL divergence of 3.76 times its reference value. Input 8 has 1 row with a different top token, and its reference has 0. Input 1 has an NMSE of the conv state of 4.51 times its reference value.

The rule of the test is that the change stays out of `dmk/stack` when a limit is not true. Change 0008 stays out, and the branch `dmk/0008-ragged-suffix` keeps it. A new limit after this result must have a new prediction and new inputs.

The server can start all questions together. [INFERENCE: this gives 1 % to 2 % of `lat_ms` at most. We did not measure it.] We did not make that change, and it gives a gain only together with ragged batching.

### Target `G`: the load pipeline of MMQ (no change document)

Phase 4 thought that a pipeline with `cp.async` loads makes MMQ 1.35 times faster at 1,025 tokens or more. The loads of the next step then run while the MMA instructions of this step run. The prediction was 14.8 % of `lat_ms`, with a confidence of 0.40.

The entry of `H` measured the ceiling of `G` with variants of the kernel. Without the loads (`no_loads`), the MMQ time is 27.6 % lower, which is 15.6 % of `lat_ms`. Phase 4 used this value, the full load time, as the gain. The warps wait for data in 21.8 % of the MMQ time (`same_loads`), which is 12.4 % of `lat_ms`. A pipeline cannot remove the instructions of the loads.

The entry set the confidence of a pipeline at 0.42 for Q4_0 and 0.17 for Q8_0. A first change that gives Q4_0 less than 1.08 times at 1,025 tokens or more is below the limit of the rule to stop. [INFERENCE: this value is 41 % of the ceiling `same_loads` for `ffn_up` at 2,048 tokens.]

We made a pipeline for Q4_0 in a micro-benchmark. The last of 6 versions loads the weights and the activations of the next step with `cp.async` while the MMA instructions run. It is commit `4f7bc1fe7` on the branch `dmk/0009-mmq-pipeline`. The shared memory goes from 57,856 bytes to 94,720 bytes at `J` of 128, and the kernel uses 255 registers. The change adds 257 lines and removes 22 lines.

The output has the same bits in all 62 cells of 13 runs. The mean ratio is 1.042 at 1,025 tokens or more, against a limit of 1.08. The largest cell is 1.088. `ffn_up` and `ffn_gate` give 1.027, and `ffn_down` gives 1.056. Version 1 loads the weights only and gives 0.934, thus the loads of the activations give the gain. The pipeline removes 18 % of the ceiling `same_loads` at 1,025 tokens or more.

We calculated the gain for each request type from the micro-benchmark. It is 2.21 % for `short1q`, 1.96 % for `short2q`, 0.95 % for `medium`, 1.84 % for `long`, and 1.65 % for `lat_ms`. The largest value is below 3 % with a confidence of 1. This is a projection and not a speed test.

A pipeline for Q8_0 must have 111,104 bytes of shared memory, and the limit is 101,376 bytes. We did not make it. We did not try other alternatives, for example a pipeline for the loads from the L2 cache or a warp that only loads. The column `ndiff` of the micro-benchmark is the only test of correctness.

We do not use `G` in this shape. We do not run a test with two arms, and we do not open a pull request. The branches `dmk/0009-mmq-pipeline` and `dmk/0009-mmq-pipeline-bench` are for reference and not for a merge.

### Target `A`: the output head for the answer tokens (change 0003)

The server gets the answer of a decision from the logits of some label tokens. The output head calculates the logits of all 248,320 tokens. Its weight has the type Q8_0, and a pass reads 675 MB for it. Change 0003 gives `llama_context` a list of tokens (`llama_set_logit_tokens`). If the list is set, the head calculates the rows of the weight for these tokens only. The server sets the list for a batch of decisions.

A new operation, `ggml_mul_mat_rows`, reads these rows. The change applies for 4 output rows at most and a list of 64 tokens at most. The first version is commit `f0a2033e7` on the branch `dmk/0003-answer-head`. The last commits are `ca69612da`, `c4175ae75` and `419b8f612` on the branch `dmk/0003-answer-head-rebased`.

We committed the [prediction](notebook/2026-10-31-0003-answer-head-prediction.md) at 06:18:10 UTC. The first run was at 10:32:12 UTC. The prediction was a gain of 1.75 ms for each pass with outputs. This is the head kernel of 1.64 ms and the copy of the logits of 0.15 ms, less the new work of 0.04 ms. The prediction for `lat_ms` was -1.5 % (3.6 ms), and for `short_ms` it was -3.9 %. It holds if `lat_ms` is -0.8 % or less and its interval is below 0.

The [result](notebook/2026-10-31-0003-answer-head-result.md) on arm `L`, which is the stack with change 0001 only, holds the prediction. `lat_ms` is 0.87 % lower (interval -1.38 % to -0.33 %). `short_ms` is 3.15 % lower, `medium_ms` is 0.84 % lower, and `long_ms` is 0.09 % lower with an interval that includes 0. `rps8` is 1.24 % higher. The gain is about 1.4 ms for each pass with outputs, against 1.75 ms in the prediction.

On the stack (arm `A3R`), `lat_ms` is 1.48 % lower (interval -2.03 % to -0.81 %). `short_ms` is 3.73 % lower, `medium_ms` is 1.33 % lower, and `long_ms` is 0.58 % lower. This test has no prediction. The tests of the branch give the same answers bit for bit, and `memcheck` found 0 errors.

The tests found two faults in the first build. The first build gave 378 `graph reallocation` lines in one run of the benchmark. Log level 4 shows these lines, and the default level does not. A first check at the default level found 0 lines, and this result was not correct. The repair is a second `sched_reserve` with a list of 64 tokens. It uses about 0.2 s one time, and the number of lines is then 0.

The second fault is in the MMVQ launch. The test `test-mul-mat-rows` gave 120 of 480 cases with different values, and all 120 have K = 1,536. The guard was not correct with `ids`. After the repair, 0 of 480 cases give different values. The head of the model has K = 2,560 and did not use this launch.

The test cases `-p same_rows` with `n_ids` of 1 had an NMSE above the limit (6 of 177 on `f0a2033e7`, and 2 on `ca69612da`). Commit `c4175ae75` changed them to `n_ids` of 10 and 33. The result is 121 of 121 correct cases in 3 runs.

The first version of the server change had a fault that no test shows. It made the list from the labels of the decision slots if the first slot of a batch was a decision. A text slot in the same batch then had no logits. [INFERENCE: we found this fault when we read the code, and not in a run.] Commit `419b8f612` repairs it. The server sets the list only if each slot of the batch holds a decision.

The model gives `HTTP 501` for each text request, thus a batch with a text slot cannot occur with this model. The two runs with text requests gave `HTTP 501` for 25 of 25 requests. They do not show the fault or its repair. We have no other model on the host to examine it.

### The two tests of the stack

Each change in the stack is `exact`. Thus the answers of the stack must be the same bit for bit as the answers of build `D` and of the image in production. The [phase 1 result](notebook/2026-10-30-phase1-equivalence-result.md) shows that build `D` and the image give the same answers bit for bit. The stack has two speed tests. Each test has 6 blocks of the private benchmark and 4 blocks of `llama bench`.

The first test compares `dmk/stack` at `fbfd9cb8c` with build `D`. The stack has the changes 0001, 0007, 0002 (test cases only), 0004 and 0005. We committed the [prediction](notebook/2026-10-31-stack-1-prediction.md) at 08:38:18 UTC, before the first run at 08:43:08 UTC. The prediction multiplies the ratios of the tests of the changes. The range is ±1.5 points.

The first line of the prediction gave the time "about 08:45". A commit at 08:38:30 UTC changed it to "about 08:35". The commit changed no other text and no number.

The [result](notebook/2026-10-31-stack-1-result.md) is a faster build with the same answers bit for bit in 12 runs. `lat_ms` is 8.04 % lower (interval -8.50 % to -7.62 %), a gain of 19.34 ms. The prediction was -6.8 % (range -5.3 % to -8.3 %). Seven of the 8 results show more gain than the prediction. `short_ms` is 6.00 % lower, and this is not in its range of -2.4 % to -5.4 %.

The result entry does not find one measured cause for `short_ms`. The range of ±1.5 points is too small for the four tests of the changes together, which have ±2.08 points. [INFERENCE: this is the largest part of the cause.] A second test of the same two arms with the client `latency_by_type.py` (2 blocks) gives -4.97 % for `short_ms`, which is in the range. We wrote no prediction for it, and we ran it after the first result.

The second test compares `dmk/stack` at `419b8f612` with the image `server-cuda-b12004`, which the server in production uses. This stack also has change 0003. We committed the [prediction](notebook/2026-10-31-stack-2-prediction.md) at 20:43:36 UTC, and the first run started at 20:43:55 UTC. This is the last speed test before the stack goes into production.

The prediction multiplies three ratios for each metric. The first ratio is build `D` against the image (phase 1). The second is the first test of the stack. The third is change 0003, and it is 1 for `llama bench` because the change does not touch that path. The range is ±2.0 points, because the first test had a range of ±1.5 points and one result that was not in it.

The first line of this prediction gives the time "about 20:45 UTC". This time is after the commit and after the first run. We did not change the file. The time of the commit is the true time.

The [result](notebook/2026-10-31-stack-2-result.md) is a faster build with the same answers bit for bit in 12 runs. All 8 results are in their ranges. Each result has less gain than the predicted value, by 0.1 to 0.7 points. The load of the host was 1.14 to 2.64, and the largest VRAM use was 4,425 MiB.

| Metric | Predicted change (%) | Range (%) | Result (%) | 95 % interval (%) |
|---|---:|---|---:|---|
| `lat_ms` | -9.8 | -11.8 to -7.8 | -9.145 | -9.280 to -8.978 |
| `short_ms` | -9.8 | -11.8 to -7.8 | -9.611 | -9.743 to -9.462 |
| `medium_ms` | -9.8 | -11.8 to -7.8 | -9.264 | -9.386 to -9.115 |
| `long_ms` | -9.4 | -11.4 to -7.4 | -8.963 | -9.077 to -8.799 |
| `rps8` | +10.7 | +8.7 to +12.7 | +10.340 | +10.209 to +10.465 |
| `llama bench`, 64 tokens | +6.0 | +4.0 to +8.0 | +5.882 | +5.770 to +5.999 |
| `llama bench`, 256 tokens | +7.7 | +5.7 to +9.7 | +7.266 | +7.177 to +7.354 |
| `llama bench`, 3,000 tokens | +9.5 | +7.5 to +11.5 | +9.388 | +9.206 to +9.503 |

Source: the [prediction](notebook/2026-10-31-stack-2-prediction.md), the [result](notebook/2026-10-31-stack-2-result.md) and the [data set](../data/2026-10-31-stack-2/README.md) of the second test of the stack. The range is the predicted change ±2.0 points. A latency less than 0 is a faster build. A number more than 0 for `rps8` and `llama bench` is also a faster build.

| Arm | `lat_ms` (ms) | `short_ms` (ms) | `medium_ms` (ms) | `long_ms` (ms) |
|---|---:|---:|---:|---:|
| `P`, the image in production | 237.015 | 71.264 | 357.505 | 781.715 |
| `S2`, the stack | 215.339 | 64.414 | 324.386 | 711.649 |

Source: the [result](notebook/2026-10-31-stack-2-result.md) of the second test of the stack. The gain of `lat_ms` is 21.68 ms.

The result of `pp256_ts` is not in the interval of the three tests together (+7.27 % against +7.57 % to +7.79 %). It is in the range of the prediction. The result of `short_ms` is 3.84 points more than in the first test, which is near the 3.73 % of change 0003 for `short_ms`. [INFERENCE: the two tests have different arms and a different host load. The data do not show which part comes from change 0003.]

### The rule to stop

The [entry of the rule to stop](notebook/2026-10-31-phase5-rule-to-stop.md) applies the rule of phase 4 again. The work stops when the expected gain is below 3 % for each request type. The entry makes two corrections.

The stack removed about 9 % of the time of each request. A target that removes the same time in ms is then a larger part of the new time. We multiply each gain of phase 4 by 1.1. The confidence of `N` goes from 0.20 to 0.10, because the test of `G` gave much less than its prediction.

| Target | Work | Largest gain in phase 4 (%) | Request type | Confidence | Expected gain (%) |
|---|---|---:|---|---:|---:|
| `N` | New MMQ kernels | 11.0 | `short2q` | 0.10 | 1.2 |
| `F` | Fusion of RMS norm kernels, element-wise kernels and quantization kernels | 3.8 | `long` | 0.50 | 2.1 |
| `L` | Smaller KV window for flash attention | 1.3 | `medium`, `long` | 0.50 | 0.7 |
| `K` | CUDA graphs for passes with more than one token | 2.2 | `short2q` | 0.30 | 0.7 |
| `M` | One kernel for each layer | 6.1 | `short2q` | 0.10 | 0.7 |
| `O` | Cache of the prompt tokens | 1.0 | `short2q` | 0.50 | 0.6 |

Source: the [entry of the rule to stop](notebook/2026-10-31-phase5-rule-to-stop.md) and `tables/ranked.tsv` of the [time model](../data/2026-10-31-phase4-model/README.md). The confidence of `N` was 0.20 in phase 4. At 0.20 the expected gain of `N` is 2.4 %.

The largest expected gain of one target for one request type is 2.1 % (`F`, `long`). It is below 3 %. Thus the rule stops the work on the kernels, and the project does not make the targets `F`, `L`, `K`, `M`, `N` and `O`.

The entry gives two more items that are not in the ranked list. The first item is the uploads that stay before the first kernel of a pass. They use 15.5 ms in a long request. If a change removes all of them, the gain is 2.2 % of `long_ms` (711.6 ms), which is below 3 % before the confidence. The second item is to start the third question of a request after the other questions. This gives 1 % to 2 % of `lat_ms` or less, and only together with ragged batching, which we do not use.

### The facts that phase 5 shows

- **The changes in the stack remove work that the answers do not use.** They write less data between kernels (`E` and `C`), send less data to the GPU (`P` and `B`), or read less of the output head (`A`). All of them are `exact`. The three targets that try to speed the matrix work (`D`, `H` and `G`) gave no gain that the project can use.
- **A value of phase 4 is a ceiling and not a prediction.** The ratio of 1.6 for `D` was an assumption, and the value of 1.35 for `G` was the full load time. The model of `B` used all upload time as the copy time of the attention mask. `B` gave 37 % to 44 % of its predicted gain. The prediction for `H` was 8 % of the MMQ time, and the oracle gave 0.54 %. The cause of `P` was not the cause of phase 4.
- **A short test before the speed test can stop a target.** The profiles of `P` and the micro-benchmarks of `D`, `H` and `G` showed the problem before a speed test. The speed test of `B` showed it after the work.
- **A limit must be against the correct reference.** The first limits of `J` were against the equal split. The control showed that the equal split is as far from a decode of one sequence at a time as ragged batching is. New limits and new inputs gave sums in the limit, but 3 of 40 pairs were above the limit of 3 times.
- **Tests find faults, and an inspection of the code finds a fault that no run shows.** The tests of `A` found two faults, and the tests of `C` found the sign of `-0.0`. The third fault of `A` is in a case that this model cannot give. [INFERENCE: we found it when we read the code, and not in a run.]
- **The range of a prediction for a stack must come from the intervals of its tests.** The first prediction had ±1.5 points, and `short_ms` was not in it. The four tests together have ±2.08 points. The second prediction had ±2.0 points, and all 8 results were in their ranges.

The stack is 9.145 % faster for `lat_ms` than the image in production, with the same answers bit for bit. The prediction of phase 4 for the total was 17.9 % with the expected gains. The sets of targets are not the same, thus we cannot compare the two numbers.

The soak test of 24 hours and the change of production are the last steps. The section [The next steps](#the-next-steps) gives them.

## The next steps

The changes of phase 5 are complete, and the rule to stop stops the work on the kernels. The [conditions of the README](../README.md#condition-of-each-change) show the state of each target. The stack at `419b8f612` has the changes 0001, 0007, 0002 (test cases only), 0004, 0005 and 0003. It is 9.145 % faster for `lat_ms` than the image in production, with the same answers bit for bit.

The rule of the project for a build has three steps before production. The first step is the tests of correctness, and the second step is the test with two arms. The third step is a long test on the requests of production. The project did the first two steps for the stack. The third step and the change of production stay, and they do not change the code.

1. Run the soak test of 24 hours on the stack at `419b8f612` with the requests of production.
2. Check that the soak test has no GPU fault and no error of `compute-sanitizer`.
3. Check that the answers are the same all the time.
4. Change the server in production to the build of the stack after a correct soak test.
5. Keep the image `server-cuda-b12004` as the build to go back to after the change of production.

The image `server-cuda-b12004` gives the same answers as build `D`, and the stack gives the same answers as the image. Each change document gives the procedure to go back to the build before its change. Change 0004 goes back with the environment variable `LLAMA_KQ_MASK_DEVICE=0`. The other changes in the stack go back with `git revert` or with a build before the change.

The lead #30489 had six steps before use in production. The project did three of them. `test-backend-ops` gave 310 of 310 correct cases for the GLU cases. The four tools of `compute-sanitizer` found no error in the new kernel. The [change document 0001](changes/0001-glu-into-quantize.md) has the equivalence class `exact`.

Three items stay open for change 0001. We cannot write a prediction entry before a measurement that we made. No test measures only change 0001 again with more blocks at a low load. The change document gives no profile of the two builds.

Source: the section ["Limits and next work"](notebook/2026-10-31-lead-30489.md#limits-and-next-work) of the entry of the lead, the [change documents](changes/README.md), and the ["Next step"](notebook/2026-10-31-stack-2-result.md#next-step) of the second test of the stack.

## Known and unknown facts

After phase 5, the project knows these facts:

- The speed and the answers of build `D` are the same as the speed and the answers of the stock image. Thus a change of the project compares with build `D`.
- The GPU is the limit of a request, and MMQ is the largest part of the kernel time.
- The MMQ kernels are at 32.5 % to 39.2 % of the int8 peak. The structure of the MMQ loop can be the cause [INFERENCE].
- No tile adjustment of MMQ gives more than 1.05 times at 256 tokens or more for the three large shapes. A load pipeline for Q4_0 gives 1.042 times at 1,025 tokens or more.
- The value-split path of `gdn_single` is 1.15 to 1.31 times as slow as the whole-head path on this GPU.
- Fusions of kernels give a gain. The lead #30489 gave 1.94 % for `lat_ms`, and change 0005 gave 3.02 %.
- A pass has 21 to 25 sync calls. All of them are before the first kernel of a pass, and the uploads of the scheduler cause them.
- The uploads that stay use 15.5 ms in a long request.
- The equal split is as far from a decode of one sequence at a time as ragged batching is from the equal split.
- The stack is 9.145 % faster for `lat_ms` than the image in production, with the same answers bit for bit in 12 runs.

The project does not know these facts:

- The cause of the faster second request with the same case.
- The cause of the high DRAM value of two MMQ kernels.
- How much of the cap of the matrix kernels comes from the higher clock.
- The width of the PCIe link of the test device [INFERENCE: PCIe 3.0 x8 fits the rate].
- The result of one change at a low load with a large number of blocks. The second test of the stack had a low load, but it tests all changes together.
- The part of each change in the gain of the stack.
- The cause of the larger gain of change 0005 than the prediction.
- The cause of the smaller gain of change 0003 for the `medium` request and the `long` request.
- The cause of the `short1q` request that is 1.38 % slower with change 0008.
- The cause of the different number of ubatches in the benchmark and in the client with change 0008.
- The cause of the small gain of the load pipeline of target `G`.
- If the repair of commit `419b8f612` shows in a run. No model on the host can examine it.
- The cause of the lower value of `pp256_ts` than the estimate (+7.27 % against +7.57 % to +7.79 %). [INFERENCE: change 0003 or the code layout of the build can cause a part.]
- The result of a test of 24 hours for the stack.

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
| 5 | The phase 4 cause of target `P` was not true. The profiles do not show that the GPU waits 40 µs to 50 µs after a sync call. | The prediction entry of change 0007 found the code. The change skips 3 uploads. | [Prediction of change 0007](notebook/2026-10-31-0007-input-copies-prediction.md) |
| 5 | Other programs ran on the GPU, with no GPU lock, in the first measurement window of target `D` (05:11 to 05:28). | We did not use that measurement window. We made the test again with the GPU lock (06:17 to 06:31). | [Result for target `D`](notebook/2026-10-31-0002-gdn-value-split-result.md) |
| 5 | Other work ran on the GPU from 05:45:30 to 05:48:30, in the first measurement window of the sweep of target `H`. It changed the rows of the Q8_0 shape with 8,192 rows. | We ran the Q8_0 shapes with 8,192 rows and 4,096 rows again (07:05 to 07:10). The tables use the second run. | [Result for target `H`](notebook/2026-10-31-0006-mmq-tiles-result.md) |
| 5 | Small tasks ran on the GPU with no GPU lock, at times with no measurement window (for example 05:26:10 to 05:27:52). Other measurement windows were open at these times. | The entry of change 0004 lists the times. The data set of change 0004 has no run from that time. | [Result of change 0004](notebook/2026-10-31-0004-gpu-mask-result.md#limits-and-next-work) |
| 5 | The prediction of change 0004 used all upload time as the copy time of the attention mask. The gain was 37 % to 44 % of the prediction. | The result entry gives the cause. The copy of the attention mask is a part of the upload time. | [Result of change 0004](notebook/2026-10-31-0004-gpu-mask-result.md) |
| 5 | A first version of the kernel of change 0005 did not add 0 at the end of each sum. It gave 6,222 different bytes. | We repaired the kernel. 470 random test cases give the same files. | [Result of change 0005](notebook/2026-10-31-0005-concat-conv-result.md) |
| 5 | Some limits of the first prediction of change 0008 were not true. They were against the equal split, and the conv states are not the same bit for bit. | The second prediction has new limits against a decode of one sequence at a time and 8 new inputs. 3 of 40 pairs are above the limit of 3 times, and we do not use change 0008. | [Result of the first test](notebook/2026-10-31-0008-ragged-suffix-result.md), [result of the second test](notebook/2026-10-31-0008-ragged-suffix-retest-result.md) |
| 5 | The first build of change 0003 gave 378 `graph reallocation` lines. A first count at the default log level was not correct. | We count at log level 4. A second `sched_reserve` with a list of 64 tokens gives 0 lines. | [Result of change 0003](notebook/2026-10-31-0003-answer-head-result.md) |
| 5 | The first version of the server change of 0003 gave no logits to a text slot in a batch with decisions. We found this fault when we read the code [INFERENCE]. | Commit `419b8f612` repairs it. No run shows the fault or its repair. | [Result of change 0003](notebook/2026-10-31-0003-answer-head-result.md) |
| 5 | The range of ±1.5 points was too small for the first test of the stack. `short_ms` was not in its range. | The second test of the stack has a range of ±2.0 points. All 8 results are in their ranges. | [Result of the first test of the stack](notebook/2026-10-31-stack-1-result.md#finding), [result of the second test of the stack](notebook/2026-10-31-stack-2-result.md) |
| 5 | The first line of the prediction of the first test of the stack gave a time after the first run (about 08:45). | A commit 12 seconds after the first commit changed it to about 08:35. No other text and no number changed. | [Result of the first test of the stack](notebook/2026-10-31-stack-1-result.md) |
| 5 | The first line of the prediction of the second test of the stack gives the time about 20:45. This time is after the commit and after the first run. | We did not change the file. The time of the commit is the true time. | [Result of the second test of the stack](notebook/2026-10-31-stack-2-result.md) |

Source: the entries in the last column.

## Where to read more

The [notebook](notebook/README.md) has the entries in full. The [data sets](../data/README.md) have the raw files, the tables, and the conditions of each measurement. The [kernels document](kernels.md) has the facts of the kernels, and the [method](method.md) has the rules of the tests.
