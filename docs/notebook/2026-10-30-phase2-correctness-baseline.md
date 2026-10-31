# Phase 2: correctness baseline of the model mix8 on build D (result)

Written 2026-10-30, after the measurement window (22:07 to 22:51 UTC).

## Task of this entry

This entry records the first measurement of the quantities that the correctness tests use. It has no prediction entry. The numbers are the baseline. Each subsequent prediction entry that changes a kernel must compare with them. The [README of the data set](../../data/2026-10-30-window1/README.md) gives the raw files, the tables and the conditions. The build is build D of the [result of phase 1](2026-10-30-phase1-equivalence-result.md).

## Result in short

- The KL divergence of mix8 from the Q8_0 model is 0.027830 on the WikiText-2 test file and 0.057587 on the benchmark states. These two numbers are the KL budget.
- The 52-decision set gives 48 correct decisions of 52 on the two arms. The four incorrect decisions are the same.
- The four tools of `compute-sanitizer` find no error in one pass of mix8 on build D. The test has a limit: it does not run all kernel paths.
- The public benchmark has a reference from the Q8_0 model. Three runs of mix8 gave `lat_ms` of 237.1 to 237.5 ms.
- The CUDA backend of build D gives the same result as the CPU backend in 181 of 187 tests of the graph of mix8. The 6 tests with `FAIL` have causes in the test method. The causes are not in the CUDA code. The section `test-backend-ops` gives the facts.

## KL budget: mix8 against the Q8_0 model

The model mix8 comes from the Q8_0 model by quantization. The KL divergence of mix8 from the Q8_0 model is the error that this quantization makes. We use it as the KL budget. A change to a kernel that gives a different number must add much less error than the KL budget.

Build D wrote the base file of the Q8_0 model. A second run of build D with mix8 read the base file. The test is the same as the test of phase 1: 20 chunks of 512 tokens, the flags `-b 2048 -ub 2048` and the flag `-fa auto`.

| Text | Mean KL divergence | Error of the mean | Largest KL divergence | 99.9 % percentile of the KL divergence |
|---|---:|---:|---:|---:|
| WikiText-2 test | 0.027830 | 0.000745 | 1.568225 | 0.783042 |
| Benchmark states (the 16 `medium` and `long` states) | 0.057587 | 0.004597 | 11.218664 | 4.251879 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `kld.tsv`.

| Text | Same top token (%) | Error of the same top token (%) | Perplexity of mix8 | Perplexity of Q8_0 | Ratio of the perplexities | Error of the ratio |
|---|---:|---:|---:|---:|---:|---:|
| WikiText-2 test | 90.843 | 0.404 | 14.894258 | 14.617107 | 1.018961 | 0.003461 |
| Benchmark states (the 16 `medium` and `long` states) | 88.588 | 0.445 | 9.177134 | 9.247825 | 0.992356 | 0.005584 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `kld.tsv`.

The KL floor of the method is much smaller than these numbers. In phase 1 it was 0.000000 for the mean KL divergence and 0.000058 or less for the largest KL divergence. The numbers in this entry are thus the error of the quantization and not the noise of the method.

On the WikiText-2 test file, the perplexity of mix8 is 1.9 % larger than the perplexity of Q8_0. On the benchmark states, the ratio is 0.992 with an error of 0.006. The difference from 1 is 1.4 times the error. We calculated this value. The perplexity does not show that mix8 is worse than Q8_0 on these states. The KL divergence and the same top token show a difference on the two texts.

## The 52-decision set

The set has 32 cases and 52 decisions with labels. The cases of the set are not in this repository. The folder `bench/dectest/` has a set with the cases of the public benchmark. We ran the set on arm S and on arm D with the client `run_dectest.py`. The test is a fast check of the quality of the answers.

| Group | Decisions | S correct | D correct |
|---|---:|---:|---:|
| All | 52 | 48 | 48 |
| `triage` | 24 | 20 | 20 |
| `check` | 8 | 8 | 8 |
| `style` | 16 | 16 | 16 |
| `commit` | 4 | 4 | 4 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `decisions52.tsv`.

The four incorrect decisions are the same on the two arms: `triage00.category`, `triage07.category`, `triage07.needs_action` and `triage08.category`. The two answer files have the same bytes. The 48 correct decisions of 52 are thus the baseline of the set. A change to a kernel must keep the 48 correct decisions. If an incorrect decision is different, find the cause first.

## Test with `compute-sanitizer`

The test runs `llama perplexity` on build D with the model mix8. It uses 1 chunk of the benchmark states and the flags `-c 256 -b 256 -ub 256`. The wrapper is `timeout 1800 compute-sanitizer --tool <tool> --error-exitcode 99`. The version of `compute-sanitizer` is 12.8.93. All four runs give the perplexity 17.5145.

| Tool | Errors (count) | Time of one pass (s) |
|---|---:|---:|
| `memcheck` | 0 | 4.11 |
| `initcheck` | 0 | 2.53 |
| `synccheck` | 0 | 1.15 |
| `racecheck` | 0 | 606.47 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `sanitizers.tsv`.

The tool `racecheck` also shows 0 problems and 0 warnings. The time of one pass of `racecheck` is about 10 minutes. This is about 150 times the time of `memcheck`. We calculated this value.

### Limit of the `compute-sanitizer` test

The test has a limit. The path of the test uses the MMQ kernels with 256 tokens and one sequence. It does not use these paths:

- The MMVQ kernels. The server uses them for a batch of 8 tokens or less.
- A batch with some sequences.
- The decision path of the server.

A kernel change must run the four tools on the paths that the change touches. The result of this entry does not show that these paths have no error.

## `test-backend-ops`

The test compares the CUDA backend of build D with the CPU backend of the same build. It does this for each operation of the graph of mix8. The program `test-export-graph-ops` wrote the different operations of the graph for a prompt and for one token for each sequence. It did this for the three values 2048, 160 and 48 of `-ub`. The three files have 97, 97 and 91 different operations. The merged file has 187 different operations.

The [README of the data set](../../data/2026-10-30-test-backend-ops/README.md) gives the commands and the conditions.

The server in production was on during the test. The test of the 187 operations ran for 64 s. A second test ran the built-in tests of `test-backend-ops` for five operations of the model.

| Test | Tests that run (count) | Tests with `OK` (count) | Tests with `FAIL` (count) | Tests with `not supported` (count) |
|---|---:|---:|---:|---:|
| The 187 operations of the graph of mix8 | 187 | 181 | 6 | 0 |
| The built-in tests of `GATED_DELTA_NET`, `CPY`, `SSM_CONV`, `L2_NORM` and `SOFTPLUS` | 390 | 390 | 0 | 310 |

Source: [README of the data set](../../data/2026-10-30-test-backend-ops/README.md) and the table `results.tsv`.

Before the measurement window, the built-in tests of `MUL_MAT` on build D gave 2,490 correct tests of 2,490. These tests have no table in a data set.

| Operation | Tensor name | Dimensions of the output | Text of the test |
|---|---|---|---|
| `CPY` | `conv_state_update-0` | 24576, 3, 1, 1 | `ERR = inf > 0.000500000` |
| `CPY` | `conv_state_update-0` | 24576, 3, 1, 1 | `ERR = inf > 0.000500000` |
| `CPY` | `conv_state_update-0` | 24576, 3, 1, 1 | `ERR = inf > 0.000500000` |
| `CPY` | `conv_state_update-0` | 24576, 3, 1, 1 | `ERR = inf > 0.000500000` |
| `CPY` | `cache_s_l0` | 524288, 3, 1, 1 | `ERR = inf > 0.000500000` |
| `GATED_DELTA_NET` | `node_49` | 4096, 2433, 1, 1 | `NaN at index 461056`: CUDA gives `nan`, the CPU gives -3.7e32 |

Source: [README of the data set](../../data/2026-10-30-test-backend-ops/README.md) and the table `failures.tsv`.

The 6 tests with `FAIL` have two causes. The two causes are in the test method. We read the code of llama.cpp at the tag `b12004`. We did not change the test and run it again.

### Cause of the 5 tests of `CPY`

In `ggml`, the output of `CPY` is a view of its second source. The CUDA backend writes the copy into that source (`ggml-cuda.cu`, line 2389). The test that `--test-file` makes gives `CPY` a new output tensor (`tests/test-backend-ops.cpp`, lines 9428 to 9487). The CUDA backend does not write that tensor. The test thus reads memory that has no value. [INFERENCE: we did not run a test that gives `CPY` the correct output.]

### Cause of the test of `GATED_DELTA_NET`

The test has 683 tokens for each sequence and 3 sequences. The test that `--test-file` makes fills each `f32` input with random values from -1 to 1 (`tests/test-backend-ops.cpp`, lines 9517 to 9582). This includes the `gate` input. In the model, this input has values of 0 or less, and it makes the state smaller. With a value of 1, the state can become 2.7 times larger at each token. After 683 tokens the values are larger than the range of `f32`.

The built-in test of this operation sets the `gate` input to a correct range. [INFERENCE: we did not run the test with a correct `gate` input.]

### Tests that the built-in tests do not run

The built-in tests of the same operations on build D all pass. This includes the tests of `GATED_DELTA_NET` with a maximum of 513 tokens for each sequence. The path with tensor cores for long inputs starts at 64 tokens. The built-in tests have no case with 683 tokens.

A prompt of the model can have 683 tokens in a batch of one sequence. The batch has a maximum of 2048 tokens. The KL divergence tests and the benchmark runs of the measurement window use batches of this type. [INFERENCE: the logs do not show the number of tokens of each batch.]

### Rule for subsequent changes

Run the file with the operations of the model and the built-in tests of each operation that the change touches. Run them on the baseline build and on the build with the change. The 6 tests with `FAIL` of this section must stay the same 6 tests.

## Driver text in the standard error

Each `.err` file and each `.server.log` file of the measurement window starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The files of the two arms have it. The runs with no `compute-sanitizer` have it also. The container set-up of the driver writes the text. It does not come from llama.cpp or from `compute-sanitizer`. We saw no effect from it on the runs.

## Public benchmark

The public benchmark in `bench/` has the same token counts as the private benchmark. The runs used build D and the flags of the server in production. The run `p2-pub-ref-q8` used the Q8_0 model and 1 timed pass. It wrote the reference answers. The three runs `p2-pub-base-1` to `p2-pub-base-3` used mix8 and 5 timed passes.

| Run | Model | Timed passes | `lat_ms` (ms) | `short_ms` (ms) | `medium_ms` (ms) | `long_ms` (ms) |
|---|---|---:|---:|---:|---:|---:|
| `p2-pub-ref-q8` | Q8_0 | 1 | 233.265 | 71.866 | 348.437 | 763.688 |
| `p2-pub-base-1` | mix8 | 5 | 237.385 | 71.460 | 357.871 | 782.487 |
| `p2-pub-base-2` | mix8 | 5 | 237.487 | 71.509 | 357.990 | 781.724 |
| `p2-pub-base-3` | mix8 | 5 | 237.068 | 71.391 | 357.578 | 781.175 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `public-bench.tsv`.

| Run | `rps8` (requests/s) | Correct decisions (of 100) | Errors (count) |
|---|---:|---:|---:|
| `p2-pub-ref-q8` | 4.294 | 93 | 0 |
| `p2-pub-base-1` | 4.233 | 94 | 0 |
| `p2-pub-base-2` | 4.234 | 94 | 0 |
| `p2-pub-base-3` | 4.238 | 94 | 0 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `public-bench.tsv`.

The three runs of mix8 ran before the reference run. They have no numbers for `agree` and `mean_dp`. They are not the baseline of the benchmark. The file `bench/reference.json` has the answers of `p2-pub-ref-q8`.

The file `bench/baseline.json` had the mark `not-measured` when we wrote this entry. It comes from the next measurement window with the procedure of the [README of the benchmark](../../bench/README.md#make-the-reference-and-the-baseline). The procedure has three runs after the reference run and a fourth run. The fourth run must give `PASS`.

On 2026-10-31, the project made the baseline in the second measurement window, with the procedure above. The fourth run gave `PASS`. The [README of the benchmark](../../bench/README.md#source-of-the-baseline) gives the numbers. The [data set](../../data/2026-10-31-public-baseline/README.md) has the raw files.

The three runs of mix8 have a mean `lat_ms` of 237.31 ms. The 6 runs of the private benchmark in blocks 4 to 6 of phase 1 have a mean of 237.36 ms. The difference is -0.02 %. The load average of the host was 2.16 to 3.56 for the three runs of the public benchmark. It was 2.13 to 2.84 for the 6 runs of the private benchmark. On a host with a low load, the public benchmark thus gives the same speed as the private benchmark.
