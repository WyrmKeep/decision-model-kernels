# 0007 Uploads of the scheduler: result

Written 2026-10-31, after the measurement window (05:53 to 06:10 UTC).

## Task of this entry

This entry gives the result of change 0007. The change makes the scheduler of ggml skip the uploads that no `COMPUTE` node reads. A `COMPUTE` node is a node with the flag `GGML_TENSOR_FLAG_COMPUTE`. The [prediction entry](2026-10-31-0007-input-copies-prediction.md) has the cause, the change, and the prediction. Its commit is `ebe92631baed2ce220fefbef0d2d1e2045efca0b`, from 05:32:15 UTC.

The first run of the new build in the test started at 05:53:42 UTC. The [data set](../../data/2026-10-31-0007-input-copies/README.md) has the raw files, the tables, and the conditions. The [change document](../changes/0007-input-copies.md) is the text of the pull request.

## Result in short

- The answers of the private benchmark are the same bit for bit in the 8 runs. The change is `exact`.
- The SASS of `libggml-cuda.so` is the same in the two builds. No `test-backend-ops` case tests this change.
- A pass of the new build has 3 uploads and 8 sync calls less. The upload phase of a pass of 65 to 256 tokens is 0.31 ms shorter.
- The private benchmark gives a gain of 1.0 % for `medium_ms` and 1.3 % for `long_ms`. The interval of each gain does not include 0.
- The gain of `lat_ms` is 0.93 %. The interval is from -2.05 % to +0.12 %, and it includes 0. The gain of `short_ms` is 0.21 %, and its interval includes 0.
- The prompt speed of `llama bench` is 0.8 % to 1.5 % higher. The three intervals do not include 0.
- The prediction is correct for 5 of 7 metrics. It is incorrect by the rule of the prediction entry for `lat_ms` and `short_ms`.

## Change in the pull request

The change is the commit `c3ec5653bda455b6d49976412ab5b7904f55fb13` of the branch `dmk/0007-input-copies`. It changes the file `ggml/src/ggml-backend.cpp` (23 lines added, 3 removed). The scheduler records, for each tensor and each backend, if a `COMPUTE` node reads the copy of the tensor. In `ggml_backend_sched_compute_splits`, the scheduler then writes only the inputs that a `COMPUTE` node reads. It uses the number of these inputs for the sync call before each part of the graph. The prediction entry gives the cause and the lines of the source.

## Equivalence class

The equivalence class is `exact`. We selected it on 2026-10-31, before the first run. The backends skip the nodes without the flag `GGML_TENSOR_FLAG_COMPUTE`. Each `COMPUTE` node gets the same inputs as before. The tests below check this.

| Test | Result |
|---|---|
| Answers of the private benchmark: `answers_compare.py` for 4 files of `L` and 4 files of `I` | `cross_identical_all` is true. `self_repeat` is true for `L` and for `I`. The largest probability difference is 0.0. 0 of 100 decisions have a different top answer. |
| Quality of the private benchmark | The 8 runs have the same numbers: 94 of 100 correct, 97 agree with the reference, mean difference 0.0133, largest difference 0.2087. No request has an error. All guards are true. |
| SASS of `libggml-cuda.so` for `sm_86` | 6,766 of 6,766 names have the same SASS and the same code of the instructions. The two builds have 6,156,888 instructions each. 0 names have different registers or memory. |
| `test-backend-ops` | Not run. The change has no operation and no kernel. The SASS is the same. No case of `test-backend-ops` tests the scheduler. |

The two files `libggml-cuda.so` have the same number of bytes and different SHA-256 values. We did not look for the cause of the different hash. The cubins have the same SASS and the same code of the instructions.

Source: [answers-compare-bench.json](../../data/2026-10-31-0007-input-copies/tables/answers-compare-bench.json), [quality-bench.tsv](../../data/2026-10-31-0007-input-copies/tables/quality-bench.tsv), and [sass-summary.tsv](../../data/2026-10-31-0007-input-copies/tables/sass-summary.tsv).

## Mechanism check

We ran each build with Nsight Systems, with 2 warm-up requests and 10 requests of the type `short2q`. This run is not a speed test. We counted the host to device copies and the sync calls of each pass. We started at the sync call before the first copy. We stopped at the sync call after the last copy.

| Run | Pass | Passes | Uploads | Sync calls | Upload phase (ms) | Uploaded (MB) |
|---|---|---:|---:|---:|---:|---:|
| `L` | 1 to 64 tokens | 7 | 10.0 | 22.0 | 0.175 | 0.41 |
| `I` | 1 to 64 tokens | 7 | 7.0 | 14.0 | 0.100 | 0.14 |
| `L` | 65 to 256 tokens | 27 | 9.6 | 21.3 | 0.538 | 2.82 |
| `I` | 65 to 256 tokens | 27 | 6.6 | 13.3 | 0.224 | 0.98 |

A pass that makes an answer has 10 uploads in arm `L` and 7 in arm `I`. The other passes have 9 and 6. Each pass has 8 sync calls less in the new build. The prediction had 7 sync calls less, and it gave 14 to 18 sync calls in a pass. A pass that makes an answer has 14 sync calls in the upload phase. A pass that makes no answer has 12.

The upload phase of the passes of 65 to 256 tokens is 0.31 ms shorter. The prediction gave 0.37 ms for a pass of 100 tokens (60 µs plus 3.05 µs for each token). The mean time of the 10 requests was 83.0 ms for `L` and 82.1 ms for `I`. One run of each build cannot give an interval.

Source: [mechanism.tsv](../../data/2026-10-31-0007-input-copies/tables/mechanism.tsv) and [mechanism-pass-types.tsv](../../data/2026-10-31-0007-input-copies/tables/mechanism-pass-types.tsv).

## Speed test with the private benchmark

The ratio is the value of arm `I` divided by the value of arm `L`, minus 1. A number less than 0 is a faster build for the metrics in ms. The estimate is the mean of the 4 ratios of the blocks. The interval is the 95 % bootstrap interval with 10,000 samples.

| Metric | Arm `L` | Arm `I` | Difference (%) | 95 % interval (%) |
|---|---:|---:|---:|---|
| `lat_ms` | 234.47 | 232.27 | -0.93 | -2.05 to +0.12 |
| `short_ms` | 71.47 | 71.32 | -0.21 | -1.43 to +1.28 |
| `medium_ms` | 352.92 | 349.32 | -1.01 | -1.90 to -0.04 |
| `long_ms` | 768.79 | 758.99 | -1.27 | -2.00 to -0.55 |
| `rps8` (requests/s) | 4.303 | 4.337 | +0.80 | +0.51 to +1.09 |

The next table gives the difference of `I` from `L` in each block. It also gives the one-minute load average of the host before the run of each arm.

| Block | First arm | Load of `L` | Load of `I` | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `I` | 21.49 | 14.63 | -2.50 | -1.84 | -2.19 | -2.26 |
| 2 | `I` | 17.34 | 19.67 | +0.51 | +1.95 | +0.35 | -0.30 |
| 3 | `L` | 6.47 | 5.09 | -1.05 | -0.74 | -1.20 | -1.28 |
| 4 | `L` | 4.86 | 3.27 | -0.69 | -0.21 | -1.01 | -1.24 |

Source: [speed-bench.tsv](../../data/2026-10-31-0007-input-copies/tables/speed-bench.tsv) and [speed-bench-blocks.tsv](../../data/2026-10-31-0007-input-copies/tables/speed-bench-blocks.tsv).

## Speed test with `llama bench`

The 8 runs used a prompt of 64, 256 and 3,000 tokens. Each run did 5 tests of each prompt. The ratio is the prompt speed of `I` divided by the prompt speed of `L`, minus 1. A number larger than 0 is a faster build.

| Prompt | Arm `L` (tokens/s) | Arm `I` (tokens/s) | Difference (%) | 95 % interval (%) |
|---|---:|---:|---:|---|
| 64 tokens | 2,875.9 | 2,897.8 | +0.76 | +0.59 to +0.93 |
| 256 tokens | 4,330.2 | 4,391.3 | +1.41 | +1.19 to +1.60 |
| 3,000 tokens | 4,731.7 | 4,802.8 | +1.50 | +1.40 to +1.65 |

All 12 block values are larger than 0. The host load was 2.8 to 3.8 in these runs.

Source: [speed-llama-bench.tsv](../../data/2026-10-31-0007-input-copies/tables/speed-llama-bench.tsv) and [speed-llama-bench-blocks.tsv](../../data/2026-10-31-0007-input-copies/tables/speed-llama-bench-blocks.tsv).

## Host load and its effect

### Facts

- The one-minute load average of the host was 14.6 to 21.5 before the runs of block 1. It was 17.3 to 19.7 before the runs of block 2. It was 3.3 to 6.5 in blocks 3 and 4.
- In blocks 1 and 2, arm `I` ran first. The load before the run of `L` was 21.5 in block 1 and 17.3 in block 2.
- The load of the 8 runs of `llama bench` was 2.8 to 3.8.
- The data do not show the cause of the load. Other agents of the project made builds on the host at this time. [INFERENCE: we did not record the programs that ran.]

### Meaning of the facts

- The 4 values of `lat_ms` are -2.50 %, +0.51 %, -1.05 % and -0.69 %. The two blocks with a high load have the largest difference between them. The interval of `lat_ms` is wide because of block 2.
- The two blocks with a low load (3 and 4) give -0.87 % for `lat_ms`, -0.47 % for `short_ms`, -1.11 % for `medium_ms` and -1.26 % for `long_ms`. These means are in the ranges of the prediction. Each mean has 2 values, and we did not declare it before the test.
- The short requests have the largest difference between the blocks (-1.84 % to +1.95 %). The interval of 4 blocks is larger than the gain that we predict for them.
- We did not run a second measurement.

## Prediction against result

| Metric | Predicted (%) | Range (%) | Result (%) | 95 % interval (%) | Prediction correct |
|---|---:|---|---:|---|---|
| `lat_ms` | -1.27 | -0.9 to -1.7 | -0.93 | -2.05 to +0.12 | No: the interval includes 0 |
| `short_ms` | -1.0 | -0.6 to -1.5 | -0.21 | -1.43 to +1.28 | No: not in the range, and the interval includes 0 |
| `medium_ms` | -1.28 | -0.9 to -1.7 | -1.01 | -1.90 to -0.04 | Yes |
| `long_ms` | -1.37 | -0.9 to -1.8 | -1.27 | -2.00 to -0.55 | Yes |
| `llama bench`, 64 tokens | +1.1 | +0.6 to +1.7 | +0.76 | +0.59 to +0.93 | Yes |
| `llama bench`, 256 tokens | +1.4 | +0.9 to +2.0 | +1.41 | +1.19 to +1.60 | Yes |
| `llama bench`, 3,000 tokens | +1.5 | +1.0 to +2.0 | +1.50 | +1.40 to +1.65 | Yes |

The prediction had no value for `rps8`. The result is +0.80 % (+0.51 % to +1.09 %).

- The estimate of `lat_ms` is in the range. The interval includes 0, and the rule of the prediction entry is that the interval must not include 0. We keep the rule. The prediction is incorrect for `lat_ms`.
- The estimate of `short_ms` is 0.4 % from the range, and the interval is 2.7 % wide. The test cannot tell the predicted gain from 0 for short requests.
- All the results that show an incorrect prediction in the prediction entry are false. The gain of `lat_ms` is 0.93 %, which is more than 0.9 %. The answers are the same. A pass of the new build has 7 uploads.
- The mechanism is correct: 3 uploads and 8 sync calls less in each pass, and a shorter upload phase. The prediction had a number of sync calls that was 1 too small.

## Limits and next work

- The measurement has 4 blocks. Two blocks have a high host load. A second measurement at a low load can make the interval of `lat_ms` and `short_ms` smaller. We did not run it.
- The speed test checks the build in the conditions of the project: CUDA and CPU backends, one GPU. The change is in the scheduler that all backends use. A backend that does not skip the nodes without `GGML_TENSOR_FLAG_COMPUTE` can read an input that the scheduler did not write. The backends that we read in the source skip these nodes. [INFERENCE: we did not run a different backend.]
- The other 14 to 18 sync calls of a pass use 0.09 to 0.12 ms in the profiles of phase 3. This is 0.2 % of `short1q`. We did not change them.
- The CPU makes the embedding rows and uploads them (10,240 bytes for each token). This is the largest upload that stays. A GPU kernel that finds the rows can remove it. This is a different change.
