# Test of the stack 2: `dmk/stack` at `419b8f612` against the image in production (result)

Written 2026-10-31, after the measurement window (20:43 to 21:06 UTC).

## Task of this entry

This entry gives the result of the second test of the stack. The [prediction](2026-10-31-stack-2-prediction.md) is in the commit `346f9833f613d48ba37937bc54c3b4b4057636e9` from 20:43:36 UTC. The first run started at 20:43:55 UTC. The [README of the data set](../../data/2026-10-31-stack-2/README.md) gives the raw files, the tables and the conditions.

The first line of the prediction gives the time "about 20:45 UTC". This time is after the commit and after the first run. We did not change the file. The time of the commit is the true time.

The test compares the build of the stack (arm `S2`, `dmk/stack` at `419b8f612`) with the image `server-cuda-b12004` (arm `P`). The image is the image that the server in production uses. The stack has the changes 0001, 0007, 0002 (test cases only), 0004, 0005 and 0003.

## Result in short

- The answers of the 12 runs of the private benchmark are the same bit for bit on 100 decisions. No decision has a different top answer. The four guards are true in all 12 runs.
- The stack (arm `S2`) is faster than the image (arm `P`) in all 8 metrics. `lat_ms` is 9.15 % lower (95 % interval: -9.28 % to -8.98 %). The gain is 21.68 ms.
- `short_ms` is 9.61 % lower, `medium_ms` is 9.26 % lower and `long_ms` is 8.96 % lower. `rps8` is 10.34 % higher.
- The prompt speed of `llama bench` is +5.88 % at 64 tokens, +7.27 % at 256 tokens and +9.39 % at 3,000 tokens.
- All 8 results are in their range in the prediction. Each result has less gain than the predicted change, by 0.1 to 0.7 points.
- The result is in the interval of the three tests together for 7 of the 8 metrics. `pp256_ts` is not in its interval (+7.27 % against +7.57 % to +7.79 %). It is in the range of the prediction. The section "Result against the estimate of the prediction" gives the numbers.
- `short_ms` is 9.61 % lower in this test. It was 6.00 % lower in the first test of the stack. The test of change 0003 on the stack gives 3.73 % for `short_ms`. The difference of the two tests is 3.84 %, near this value. [INFERENCE: the two tests have different arms and a different host load. The data do not show which part comes from change 0003.]
- The next step is the soak test of 24 hours.

## Prediction against result

The change is the value of arm `S2` divided by the value of arm `P`, minus 1. For the metrics in ms, a number less than 0 is a faster build. For `rps8` and `llama bench`, a number more than 0 is a faster build.

| Metric | Predicted (%) | Range (%) | Result (%) | 95 % interval (%) | Result minus predicted (points) | In range |
|---|---:|---|---:|---|---:|---|
| `lat_ms` | -9.8 | -11.8 to -7.8 | -9.145 | -9.280 to -8.978 | +0.655 | Yes |
| `short_ms` | -9.8 | -11.8 to -7.8 | -9.611 | -9.743 to -9.462 | +0.189 | Yes |
| `medium_ms` | -9.8 | -11.8 to -7.8 | -9.264 | -9.386 to -9.115 | +0.536 | Yes |
| `long_ms` | -9.4 | -11.4 to -7.4 | -8.963 | -9.077 to -8.799 | +0.437 | Yes |
| `rps8` | +10.7 | +8.7 to +12.7 | +10.340 | +10.209 to +10.465 | -0.360 | Yes |
| `llama bench`, 64 tokens | +6.0 | +4.0 to +8.0 | +5.882 | +5.770 to +5.999 | -0.118 | Yes |
| `llama bench`, 256 tokens | +7.7 | +5.7 to +9.7 | +7.266 | +7.177 to +7.354 | -0.434 | Yes |
| `llama bench`, 3,000 tokens | +9.5 | +7.5 to +11.5 | +9.388 | +9.206 to +9.503 | -0.112 | Yes |

Source: the table [prediction-vs-result.tsv](../../data/2026-10-31-stack-2/tables/prediction-vs-result.tsv). The predicted values and the ranges are from the prediction entry.

The interval of each result is in the range of the prediction. The result of each metric is on the side of less gain. The largest difference is 0.655 points (`lat_ms`).

The prediction entry tells that if the answers are not the same, two changes have an effect on each other. The answers are the same. The next section gives the data.

## Answers and guards

The equivalence class of the stack is `exact`. We selected it before the measurement.

| Test | Result |
|---|---|
| Runs of arm `P` | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Runs of arm `S2` | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Arm `P` and arm `S2` | The answers are the same bit for bit. The largest probability difference is 0.0. No decision has a different top answer. |
| Quality of the 12 runs | All runs have 94 of 100 decisions correct, 97 of 100 with the top answer of the reference, `mean_dp` 0.013322 and `max_dp` 0.2087. No request has an error. |
| Guards of the 12 runs | The guards `errors`, `correct`, `agree` and `mean_dp` are true in all 12 runs. |
| VRAM guard | The largest `memory.used` is 4,415 MiB for arm `P` and 4,425 MiB for arm `S2`. The limit is 6,144 MiB. |

Source: the tables [answers-compare-bench.json](../../data/2026-10-31-stack-2/tables/answers-compare-bench.json), [quality-bench.tsv](../../data/2026-10-31-stack-2/tables/quality-bench.tsv) and [guards-bench.tsv](../../data/2026-10-31-stack-2/tables/guards-bench.tsv).

The stack gives the same answers as the image. Thus the changes of the stack have no effect on each other that changes an answer. The 3 decisions with a different top answer than the reference are the same in all 12 runs. They are `triage00.needs_action`, `style03.secret` and `long07-none.worst`.

The tests of change 0003 are in the [data set of change 0003](../../data/2026-10-31-0003-answer-head/README.md). This data set has no run of `compute-sanitizer` and no run of `test-backend-ops` on the build of the stack.

## Speed test with the private benchmark

Each block has one run of `P` and one run of `S2`. The estimate is the mean of the 6 block ratios. A block ratio is the value of `S2` divided by the value of `P`, minus 1. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples.

| Metric | `P` mean | `S2` mean | `S2` against `P` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` (ms) | 237.015 | 215.339 | -9.145 | -9.280 | -8.978 |
| `short_ms` (ms) | 71.264 | 64.414 | -9.611 | -9.743 | -9.462 |
| `medium_ms` (ms) | 357.505 | 324.386 | -9.264 | -9.386 | -9.115 |
| `long_ms` (ms) | 781.715 | 711.649 | -8.963 | -9.077 | -8.799 |
| `rps8` (requests/s) | 4.232 | 4.670 | +10.340 | +10.209 | +10.465 |

Source: the table [speed-bench.tsv](../../data/2026-10-31-stack-2/tables/speed-bench.tsv).

| Block | First run | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) | `rps8` (%) | Load before `P` | Load before `S2` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `P` | -8.788 | -9.284 | -8.949 | -8.581 | +10.085 | 1.14 | 1.53 |
| 2 | `S2` | -9.275 | -9.776 | -9.365 | -9.061 | +10.453 | 1.96 | 2.34 |
| 3 | `S2` | -9.343 | -9.628 | -9.426 | -9.082 | +10.398 | 2.18 | 2.42 |
| 4 | `S2` | -9.010 | -9.482 | -9.136 | -8.911 | +10.178 | 1.76 | 1.68 |
| 5 | `S2` | -9.237 | -9.778 | -9.396 | -9.098 | +10.557 | 2.40 | 1.66 |
| 6 | `P` | -9.217 | -9.720 | -9.310 | -9.044 | +10.369 | 2.58 | 2.64 |

Source: the table [speed-bench-blocks.tsv](../../data/2026-10-31-stack-2/tables/speed-bench-blocks.tsv). The load is the one-minute load average of the host before the run.

The change has the same sign in all 6 blocks for all 5 metrics. The `lat_ms` of arm `P` was 235.958 to 237.381 ms in all blocks. The `lat_ms` of arm `S2` was 214.946 to 215.932 ms. The 6 blocks are near each other. For `short_ms`, they are 0.49 points apart (-9.78 % to -9.28 %). In the first test of the stack, they were 2.64 points apart.

Source: the table [runs.tsv](../../data/2026-10-31-stack-2/tables/runs.tsv).

## Speed test with `llama bench`

The test has 4 blocks with seed 62002. Each run measures each prompt length 5 times. A speed in tokens/s that is more is faster.

| Prompt length (tokens) | `P` mean (tokens/s) | `S2` mean (tokens/s) | `S2` against `P` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,839.148 | 3,006.143 | +5.882 | +5.770 | +5.999 |
| 256 | 4,235.429 | 4,543.163 | +7.266 | +7.177 | +7.354 |
| 3,000 | 4,610.346 | 5,043.166 | +9.388 | +9.206 | +9.503 |

Source: the table [speed-llama-bench.tsv](../../data/2026-10-31-stack-2/tables/speed-llama-bench.tsv). In blocks 1, 2 and 3, arm `S2` ran first. In block 4, arm `P` ran first.

The ratio is more than 0 in all 4 blocks for all 3 prompt lengths (table [speed-llama-bench-blocks.tsv](../../data/2026-10-31-stack-2/tables/speed-llama-bench-blocks.tsv)). The load of the host was 1.86 to 2.19 in these 8 runs. The three results are in their ranges.

## Host load and its effect

### Facts

- The one-minute load average of the host was 1.14 to 2.64 in the 12 runs of the private benchmark. It was 1.86 to 2.19 in the 8 runs of `llama bench`. The load was lower than in the first test of the stack (1.22 to 18.08).
- In a block, the load before the run of `P` and the load before the run of `S2` are 0.06 to 0.74 apart. The largest difference is in block 5.
- In block 1, the change is the smallest for all 5 metrics of the private benchmark. In block 1, arm `P` ran first, and it was the first run of the measurement window. The `lat_ms` of arm `P` in block 1 is 235.958 ms. The mean of the 5 other runs of arm `P` is 237.226 ms. The `lat_ms` of arm `S2` in block 1 is 215.221 ms. The mean of its 5 other runs is 215.362 ms.
- The data do not show the cause of the lower value of arm `P` in block 1.
- The mean of the changes of the 2 blocks where `P` ran first is -9.00 % for `lat_ms`. The mean of the 4 blocks where `S2` ran first is -9.22 %.

### Meaning of the facts

- The load was low, and the two arms had almost the same load in each block. [INFERENCE: a difference of load between the arms did not cause the result.]
- The first test of the stack had a load of 1.22 to 18.08, and more gain than the estimate of its four tests of changes. Its block 6 had the lowest load (2.03 to 3.11), and `lat_ms` was -7.75 % in this block against -8.04 % for the 6 blocks. Its block 1 had the largest gain and the largest difference of load between the arms. In this test, the load was low, and the result has less gain than the estimate of the three tests. [INFERENCE: the load can change the ratio of the two arms. No test changed only the load.]
- Arm `P` ran first in 2 blocks, and arm `S2` ran first in 4 blocks. The seed set the sequence. The data do not show an effect of the sequence.

## Result against the estimate of the prediction

The prediction entry multiplies the ratios of three tests. These tests are phase 1 (build `D` against the image), the first test of the stack (the stack at `fbfd9cb8c` against build `D`) and the test of change 0003 on the stack at `fbfd9cb8c`. For `llama bench`, the entry uses the ratio 1 for change 0003.

| Metric | Phase 1 (%) | First test of the stack (%) | Change 0003 (%) | Estimate (%) | Interval of the estimate (%) | Result (%) | Result minus estimate (points) | In the interval |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `lat_ms` | -0.397 | -8.039 | -1.481 | -9.761 | -10.586 to -8.936 | -9.145 | +0.616 | Yes |
| `short_ms` | -0.314 | -6.002 | -3.725 | -9.788 | -10.989 to -8.586 | -9.611 | +0.177 | Yes |
| `medium_ms` | -0.276 | -8.298 | -1.332 | -9.769 | -10.441 to -9.098 | -9.264 | +0.505 | Yes |
| `long_ms` | -0.208 | -8.710 | -0.576 | -9.425 | -9.916 to -8.933 | -8.963 | +0.462 | Yes |
| `rps8` | +0.051 | +8.832 | +1.630 | +10.662 | +10.292 to +11.033 | +10.340 | -0.322 | Yes |
| `llama bench`, 64 tokens | +0.238 | +5.758 | Not used | +6.010 | +5.625 to +6.394 | +5.882 | -0.128 | Yes |
| `llama bench`, 256 tokens | +0.010 | +7.668 | Not used | +7.679 | +7.567 to +7.791 | +7.266 | -0.413 | No |
| `llama bench`, 3,000 tokens | +0.052 | +9.480 | Not used | +9.537 | +9.353 to +9.721 | +9.388 | -0.149 | Yes |

Source: the table [stack-estimate.tsv](../../data/2026-10-31-stack-2/tables/stack-estimate.tsv). The interval is the estimate, minus and plus the value of `sqrt(h1^2 + h2^2 + h3^2)`. The values `h1` to `h3` are one side of the interval of each of the three tests. For `llama bench`, the value has the first two terms only.

- All 8 results show less gain than the estimate. The difference is 0.18 to 0.62 points for the metrics in ms, 0.32 points for `rps8` and 0.13 to 0.41 points for `llama bench`.
- In the first test of the stack, 7 of 8 results showed more gain than the estimate of its four tests of changes. In this test, all 8 results show less gain than the estimate of the three tests.
- `pp256_ts` is 0.30 points below the low end of its interval. The interval is small (±0.11 points). It has two intervals of `llama bench`: phase 1 and the first test of the stack. Each one is ±0.08 points. The prediction used a range of ±2.0 points. The result is in this range.
- The [result of change 0003](2026-10-31-0003-answer-head-result.md) shows -0.60 % for 256 tokens in `llama bench` (arm `A3` against arm `L`, 4 blocks). The change does not touch this path. The prediction used the ratio 1. [INFERENCE: a part of the difference of `pp256_ts` can come from a true effect of the change or from the code layout of the build. This test has no run that shows which one.]

## Difference from the first test of the stack and from change 0003

### Facts

The first test of the stack compared the stack at `fbfd9cb8c` with build `D`. This test compares the stack at `419b8f612` with the image. The stack at `419b8f612` has change 0003 on top of the stack at `fbfd9cb8c`. The [result of change 0003](2026-10-31-0003-answer-head-result.md) has a test on the stack at `fbfd9cb8c` (arm `A3R`, 4 blocks).

| Metric | First test of the stack (%) | This test (%) | This test divided by the first test (%) | Change 0003 in its test (%) | Interval of change 0003 (%) | In the interval |
|---|---:|---:|---:|---:|---|---|
| `lat_ms` | -8.039 | -9.145 | -1.203 | -1.481 | -2.032 to -0.808 | Yes |
| `short_ms` | -6.002 | -9.611 | -3.839 | -3.725 | -4.597 to -2.827 | Yes |
| `medium_ms` | -8.298 | -9.264 | -1.053 | -1.332 | -1.819 to -0.822 | Yes |
| `long_ms` | -8.710 | -8.963 | -0.277 | -0.576 | -0.918 to -0.234 | Yes |
| `rps8` | +8.832 | +10.340 | +1.386 | +1.630 | +1.397 to +1.862 | No: 0.011 points below the low end |
| `llama bench`, 64 tokens | +5.758 | +5.882 | +0.117 | Not used | | |
| `llama bench`, 256 tokens | +7.668 | +7.266 | -0.373 | Not used | | |
| `llama bench`, 3,000 tokens | +9.480 | +9.388 | -0.084 | Not used | | |

Source: the table [comparison-stack-1.tsv](../../data/2026-10-31-stack-2/tables/comparison-stack-1.tsv). The value "This test divided by the first test" is (1 plus the change of this test) divided by (1 plus the change of the first test), minus 1. The two tests have different arms. Thus the value has the effect of change 0003 and the difference of build `D` and the image.

The gain in ms of the stack, and the gains of the three tests:

| Metric | Phase 1 (ms) | First test of the stack (ms) | Change 0003 (ms) | Sum (ms) | This test (ms) | This test minus sum (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `lat_ms` | 0.954 | 19.341 | 3.298 | 23.593 | 21.676 | -1.917 |
| `short_ms` | 0.230 | 4.387 | 2.584 | 7.201 | 6.850 | -0.351 |
| `medium_ms` | 1.002 | 30.053 | 4.448 | 35.503 | 33.119 | -2.384 |
| `long_ms` | 1.641 | 68.783 | 4.175 | 74.599 | 70.066 | -4.533 |

Source: the table [gain-ms.tsv](../../data/2026-10-31-stack-2/tables/gain-ms.tsv). Each gain is the mean of arm A minus the mean of arm B in the test. The three tests had different baseline arms and a different host load.

- `short_ms` is 9.611 % lower in this test and 6.002 % lower in the first test of the stack. The difference is 3.6 points. The gain in ms is 6.850 ms and 4.387 ms. The difference is 2.463 ms.
- The test of change 0003 on the stack gives -3.725 % (interval: -4.597 % to -2.827 %) and 2.584 ms for `short_ms`. This is the largest effect of change 0003 in %, for the four metrics in ms.
- The ratio of this test to the first test is -3.839 % for `short_ms`. It is in the interval of the test of change 0003. The ratio is in the interval of the test of change 0003 for `lat_ms`, `medium_ms` and `long_ms`. It is not in the interval for `rps8`, by 0.011 points.
- The estimate of the three tests gives -9.788 % for `short_ms`. The result is 0.177 points from it. In the first test of the stack, the result of `short_ms` was 2.1 points from the estimate of the four tests of changes.
- The mean of arm `P` in this test is 1.46 % lower than the mean of arm `D` in the first test for `lat_ms` (237.015 ms and 240.527 ms). For `short_ms`, it is 2.46 % lower (71.264 ms and 73.063 ms). Phase 1 gives build `D` 0.40 % lower than the image for `lat_ms` and 0.31 % lower for `short_ms`. Here, arm `P` has the lower mean.
- The mean of arm `S2` is 2.64 % lower than the mean of arm `S` in the first test for `lat_ms`. For `short_ms`, it is 6.21 % lower (64.414 ms and 68.676 ms).

### Meaning of the facts

- The result of `short_ms` is near the estimate of the three tests. Change 0003 is not in the stack of the first test of the stack. Its test gives -3.725 % for `short_ms`, and the ratio of this test to the first test gives -3.839 %. [INFERENCE: the gain of change 0003 is the largest part of the difference between the two tests for `short_ms`. The tests have different arms and a different host load. They do not show which part comes from change 0003.]
- The change 0003 has a large effect on `short_ms` and a small effect on `long_ms`. The [result of change 0003](2026-10-31-0003-answer-head-result.md) tells that a short request has 1 or 2 passes with outputs and no prefix of 3,000 tokens. [INFERENCE: the gain of the head is a larger part of a short request than of a long request. We did not measure this part in this test.]
- In the first test of the stack, `short_ms` was better than the range of its prediction. The cause was not one measured fact. This test does not measure that result again, because its arm `S` is not an arm of this test. It gives no new cause. The result of this test for `short_ms` is in the range, and it is near the estimate.
- The means of arm `P` and arm `S2` are lower than the means of arm `D` and arm `S` in the first test. The load of the host was lower in this test (1.14 to 2.64 against 1.22 to 18.08). [INFERENCE: the lower load can cause a part of the difference of the means. No test changed only the load.]
- The results of the 5 metrics of the private benchmark show less gain than the estimate of the three tests, by 0.18 to 0.62 points. The intervals of the tests are the intervals of 4 to 6 blocks. The result entry of the first test of the stack tells that such an interval can be too small.

## Next step

The rule of the project for a build has three steps before production. The first step is the tests of correctness. The second step is the test with two arms. The third step is a long test on the requests of production. The [method](../method.md) gives the long test: 24 h, with no GPU fault, no error of `compute-sanitizer`, and the same answers all the time.

The tests of correctness of each change are in the data sets of the changes. This entry is the test with two arms for the stack at `419b8f612` against the image in production. The result is a faster build with the same answers bit for bit. The next step is the soak test of 24 hours.

## Limits and next work

- The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`, one benchmark and one GPU.
- The load of the host was low in all runs. The first test of the stack had a larger load. The load in production can be different.
- The answers are the same on 100 decisions of one benchmark. The tests do not show other inputs. The model gives no text. A request for text gets `HTTP 501` (see the result of change 0003).
- This test has no run of the client of the request types. The gain for each request type of the stack at `419b8f612` is not in the data.
- The three tests of the estimate have 4 to 6 blocks each. Their intervals are small, and the interval of the estimate can be too small. The `pp256_ts` result is not in the interval of the estimate.
- This data set has no profile of the stack. A profile can show the time that each change removes in the stack, and the time that change 0003 removes for each request type.

The next step is the soak test of 24 hours. It can show an error that 12 short runs cannot show.
