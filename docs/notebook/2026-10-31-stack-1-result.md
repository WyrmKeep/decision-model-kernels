# Test of the stack 1: `dmk/stack` at `fbfd9cb8c` against build `D` (result)

Written 2026-10-31, after the measurement window (08:42 to 09:06 UTC).

## Task of this entry

This entry gives the result of the first test of the stack. The [prediction](2026-10-31-stack-1-prediction.md) is in the commit `fb4dd0c9074896c716c4477d3a24a6965a4cc6ff` from 08:38:18 UTC. The commit `d4a57934cd72bbdf9a7e81c5f6680973ce59351a` from 08:38:30 UTC changed the time in the first line of the prediction, from "about 08:45" to "about 08:35". It changed no other text and no number. The first run started at 08:43:08 UTC. The [README of the data set](../../data/2026-10-31-stack-1/README.md) gives the raw files, the tables and the conditions.

## Result in short

- The answers of the 12 runs of the private benchmark are the same bit for bit on 100 decisions. No decision has a different top answer. The four guards are true in all 12 runs.
- The stack (arm `S`) is faster than build `D` in all 8 metrics. `lat_ms` is 8.04 % lower (95 % interval: -8.50 % to -7.62 %). The gain is 19.34 ms.
- `short_ms` is 6.00 % lower, `medium_ms` is 8.30 % lower and `long_ms` is 8.71 % lower. `rps8` is 8.83 % higher.
- The prompt speed of `llama bench` is +5.76 % at 64 tokens, +7.67 % at 256 tokens and +9.48 % at 3,000 tokens.
- 7 of the 8 results are in their range in the prediction. `short_ms` is better than its range: -6.00 % against -2.4 % to -5.4 %.
- `lat_ms` is in its range (-5.3 % to -8.3 %), but near the end of the range. The low end of its interval (-8.50 %) is 0.2 points more than the end.
- 7 of the 8 results show more gain than the predicted gain. The ratios of the tests of the four changes, multiplied, give less gain than the stack.
- The cause of `short_ms` is not one measured fact. The range of the prediction (±1.5 points) is too small for the four tests of the changes together (±2.1 points). The result is at the edge of that interval. [INFERENCE: this is the largest part of the cause.]
- The stack can go to the next step, a long test on the requests of production. The section "Next step" gives the rule.

## Prediction against result

The change is the value of arm `S` divided by the value of arm `D`, minus 1. For the metrics in ms, a number less than 0 is a faster build. For `rps8` and `llama bench`, a number more than 0 is a faster build.

| Metric | Predicted (%) | Range (%) | Result (%) | 95 % interval (%) | Result minus predicted (points) | In range |
|---|---:|---|---:|---|---:|---|
| `lat_ms` | -6.8 | -5.3 to -8.3 | -8.04 | -8.50 to -7.62 | -1.24 | Yes, near the end of the range |
| `short_ms` | -3.9 | -2.4 to -5.4 | -6.00 | -6.78 to -5.30 | -2.10 | No: better than the range by 0.6 points |
| `medium_ms` | -7.2 | -5.7 to -8.7 | -8.30 | -8.73 to -7.99 | -1.10 | Yes |
| `long_ms` | -8.0 | -6.5 to -9.5 | -8.71 | -9.03 to -8.47 | -0.71 | Yes |
| `rps8` | +8.3 | +6.8 to +9.8 | +8.83 | +8.58 to +9.14 | +0.53 | Yes |
| `llama bench`, 64 tokens | +5.2 | +3.7 to +6.7 | +5.76 | +5.59 to +5.99 | +0.56 | Yes |
| `llama bench`, 256 tokens | +7.5 | +6.0 to +9.0 | +7.67 | +7.57 to +7.73 | +0.17 | Yes |
| `llama bench`, 3,000 tokens | +9.7 | +8.2 to +11.2 | +9.48 | +9.36 to +9.59 | -0.22 | Yes |

Source: the table [prediction-vs-result.tsv](../../data/2026-10-31-stack-1/tables/prediction-vs-result.tsv). The predicted values and the ranges are from the prediction entry.

The high end of the interval of `short_ms` is -5.30 %. This value is in the range. The result of each other metric is in its range.

The prediction entry tells that if the answers are not the same, two changes have an effect on each other. The answers are the same. The next section gives the data.

## Answers and guards

The equivalence class of the stack is `exact`. We selected it before the measurement.

| Test | Result |
|---|---|
| Runs of arm `D` | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Runs of arm `S` | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Arm `D` and arm `S` | The answers are the same bit for bit. The largest probability difference is 0.0. No decision has a different top answer. |
| Quality of the 12 runs | All runs have 94 of 100 decisions correct, 97 of 100 with the top answer of the reference, `mean_dp` 0.013322 and `max_dp` 0.2087. No request has an error. |
| Guards of the 12 runs | The guards `errors`, `correct`, `agree` and `mean_dp` are true in all 12 runs. |
| VRAM guard | The largest `memory.used` is 4,415 MiB for arm `D` and 4,437 MiB for arm `S`. The limit is 6,144 MiB. |

Source: the tables [answers-compare-bench.json](../../data/2026-10-31-stack-1/tables/answers-compare-bench.json), [quality-bench.tsv](../../data/2026-10-31-stack-1/tables/quality-bench.tsv) and [guards-bench.tsv](../../data/2026-10-31-stack-1/tables/guards-bench.tsv).

The stack gives the same answers as build `D`. Thus the changes of the stack have no effect on each other that changes an answer. The 3 decisions with a top answer that is not the top answer of the reference are the same in all 12 runs.

This data set has no run of `compute-sanitizer` and no run of `test-backend-ops` on the build of the stack. Each change has these tests in the data set of the change.

## Speed test with the private benchmark

Each block has one run of `D` and one run of `S`. The estimate is the mean of the 6 block ratios. A block ratio is the value of `S` divided by the value of `D`, minus 1. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples.

| Metric | `D` mean | `S` mean | `S` against `D` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` (ms) | 240.527 | 221.186 | -8.039 | -8.503 | -7.621 |
| `short_ms` (ms) | 73.063 | 68.676 | -6.002 | -6.780 | -5.298 |
| `medium_ms` (ms) | 362.092 | 332.039 | -8.298 | -8.725 | -7.990 |
| `long_ms` (ms) | 789.513 | 720.730 | -8.710 | -9.032 | -8.466 |
| `rps8` (requests/s) | 4.197 | 4.568 | +8.832 | +8.575 | +9.137 |

Source: the table [speed-bench.tsv](../../data/2026-10-31-stack-1/tables/speed-bench.tsv).

| Block | First run | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) | `rps8` (%) | Load before `D` | Load before `S` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `S` | -8.945 | -7.690 | -9.252 | -9.413 | +8.749 | 6.95 | 1.22 |
| 2 | `D` | -7.256 | -5.046 | -7.971 | -8.476 | +9.036 | 15.37 | 15.94 |
| 3 | `D` | -8.138 | -5.178 | -8.189 | -8.707 | +9.505 | 17.57 | 11.39 |
| 4 | `S` | -8.464 | -6.761 | -8.485 | -8.861 | +8.755 | 10.79 | 11.14 |
| 5 | `D` | -7.683 | -5.671 | -7.967 | -8.384 | +8.428 | 18.08 | 5.36 |
| 6 | `S` | -7.750 | -5.664 | -7.922 | -8.422 | +8.519 | 2.15 | 3.11 |

Source: the table [speed-bench-blocks.tsv](../../data/2026-10-31-stack-1/tables/speed-bench-blocks.tsv). The load is the one-minute load average of the host before the run.

The change has the same sign in all 6 blocks for all 5 metrics. The `lat_ms` of arm `D` was 237.2 to 242.5 ms in all blocks. The `lat_ms` of arm `S` was 218.9 to 224.6 ms. The gain of the stack in ms is more than the sum of the gains in the tests of the four changes.

| Metric | Sum of the gains of the four changes (ms) | Gain of the stack (ms) | Stack minus sum (ms) |
|---|---:|---:|---:|
| `lat_ms` | 16.516 | 19.341 | +2.825 |
| `short_ms` | 2.839 | 4.387 | +1.548 |
| `medium_ms` | 26.195 | 30.053 | +3.858 |
| `long_ms` | 63.625 | 68.783 | +5.158 |

Source: the table [gain-ms.tsv](../../data/2026-10-31-stack-1/tables/gain-ms.tsv). Each gain is the mean of the baseline arm minus the mean of the arm with the change, in the test of the change. The four tests had different baseline arms and a different host load.

## Speed test with `llama bench`

The test has 4 blocks with seed 61002. Each run measures each prompt length 5 times. A speed in tokens/s that is more is faster.

| Prompt length (tokens) | `D` mean (tokens/s) | `S` mean (tokens/s) | `S` against `D` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,842.358 | 3,006.013 | +5.758 | +5.593 | +5.994 |
| 256 | 4,230.977 | 4,555.410 | +7.668 | +7.565 | +7.731 |
| 3,000 | 4,615.726 | 5,053.290 | +9.480 | +9.359 | +9.593 |

Source: the table [speed-llama-bench.tsv](../../data/2026-10-31-stack-1/tables/speed-llama-bench.tsv). In blocks 1 and 3, arm `D` ran first. In blocks 2 and 4, arm `S` ran first.

The ratio is more than 0 in all 4 blocks for all 3 prompt lengths. The load of the host was 1.71 to 1.97 in these 8 runs. The three results are in their ranges.

## Host load and its effect

### Facts

- The one-minute load average of the host was 1.22 to 18.08 in the 12 runs of the private benchmark. It was 1.71 to 1.97 in the 8 runs of `llama bench`.
- The load in block 1 starts at 1.22 and ends at 15.37. Blocks 2 to 4 have a load of 10.79 to 18.08. The load in block 5 starts at 18.08 and ends at 3.11. The load in block 6 is 2.03 to 3.11.
- The data do not show the cause of the load. Other work of the project ran on the host at this time. [INFERENCE: we did not record the programs that ran.]
- In block 1, the load is 1.22 before `S` and 6.95 before `D`. This is the largest difference of load between the two arms. The gain in block 1 is the largest for all 4 metrics in ms.
- The 5 blocks 2 to 6 give -7.86 % for `lat_ms`, -5.66 % for `short_ms`, -8.11 % for `medium_ms` and -8.57 % for `long_ms`.
- In the 3 blocks where `S` ran first, `short_ms` is -6.71 %. In the 3 blocks where `D` ran first, it is -5.30 %. For `lat_ms`, the two values are -8.39 % and -7.69 %.

### Meaning of the facts

- The load can change the ratio. [INFERENCE: no test changed only the load.] Block 1 has the largest gain and the largest difference of load.
- The bootstrap uses only 6 block ratios. The interval can be too small.
- A random seed gives the sequence of the two arms in a block. The mean of 6 blocks has the two sequences in the same number. The two groups above have 3 blocks each, and we did not declare them before the test.
- The test of `llama bench` ran at a low load. Its intervals are the smallest (0.1 to 0.2 points on each side).

## Cause of the result of `short_ms`

The prediction entry tells: if a result is not in its range, the result entry gives the cause. This also applies to a result that is better than the range. We did not measure the cause. This section gives the facts. It checks three possible causes with the documents of the project. Each possible cause is an [INFERENCE].

### Facts

- The result is -6.00 % (interval: -6.78 % to -5.30 %). The predicted change is -3.9 % and the range is -2.4 % to -5.4 %. The result is 2.1 points from the predicted change and 0.6 points from the range.
- The gain is 4.39 ms. The sum of the gains of the four tests of the changes is 2.84 ms. The gain of the stack is 1.55 ms more.
- 4 of the 6 blocks have more gain than the end of the range (-7.69 %, -6.76 %, -5.67 % and -5.66 %). 2 blocks are in the range (-5.05 % and -5.18 %).
- The other 4 metrics of the private benchmark also give more gain than the ratios of the changes, multiplied. The differences are 1.26 points for `lat_ms`, 1.15 points for `medium_ms`, 0.77 points for `long_ms` and 0.54 points for `rps8`.
- The difference of `short_ms` is the largest, in points and as a part of the predicted gain.

### Possible cause 1: the uploads of the changes 0004 and 0007

Each of the changes 0004 and 0007 removes a part of the uploads before the first kernel of a short pass. The possible cause is that the two changes together remove all of them. The gains then add to more than the ratios, multiplied. We check this with the documents.

- The result entry of change 0007 shows that a pass of the new build has 7 uploads, not 0. The upload phase is 0.175 ms to 0.100 ms for a pass of 1 to 64 tokens. It is 0.538 ms to 0.224 ms for a pass of 65 to 256 tokens.
- The result entry of change 0004 tells that the new input `info` has the same number of copies as the attention mask had. Thus change 0004 does not decrease the number of uploads.
- The upload time of a short request in phase 4 is 0.71 ms for `short1q` and 1.08 ms for `short2q`. The result entry of change 0004 gives these values. The mean for the 12 cases of `short1q` and the 20 cases of `short2q` is 0.94 ms.
- The gain of the stack is 1.55 ms more than the sum of the gains. A removal of all upload time gives 0.94 ms at most. The stack does not remove all uploads. [INFERENCE: the upload time of phase 4 is the largest gain that a change of the uploads gives.] Thus the uploads can cause a part of the 1.55 ms at most.
- In the tests of the changes, change 0007 gave 0.16 ms for `short_ms` and change 0004 gave -0.06 ms. The prediction of change 0007 had 0.51 ms for `short1q` and 0.85 ms for `short2q`.

The possible cause is not correct as the text gives it, because uploads stay in the new build. It can cause a part of the difference. The data of this test do not show the part. A test of the stack with 0004 and 0007 against the stack without them can show it.

### Possible cause 2: the range of the prediction was too small

The prediction entry gives a range of ±1.5 points. The cause that it gives is that each test of a change had 4 blocks and a different host load. We compare the range with the interval of each test of a change. The table gives the `short_ms` of the four tests.

| Change | `short_ms` (%) | 95 % interval (%) | One side of the interval (points) |
|---|---:|---|---:|
| 0001, against build `D` | -1.23 | -2.14 to -0.34 | 0.90 |
| 0007, against the stack with 0001 | -0.21 | -1.43 to +1.28 | 1.35 |
| 0004, against the stack with 0001 | +0.08 | -0.72 to +0.88 | 0.80 |
| 0005, against the stack with 0001 | -2.53 | -3.37 to -1.31 | 1.03 |

Source: the table [change-intervals.tsv](../../data/2026-10-31-stack-1/tables/change-intervals.tsv). It has the same table for all 8 metrics, from the data sets of the changes.

- The four tests together give a total interval of ±2.08 points. The value is `sqrt(h1^2 + h2^2 + h3^2 + h4^2)`, where `h1` to `h4` are the four values in the last column. The method thinks that the four tests have no effect on each other. The range of ±1.5 points is too small for this value.
- The total interval is -5.94 % to -1.77 %, around the ratios, multiplied, of -3.86 %. The result of -6.00 % is 0.06 points more than the low end of the total interval.
- For `lat_ms`, the same method gives ±1.65 points. The result is in the interval of -8.43 % to -5.13 %. For `medium_ms`, `long_ms` and `rps8`, the method gives ±1.21, ±1.08 and ±0.56 points. The range of ±1.5 points was wide for these three metrics, and their results are in their ranges. The table `stack-estimate.tsv` has the values for all 8 metrics.
- The test of change 0007 has the widest interval for `short_ms`. Its result (-0.21 %) is 0.8 points less than its predicted gain (-1.0 %, range -0.6 % to -1.5 %). The prediction came from the profiles of the uploads. In the result entry of change 0007, blocks 1 and 2 have a host load of 14.6 to 21.5.
- If the gain of change 0007 for `short_ms` is its predicted -1.0 %, the ratios, multiplied, give -4.6 %. The result of -6.00 % is then 1.4 points from this value. This is in the range of ±1.5 points. [INFERENCE: this uses the predicted value as the true value. We did not measure it.]
- Each interval is a bootstrap of 4 blocks. The [result of change 0005](2026-10-31-0005-concat-conv-result.md) tells that the interval can be too small. [INFERENCE: the true interval is more than ±2.08 points.]

The range was too small for `short_ms`. This is a fact about the documents. The result is 0.06 points more than the end of the total interval. Thus this cause does not give all of the difference.

### Possible cause 3: the host load

- Blocks 5 and 6 have the lowest load. Their `short_ms` is -5.67 % and -5.66 %. Each value has more gain than the end of the range (-5.4 %).
- Without block 1, `short_ms` is -5.66 %. This value also has more gain than the end of the range.
- The host load can change the ratio. The data do not show a load that gives the result. [INFERENCE: no test changed only the load.]

### Gain for each request type

We did not measure the gain of the stack for each request type (`short1q`, `short2q`, `medium` and `long`) in this test. The benchmark gives `short_ms` for the two short types together. Thus we cannot tell which short type has the larger gain.

### Finding

The cause of the result is not one measured fact. The documents show that the range of the prediction was too small for `short_ms`. The test of change 0007 has the widest interval. The uploads can cause a part of the difference. The host load does not give the result. All three are [INFERENCE].

The result is better than the prediction. It does not change the decision about the stack. For the next prediction of a stack, use the interval of the table `stack-estimate.tsv` and not the range of ±1.5 points.

## Next step

The rule of the project for a build has three steps before production. The first step is the tests of correctness. The second step is the test with two arms. The third step is a long test on the requests of production.

The tests of correctness of each change are in the data sets of the changes. This entry is the test with two arms for the full stack. The result is a faster build with the same answers bit for bit. Thus the stack can go to the long test.

## Limits and next work

- The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`, one benchmark and one GPU.
- The host load was different in the 6 blocks. The load in block 6 is the lowest, and the change in block 6 has the same sign as in the other blocks.
- The ratios of the changes, multiplied, give less gain than the result for 7 of the 8 metrics, by 0.13 to 2.15 points. One test of the stack cannot show if the method has a bias. The next tests of the stack can show it.
- The possible cause 1 has a test: the stack without the changes 0004 and 0007 against the full stack.
- This data set has no profile of the stack. A profile can show the time that each change removes in the stack.
