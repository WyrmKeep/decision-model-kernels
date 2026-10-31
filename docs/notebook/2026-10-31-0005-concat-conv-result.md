# Change 0005: concat folded into `ssm_conv` (result)

Written 2026-10-31, after the measurement window (06:34 to 07:05 UTC).

## Task of this entry

This entry gives the result of change 0005. We committed the [prediction](2026-10-31-0005-concat-conv-prediction.md) on 2026-10-31 at 05:15:13 UTC, before the first run. The [change document](../changes/0005-concat-conv.md) is the text of the pull request. The [README of the data set](../../data/2026-10-31-0005-concat-conv/README.md) gives the raw files, the tables and the conditions.

## Result in short

- The build with the change (arm `C`) is faster than the baseline (arm `L`, build `lead-30489`). `lat_ms` is 3.02 % lower (95 % interval: -3.88 % to -1.81 %). The gain is 7.13 ms.
- The change is -2.53 % for the short request, -3.21 % for the medium request and -3.15 % for the long request.
- The prompt speed of `llama bench` is +3.35 % at 64 tokens, +3.54 % at 256 tokens and +3.86 % at 3,000 tokens. No interval includes 0.
- The answers of the 8 runs are the same bit for bit on 100 decisions. The new kernel and the path with no fusion give the same files in 470 test cases.
- 6,766 names have the same SASS in the two builds. The only difference is 6 new names for the new kernel.
- The prediction is true for `lat_ms`. The gain is 1.11 times the predicted gain. The gain of the prompt speed is larger than the predicted range.
- `initcheck` on the model run with 300 tokens finds errors in the kernel `mul_mat_q`. The baseline has the same errors. With 256 tokens, the new build has 0 errors.
- The host load was 6 at the start of the schedule and 15 to 16 at the end. We did not control it.

## Equivalence class

We selected the equivalence class `exact` before the measurement. The new kernel does the same multiply-add operations as `ssm_conv`, in the same sequence for each output value.

The kernel adds the value 0 at the end of each sum, as `ssm_conv` does when it has no bias. This changes the sign of a sum that is `-0.0`. A first version of the kernel did not do this. A first test found 6,222 different bytes. The first 40 of them were the sign bit of a zero in the output. We added the instruction, and the test then gave the same files.

### Answers

| Test | Result |
|---|---|
| Runs of arm `L` | The 4 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Runs of arm `C` | The 4 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Arm `L` and arm `C` | The answers are the same bit for bit. No decision has a different top answer. |

Source: [README of the data set](../../data/2026-10-31-0005-concat-conv/README.md) and the table `answers-compare-bench.json`. All 8 runs have 94 of 100 decisions correct, 97 of 100 with the top answer of the reference, and `mean_dp` 0.013322.

### Test with random data

A test program ran the fused path and the path with no fusion in two programs. It has 470 cases. They have `d_conv` of 3, 4, 5, 7, 9 and 15, 1 to 2,048 tokens and 1 to 3 sequences. The cases have `transposed` set to true and to false.

The data have 0, `-0.0`, very small values and large values. The program wrote the output and all of the cache. The two files are the same bit for bit. The source is in `code/bitcheck.cpp` of the data set.

### SASS

| Item | Build `L` | Build `C` |
|---|---:|---:|
| Cubins | 146 | 146 |
| Names | 6,766 | 6,772 |
| SASS instructions | 6,156,888 | 6,166,656 |

The 6,766 names in the two builds have the same SASS and the same code of the instructions. The registers, the `stack`, the shared memory and the local memory are the same. The 6 new names are `ssm_conv_concat_silu_f32` for `d_conv` 3, 4, 5, 7, 9 and 15. They use 88, 92, 96, 94, 114 and 112 registers, and no `stack` and no local memory. The kernels `ssm_conv_f32`, `ssm_conv_long_token_f32` and `concat_non_cont` did not change.

Source: the tables `sass-summary.tsv` and `sass-names-changed.tsv`.

### Tests of the operations and `compute-sanitizer` on the test cases

| Test | Result |
|---|---|
| `test-backend-ops` `SSM_CONV_CONCAT_CPY` (new) | 122 of 122 cases pass |
| `test-backend-ops` `SSM_CONV` | 60 of 60 cases pass |
| `test-backend-ops` `SSM_CONV_BIAS_SILU` | 120 of 120 cases pass |
| `test-backend-ops` `CONCAT` | 211 of 211 cases pass |
| `compute-sanitizer` with 4 tools on `SSM_CONV_CONCAT_CPY` | 0 errors and 0 warnings in each tool |

Source: the table `ops-tests.tsv`. The tests compare the CUDA result with the CPU result. The limit is NMSE 1e-7.

## Tests with `compute-sanitizer` on the model

One chunk of 300 tokens ran with each tool. The value 300 has `n_t % 32 != 0`, thus the last tile of the kernel is not full. The perplexity of the chunk is 16.9310 in all runs.

| Tool | Errors | Warnings |
|---|---:|---:|
| `memcheck` | 0 | 0 |
| `racecheck` | 0 | 0 |
| `synccheck` | 0 | 0 |
| `initcheck` | 136,608 | not given |

The tool shows 100 of the 136,608 errors. They are reads of `uninitialized` global memory in the kernel `mul_mat_q`. None of the 100 errors is in the new kernel.

Two control runs used `initcheck`. Arm `L` with 300 tokens has 136,628 errors, and the tool shows 100 reads in `mul_mat_q`. Arm `C` with 256 tokens has 0 errors. Thus the errors are in `mul_mat_q` of the two builds.

## Speed test with the private benchmark

Each block has one run of `L` and one run of `C`. The estimate is the mean of the 4 block ratios. A block ratio is the value of `C` divided by the value of `L`, minus 1. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples.

| Metric | `L` mean (ms) | `C` mean (ms) | `C` against `L` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` | 235.310 | 228.176 | -3.022 | -3.884 | -1.805 |
| `short_ms` | 72.212 | 70.376 | -2.531 | -3.365 | -1.309 |
| `medium_ms` | 354.391 | 343.016 | -3.205 | -3.816 | -2.490 |
| `long_ms` | 771.001 | 746.710 | -3.146 | -3.714 | -2.272 |
| `rps8` (requests/s) | 4.270 | 4.449 | +4.188 | +3.951 | +4.415 |

Source: the table `speed-bench.tsv`. The table `speed-bench-blocks.tsv` gives each block.

| Block | First run | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) | Load before `L` | Load after `C` |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `L` | -1.233 | -0.703 | -2.122 | -1.873 | 6.42 | 10.23 |
| 2 | `L` | -3.231 | -2.710 | -3.067 | -3.435 | 10.23 | 12.50 |
| 3 | `C` | -3.521 | -3.126 | -3.594 | -3.468 | 13.07 | 16.38 |
| 4 | `L` | -4.102 | -3.584 | -4.038 | -3.807 | 16.38 | 15.21 |

Source: the table `speed-bench-blocks.tsv`. The load is the one-minute load average of the host.

The ratio is less than 0 in all 4 blocks for all metrics. The ratio of block 1 is smaller than the ratio of the other blocks. The `lat_ms` of arm `L` was 231.5 ms in block 1 and 236.3 to 236.9 ms in blocks 2 to 4. The `lat_ms` of arm `C` was 227.2 to 228.7 ms in all blocks.

## Speed test with `llama bench`

The test has 4 blocks with seed 50052. Each run measures each prompt length 5 times. A speed in tokens/s that is more is faster.

| Prompt length (tokens) | `L` mean (tokens/s) | `C` mean (tokens/s) | `C` against `L` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,830.591 | 2,925.294 | +3.348 | +2.712 | +4.243 |
| 256 | 4,306.290 | 4,458.565 | +3.537 | +3.304 | +3.785 |
| 3,000 | 4,707.411 | 4,889.227 | +3.863 | +3.680 | +4.032 |

Source: the table `speed-llama-bench.tsv`. In blocks 1 to 3, arm `L` ran first. In block 4, arm `C` ran first.

## Host load and its effect

The host did other work during the schedule. The load average of the host was 6.42 at the start and 14.71 to 16.69 in all runs of `llama bench`. The cause is not in the data.

- The `lat_ms` of arm `L` is 4.8 ms to 5.4 ms more in blocks 2 to 4 than in block 1. The `lat_ms` of arm `C` has a range of 1.5 ms in the 4 blocks.
- The first run of the schedule had the lowest load and the smallest difference. [INFERENCE: the load changes the ratio. No test changed only the load.]
- The bootstrap uses only 4 block ratios. The interval can be too small.

## Prediction against result

| Item | Prediction | Result | Same as the prediction |
|---|---|---|---|
| Equivalence class | `exact` | Same answers bit for bit, 470 test cases with the same files | Yes |
| `lat_ms` gain | 6.4 ms (2.7 %), limits 3.2 to 10.2 ms | 7.13 ms (3.02 %), interval -3.88 % to -1.81 % | Yes. The gain is 1.11 times the prediction. |
| `short_ms` gain | 1.8 ms (2.5 %) | 1.84 ms (2.53 %) | Yes |
| `medium_ms` gain | 9.8 ms (2.7 %) | 11.38 ms (3.21 %) | More by 16 % |
| `long_ms` gain | 21.4 ms (2.7 %) | 24.29 ms (3.15 %) | More by 14 % |
| Prompt speed at 64 tokens | +2.1 % to +2.8 % | +3.35 % | More than the range |
| Prompt speed at 256 tokens | +2.5 % | +3.54 % | More |
| Prompt speed at 3,000 tokens | +2.7 % | +3.86 % | More |
| Tests | `test-backend-ops` and `compute-sanitizer` with 0 errors | 0 errors in all tests. `initcheck` on the model with 300 tokens has errors in `mul_mat_q`, and the baseline has them too. | Yes for the new kernel |

The prediction used 90 % of the `concat` time and it did not include a faster kernel than `ssm_conv`. The cause of the larger gain can be that the new kernel reads with adjacent threads, and `ssm_conv` ran at 67 % of the bandwidth. [INFERENCE: no profile shows this.]

## Limits and next work

- The test has 4 blocks, one benchmark and one GPU.
- We did not profile the kernels. The `nsys` run did not start. A profile can show the time of the new kernel and the cause of the larger gain.
- Only the model `qwen35` ran. The changes to `qwen35moe` and `qwen3next` have no test.
- The kernel `ssm_conv_long_token_f32` of the path with no fusion reads after the end of its input when `n_t` is more than 32 and `n_t % 32 != 0`. `memcheck` shows this in the test case with 40 tokens. The file `ops/memcheck-test-with-40-tokens.txt` of the data set has the report. The new kernel does not do this.
