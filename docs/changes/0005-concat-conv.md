# Change 0005: concat folded into `ssm_conv`

## Summary

Each Gated DeltaNet layer of the model makes the input of `ssm_conv` with a `concat` kernel. The new input is the transposed output of the `QKV` projection. The threads of a warp read it with a `stride`. A second kernel then copies the last 3 columns to the cache.

One new kernel replaces `concat`, the copy of the conv state, `ssm_conv` and `SILU`. It reads the conv state and the new input where they are. It does not write the tensor of the `concat`. In the private benchmark, `lat_ms` is 3.02 % lower (95 % interval: -3.88 % to -1.81 %). The answers are the same bit for bit.

## Equivalence class

The equivalence class is `exact`. We selected it on 2026-10-31, before the first run. The new kernel does the same multiply-add operations as `ssm_conv`. It does them in the same sequence for each output value. The new conv state is a copy of 3 values for each row.

The [method](../method.md) gives the tests for `exact`. The answers are the same bit for bit as the answers of the baseline. `test-backend-ops` and `compute-sanitizer` have no error. The section "Correctness" gives the results.

## Results

The baseline is arm `L`, build `lead-30489`. It is the branch `dmk/stack` without this change. The new build is arm `C`. Each arm ran 4 times in 4 blocks, in a random sequence in each block.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| `lat_ms` (all requests) | 235.310 | 228.176 | -3.02 | -3.88 to -1.81 | 4 | 200 | 1,845 |
| `short_ms` | 72.212 | 70.376 | -2.53 | -3.37 to -1.31 | 4 | 200 | 1,845 |
| `medium_ms` | 354.391 | 343.016 | -3.21 | -3.82 to -2.49 | 4 | 200 | 1,845 |
| `long_ms` | 771.001 | 746.710 | -3.15 | -3.71 to -2.27 | 4 | 200 | 1,845 |

Source: [speed-bench.tsv](../../data/2026-10-31-0005-concat-conv/tables/speed-bench.tsv). The clock is the median SM clock of the runs. The power limit and the clock are in [runs.tsv](../../data/2026-10-31-0005-concat-conv/tables/runs.tsv).

The speed of the prompt in `llama bench` also increases. A speed that is more is faster.

| Prompt (tokens) | Before (tokens/s) | After (tokens/s) | Difference (%) | 95 % interval (%) | Runs |
|---:|---:|---:|---:|---|---:|
| 64 | 2,830.591 | 2,925.294 | +3.35 | +2.71 to +4.24 | 4 |
| 256 | 4,306.290 | 4,458.565 | +3.54 | +3.30 to +3.79 | 4 |
| 3,000 | 4,707.411 | 4,889.227 | +3.86 | +3.68 to +4.03 | 4 |

Source: [speed-llama-bench.tsv](../../data/2026-10-31-0005-concat-conv/tables/speed-llama-bench.tsv).

The interval uses 4 block ratios only. The host load was 6 in block 1 and 15 to 16 in block 4. The ratio of `lat_ms` was -1.23 % in block 1 and -3.23 % to -4.10 % in blocks 2 to 4. The data do not show the cause.

## Prediction

We committed the [prediction](../notebook/2026-10-31-0005-concat-conv-prediction.md) on 2026-10-31 at 05:15:13 UTC, before the first run at 06:34 UTC. The [results entry](../notebook/2026-10-31-0005-concat-conv-result.md) gives all the numbers.

| Metric | Predicted gain | Measured gain | Measured / predicted |
|---|---:|---:|---:|
| `lat_ms` | 6.4 ms (2.7 %) | 7.13 ms (3.02 %) | 1.11 |
| `short_ms` | 1.8 ms (2.5 %) | 1.84 ms (2.53 %) | 1.02 |
| `medium_ms` | 9.8 ms (2.7 %) | 11.38 ms (3.21 %) | 1.16 |
| `long_ms` | 21.4 ms (2.7 %) | 24.29 ms (3.15 %) | 1.14 |

The prediction is true for `lat_ms`. The gain is 1.11 times the predicted gain, and the limits were 0.5 and 1.6 times. The 95 % interval does not include 0.

The gain of the prompt speed is larger than the predicted range at all three prompt lengths. The prediction did not include a kernel that is faster than `ssm_conv`. We did not profile the new kernel. We do not know the cause of the difference.

## Correctness

| Test | Threshold | Result |
|---|---|---|
| Answers of 100 decisions, 4 runs of each arm | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true | pass |
| Answers of the new kernel against the `unfused` path, 470 cases | The files of the two paths are the same bit for bit, with the sign of zero | pass |
| `test-backend-ops` `SSM_CONV_CONCAT_CPY` (the new case) | All cases pass, NMSE 1e-7 | pass, 122 of 122 |
| `test-backend-ops` `SSM_CONV`, `SSM_CONV_BIAS_SILU` and `CONCAT` | All cases pass | pass, 60, 120 and 211 cases |
| `compute-sanitizer` `memcheck`, `racecheck`, `initcheck`, `synccheck` on the new case | 0 errors | pass |
| `compute-sanitizer` `memcheck`, `racecheck` and `synccheck` on a model run with 300 tokens | 0 errors | pass |
| `compute-sanitizer` `initcheck` on a model run with 256 tokens | 0 errors | pass |
| `compute-sanitizer` `initcheck` on the model run with 300 tokens | 0 errors | 136,608 errors in `mul_mat_q`. The baseline has 136,628. See the section "Limits". |

Source: [README of the data set](../../data/2026-10-31-0005-concat-conv/README.md), the tables `answers-compare-bench.json`, `ops-tests.tsv`, `sanitizer-model.tsv` and `bitcheck.tsv`.

The SASS of the 6,766 names that are in the two builds is the same. The 6 new names are the kernel `ssm_conv_concat_silu_f32` for 6 values of `d_conv`. The kernel for `d_conv` 4 uses 92 registers and no local memory.

## Reproduce

The fork is the fork of llama.cpp of this project. The branch is `dmk/0005-concat-conv`. The commit is `a67acc087906d2049c52d7ace2a275123656eb09`. It is on top of `6b8b65e6672f07805000d3c9683b65c72cc62303`, which is `dmk/stack` with change 0001.

1. Build the commit with `harness/build/build.sh`. The flags are in the data set.
2. Put the build in `/opt/dmk` of the server image. The image digest is in the data set.
3. Run the schedule from `harness/make_schedule.py` with the two commands in the data set.
4. Run `python3 data/2026-10-31-0005-concat-conv/make_tables.py` to make the tables.

The test of the new case is `test-backend-ops test -o SSM_CONV_CONCAT_CPY`. To run the path with no fusion, set `GGML_CUDA_DISABLE_FUSION`.

## Limits

- The kernel fuses only a graph with one copy of the conv state. A graph with more copies (for rollback) uses the kernels with no fusion.
- The fusion is active only if the rule in `graph_optimize` keeps the two inputs of the `concat` in memory. If an output uses the memory of an input, the program uses the kernels with no fusion. A test has this case.
- The kernel works for `d_conv` of 3, 4, 5, 7, 9 and 15, and for a number of rows (`nr`) that `nr % 128 == 0` accepts. The model uses `d_conv` 4 and `nr` 8,192.
- We changed the sequence of the nodes in `qwen35`, `qwen35moe` and `qwen3next`. Only the model `qwen35` ran. The other two have no test.
- The tensor of the `concat` stays in the memory of the graph. The largest memory that the server used was 4,415 MiB (arm `L`) and 4,413 MiB (arm `C`) in block 1.
- `initcheck` on the model run with 300 tokens finds 136,608 errors in the kernel `mul_mat_q`. The baseline build finds 136,628 errors in the same run. The new kernel has no error in the list. With 256 tokens, the new build has 0 errors.
- `memcheck` finds reads after the end of the input in the kernel `ssm_conv_long_token_f32` that runs with no fusion. This is true when `n_t` is more than 32 and `n_t % 32 != 0`. The new test case does not use such a value of `n_t` for that kernel. We did not change that kernel.
- We did not profile the kernel. The `nsys` run did not start because the folder of the tool has a different layout. The speed numbers are for full requests.
- To go back to the baseline, remove the commit with `git revert`, or use the build `lead-30489`.

## Data

- [Data set](../../data/2026-10-31-0005-concat-conv/README.md)
- [Prediction](../notebook/2026-10-31-0005-concat-conv-prediction.md)
- [Results entry](../notebook/2026-10-31-0005-concat-conv-result.md)
