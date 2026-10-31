# Change 0003: output head for the answer tokens only

## Summary

The server gets the answer of a decision from the logits of some label tokens. The output head calculates the logits of all 248,320 tokens. Its weight has the type Q8_0, and a pass reads 675 MB for it. The ablation of phase 4 shows that the head uses about 1.6 ms of each pass that has outputs.

The change gives `llama_context` a list of tokens. If the list is set, the head calculates the rows of the weight for these tokens only. The server sets the list for a batch of decisions. In the private benchmark, `lat_ms` is 0.87 % lower (95 % interval: -1.38 % to -0.33 %). The answers are the same bit for bit.

## Equivalence class

The equivalence class is `exact`. We selected it on 2026-10-31, before the first run. The new path uses the MMVQ kernel for each output row with the same weight rows as the full head. The kernel does the same sums in the same sequence for each output value.

The [method](../method.md) gives the tests for `exact`. The answers are the same bit for bit as the answers of the baseline. `test-backend-ops`, the new program `test-mul-mat-rows` and `compute-sanitizer` have no error. The section "Correctness" gives the results.

## Results

The baseline is arm `L`, build `lead-30489`. It is the branch `dmk/stack` with change 0001. The new build is arm `A3`. Each arm ran 4 times in 4 blocks, in a random sequence in each block.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| `lat_ms` (all requests) | 233.024 | 231.007 | -0.87 | -1.38 to -0.33 | 4 | 200 | 1,845 |
| `short_ms` | 70.933 | 68.701 | -3.15 | -4.07 to -2.24 | 4 | 200 | 1,845 |
| `medium_ms` | 351.291 | 348.338 | -0.84 | -1.38 to -0.31 | 4 | 200 | 1,845 |
| `long_ms` | 765.360 | 764.644 | -0.09 | -0.46 to +0.28 | 4 | 200 | 1,845 |

Source: [speed-bench.tsv](../../data/2026-10-31-0003-answer-head/tables/speed-bench.tsv). The clock is the median SM clock of the runs. One run had 1,875 MHz. The power limit and the clock are in [runs.tsv](../../data/2026-10-31-0003-answer-head/tables/runs.tsv).

The number of requests each second with 8 requests at the same time is 4.309 for `L` and 4.363 for `A3`. This is +1.24 % (95 % interval: +1.03 % to +1.41 %). The prompt speed of `llama bench` does not use the change. A speed that is more is faster.

| Prompt (tokens) | Before (tokens/s) | After (tokens/s) | Difference (%) | 95 % interval (%) | Runs |
|---:|---:|---:|---:|---|---:|
| 64 | 2,843.051 | 2,819.834 | -0.82 | -1.16 to -0.40 | 4 |
| 256 | 4,305.367 | 4,279.718 | -0.60 | -1.05 to -0.18 | 4 |
| 3,000 | 4,699.300 | 4,691.886 | -0.16 | -0.49 to +0.08 | 4 |

Source: [speed-llama-bench.tsv](../../data/2026-10-31-0003-answer-head/tables/speed-llama-bench.tsv).

The interval uses 4 block ratios only. The host load was 1.80 to 15.64. In the two blocks with a load below 5 for the run of `A3`, the ratio of `lat_ms` is -1.36 % and -1.40 %. In the two other blocks it is -0.73 % and +0.02 %. The data do not show the cause.

The branch on the stack is the build for the pull request. It has the same change on top of `dmk/stack` at `fbfd9cb8c`. The baseline is arm `S`, the build `stack-fbfd9cb`. The new build is arm `A3R`. The speed test has 4 blocks. We have no prediction for this test.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| `lat_ms` (all requests) | 221.977 | 218.679 | -1.48 | -2.03 to -0.81 | 4 | 200 | 1,845 |
| `short_ms` | 69.130 | 66.546 | -3.73 | -4.60 to -2.83 | 4 | 200 | 1,845 |
| `medium_ms` | 333.260 | 328.812 | -1.33 | -1.82 to -0.82 | 4 | 200 | 1,845 |
| `long_ms` | 722.721 | 718.546 | -0.58 | -0.92 to -0.23 | 4 | 200 | 1,845 |

Source: [speed-rebased.tsv](../../data/2026-10-31-0003-answer-head/tables/speed-rebased.tsv). The number of requests each second with 8 requests at the same time is 4.561 for `S` and 4.635 for `A3R`. This is +1.63 % (95 % interval: +1.40 % to +1.86 %). The host load was 3.64 to 20.11.

The pull request has 2 more commits on top of `A3R`. The commit `c4175ae75` changes the cases of `test_mul_mat_rows` in `test-backend-ops`. The commit `419b8f612` changes the server. It is the last commit of the branch. The speed test did not use these two commits.

The server gives the list of tokens to `llama_context` only if each slot of the batch holds a decision. Thus a batch with decisions only has the same list as before, and the speed test is correct for the last commit. [INFERENCE: this comes from the `git diff` of the commit. We did not run the speed test again.] Two passes of the private benchmark of the last commit gave `lat_ms` 214.731 ms and 215.553 ms. These two passes are not a test of two arms, because no run of `S` ran in the same time.

## Prediction

We committed the [prediction](../notebook/2026-10-31-0003-answer-head-prediction.md) on 2026-10-31 at 06:18:10 UTC, before the first run of the speed test at 10:32 UTC. The [results entry](../notebook/2026-10-31-0003-answer-head-result.md) gives all the numbers.

| Metric | Predicted gain | Measured gain | Measured / predicted |
|---|---:|---:|---:|
| `lat_ms` | 3.6 ms (1.5 %) | 2.02 ms (0.87 %) | 0.56 |
| `short_ms` | 2.8 ms (3.9 %) | 2.23 ms (3.15 %) | 0.80 |
| `medium_ms` | 5.3 ms (1.5 %) | 2.95 ms (0.84 %) | 0.56 |
| `long_ms` | 5.3 ms (0.7 %) | 0.72 ms (0.09 %) | 0.14 |

The prediction is true for `lat_ms` by its rule: the estimate is -0.8 % or less, and the interval is less than 0. The gain is 0.56 times the predicted gain. The gain of the medium request and of the long request is less than the prediction. We did not find the cause.

The decisions of one request can be in the same pass, and then the head runs one time for them. [INFERENCE: this is one possible cause. No test shows it.]

The prediction recorded a risk: the memory of the graph changes. The first build had this fault. The log with the log level 4 showed 378 lines `graph reallocation` in one pass of the private benchmark, and the parent build showed 0. The commit `f0a2033e7` repairs it. The speed test used the build with the repair.

## Correctness

| Test | Threshold | Result |
|---|---|---|
| Answers of 100 decisions, 4 runs of each arm (`L` against `A3`) | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true | pass |
| Quality of the 8 runs | The 4 guards of the benchmark are true | pass. 94 of 100 correct, 97 of 100 agree, `mean_dp` 0.013322. |
| `test-mul-mat-rows`, `ggml_mul_mat_rows` against `ggml_mul_mat`, 480 cases | 0 cases with different values | pass |
| `test-mul-mat-rows --full`, 512 cases with the dimensions of the head | 0 cases with different values | pass |
| `test-backend-ops test -b CUDA0 -o MUL_MAT_ID`, with the new cases that have `m = 6` | All cases pass | pass, 934 of 934 |
| `test-backend-ops test -b CUDA0 -p same_rows`, the new cases of `test_mul_mat_rows` (`CONT` is the name of these cases in `test-backend-ops`) | All cases pass, NMSE 5e-4 | No on `f0a2033e7`: 171 of 177 cases pass. The 6 cases with an NMSE above the limit have `n_ids = 1`. They have 1 to 4 values, and the NMSE of a small number of values is not stable. The commit `c4175ae75` changed the cases to `n_ids` 10 or 33. On `419b8f612`: 121 of 121 cases pass in 3 runs. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT` | All cases pass | pass, 2,490 of 2,490 |
| `compute-sanitizer --tool memcheck` on the server, one request of each type | `ERROR SUMMARY: 0 errors` | pass |
| Lines `graph reallocation` in the log (log level 4), one pass of the private benchmark | 0 | pass |
| Graph with the full head and the rows of the tokens (6 slots, one request at a time), `A3` against `L`, 4 pairs of runs | The answers are the same bit for bit | pass, 48 of 48 cases in each pair |
| The same test, with the weight of the head in a CPU buffer | The answers are the same bit for bit | not done. The flags `-ot` and `-ngl 32` did not move the weight. |
| Branch on the stack: answers of 100 decisions, `A3R` against `S`, 4 runs of each arm | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true | pass |
| Branch on the stack: the same test with 6 slots and one request at a time, 4 pairs of runs | The answers are the same bit for bit | pass, 48 of 48 cases in each pair |
| Branch on the stack: `compute-sanitizer --tool memcheck` on the server, one request of each type | `ERROR SUMMARY: 0 errors` | pass |
| Branch on the stack: lines `graph reallocation` in the log (log level 4), 2 runs of one pass | 0 | pass |
| Branch on the stack: `test-mul-mat-rows`, `test-mul-mat-rows --full`, `test-backend-ops` (`MUL_MAT_ID`, `MUL_MAT` and `-p same_rows`) | 0 cases with different values, all cases pass | Pass for the first three: 480 and 512 cases with 0 different values, 934 of 934, 2,490 of 2,490. `-p same_rows` on `ca69612da`: 175 of 177 cases pass. The 2 cases with an NMSE above the limit have `n_ids = 1`. |
| Last commit `419b8f612`: `test-mul-mat-rows`, `test-mul-mat-rows --full`, `test-backend-ops` (`MUL_MAT_ID`, `MUL_MAT` and `-p same_rows`, 3 runs) | 0 cases with different values, all cases pass | pass: 480 and 512 cases with 0 different values, 934 of 934, 2,490 of 2,490, and 121 of 121 in each of the 3 runs |
| Last commit: answers of 100 decisions, 2 passes of `F` against the 4 runs of `S` of the speed test of the stack | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true | pass. The largest probability difference is 0.0. |
| Last commit: 48 decision requests one at a time, with 20 `POST /completion` requests and 5 `POST /v1/chat/completions` requests at the same time, `F` and `A3R` | All decision requests give `HTTP 200`. All text requests give `HTTP 200` with text. | Decisions: pass, 48 of 48 in each run, and the answers are the same as the answers of `S`. Text requests: no pass in the two runs. All 25 text requests gave `HTTP 501`, because the model does not give text. These runs do not examine a batch with a slot that is not a decision. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md), the tables `answers-compare-bench.json`, `answers-compare-final.json`, `quality-bench.tsv`, `unit-tests.tsv`, `correctness-runs.tsv`, `answers-client-runs.tsv`, `memcheck-requests.tsv` and `mixed-load.tsv`.

The program `test-mul-mat-rows --all` stops with `GGML_ASSERT` for the type F16 with equal `ids`. The graph does not use this case. The same program compares 5 to 8 columns and finds differences of about 1e-5. The graph does not use the new path for more than 4 output rows.

The MMVQ kernel had a fault that the new operation shows. For a small K, the kernel uses 4 rows for each thread block. With `ids`, the guard at the end of the rows used the distance between two columns of the result. It did not use the number of rows. A thread block with rows after the last row wrote into the results of the next `ids`.

The first build of `test-mul-mat-rows` had 120 of 480 cases with different values. All 120 cases have K = 1,536. The repair does not use the launch with 4 rows if `ids` is set and `nrows_x % 4 != 0`. After the repair, 0 of 480 cases have different values. The head of the model (type Q8_0, K = 2,560) did not use this launch.

## Reproduce

The fork is the fork of llama.cpp of this project. The branch of the pull request is `dmk/0003-answer-head-rebased`. The last commit is `419b8f612780dbe48e21ebb80e7eacc64e88995a`, on top of `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978` (`dmk/stack`). The first commit of the branch is `ca69612da671ed8d185ce39dcf8dae0a1023a570`. The speed test of the stack used it.

The branch `dmk/0003-answer-head` has the same change on top of `6b8b65e6672f07805000d3c9683b65c72cc62303`. The speed test above used it, at the commit `f0a2033e7966e37093b0187743b0fab87720f4e2`.

1. Build the commit with `harness/build/build.sh`. The flags are in the data set.
2. Put the build in `/opt/dmk` of the server image. The image digest is in the data set.
3. Run the schedule from `harness/make_schedule.py` with the commands in the data set.
4. Run `python3 data/2026-10-31-0003-answer-head/make_tables.py` to make the tables.

The new tests are `test-mul-mat-rows`, the cases `test_mul_mat_rows` in `test-backend-ops` and the cases with `m = 6` in `test-backend-ops test -o MUL_MAT_ID`. The flag `-p same_rows` selects the cases of `test_mul_mat_rows`, because the flag `-o` does not select them. The new functions are `llama_set_logit_tokens`, `llama_get_logit_tokens` and `llama_get_logit_tokens_ith` in `llama-ext.h`. They are not part of `llama.h`.

## Limits

- The path with the rows of the weight is for a head with the type Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0 in a CUDA buffer. The weight must be in the memory of one GPU. `n_outputs_max` must be 4 or less. The server sets it to the number of slots. The list must have 64 tokens or less. We ran the tests on `sm_86` only, with the Q8_0 head of the model.
- A pass with 5 to 8 output rows has different sums in the MMVQ kernel (differences of about 1e-5). The graph then has the full head and gets the rows of the tokens from the logits.
- The type F16 with equal `ids` stops with `GGML_ASSERT` in the `MUL_MAT_ID` path of CUDA. The graph does not use it.
- `llama_get_logits` and `llama_get_logits_ith` give `nullptr` while a list is in use. The server gives the list to a batch only if each slot of the batch holds a decision. The commit `419b8f612` repairs a fault of the first version. In that version, the list had the labels of the decision slots if the first slot of the batch was a decision. A slot of a `POST /completion` request or of a `POST /v1/chat/completions` request in the same batch had no logits. [INFERENCE: we found this fault when we read the code, and not in a run.]
- The model of the project does not give text. The server answers each text request with `HTTP 501` and the text `This model is a decision model without text generation, use /v1/systemone`. A batch with a slot that is not a decision cannot occur with this model. Thus the server in production cannot use this path.
- The commit `419b8f612` is a guard for a model with text output and decisions. The two runs with decisions and text requests (`F` and `A3R`) gave `HTTP 501` for all 25 text requests. They do not show the fault or its repair. We have no other model on the host to examine it.
- The first list makes `llama_context` do `sched_reserve` again with a list of 64 tokens. This is about 0.2 s, one time. A batch with no list, after this, gives a `graph reallocation`. A server that sends decisions and other batches in turn has a `graph reallocation` at each change. [INFERENCE: the server in production sends decisions only, because it gives `HTTP 501` for text requests.]
- `llama_context` does not use a list of more than 64 tokens. The server then uses the full logits, and `llama_context` writes a line in the log one time.
- The MMVQ repair changes the guard for the launch with 4 rows. It has an effect only if `ids` is set and `nrows_x % 4 != 0`. The program `test-mul-mat-rows` had 120 of 480 cases with different values before the repair and 0 after it.
- The test with 6 requests at the same time cannot show a result bit for bit. Two runs of the parent build give different answers (the largest probability difference is 0.07). The tests with one request at a time replace it.
- No test has the weight of the head in a CPU buffer. The flags `-ot output.weight=CPU` and `-ngl 32` did not move the weight.
- The `compute-sanitizer` run has 4 requests, one of each type. It does not run the full benchmark.
- The speed test has 4 blocks, one benchmark and one GPU. We did not control the load of the host. The `llama bench` result has intervals that do not include 0 for the two shorter prompts (-0.82 % and -0.60 %). The graph of `llama bench` has no new node, and we did not find the cause.
- To go back to the baseline, use the build without the change. Use `lead-30489` for the branch on `6b8b65e66`, or `stack-fbfd9cb` for the branch on the stack. You can also remove the commits with `git revert`. The server then reads the full logits.

## Data

- [Data set](../../data/2026-10-31-0003-answer-head/README.md)
- [Prediction](../notebook/2026-10-31-0003-answer-head-prediction.md)
- [Results entry](../notebook/2026-10-31-0003-answer-head-result.md)
