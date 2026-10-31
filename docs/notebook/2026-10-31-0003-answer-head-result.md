# 0003 Answer head: result

Written 2026-10-31, after the measurement windows (07:33 to 20:30 UTC).

## Task of this entry

This entry gives the result of change 0003. The change calculates the output head for the answer tokens only. It is target `A` of the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md). The [prediction entry](2026-10-31-0003-answer-head-prediction.md) is in the commit `5cc6cad2f241c3158b37acabdc6ac1d7501c80f7` from 06:18:10 UTC. The first run of the speed test started at 10:32:12 UTC.

The [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) gives the raw files, the tables and the conditions. The [change document](../changes/0003-answer-head.md) is the text of the pull request.

## Result in short

- The answers of arm `A3` are the same as the answers of arm L bit for bit on all 100 decisions. The 4 runs of each arm give the same answers. The equivalence class `exact` holds for these decisions.
- `lat_ms` is 0.87 % lower (95 % interval: -1.38 % to -0.33 %). The gain is 2.0 ms. The prediction was -1.5 %, or 3.6 ms. The gain is 56 % of the predicted gain.
- `short_ms` is 3.15 % lower, `medium_ms` is 0.84 % lower and `long_ms` is 0.09 % lower. The interval of `long_ms` includes 0. `rps8` is 1.24 % higher.
- The prompt speed of `llama bench` is -0.82 % at 64 tokens, -0.60 % at 256 tokens and -0.16 % at 3,000 tokens. The change does not touch this path. The three values are in the range of ±1 %.
- The prediction holds by its rule. The estimate of `lat_ms` is -0.8 % or less, and the interval is less than 0. The gain for the medium request and the long request is less than we predicted.
- The first build of the change had a fault that the prediction recorded as a risk. The scheduler gave memory to the graph again in about each second pass. We found it in the log with the log level 4. We repaired the fault before the speed test. The speed test used the build with the repair.
- The load of the host changed from run to run. In the two blocks with a low load for `A3`, `lat_ms` is 1.36 % and 1.40 % lower. These two values are near the prediction.
- The graph with the full head (6 slots) gives the same answers as arm L bit for bit, with one request at a time. The test with 6 requests at the same time cannot show this, because two runs of arm L give different answers.
- The same change on top of the stack (arm `A3R`, the commit `ca69612da`) gives the same answers as the stack (arm `S`) on 100 decisions. `lat_ms` is 1.48 % lower (95 % interval: -2.03 % to -0.81 %). The test has no prediction.
- The last commit of the pull request is `419b8f612` (arm `F`). It gives the same answers as the stack (arm `S`) bit for bit (2 runs, 100 decisions). The unit tests pass: 121 of 121 for `-p same_rows` (3 runs), and all other tests. The model of the project gives no text, and the server answers a text request with `HTTP 501`. Thus a batch with a slot that is not a decision cannot occur with this model. No run shows the fault that the commit repairs, and we checked the commit when we read the code.

## Change

The output head is the last matrix multiply of a pass. A decision uses the logits of its label tokens only. The change gives `llama_context` a list of tokens. If the list is set, the head reads the rows of the weight for these tokens only. The server sets the list for a batch of decisions. The [change document](../changes/0003-answer-head.md) has the description of the change.

The change has one more part. The MMVQ kernel had a fault for a weight with one row for each expert. The fault has no effect on the head of the model. It has an effect on the new operation `ggml_mul_mat_rows`. The section "Unit tests" gives the results before and after the repair.

The build for the speed test has the commit `f0a2033e7`. The first build for the pull request has the commit `ca69612da`. It has the same change on top of the stack `fbfd9cb8c`. The last commit of the pull request is `419b8f612`. It has 2 more commits than `ca69612da` (tests and server). The section "The branch on the stack" gives the tests of `ca69612da`, and its part "The last commit" gives the tests of `419b8f612`.

## Equivalence class

We selected the equivalence class `exact` before the measurement. The prediction entry gives the cause. The tests in this section agree with it.

### Unit tests

The program `test-mul-mat-rows` compares `ggml_mul_mat_rows` with `ggml_mul_mat` on the GPU, value by value.

| Build | Test | Result |
|---|---|---|
| First build, before the repair of `mmvq.cu` | `test-mul-mat-rows` | 480 cases. 120 cases have different values. All 120 cases have K = 1,536. |
| `c1d3d481b`, `f0a2033e7` | `test-mul-mat-rows` | 480 cases. 0 cases have different values. |
| `c1d3d481b`, `f0a2033e7` | `test-mul-mat-rows --full` | 512 cases, with the dimensions of the head of the model. 0 cases have different values. |
| `c1d3d481b`, `f0a2033e7` | `test-backend-ops test -b CUDA0 -o MUL_MAT_ID` | 934 of 934 tests pass. This includes the new cases with `m = 6`. |
| `c1d3d481b`, `f0a2033e7` | `test-backend-ops test -b CUDA0 -o MUL_MAT` | 2,490 of 2,490 tests pass. |
| `f0a2033e7` | `test-backend-ops test -b CUDA0 -p same_rows` | 171 of 177 tests pass. The 6 tests with an NMSE above the limit have `n_ids = 1` and 1 to 4 values. |
| `c1d3d481b`, `f0a2033e7` | `test-mul-mat-rows --all` | The program stops with `GGML_ASSERT`. The cause is the type F16 with equal `ids`. The graph does not use this case. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `unit-tests.tsv`.

The first build had a fault in the MMVQ kernel. For a small K, the kernel uses 4 rows for each thread block. With `ids`, the guard at the end of the rows used the distance between two columns of the result. It did not use the number of rows. A thread block with rows after the last row then wrote into the results of the next `ids`. It also read after the weight.

The head of the model has the type Q8_0 and K = 2,560, and it does not use this launch. The new operation uses it, because its weight has one row for each expert.

The repair does not use the launch with 4 rows if `ids` is set and `nrows_x % 4 != 0`. The sequence of the sums is the same in the two launches. After the repair, the same program has 0 cases with different values.

With `--all`, the program also compares 5 to 8 columns. These cases have differences of about 1e-5. The kernel for 5 to 8 columns uses a different sequence of sums. This is the cause of the limit of 4 output rows. The graph uses the full head for a larger number of output rows.

The flag `-p same_rows` selects the cases of `test_mul_mat_rows` in `test-backend-ops`. These cases compare the result on the GPU with the result on the CPU, with the limit NMSE 5e-4. The first runs had cases with an NMSE above the limit. All of them have `n_ids = 1`. The build `f0a2033e7` had 6 such cases and `ca69612da` had 2 (of 56 cases with `n_ids = 1` for each build).

We ran `ca69612da` 10 times again with the cases with `n_ids = 1`: 34 of 560 results had an NMSE above the limit (6.1 %). We ran it 10 times again with the cases with `n_ids = 10`: 0 of 640 results. [INFERENCE: a case with `n_ids = 1` has 1 to 4 values. The NMSE of a small number of values is not stable.] `test-mul-mat-rows` compares the values bit for bit, also for `n_ids` 1 and 2, and it found no difference. The commit `c4175ae75` changed the cases to `n_ids` 10 and 33, and the section "The last commit" gives the result.

### Answers of the private benchmark

Each run of the private benchmark wrote its answers with the flag `--write-reference`. The tool `harness/analysis/answers_compare.py` compared the answers of the 8 runs of the speed test.

| Test | Result |
|---|---|
| Runs of arm L | The 4 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Runs of arm `A3` | The 4 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Arm L and arm `A3` | The answers are the same bit for bit. The largest probability difference is 0.0. No decision has a different top answer. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `answers-compare-bench.json`.

All 8 runs give the same quality numbers. 94 of 100 decisions are correct, and 97 of 100 agree with the reference. `mean_dp` is 0.013322. All guards are true and the number of errors is 0. The runs of the measurement window 4 give the same result for 2 runs of each arm (the table `answers-compare-window4.json`). The runs of the measurement window 1 used the commit `c1d3d481b` and give the same result (the table `answers-compare-window1.json`).

### Graph with the full head

A server with 5 or more slots can have 5 or more output rows in a pass. This is more than the limit of 4. The graph then has the full head, and it gets the rows of the tokens from the logits. The answers must be the same as the answers of arm L for this graph. The prediction gives this rule.

The first test used 6 requests at the same time. The answers of arm L are not the same in two runs of this test. The cause is the sequence of the requests in the batches. The table gives the largest probability difference of the 48 cases for pairs of runs in the measurement window 4.

| Pair | Largest probability difference |
|---|---:|
| L and L | 0.0747 |
| `A3` and L (4 pairs) | 0.0332 to 0.0415 |
| `A3` and `A3` | 0.0180 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `answers-client-runs.tsv`.

The differences between `A3` and L are less than the difference between two runs of L. This test cannot show a result bit for bit. Thus we ran a second test with 6 slots and one request at a time. The batches are then the same in each run. The section "Six slots and one request at a time" gives the result.

### Six slots and one request at a time

The second test used a server with 6 slots (`-np 6`) and one request at a time. The graph of arm `A3` has the full head, because `n_outputs_max` is 6. The log of each run of `A3` and of `A3R` has the line `the full head, then the values of` one time, and it has no line `graph reallocation`. Only one request is in the server at one time, thus the batches are the same in each run. [INFERENCE: this is the cause of the same answers in two runs of the same arm.]

| Pair of runs | Pairs | Cases with the same answers (of 48) | Largest probability difference |
|---|---:|---:|---:|
| L and L | 1 | 48 | 0.0 |
| `A3` and `A3` | 1 | 48 | 0.0 |
| `A3` and L | 4 | 48 in each pair | 0.0 |
| `S` and `S` | 1 | 48 | 0.0 |
| `A3R` and `A3R` | 1 | 48 | 0.0 |
| `A3R` and `S` | 4 | 48 in each pair | 0.0 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `answers-client-runs.tsv`.

The answers of the graph with the full head are the same bit for bit as the answers of the baseline. This is true for `A3` against L and for `A3R` against `S`.

We tried two flags to put the weight of the head in a CPU buffer. The flag `-ot output.weight=CPU` (measurement window 4) moved 8 other weights to the CPU. The flag `-ngl 32` (measurement window 6) kept the output layer on the GPU, because the log shows `offloading output layer to GPU`. In the 4 runs of each flag, the log of `A3` has the line of the head with the rows of the tokens. The answers of `A3` and L are the same bit for bit in these runs (48 of 48 cases in each pair). We have no test with the weight of the head in a CPU buffer.

### Test with `compute-sanitizer`

The server of arm `A3` ran in `compute-sanitizer --tool memcheck`. The client sent one request of each type: `short1q`, `short2q`, `medium` and `long`. All 4 requests gave `HTTP 200`. In the run on `f0a2033e7`, the summary shows `ERROR SUMMARY: 0 errors`.

The wrapper that we used in the measurement windows 3 and 4 stops the server only when `docker stop` sends `TERM`. Then `compute-sanitizer` ends and writes its summary. The first two measurement windows did not have the wrapper, and `compute-sanitizer` wrote no summary there. The logs of these two runs have no error report.

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the tables `memcheck-requests.tsv` and `correctness-runs.tsv`.

## Graph memory

The prediction recorded this risk: the memory of the graph changes. We counted the lines `graph reallocation` in the log of the server. The library writes these lines at the log level 4 only. A first count with the default level showed no lines, but this count was not correct.

| Build | Run | Lines `graph reallocation` |
|---|---|---:|
| `4bd0c75b5` | One pass of the private benchmark, arm `A3` | 378 |
| `4bd0c75b5` | 6 requests at the same time, arm `A3` | 53 |
| `6b8b65e66` | The same runs, arm L | 0 |
| `f0a2033e7` | One pass of the private benchmark, 2 runs, arm `A3` | 0 |
| `f0a2033e7` | 6 requests at the same time, 2 runs, arm `A3` | 0 |
| `f0a2033e7` | `compute-sanitizer`, arm `A3` | 0 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `correctness-runs.tsv`.

The server gives memory to the graphs of the largest batches when it starts. The log shows this step as `sched_reserve`. The graph of the first build had the full head at that time. The graphs of the batches had the head for the tokens of the list. These graphs have different nodes. Some tensors had a dimension that is more than the dimension at `sched_reserve`.

In this case the scheduler gives memory to the graph of the batch only. It does not keep the larger dimensions of the other graphs. [INFERENCE: we read the code. We did not make a test for this cause.]

The repair has three parts. The first list makes `llama_context` do `sched_reserve` again with a list of 64 tokens. The largest number of outputs of `llama_context` sets the path of the head, and the graph does not set it. `llama_context` does not use a list of more than 64 tokens. Thus each graph has the nodes of the graphs of `sched_reserve`, and no tensor is larger than the tensor of `sched_reserve`. The lines `graph reallocation` went from 378 to 0.

The repair adds one step. In the run `0003-r-bench-A3-1`, the second `sched_reserve` ran for 0.22 s. The first one, at the start, ran for 0.16 s. [INFERENCE: the server in production receives decisions only. It does the second `sched_reserve` one time after the start.]

## Speed test with the private benchmark

The test has 4 blocks with seed 3001. Each block has one run of L and one run of `A3` in a random sequence. The estimate is the mean of the 4 block ratios. A block ratio is the value of `A3` divided by the value of L, minus 1. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples. A latency that is less is faster, and a rate that is more is faster.

| Metric | L mean (ms) | `A3` mean (ms) | `A3` against L (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` | 233.024 | 231.007 | -0.867 | -1.382 | -0.326 |
| `short_ms` | 70.933 | 68.701 | -3.153 | -4.069 | -2.237 |
| `medium_ms` | 351.291 | 348.338 | -0.842 | -1.380 | -0.305 |
| `long_ms` | 765.360 | 764.644 | -0.094 | -0.464 | +0.276 |

| Metric | L mean (requests/s) | `A3` mean (requests/s) | `A3` against L (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `rps8` | 4.309 | 4.363 | +1.240 | +1.027 | +1.412 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `speed-bench.tsv`.

The next table gives each run in time sequence. The load is the 1-minute load average of the host before and after the run.

| Start (UTC) | Run | Arm | `lat_ms` (ms) | `short_ms` (ms) | `medium_ms` (ms) | `long_ms` (ms) | `rps8` (requests/s) | Load before | Load after |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 10:32:12 | `b1-L` | L | 234.0 | 71.7 | 351.9 | 765.6 | 4.304 | 14.16 | 10.84 |
| 10:34:03 | `b1-A3` | `A3` | 232.3 | 69.7 | 350.1 | 767.4 | 4.344 | 10.84 | 15.64 |
| 10:35:51 | `b2-L` | L | 232.2 | 70.3 | 350.3 | 763.6 | 4.326 | 15.64 | 4.10 |
| 10:37:39 | `b2-A3` | `A3` | 229.1 | 67.5 | 345.3 | 760.1 | 4.389 | 4.10 | 1.89 |
| 10:39:25 | `b3-A3` | `A3` | 229.1 | 67.5 | 345.6 | 760.8 | 4.379 | 1.89 | 1.80 |
| 10:41:10 | `b3-L` | L | 232.3 | 70.4 | 350.4 | 764.3 | 4.323 | 1.80 | 1.98 |
| 10:42:57 | `b4-L` | L | 233.6 | 71.3 | 352.7 | 767.9 | 4.284 | 1.98 | 7.57 |
| 10:44:45 | `b4-A3` | `A3` | 233.6 | 70.1 | 352.3 | 770.4 | 4.339 | 7.57 | 11.08 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `runs.tsv`. The source table has more decimal places.

The next table gives the ratio of each block and the load of the two runs of the block.

| Block | First run | L load before | L load after | `A3` load before | `A3` load after | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) | `rps8` (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | L | 14.16 | 10.84 | 10.84 | 15.64 | -0.725 | -2.750 | -0.496 | +0.229 | +0.939 |
| 2 | L | 15.64 | 4.10 | 4.10 | 1.89 | -1.364 | -4.019 | -1.401 | -0.463 | +1.458 |
| 3 | `A3` | 1.80 | 1.98 | 1.89 | 1.80 | -1.400 | -4.118 | -1.359 | -0.466 | +1.291 |
| 4 | L | 1.98 | 7.57 | 7.57 | 11.08 | +0.020 | -1.725 | -0.114 | +0.322 | +1.272 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `speed-bench-blocks.tsv`.

The bootstrap has only 4 block ratios. Its interval cannot be wider than the range of these 4 ratios. The ratio of `lat_ms` is less than 0 in 3 blocks and +0.02 % in block 4. The ratio of `short_ms` is less than 0 in all 4 blocks. The ratio of `long_ms` is between -0.47 % and +0.32 %.

## Speed test with `llama bench`

The test has 4 blocks with seed 3002. Each run measures each prompt length 5 times. The command is below. In the command, `mix8` is the path of the model file.

```
llama bench -m mix8 -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json
```

A speed in tokens/s that is more is faster. `llama bench` does not set a list of tokens. The graph is the same as in arm L. This test is a control.

| Prompt length (tokens) | L mean (tokens/s) | `A3` mean (tokens/s) | `A3` against L (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---:|---:|---:|---:|---:|---:|
| 64 | 2,843.051 | 2,819.834 | -0.816 | -1.163 | -0.403 |
| 256 | 4,305.367 | 4,279.718 | -0.595 | -1.052 | -0.182 |
| 3,000 | 4,699.300 | 4,691.886 | -0.157 | -0.492 | +0.078 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `speed-llama-bench.tsv`.

The three estimates are in the range of ±1 %. The intervals of the two shorter prompts do not include 0. The graph of this test has no new node, thus the cause of the difference is not in the graph. [INFERENCE: it can be the load of the host (9.84 to 15.80 in the 8 runs) or a different code layout in the two builds. We did not run a test for these causes.]

The value at 3,000 tokens is the nearest to 0. The ratio at 64 tokens is between -0.22 % and -1.28 % in the 4 blocks.

## Host load

The host did other work during the schedule. The data does not show the cause.

- The load was 1.80 to 15.64 in the runs of the private benchmark. It was 9.84 to 15.80 in the 8 runs of `llama bench`.
- In block 1 and block 4, the load before the run of `A3` was 10.84 and 7.57. In block 2 and block 3, it was 4.10 and 1.89.
- The ratio of `lat_ms` is -1.364 % in block 2 and -1.400 % in block 3. It is -0.725 % in block 1 and +0.020 % in block 4.
- In block 3, the load is low for the two arms (1.80 to 1.98). Its ratios are -1.40 % for `lat_ms`, -4.12 % for `short_ms`, -1.36 % for `medium_ms` and -0.47 % for `long_ms`.

[INFERENCE: a high load makes the requests slower. It can change the gain of the two arms. No test changed the load only.] The prediction for `lat_ms` is -1.5 %. In block 3, the conditions are the nearest to the conditions of the prediction. The block gives -1.40 %.

One block is not sufficient for a decision. A test with a low load and more blocks can show if the estimate is nearer to -1.4 % than -0.87 %.

## The branch on the stack

The build for the pull request is arm `A3R`. It has the commit `ca69612da` of the branch `dmk/0003-answer-head-rebased`. The commit has the 3 commits of arm `A3` as one commit on top of the stack at `fbfd9cb8c` (arm `S`). The rebase did not stop. We ran the same tests as for arm `A3`, with arm `S` as the baseline.

The prediction entry has no prediction for this test. The prediction is for arm L and arm `A3`. We ran the speed test of `S` against `A3R` because the project can use the stack and not arm L in production.

### Answers and graph memory

The measurement window 5 ran from 10:48 to 11:09 UTC.

| Test | Result |
|---|---|
| One pass of the private benchmark, 2 runs of each arm, log level 4 | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true. The largest probability difference is 0.0 on 100 decisions. |
| The 8 runs of the speed test below | `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true. The largest probability difference is 0.0 on 100 decisions. |
| Lines `graph reallocation`, 2 runs of `A3R` with log level 4 | 0 and 0. Each run has one line of the head path (the rows of the tokens). |
| `compute-sanitizer --tool memcheck`, one request of each type, arm `A3R` | 4 requests with `HTTP 200`. `ERROR SUMMARY: 0 errors`. |
| 48 cases, one request at a time, arm `A3R` and arm `S` (4 runs) | The 4 runs give the same answers bit for bit. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the tables `answers-compare-rebased-bench.json`, `answers-compare-rebased-ab.json`, `correctness-runs.tsv`, `memcheck-requests.tsv` and `answers-client-runs.tsv`.

The last test in the table used the flag `-ot ^output[.]weight=CPU`. The flag does not change the buffers of the model, because the server log shows the same buffers with and without it. [INFERENCE: the model has one tensor for the input embedding and the head, and the flag does not select its copy on the GPU.] Thus the 4 runs test the path with the rows of the weight. They give no test of the path with the full head. The section "Six slots and one request at a time" gives tests of the full head.

### Unit tests

| Test | Result |
|---|---|
| `test-mul-mat-rows` | 480 cases. 0 cases have different values. |
| `test-mul-mat-rows --full` | 512 cases. 0 cases have different values. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT_ID` | 934 of 934 tests pass. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT` | 2,490 of 2,490 tests pass. |
| `test-backend-ops test -b CUDA0 -p same_rows` | 175 of 177 tests pass. The 2 tests with an NMSE above the limit have `n_ids = 1` and one value each (type Q4_0 and type Q4_1, one column). The last commit has no case with `n_ids = 1`. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `unit-tests.tsv`.

### Speed test

The test has 4 blocks with seed 3003. Each block has one run of `S` and one run of `A3R` in a random sequence. The estimate, the interval and the unit are the same as in the speed test of `A3`. The ratio is the value of `A3R` divided by the value of `S`, minus 1.

| Metric | `S` mean (ms) | `A3R` mean (ms) | `A3R` against `S` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `lat_ms` | 221.977 | 218.679 | -1.481 | -2.032 | -0.808 |
| `short_ms` | 69.130 | 66.546 | -3.725 | -4.597 | -2.827 |
| `medium_ms` | 333.260 | 328.812 | -1.332 | -1.819 | -0.822 |
| `long_ms` | 722.721 | 718.546 | -0.576 | -0.918 | -0.234 |

| Metric | `S` mean (requests/s) | `A3R` mean (requests/s) | `A3R` against `S` (%) | 95 % interval, low end (%) | 95 % interval, high end (%) |
|---|---:|---:|---:|---:|---:|
| `rps8` | 4.561 | 4.635 | +1.630 | +1.397 | +1.862 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `speed-rebased.tsv`.

The next table gives the ratio of each block and the load of the two runs of the block.

| Block | First run | `S` load before | `S` load after | `A3R` load before | `A3R` load after | `lat_ms` (%) | `short_ms` (%) | `medium_ms` (%) | `long_ms` (%) | `rps8` (%) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `S` | 16.60 | 15.72 | 15.72 | 14.31 | -2.202 | -4.919 | -2.023 | -0.911 | +1.455 |
| 2 | `A3R` | 6.44 | 3.64 | 14.31 | 6.44 | -0.456 | -2.344 | -0.595 | -0.032 | +1.987 |
| 3 | `S` | 3.64 | 9.08 | 9.08 | 19.42 | -1.403 | -3.362 | -1.207 | -0.435 | +1.737 |
| 4 | `A3R` | 20.11 | 16.47 | 19.42 | 20.11 | -1.863 | -4.275 | -1.502 | -0.925 | +1.340 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `speed-rebased-blocks.tsv`.

The gain of `lat_ms` is 3.3 ms (1.48 %). It is 1.6 times the gain of arm `A3` against arm L (2.0 ms). The ratio of each block of the 4 metrics `lat_ms`, `short_ms`, `medium_ms` and `long_ms` is less than 0. The interval of `long_ms` does not include 0 in this test.

[INFERENCE: the stack is faster than arm L, thus the output head is a larger part of each pass. This can make the gain larger. We did not run a test for this cause.] The prediction for arm L was -1.5 % for `lat_ms`. The estimate here is -1.48 %, but the test has no prediction. We did not select this value before the run.

The load of the host changed from 3.64 to 20.11 in the 8 runs. The data do not show the effect of the load.

### The last commit

The last commit of the pull request is `419b8f612` (arm `F`) on the branch `dmk/0003-answer-head-rebased`. The branch is on top of `fbfd9cb8c`. It has 3 commits: `ca69612da` (the change), `c4175ae75` (the cases of `test_mul_mat_rows` have `n_ids` 10 or 33) and `419b8f612` (the server).

The server gives the list of tokens to `llama_context` only if each slot of the batch holds a decision. Before this commit, the list had the labels of the decision slots if the first slot of the batch was a decision. A slot of a `POST /completion` request or of a `POST /v1/chat/completions` request samples from the full logits. [INFERENCE: such a slot got `nullptr` from `llama_get_logits_ith`. We did not see this in a run.]

The measurement window 7 ran from 20:25 to 20:30 UTC with the flags of production. The load of the host was 1.38 to 2.92.

The speed test of `A3R` against `S` sent decision requests only. In each batch of that test, each slot holds a decision. For such a batch, the new code makes the same list as the code before. The list has the labels of the decision slots, and no label two times. Only a batch with a slot that is not a decision has a different list.

[INFERENCE: this comes from the `git diff` of `419b8f612` (18 lines added and 11 lines removed in `server-context.cpp`). We did not run the speed test again.] The 2 passes of `F` gave `lat_ms` 214.731 ms and 215.553 ms. They are not a test of two arms, because no run of `S` ran in the same measurement window.

Unit tests of the build `0003-answer-head-final`:

| Test | Result |
|---|---|
| `test-mul-mat-rows` | 480 cases. 0 cases have different values. |
| `test-mul-mat-rows --full` | 512 cases. 0 cases have different values. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT_ID` | 934 of 934 tests pass. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT` | 2,490 of 2,490 tests pass. |
| `test-backend-ops test -b CUDA0 -p same_rows`, 3 runs | 121 of 121 tests pass in each run. The cases of `test_mul_mat_rows` have `n_ids` 10 or 33. |
| `test-backend-ops test -b CUDA0 -o MUL_MAT_ROWS`, 3 runs | 0 of 0 tests. This flag selects no test. Only the flag `-p same_rows` selects the cases of `test_mul_mat_rows`. |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `unit-tests.tsv`.

The tool `answers_compare.py` compared the answers of the 2 passes of `F` with the 4 runs of `S` (`answers-compare-final.json`). The values `cross_identical_all`, `self_repeat_a` and `self_repeat_b` are true. The largest probability difference is 0.0 on 100 decisions. No decision has a different top answer. Each pass has 94 of 100 decisions correct, 97 of 100 that agree with the reference, and `mean_dp` 0.013322. They have the same values as the 8 runs of the first speed test.

The runs with decisions and text requests send the 48 cases of the private benchmark one at a time. At the same time, other threads send 20 `POST /completion` requests (`n_predict` 16) and 5 `POST /v1/chat/completions` requests (`max_tokens` 16). We ran this on `F` and on `A3R`, the build before the repair.

| Build | Request type | Requests | `HTTP 200` | `HTTP 501` | With text |
|---|---|---:|---:|---:|---:|
| `F` (`419b8f612`) | decision | 48 | 48 | 0 | not used |
| `F` (`419b8f612`) | `POST /completion` | 20 | 0 | 20 | 0 |
| `F` (`419b8f612`) | `POST /v1/chat/completions` | 5 | 0 | 5 | 0 |
| `A3R` (`ca69612da`, before the repair) | decision | 48 | 48 | 0 | not used |
| `A3R` (`ca69612da`, before the repair) | `POST /completion` | 20 | 0 | 20 | 0 |
| `A3R` (`ca69612da`, before the repair) | `POST /v1/chat/completions` | 5 | 0 | 5 | 0 |

Source: [README of the data set](../../data/2026-10-31-0003-answer-head/README.md) and the table `mixed-load.tsv`.

The two builds gave the same result. The server answers each text request with `HTTP 501` and the text `This model is a decision model without text generation, use /v1/systemone`. The code is in `process_single_task` of `server-context.cpp`: a task that samples tokens gets this error if the output of the model has no text. The output of the decision type `h2o-lightning` is `decisions` only. A task with this error does not get a slot.

Thus no batch of these runs had a slot that is not a decision. The runs do not show the fault that `419b8f612` repairs. The 48 decision answers of each run are the same as the answers of `S` with one request at a time (48 of 48 cases). The log of each server shows one start of the server.

[INFERENCE: with this model, a batch with a slot that is not a decision cannot occur. Only a model with text output and decisions can have it. We have no such model on the host.] The commit `419b8f612` is a guard for those models. We checked it when we read the code, and not in a run. To go back, use the build without the change.

## Prediction against result

| Item | Prediction | Result | Same as the prediction |
|---|---|---|---|
| Equivalence class | `exact` | Same answers bit for bit on 100 decisions in 8 runs. 0 cases with different values in `test-mul-mat-rows`. | Yes |
| Gain of one pass with outputs | 1.75 ms | About 1.4 ms. [INFERENCE: the gain of `short_ms` (2.2 ms) divided by the 1.6 passes with outputs of the mean short request.] | Less. It is about 80 %. |
| `lat_ms` | -1.5 % (-3.6 ms) | -0.87 % (-2.0 ms). Interval -1.38 % to -0.33 %. | Yes, by the rule. The gain is 56 % of the predicted gain. |
| `short_ms` | -3.9 % (-2.8 ms) | -3.15 % (-2.2 ms). Interval -4.07 % to -2.24 %. | Yes. The prediction is in the interval. |
| `medium_ms` | -1.5 % (-5.3 ms) | -0.84 % (-3.0 ms). Interval -1.38 % to -0.31 %. | No. The prediction is 0.12 points less than the low end of the interval. |
| `long_ms` | -0.7 % (-5.3 ms) | -0.09 % (-0.7 ms). Interval -0.46 % to +0.28 %. | No. The interval includes 0 and not the prediction. |
| `rps8` | +0.5 % to +1.5 % | +1.24 %. Interval +1.03 % to +1.41 %. | Yes |
| `llama bench` | 0, or in the range of ±1 % | -0.82 %, -0.60 %, -0.16 % | Yes, by the rule. |
| `lat_ms` of -2.3 % or more | A second cause | Not seen. | Not applicable |
| Graph memory | A risk | 378 lines `graph reallocation` in one pass of the first build. 0 in the build of the speed test. | The fault was in the first build. We repaired it. |
| Graph with the full head | Same answers as L | 6 requests at the same time: no result, because the answers of two runs of L are not the same. 6 slots and one request at a time: the same answers bit for bit (48 of 48 cases in 4 pairs). | Yes, in the test with one request at a time. |

Source: the tables in this entry, and the [prediction entry](2026-10-31-0003-answer-head-prediction.md).

The prediction had 3 passes with outputs for the medium request and for the long request. At 1.4 ms, 3 passes give 4.2 ms. The gain that we measured is 3.0 ms for the medium request and 0.7 ms for the long request. We did not find the cause of the difference. The causes can be these. [INFERENCE: we did not run a test for one of them.]

- Decisions of one request can be in the same pass. The head then runs one time for them, and the number of passes with outputs is less than 3.
- The noise of `long_ms`. The ratios of the 4 blocks are between -0.47 % and +0.32 %. The interval cannot show a gain of 0.7 ms with 4 blocks.
- The load of the host.

The result for the short request is near the prediction. A short request has 1 or 2 passes with outputs and no prefix of 3,000 tokens.

## Limits and next work

- The path with the rows of the weight is for a head with the type Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0 in a CUDA buffer. `n_outputs_max` must be 4 or less. The list must have 64 tokens or less. We ran the tests on `sm_86` only, with the Q8_0 head of the model.
- A pass with 5 to 8 output rows is not in this path. The kernel for 5 or more columns uses a different sequence of sums. The graph then has the full head.
- `llama_get_logits` and `llama_get_logits_ith` give `nullptr` while a list is in use. The server gives the list to a batch only if each slot of the batch holds a decision (commit `419b8f612`). Thus the server in production cannot use the path with a batch of mixed slots. It gives `HTTP 501` for a text request.
- `llama_context` does `sched_reserve` again at the first list. A server that sends decisions and other batches in turn has a `graph reallocation` at each change. The runs `0003-final-mixed-*` show that the server in production gives `HTTP 501` for text requests. [INFERENCE: it sends decisions only.]
- No run shows the effect of the commit `419b8f612`. A text request gets `HTTP 501` before it gets a slot. The commit is a guard for a model with text output and decisions. We checked it when we read the code. No model on the host can examine it.
- The speed test has 4 blocks, one benchmark and one GPU. We did not control the load of the host.
- The answers are the same on 100 decisions of one benchmark. The tests do not show other inputs.
- The tests of the graph with the full head have one request at a time. No test has the weight of the head in a CPU buffer, because the two flags that we tried did not move it.
- To go back to the baseline, use the build without the commit. The server then reads the full logits. No other change of the project uses the new functions.

Before the project uses the change in production, it must do these steps.

1. A test with 6 blocks of the full stack with this change, against the image in production.
2. A soak test of 24 hours.
