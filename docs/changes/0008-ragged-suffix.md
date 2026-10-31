# Ragged batching of the question suffixes

## Summary

A request with 2 or 3 questions has one suffix for each question, and the suffixes have different lengths. The equal split of the hybrid memory makes one pass for each different length, and each pass reads all weights again. The change puts all suffixes in one pass. It adds two operations for the recurrent layers: `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`. The program uses the equal split when a condition of the change is not true.

The change had two tests. The second test is on the branch `dmk/stack` with the changes 0001, 0007, 0002 (tests), 0004 and 0005. It gives -2.46 % for `lat_ms`. One limit for the decision is not true: the limit of 3 times for one input. The change stays out of `dmk/stack`. It is a Gitea pull request, number 6.

## Equivalence class

`re-associated`. We selected this equivalence class on 2026-10-31, before the first run. A pass with more tokens uses other matrix-multiply kernels: MMQ for more than 8 tokens, and MMVQ for 8 tokens or less. The flash attention kernel can also be different. The sums then occur in a different sequence, and the last bits of the results can be different.

## Results of the second test

Build `A` is `build:stack-fbfd9cb`, the branch `dmk/stack` at commit `fbfd9cb8c`. Build `B` is `build:0008-ragged-rebased`, the branch `dmk/0008-ragged-suffix` at commit `7b3698d45`. The test has 4 blocks. Each block has one run of each build. The difference is the mean of the 4 block ratios. The interval is the 95 % bootstrap interval of the blocks.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| All 48 cases (`lat_ms`) | 222.498 | 217.022 | -2.46 | -3.80 to -1.46 | 4 | 200 | 1,845 to 1,920 |
| Short requests | 69.386 | 67.993 | -2.00 | -4.43 to -0.19 | 4 | 200 | 1,845 to 1,920 |
| Medium requests | 334.483 | 318.830 | -4.68 | -5.86 to -3.91 | 4 | 200 | 1,845 to 1,920 |
| Long requests | 724.024 | 711.359 | -1.75 | -2.68 to -1.07 | 4 | 200 | 1,845 to 1,920 |

Source: table `speed-bench.tsv` in the [data set of the second test](../../data/2026-10-31-0008-ragged-suffix-retest/README.md).

The next table has the benchmark for 8 requests at the same time and the prompt speed of `llama bench`. A larger number is better.

| Measurement | Before | After | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| Requests/s, 8 at the same time | 4.543 | 4.761 | +4.80 | +4.44 to +5.10 | 4 | 200 | 1,845 to 1,920 |
| Tokens/s, prompt of 64 tokens | 2,970.301 | 2,975.895 | +0.19 | -0.46 to +0.88 | 4 | 200 | 1,845 to 1,920 |
| Tokens/s, prompt of 256 tokens | 4,517.550 | 4,533.622 | +0.36 | +0.12 to +0.66 | 4 | 200 | 1,845 to 1,920 |
| Tokens/s, prompt of 3,000 tokens | 5,042.150 | 5,044.875 | +0.05 | -0.13 to +0.23 | 4 | 200 | 1,845 to 1,920 |

Source: tables `speed-bench.tsv` and `speed-llama-bench.tsv` in the data set of the second test.

The client `latency_by_type.py` runs the phase `seq` and gives the mean latency for each request type. It is a different run, with a different seed.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| Short, 1 question | 45.093 | 45.716 | +1.38 | +1.11 to +1.80 | 4 | 200 | 1,845 to 1,920 |
| Short, 2 questions | 82.273 | 77.442 | -5.87 | -6.17 to -5.58 | 4 | 200 | 1,845 to 1,920 |
| Medium requests | 331.600 | 317.310 | -4.31 | -4.43 to -4.11 | 4 | 200 | 1,845 to 1,920 |
| Long requests | 721.279 | 707.660 | -1.89 | -2.03 to -1.75 | 4 | 200 | 1,845 to 1,920 |

Source: table `speed-type.tsv` in the data set of the second test.

The [notebook entry of the result](../notebook/2026-10-31-0008-ragged-suffix-retest-result.md) gives the table of each run, the ratio of each block, and the host load.

## Prediction of the second test

We committed the [prediction entry](../notebook/2026-10-31-0008-ragged-suffix-retest-prediction.md) on 2026-10-31 at 09:19:03 UTC. This is before the first run of the second test.

| Item | Prediction | Result | Same as the prediction |
|---|---|---|---|
| `lat_ms` | -3.3 % (-1.8 % to -4.5 %) | -2.46 % | Yes. The gain is 2.0 % or more. |
| `short_ms` | -3.5 % (-1.5 % to -5.5 %) | -2.00 % | Yes |
| `medium_ms` | -5.0 % (-3.5 % to -6.5 %) | -4.68 % | Yes |
| `long_ms` | -2.0 % (-1.0 % to -3.0 %) | -1.75 % | Yes |
| Short, 2 questions | -8 % (-5 % to -10 %) | -5.87 % | Yes |
| Short, 1 question | 0 % (-1 % to +1 %) | +1.38 % | No |
| `rps8` | +5 % (0 % to +8 %) | +4.80 % | Yes |
| `llama bench` | 0 % (-1 % to +1 %) | +0.05 % to +0.36 % | Yes |
| Limits of the logits and the states, probability 0.65 | All true | The limit of 3 times for one input is not true | No |
| Limits of the answers, probability 0.7 | True | True | Yes |

Source: the [notebook entry of the second result](../notebook/2026-10-31-0008-ragged-suffix-retest-result.md).

## Correctness of the second test

The limits are the limits of the prediction entry of the second test. The reference is the decode of one sequence at a time.

| Test | Limit | Result |
|---|---|---|
| Logits and layer states of 8 inputs, sum of `d(ragged)` | 1.5 times the sum of `d(equal)`, for each of 5 quantities | pass: ratios of 0.78, 1.29, 1.17, 0.92 and 1.08 |
| The same quantities, each input | 3 times `d(equal)` | `FAIL`: 3 of 40 pairs. They are input 7 (largest KL divergence, ratio 3.76), input 8 (top token, 1 against 0) and input 1 (NMSE of the conv state, ratio 4.51). |
| Mean KL divergence of ragged batching, each input | 10 % or less of the KL budget | pass: 2.1e-4 to 1.0e-3 |
| Answers, largest and mean difference of the probability | 1.5 times the value of `A` with `-np 3` against `A` with `-np 1` | pass: 0.019074 against 0.022426, and 0.002036 against 0.002342 |
| Answers, top answer, `self_repeat`, guards | No decision with a different top answer that `A` does not have, true, all true | pass |
| The 52-decision set | The same number of correct decisions | pass: 48 of 52 on the two builds |
| Answers of `B` with `-np 1` against `A` with `-np 1` | The same bit for bit | pass |
| `test-backend-ops`: the new operations, 592 built-in cases, 187 operations of the graph | As build `A` | pass: 18 of 18, 5 of 5, 3 of 3, 592 of 592, and 181 of 187 as on build `A` |
| `compute-sanitizer --tool memcheck`, one request of each type | 0 errors | pass: 0 errors |
| Ubatches with ragged batching in the phase `seq` | More than 0 | pass: 20 |
| `test-batch-alloc` | Not declared. All pass. | pass: 58 tests and 436 checks |

Source: the tables `logits-verdict.tsv`, `answers-verdict.tsv`, `dectest.tsv`, `ops.tsv`, `memcheck.tsv`, `ubatches.tsv` and `unit-tests.tsv` in the data set of the second test.

The result entry has the text about the other changes. The fusion of change 0005 does not apply in a pass with ragged batching, because the graph uses `SSM_CONV_RAGGED`. Passes with the equal split keep it. The attention mask of change 0004 does not use the length of the sequences.

## First test

The first test was on the branch `dmk/stack` at commit `6b8b65e66`, with build `A` as `build:lead-30489`. It gave -3.32 % for `lat_ms` (interval -4.18 % to -2.63 %) against the predicted -6.6 %, and +5.96 % for `rps8`. Some limits of the first [prediction entry](../notebook/2026-10-31-0008-ragged-suffix-prediction.md) were not true. They are the mean KL divergence, the NMSE of the layer states and the largest difference of the probability of an answer (0.0224).

The limits were smaller than the noise of the equal split against a decode of one sequence at a time. The [first result entry](../notebook/2026-10-31-0008-ragged-suffix-result.md) and the [data set of the first test](../../data/2026-10-31-0008-ragged-suffix/README.md) show the first test. The commits of the first test are on the branch `dmk/0008-ragged-suffix-v1` of the fork of llama.cpp of this project.

The census of the first test tells the cause of the gain. The phase `seq` of the benchmark has 20 ubatches with ragged batching, and the change removes 23 passes in one pass of 48 requests. The client pass has 32 such ubatches. The second test gives 20 ubatches again, with 17 of 2 sequences and 3 of 3 sequences.

## Follow-up: start all child tasks together

The server starts the questions of a request as child tasks. The log shows the line `waiting for parent slot to complete` for the third question. A change of the server could start all child tasks together. It can remove 1 more pass in at most 13 of the 16 requests with 3 questions of the benchmark. [INFERENCE: this adds 1 % to 2 % of `lat_ms` at most.] We did not make this change.

## Reproduce

1. Get the code from the branch `dmk/0008-ragged-suffix` of the fork of llama.cpp of this project. The commit before the change is `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978` of the branch `dmk/stack`. The change has four commits.

```
git fetch <fork> dmk/0008-ragged-suffix
git checkout 7b3698d452ca8a219b581974a32f3ae6c7456c54
```

| Commit | Contents |
|---|---|
| `f210825e6ffb9114710b1ec32d01c2f755f8b491` | The change: two new operations, `split_equal`, the hybrid memory, the graph of the model, and the tests |
| `93d51f9e24a7e0fc4073b777ba61862a14982739` | A log line for each ubatch with suffixes of different lengths. `test-ragged-suffix` gets its limits from the KL floor. |
| `e34f9dedef9b0784a85b827dbb989da978177dcd` | `test-ragged-suffix` writes the NMSE of the states for each recurrent layer |
| `7b3698d452ca8a219b581974a32f3ae6c7456c54` | `test-ragged-suffix` compares ragged batching and the equal split with a decode of one sequence at a time |

2. Make build `A` from the commit `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978`. Make build `B` from the commit `7b3698d452ca8a219b581974a32f3ae6c7456c54`. Use `harness/build/build.sh`. It uses the builder image of the project. The `cmake` flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

3. Make the three schedules. The seeds are 80090, 80091 and 80092.

```
python3 harness/make_schedule.py --prefix ragged0008r-bench --blocks 4 --seed 80090 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix ragged0008r-llb --blocks 4 --seed 80091 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
python3 harness/make_schedule.py --prefix ragged0008r-type --blocks 4 --seed 80092 \
    --arm A=build:stack-fbfd9cb --arm B=build:0008-ragged-rebased --kind tool \
    --args "/extra/0008/latency_by_type.py 18080 5"
```

4. Run the schedules and make the tables. The [data set](../../data/2026-10-31-0008-ragged-suffix-retest/README.md) gives the commands, and the commands of the tests of correctness.

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
python3 data/2026-10-31-0008-ragged-suffix-retest/make_tables.py
```

The server flags of the benchmark are `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`. The environment variable `LLAMA_RAGGED_SPLIT=0` selects the equal split. The environment variable `LLAMA_RAGGED_STEP` sets the limit for the step between the lengths. Its default is 128.

## Limits

- **Where it applies.** A ubatch uses ragged batching when all of these conditions are true:
  - The model is a hybrid model with Gated DeltaNet layers, and the graph is the graph of `qwen35`. All other models use the equal split.
  - Each set has one sequence, and the lengths are different.
  - All tokens fit in the ubatch.
  - The memory has no snapshots (`n_rs_seq` is 0), and the server uses `--kv-unified`.
  - Each step between the lengths, from 0 to the longest length, is 128 tokens or less. The kernel of Gated DeltaNet runs one token at a time, and this limit keeps the time small.
  - The backend of each recurrent layer has the two new operations. Only the CPU backend and the CUDA backend have them. The memory tests this when it starts.
- **Where it applies in the benchmark.** The phase `seq` has 20 ubatches with ragged batching in one pass. The benchmark has 36 requests with 2 or 3 questions.
- **Server.** The third question of a request can start after the other two. This limits the gain. A change of the server can start all questions together. We did not make it.
- **Numbers.** The limit of 3 times for one input is not true for 3 of 40 pairs. The sums, the KL budget and the answers are in their limits. The change stays out of `dmk/stack`. A new limit must have a new prediction and new inputs.
- **Short requests with 1 question.** They are 1.38 % slower on build `B`. The change does not apply to them. We do not know the cause.
- **Two measurements for short requests with 2 questions.** The first test gave -6.7 % in the client and -3.9 % in the benchmark. The census gave 32 and 20 ubatches. We do not know the cause of the different flow of the requests.
- **Host load.** The load of the host was 1.91 to 18.34 in the private benchmark of the second test. The sign of the difference of `lat_ms` is the same in all 4 blocks.
- **Not in the tests.** Other models, speculative decoding (`n_rs_seq` more than 0), and more than one GPU.
- **Operations.** The change adds two operations. On `dmk/stack`, `GGML_OP_COUNT` is 105 and `RPC_PROTO_PATCH_VERSION` is 3. Other changes that add operations change the same lines of `ggml.h` and `ggml-rpc.h`.
- **Other changes.** Changes 0002 to 0005 change the same files. A merge can give different lines in `ggml-cuda.cu`, in `qwen35.cpp` and in the graph code.
- **Go back to the baseline.** There are two methods:
  - Set `LLAMA_RAGGED_SPLIT=0`. The program then uses the equal split, with the new code in the build.
  - Use `git revert f210825e6ffb9114710b1ec32d01c2f755f8b491` on a branch that has this change.

## Data

- [Second test: speed, tests of correctness and census of the ubatches](../../data/2026-10-31-0008-ragged-suffix-retest/README.md)
- [Notebook entry of the prediction of the second test](../notebook/2026-10-31-0008-ragged-suffix-retest-prediction.md)
- [Notebook entry of the result of the second test](../notebook/2026-10-31-0008-ragged-suffix-retest-result.md)
- [First test: data set](../../data/2026-10-31-0008-ragged-suffix/README.md)
- [Notebook entry of the first prediction](../notebook/2026-10-31-0008-ragged-suffix-prediction.md)
- [Notebook entry of the first result](../notebook/2026-10-31-0008-ragged-suffix-result.md)
