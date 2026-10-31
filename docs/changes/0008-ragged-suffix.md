# Ragged batching of the question suffixes

## Summary

A request with 2 or 3 questions has one suffix for each question, and the suffixes have different lengths. The equal split of the hybrid memory makes one pass for each different length, and each pass reads all weights again. The change puts all suffixes in one pass. It adds two operations for the recurrent layers: `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`. The program uses the equal split when a condition of the change is not true.

## Equivalence class

`re-associated`. We selected this equivalence class on 2026-10-31, before the first run. A pass with more tokens uses other matrix-multiply kernels: MMQ for more than 8 tokens, and MMVQ for 8 tokens or less. The flash attention kernel can also be different. The sums then occur in a different sequence, and the last bits of the results can be different.

## Results

Build `A` is `build:lead-30489`, the branch `dmk/stack`. Build `B` is `build:0008-ragged-suffix`, the branch `dmk/0008-ragged-suffix`. The test has 4 blocks. Each block has one run of each build. The difference is the mean of the 4 block ratios. The interval is the 95 % bootstrap interval of the blocks.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| All 48 cases (`lat_ms`) | 235.894 | 228.058 | -3.32 | -4.18 to -2.63 | 4 | 200 | 1,845 to 1,860 |
| Short requests | 72.224 | 69.395 | -3.91 | -5.06 to -3.04 | 4 | 200 | 1,845 to 1,860 |
| Medium requests | 354.685 | 335.332 | -5.46 | -6.08 to -4.98 | 4 | 200 | 1,845 to 1,860 |
| Long requests | 771.258 | 755.304 | -2.07 | -2.63 to -1.67 | 4 | 200 | 1,845 to 1,860 |

Source: table `speed-bench.tsv` in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md). The clock is the median SM clock of each run.

The next table has the numbers of the benchmark for 8 requests at the same time, and the prompt speed of `llama bench`. A larger number is better.

| Measurement | Before | After | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| Requests/s, 8 at the same time | 4.280 | 4.535 | +5.96 | +5.60 to +6.31 | 4 | 200 | 1,845 to 1,860 |
| Tokens/s, prompt of 64 tokens | 2,834.659 | 2,829.112 | -0.20 | -0.58 to +0.19 | 4 | 200 | 1,860 to 1,905 |
| Tokens/s, prompt of 256 tokens | 4,309.650 | 4,296.765 | -0.30 | -0.44 to -0.15 | 4 | 200 | 1,860 to 1,905 |
| Tokens/s, prompt of 3,000 tokens | 4,703.247 | 4,699.538 | -0.08 | -0.55 to +0.29 | 4 | 200 | 1,860 to 1,905 |

Source: tables `speed-bench.tsv` and `speed-llama-bench.tsv` in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md).

The benchmark does not give the short requests with one question and the short requests with two questions as different groups. The client `latency_by_type.py` runs the phase `seq` and gives the mean latency for each type. It is a different run, with a different seed.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| Short, 1 question | 46.852 | 46.890 | +0.09 | -0.62 to +0.80 | 4 | 200 | 1,845 to 1,875 |
| Short, 2 questions | 85.903 | 78.710 | -8.37 | -8.90 to -7.94 | 4 | 200 | 1,845 to 1,875 |
| Medium requests | 351.971 | 333.051 | -5.37 | -5.74 to -5.01 | 4 | 200 | 1,845 to 1,875 |
| Long requests | 768.208 | 748.626 | -2.55 | -2.78 to -2.32 | 4 | 200 | 1,845 to 1,875 |

Source: table `speed-type.tsv` in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md).

The [notebook entry](../notebook/2026-10-31-0008-ragged-suffix-result.md) gives the table of each run, the ratio of each block, and the host load.

## Prediction

We committed the [prediction entry](../notebook/2026-10-31-0008-ragged-suffix-prediction.md) on 2026-10-31 at 07:02:58 UTC. This is before the first run of build `B`. The gain is about 50 % of the predicted gain. Some correctness tests are not in the tolerances that we declared.

| Item | Prediction | Result | Same as the prediction |
|---|---|---|---|
| Equivalence class | `re-associated` | The answers are not the same bit for bit. The largest probability difference is 0.0224 on 100 decisions. | Yes |
| `lat_ms` | -6.6 % | -3.32 % (interval -4.18 % to -2.63 %) | No. The gain is 50 % of the predicted gain. It is not less than the limit of 3.3 %. |
| `short_ms` | -10.1 % | -3.91 % | No. The gain is 39 % of the predicted gain. |
| `medium_ms` | -9.1 % | -5.46 % | No. The gain is 60 % of the predicted gain. |
| `long_ms` | -4.1 % | -2.07 % | No. The gain is 50 % of the predicted gain. |
| Short, 2 questions | -13.7 % | -8.37 % | No. The gain is 61 % of the predicted gain. |
| Short, 1 question | 0 % | +0.09 % | Yes |
| `rps8` | 0 % to +6.6 % | +5.96 % | Yes |
| `llama bench` | A change of 1 % or less | -0.08 % to -0.30 % | Yes |
| Ubatches with ragged batching in the phase `seq` | More than 0 | 20 in one pass | Yes |
| Confidence of 0.45 that the tests are in their tolerances and the gain is 50 % or more | | The gain is 50.3 % of the predicted gain. Some tests are not in their tolerances. | No |

Source: the [notebook entry of the result](../notebook/2026-10-31-0008-ragged-suffix-result.md).

The cause of the smaller gain is the number of ubatches with suffixes of different lengths. The prediction used 36 requests with 2 or 3 questions in the phase `seq`. It removed 52 passes in one pass. The pass of the benchmark with the log has 20 such ubatches, and the change removes 23 passes at most. The pass of the client `latency_by_type.py` with the log has 32 such ubatches, and the change removes 39 passes at most.

The log of the server shows one cause of the small number of ubatches with 3 sequences. The third question of a request can wait. The log line is `waiting for parent slot to complete`. Ragged batching cannot attach the third question to the other two.

The gain for each pass that the change removes is near the prediction in the client. The client removes 451 ms in one pass of 48 requests. This is 11.6 ms or more for each pass that the change removes. The prediction used 10.4 ms to 11.5 ms. The benchmark removes 376 ms, or 16.4 ms or more for each pass.

[INFERENCE: the speed test has a different number of ubatches than the passes with the log. The log of level 5 writes the prompts and can change the time of each task of the server.]

The short requests give -6.7 % in the client and -3.9 % in the benchmark. The client has more ubatches with ragged batching, and this agrees with its larger gain. We do not know the cause of the different flow of the requests.

## Correctness

Each result has a source in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md). The threshold is the one in the prediction entry.

| Test | Threshold | Result |
|---|---|---|
| `test-backend-ops`, `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`, against the CPU | NMSE of 5e-4 or less. All cases are in the tolerance. | pass: 18 of 18 cases, 5 of 5 cases, and 3 of 3 cases with the fusion into `CPY` |
| `test-backend-ops`, 187 operations of the graph of mix8 | 181 of 187 pass. The same 6 give `FAIL`. | pass: 181 of 187. The 6 are 5 `CPY` and 1 `GATED_DELTA_NET`, as on build `A`. |
| `test-backend-ops`, 390 built-in cases of the five operations of the layer | Not declared. All pass. | pass: 390 of 390 |
| Layer states, ragged batching against the equal split, 24 layers | NMSE of 5e-4 or less for each layer | `FAIL`: the largest NMSE is 1.45e-2 (conv state) and 1.73e-3 (SSM state) |
| Logits, mean KL divergence | 1e-5 or less | `FAIL` in 4 cases: 3.9e-4, 7.3e-4, 1.2e-3 and 1.8e-3 |
| Logits, largest KL divergence | 1e-3 or less | `FAIL` in 2 of 4 cases: 2.6e-3 and 3.5e-2. The other 2 cases give 6.8e-4 and 7.4e-4. |
| Logits, top token | The same in all rows | `FAIL` in 1 of 4 cases: 161 of 163 tokens. The other 3 cases have the same top token in all rows. |
| `llama perplexity`, mean KL divergence against the base file of build `A` | The KL floor plus 1e-5 | pass: 0.000000 on the two texts, as the KL floor |
| Answers, `self_repeat` | True for the two builds | pass |
| Answers, largest probability difference between the builds | 0.01 or less | `FAIL`: 0.0224 on 100 decisions, and 0.0416 on the 52-decision set |
| Answers, top answer | No decision has a different top answer | pass: 0 decisions in the two sets |
| Guards of the benchmark | All true in all runs | pass: all guards true in 11 runs, with 94 correct decisions of 100 |
| The 52-decision set | 48 correct, and the same 4 incorrect | pass: 48 of 52 on the two builds, with the same 4 incorrect decisions |
| `compute-sanitizer --tool memcheck`, one request of each type | 0 errors | pass: 0 errors |
| Ubatches with ragged batching in the phase `seq` | More than 0 | pass: 20 |
| `test-batch-alloc`, the unit test of `split_equal` | Not declared. All pass. | pass: 58 tests and 436 checks |

Source: the tables `ops-tests.tsv`, `suffix-compare.tsv`, `suffix-checks.tsv`, `kl-perplexity.tsv`, `quality-bench.tsv`, `check-runs.tsv`, `decisions-52.tsv`, `memcheck.tsv`, `unit-tests.tsv` and `ragged-ubatches.tsv`, and the files `answers-compare-*.json`, in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md).

The prediction declared that the conv states are the same bit for bit. This is not true. The 5 results with `FAIL` come from a difference of the kernels in the matrix multiplies. A control in the test program `test-ragged-suffix` shows this. The equal split gives a difference of the same range against a decode of one sequence at a time.

| Case (lengths) | Mean KL divergence: ragged batching against the equal split | Equal split against one sequence at a time | Ragged batching against one sequence at a time |
|---|---:|---:|---:|
| 43, 47, 73 | 3.9e-4 | 1.0e-3 | 1.0e-3 |
| 51, 120 | 7.3e-4 | 6.6e-4 | 6.1e-4 |
| 2, 33, 2, 64, 31 | 1.2e-3 | 1.4e-3 | 2.1e-3 |
| 43, 47, 73, all 163 logits | 1.8e-3 | 6.5e-3 | 3.9e-3 |

For the lengths 51 and 120, the NMSE of the layer state at layer 0 is 1.1e-14 (conv state) and 6.0e-12 (SSM state). This shows that the two new operations give the same values as the operations of build `A`. [INFERENCE: for the other lengths, the equal split makes a pass of 8 tokens or less, and this pass uses MMVQ. We did not run a test that selects one kernel.]

The declared limits were too small for a change of the contents of a batch. A decision on the limits is necessary before a merge.

## Reproduce

1. Get the code from the branch `dmk/0008-ragged-suffix` of the fork of llama.cpp of this project. The commit before the change is `6b8b65e6672f07805000d3c9683b65c72cc62303` of the branch `dmk/stack`. The change has three commits.

```
git fetch <fork> dmk/0008-ragged-suffix
git checkout 82eb3a931a7dca82388a847dfc79f77746280b4b
```

| Commit | Contents |
|---|---|
| `c175584ef1c7ef486ce2ccb1b6698daad233756d` | The change: two new operations, `split_equal`, the hybrid memory, the graph of the model, and the tests |
| `766629ebcb235a12aae73d3f8db4215342c17b16` | A log line for each ubatch with suffixes of different lengths. `test-ragged-suffix` gets its limits from the KL floor. |
| `82eb3a931a7dca82388a847dfc79f77746280b4b` | `test-ragged-suffix` writes the NMSE of the states for each recurrent layer |

2. Make build `A` from the commit `6b8b65e6672f07805000d3c9683b65c72cc62303`. Make build `B` from the commit `82eb3a931a7dca82388a847dfc79f77746280b4b`. Use `harness/build/build.sh`. It uses the builder image of the project. The `cmake` flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

3. Make the three schedules. The seeds are 80080, 80081 and 80082.

```
python3 harness/make_schedule.py --prefix ragged0008-bench --blocks 4 --seed 80080 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix ragged0008-llb --blocks 4 --seed 80081 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
python3 harness/make_schedule.py --prefix ragged0008-type --blocks 4 --seed 80082 \
    --arm A=build:lead-30489 --arm B=build:0008-ragged-suffix --kind tool \
    --args "/extra/0008/latency_by_type.py 18080 5"
```

4. Run the schedules and make the tables. The [data set](../../data/2026-10-31-0008-ragged-suffix/README.md) gives the commands, and the commands of the tests of correctness.

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
python3 data/2026-10-31-0008-ragged-suffix/make_tables.py
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
- **Numbers.** The tests of the layer states, the logits and the answers are not in the declared limits. The answers of 100 decisions have no different top answer. A decision on the limits is necessary before a merge.
- **Two measurements for short requests.** The client and the benchmark give -6.7 % and -3.9 %. The census gives 32 and 20 ubatches. We do not know the cause of the different flow of the requests.
- **Host load.** The load of the host was 1.65 to 17.04 in the private benchmark. The sign of the difference of `lat_ms` is the same in all 4 blocks.
- **Not in the tests.** Other models, speculative decoding (`n_rs_seq` more than 0), and more than one GPU.
- **Operations.** The change adds two operations. Thus `GGML_OP_COUNT` is 104, and `RPC_PROTO_PATCH_VERSION` is 2. Other changes that add operations change the same lines of `ggml-rpc.h`.
- **Other changes.** Changes 0002 to 0005 change the same files. A merge can give different lines in `ggml-cuda.cu`, in `qwen35.cpp` and in the graph code.
- **Go back to the baseline.** There are two methods:
  - Set `LLAMA_RAGGED_SPLIT=0`. The program then uses the equal split, with the new code in the build.
  - Use `git revert c175584ef1c7ef486ce2ccb1b6698daad233756d` on a branch that has this change.

## Data

- [Speed test, tests of correctness and census of the ubatches](../../data/2026-10-31-0008-ragged-suffix/README.md)
- [Notebook entry of the prediction](../notebook/2026-10-31-0008-ragged-suffix-prediction.md)
- [Notebook entry of the result](../notebook/2026-10-31-0008-ragged-suffix-result.md)
