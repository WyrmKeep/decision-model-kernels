# Change 0008: ragged batching of the suffixes, second test (result)

Written 2026-10-31, after the second test.

## Result

One limit of the [prediction entry](2026-10-31-0008-ragged-suffix-retest-prediction.md) is not true for build `B`. It is the limit of 3 times for one input. 3 of 40 pairs of an input and a quantity are above it. All other limits are true. The rule for this test is that the change stays out of `dmk/stack` when a limit is not true. Change 0008 stays out.

The speed gain is -2.46 % for `lat_ms`. This is less than the predicted -3.3 %, and it is in the predicted range of -1.8 % to -4.5 %. The interval of the private benchmark is -3.80 % to -1.46 %. It does not include 0.

## Task of this entry

This entry gives the result of the second test of change 0008. It compares the result with the prediction that we committed before the first run. The [data set](../../data/2026-10-31-0008-ragged-suffix-retest/README.md) has the raw files and the tables. We did not change a limit or a prediction after the first run.

## Prediction, builds and dates

| Item | Value |
|---|---|
| Commit of the prediction entry | `1f1a5cd7afb3fc7f0d08518f5530dcfd04e82f1b`, 2026-10-31 09:19:03 UTC |
| Arm `A` | `build:stack-fbfd9cb`, the branch `dmk/stack` at commit `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978` |
| Arm `B` | `build:0008-ragged-rebased`, the branch `dmk/0008-ragged-suffix` at commit `7b3698d452ca8a219b581974a32f3ae6c7456c54` |
| Build of arm `B` | 2026-10-31 09:21:20 UTC, after the prediction |
| Correctness measurement window | 09:22:50 to 09:36:02 UTC |
| Speed measurement window | 09:36:02 to 09:59:32 UTC |

The first test is in the [first result entry](2026-10-31-0008-ragged-suffix-result.md). Its commits are on the branch `dmk/0008-ragged-suffix-v1` of the fork of llama.cpp of this project. The code of the change is the same code, moved onto `dmk/stack`. Two files had different lines: `ggml.c` and `ggml-rpc.h`. The number of operations is 105, and `RPC_PROTO_PATCH_VERSION` is 3.

## Limits for the decision

| Test | Limit | Result |
|---|---|---|
| Logits and layer states of the 8 inputs, sum of `d(ragged)` against 1.5 times the sum of `d(equal)` | Each of the 5 quantities | `pass`: the ratios are 0.78, 1.29, 1.17, 0.92 and 1.08 |
| The same quantities, each input, 3 times | `d(ragged)` is 3 times `d(equal)` or less | `FAIL`: 3 of 40 pairs. They are input 7 (largest KL divergence), input 8 (top token) and input 1 (NMSE of the conv state). |
| Mean KL divergence of ragged batching against the reference, each input | 10 % or less of the KL budget | `pass`: 2.1e-4 to 1.0e-3 against 0.002783 and 0.005759 |
| Answers, largest difference of the probability, `B` with `-np 3` against `A` with `-np 1` | 1.5 times the value of `A` with `-np 3` against `A` with `-np 1` | `pass`: 0.019074 against 0.022426, ratio 0.85 |
| Answers, mean difference of the probability | 1.5 times, as above | `pass`: 0.002036 against 0.002342, ratio 0.87 |
| Answers, top answer | No decision that `A` with `-np 3` does not also have | `pass`: no decision in the two pairs |
| Answers, `self_repeat` of arm `B` | True | `pass` |
| The 52-decision set, build `B` against build `A` | The same number of correct decisions | `pass`: 48 of 52 on the two builds. The same 4 decisions are incorrect. No item has a different top answer. |
| Guards of the benchmark | All true in all runs | `pass`: 15 runs, all true, 0 errors |

Source: the tables `logits-verdict.tsv`, `answers-verdict.tsv`, `dectest.tsv` and `dectest-items.tsv` in the [data set](../../data/2026-10-31-0008-ragged-suffix-retest/README.md).

The three limits of `d(ragged)` use the decode of one sequence at a time as the reference. We accept build `B` when it is not more than 1.5 times as far from the reference as the equal split of production. The cause of this rule is in the prediction entry.

## Logits and states of the 8 inputs

The next table has the sums for the 8 inputs. A smaller number is better. The limit for the ratio is 1.5.

| Quantity | Sum of `d(ragged)` | Sum of `d(equal)` | Ratio | Result |
|---|---:|---:|---:|---|
| Mean KL divergence | 5.96e-3 | 7.68e-3 | 0.78 | `pass` |
| Largest KL divergence | 6.12e-2 | 4.74e-2 | 1.29 | `pass` |
| Rows with a different top token | 7 | 6 | 1.17 | `pass` |
| NMSE of the conv state | 4.90e-3 | 5.31e-3 | 0.92 | `pass` |
| NMSE of the SSM state | 2.31e-3 | 2.14e-3 | 1.08 | `pass` |

Source: the table `logits-verdict.tsv`. The table `logits-inputs.tsv` has the values of each input.

The three pairs above the limit of 3 times are in the next table.

| Input | Quantity | `d(ragged)` | `d(equal)` | Ratio |
|---|---|---:|---:|---:|
| 7: WikiText-2, lengths 33, 36 and 81, prefix 250, 150 rows | Largest KL divergence | 3.55e-2 | 9.43e-3 | 3.76 |
| 8: WikiText-2, lengths 61, 64 and 99, prefix 3000, 3 rows | Rows with a different top token | 1 | 0 | The limit is 0 |
| 1: WikiText-2, lengths 38, 66 and 102, prefix 800, 3 rows | NMSE of the conv state | 1.44e-3 | 3.20e-4 | 4.51 |

For input 7, the mean KL divergence of the 150 rows is 8.1e-4 for ragged batching and 8.5e-4 for the equal split. The largest value is one row. For input 8, one of 3 rows has a different top token. The equal split has the same top token in the 3 rows.

For input 1, the NMSE of the conv state of ragged batching becomes larger from layer 17 to layer 22. At layer 22 it is 1.44e-3. The equal split stays at 2e-4 to 3e-4 in the layers from 2 to 30.

[INFERENCE: each of the three quantities is the largest value of a set of rows or layers, or a number of 0 or 1. A quantity of this type changes more from one input to the next than a sum does. The cause is not in the data.]

The information-only rules of the first rules give the next numbers of the 8 inputs. A value of `d(ragged vs equal)` is 1 times `d(equal)` or less, 2 times or less, or more.

| Quantity | 1 time or less | 2 times or less | More |
|---|---:|---:|---:|
| Mean KL divergence | 6 | 1 | 1 |
| Largest KL divergence | 6 | 1 | 1 |
| Rows with a different top token | 7 | 0 | 1 |
| NMSE of the conv state | 4 | 3 | 1 |
| NMSE of the SSM state | 2 | 6 | 0 |

The other tests of the 8 inputs are `pass` for all 8 inputs. Ragged batching decodes the suffixes in one pass. The equal split has more than one pass. A second run of ragged batching gives the same logits and states bit for bit. The floor of the equal split is 0.

## Answers of the benchmark

The next table has the 100 decisions of the private benchmark, in pairs of runs. A decision has one answer field. The difference of the probability of a decision is the largest value in the answer keys.

| Pair | Same bit for bit | Largest difference | Mean difference | Decisions with a different top answer |
|---|---|---:|---:|---:|
| `A` with `-np 3`, run a against run b | Yes | 0 | 0 | 0 |
| `B` with `-np 3`, run a against run b | Yes | 0 | 0 | 0 |
| `A` with `-np 3` against `A` with `-np 1` | No | 0.022426 | 0.002342 | 0 |
| `B` with `-np 3` against `A` with `-np 1` | No | 0.019074 | 0.002036 | 0 |
| `B` with `-np 1` against `A` with `-np 1` | Yes | 0 | 0 | 0 |
| `B` against `A`, with `-np 3` for the two | No | 0.022434 | 0.001101 | 0 |

Source: the table `answers-compare.tsv`. The two runs of each arm with `-np 3` give the same values.

With one slot, ragged batching does not apply, and the answers of `B` are the same bit for bit as the answers of `A`. This shows that the new code does not change the other passes. It includes the code of the changes 0004 and 0005 in the passes with the equal split.

## Other tests

| Test | Result |
|---|---|
| `test-backend-ops`, `SSM_CONV_RAGGED` | `pass`: 18 of 18 cases |
| `test-backend-ops`, `GATED_DELTA_NET_RAGGED` | `pass`: 5 of 5 cases, and 3 of 3 cases with the fusion into `CPY` |
| `test-backend-ops`, built-in cases of `GATED_DELTA_NET`, `CPY`, `SSM_CONV`, `KQ_MASK` and `CONCAT` | `pass`: 592 of 592 cases on `A` and on `B`. The two logs are the same. |
| `test-backend-ops`, 187 operations of the graph | `pass`: 181 of 187 on `A` and on `B`. The same 6 cases give `FAIL`: 5 `CPY` and 1 `GATED_DELTA_NET`. |
| `test-batch-alloc` | `pass`: 58 tests and 436 checks |
| `compute-sanitizer --tool memcheck`, one request of each type | `pass`: 0 errors |
| Ubatches with ragged batching in the phase `seq` | `pass`: 20, with 17 of 2 sequences and 3 of 3 sequences |

The file of the first test with the 187 operations stopped the two builds with exit code 139. The cause is in the numbers of the operations. Change 0004 added the operation `KQ_MASK` at position 92, and the numbers above 91 moved by 1. We added 1 to each number of 92 or more in 26 lines, and we ran the test again. The two builds then give 181 of 187.

## Speed

Arm `A` is `build:stack-fbfd9cb`. Arm `B` is `build:0008-ragged-rebased`. The test has 4 blocks. Each block has one run of each arm, in a random sequence.

The difference is the mean of the 4 block ratios. The interval is the 95 % bootstrap interval. The power limit was 200 W, and the median SM clock was 1,845 to 1,920 MHz.

| Metric | Predicted (%) | Predicted range (%) | Result (%) | Interval (%) | In the range |
|---|---:|---|---:|---|---|
| `lat_ms` | -3.3 | -1.8 to -4.5 | -2.46 | -3.80 to -1.46 | Yes |
| `short_ms` | -3.5 | -1.5 to -5.5 | -2.00 | -4.43 to -0.19 | Yes |
| `medium_ms` | -5.0 | -3.5 to -6.5 | -4.68 | -5.86 to -3.91 | Yes |
| `long_ms` | -2.0 | -1.0 to -3.0 | -1.75 | -2.68 to -1.07 | Yes |
| Short, 2 questions (client) | -8 | -5 to -10 | -5.87 | -6.17 to -5.58 | Yes |
| Short, 1 question (client) | 0 | -1 to +1 | +1.38 | +1.11 to +1.80 | No |
| `rps8` | +5 | 0 to +8 | +4.80 | +4.44 to +5.10 | Yes |
| `llama bench`, prompt of 64, 256 and 3,000 tokens | 0 | -1 to +1 | +0.19, +0.36 and +0.05 | -0.46 to +0.88, +0.12 to +0.66 and -0.13 to +0.23 | Yes |

Source: the tables `speed-bench.tsv`, `speed-llama-bench.tsv` and `speed-type.tsv`. The client gives -4.31 % for medium requests and -1.89 % for long requests.

The condition for an incorrect prediction was a gain of `lat_ms` of less than 2.0 %, or an interval that includes 0. The gain is 2.46 %, and the interval is -3.80 % to -1.46 %. The prediction of the speed is true. The sign of the difference of `lat_ms` is the same in all 4 blocks: -4.37 %, -2.08 %, -2.16 % and -1.23 %. The host load was 1.91 to 18.34.

The short requests with 1 question are 1.38 % slower on `B`. This is 0.6 ms for each request. The change does not apply to these requests, and the sign is the same in all 4 blocks (+1.07 % to +1.99 %). The cause is not in the data. In the first test the same type gave +0.09 % against a different build.

The census of the private benchmark has 20 ubatches with ragged batching in the phase `seq`, as in the first test. The change removes 23 passes in one pass of 48 requests. The gain of `lat_ms` is 5.48 ms for each request, or 263 ms in one pass of 48 requests. This is 11.4 ms or more for each pass that the change removes.

## Predicted and found

| Item | Prediction | Result | True |
|---|---|---|---|
| All limits for the decision of the logits and the states | True, probability 0.65 | The limit of 3 times for one input is not true | No |
| Mean KL divergence of each input at 10 % of the KL budget or less | True, probability 0.85 | True | Yes |
| Limits of the answers | True, probability 0.7 | True. The largest difference is 0.019 against 0.022. | Yes |
| Largest difference of the probability of `A` with `-np 3` against `A` with `-np 1` | About 0.02 | 0.022 | Yes |
| Gain of `lat_ms` of 2.0 % or more, and an interval without 0 | True, probability 0.65 | -2.46 %, interval -3.80 % to -1.46 % | Yes |

## The changes 0004 and 0005

The fusion of change 0005 looks for the operations `CONCAT`, `CPY`, `SSM_CONV` and `UNARY`. The graph of ragged batching uses `SSM_CONV_RAGGED`, and the fusion does not apply in a pass with ragged batching. Passes with the equal split keep the fusion. The 592 built-in cases of these operations are `pass`, and the answers with `-np 1` are the same bit for bit as the answers of `A`.

The attention mask of change 0004 uses the position and the sequence of each token. It does not think that the sequences have the same length. [INFERENCE: this is from the code. The test with the three questions of the 8 inputs and the answers with `-np 3` do not show an error of the attention mask.]

## Follow-up: start all child tasks together

We did not make this change. It is not part of this test.

The server starts the questions of a request as child tasks. The log of the server shows the line `waiting for parent slot to complete` for the third question of a request. The benchmark has 16 requests with 3 questions in the phase `seq`. The census has only 3 ubatches with 3 sequences. At most 13 of the 16 requests are not in a ubatch with 3 sequences.

A change of the server could start all child tasks of a request together. Then a request with 3 questions can remove 1 more pass. This is at most 13 requests in the benchmark. It is at most 9 requests in the client pass, which has 7 ubatches with 3 sequences.

| Source of the number of passes | Passes removed at most | Gain for each pass | Gain of `lat_ms` |
|---|---:|---:|---|
| Benchmark, this test | 13 | 11.4 ms | 1.4 % |
| Benchmark, first test | 13 | 16.4 ms | 2.0 % |
| Client, first test | 9 | 11.6 ms | 1.0 % |

[INFERENCE: the gain is 1 % to 2 % of `lat_ms` at most. The gain for each pass is the lower limit of the first test and of this test. The number of passes is the largest number. The third question of a request also waits less. We did not measure this.]

## Risks and open items

- The limit of 3 times for one input is not true. A new limit after this result must have a new prediction and new inputs.
- The change is in the equivalence class `re-associated`. A pass with more tokens uses other matrix-multiply kernels, and the last bits change. The answers are not the same bit for bit as the answers of `A` with `-np 1`.
- Short requests with 1 question are 1.38 % slower. We do not know the cause.
- The tests have no case with `n_rs_seq` more than 0, no other model, and no run with more than one GPU.
- The two new operations are in the CPU backend and in the CUDA backend. A model with a different backend uses the equal split.
- The Gated DeltaNet kernel runs one token at a time. The limit of 128 tokens for the step between the lengths keeps the time small.
- The benchmark and the client give different numbers of ubatches with ragged batching in the first test. We do not know the cause.

## Reproduce

1. Make build `A` from commit `fbfd9cb8c4ecfe3d7f098c8a2040fb6682603978`. Make build `B` from commit `7b3698d452ca8a219b581974a32f3ae6c7456c54` of the branch `dmk/0008-ragged-suffix` of the fork of llama.cpp of this project.
2. Make the three speed schedules with `harness/make_schedule.py`. The seeds are 80090 (private benchmark), 80091 (`llama bench`) and 80092 (client `latency_by_type.py`). The [data set](../../data/2026-10-31-0008-ragged-suffix-retest/README.md) has the commands and the schedule files.
3. Run the tests of correctness. The data set has the commands of the 8 inputs of `test-ragged-suffix`, of `test-backend-ops`, of `compute-sanitizer` and of the runs with `-np 3` and `-np 1`.
4. Make the tables with `python3 data/2026-10-31-0008-ragged-suffix-retest/make_tables.py`.
