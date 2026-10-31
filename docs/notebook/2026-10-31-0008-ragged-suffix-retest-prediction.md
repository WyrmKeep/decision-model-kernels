# Change 0008: ragged batching of the suffixes, second test (prediction)

Written 2026-10-31, before the first run of the second test.

## Task of this entry

This entry gives the prediction and the new limits for the second test of change `0008`. The [result entry of the first test](2026-10-31-0008-ragged-suffix-result.md) shows that some limits of the first [prediction](2026-10-31-0008-ragged-suffix-prediction.md) were incorrect. We do not change those limits after we saw the data. We give new limits here, and we use new inputs. We commit this entry before the first run of the second test. We do not change it after that.

## Cause of the incorrect limits

The first prediction had limits for ragged batching against the equal split. The mean KL divergence was 1e-5 or less. The largest KL divergence was 1e-3 or less. The NMSE of the layer states was 5e-4 or less. The difference of the probability of an answer was 0.01 or less.

The control of the first test shows the cause. The equal split is as far from a decode of one sequence at a time as ragged batching is from the equal split.

| Case (lengths) | Mean KL divergence: ragged batching against the equal split | Equal split against one sequence at a time | Largest NMSE of the conv state: equal split against one sequence at a time |
|---|---:|---:|---:|
| 43, 47, 73 | 3.9e-4 | 1.0e-3 | 2.8e-2 |
| 51, 120 | 7.3e-4 | 6.6e-4 | 7.4e-4 |
| 2, 33, 2, 64, 31 | 1.2e-3 | 1.4e-3 | 1.3e-3 |
| 43, 47, 73, all 163 logits | 1.8e-3 | 6.5e-3 | 1.8e-3 |

Source: the table `suffix-compare.tsv` in the [data set of the first test](../../data/2026-10-31-0008-ragged-suffix/README.md).

The limits were smaller than the noise that each change of the contents of a batch gives. The kernels of the matrix multiplies give different sums when a pass has more tokens, and the last bits of the results change. The first prediction also gave the conv states as the same bit for bit. This was not true.

## Change of the test

The code of the change is the code of the first test. We put it onto the branch `dmk/stack` at commit `fbfd9cb8c`. This branch has the tag `b12004` and changes 0001, 0007, 0002 (tests), 0004 (attention mask on the GPU) and 0005 (concat in `ssm_conv`). Two differences occurred in the tables of the operations. The number of operations is 105, and the version of the `RPC` patch is 3.

- Change 0005 fuses the nodes `CONCAT`, `CPY`, `SSM_CONV` and `SILU`. The function that finds them looks for the operation `SSM_CONV`. The graph of ragged batching uses `SSM_CONV_RAGGED`, thus the fusion does not apply in a pass with ragged batching. Passes with the equal split keep the fusion.
- Change 0004 makes the attention mask on the GPU from the position and the sequence of each token. It does not use the length of the sequences of a ubatch.

The test program `test-ragged-suffix` has a new commit. It gives the numbers of this entry for each input. The build for the second test is `build:0008-ragged-rebased`.

## Arms and schedule

| Arm | Build | Meaning |
|---|---|---|
| A | `build:stack-fbfd9cb` | The branch `dmk/stack` at commit `fbfd9cb8c`. |
| B | `build:0008-ragged-rebased` | The branch `dmk/0008-ragged-suffix` after the change to `dmk/stack`. Ragged batching is on. |

The speed test has 4 blocks of the private benchmark with seed 80090. It has 4 blocks of `llama bench` with seed 80091, and 4 blocks of the client `latency_by_type.py` with seed 80092. A block has one run of each arm in a random sequence. The estimate is the mean of the 4 ratios of arm B to arm A. The interval is the 95 % bootstrap interval with 10,000 samples (`harness/analysis/ab_stats.py`). We do not give an equivalence margin.

## Inputs of the test of the logits

The inputs are new. The first test used the text of the states of the benchmark. It used the prefixes of 60, 300, 600 and 1000 tokens. The lengths were 43, 47, 73, 51, 120, 2, 33, 2, 64 and 31. The second test uses WikiText-2 (new text) and the text of the states with new prefixes and new lengths.

| Input | Text | Suffix lengths | Prefix (tokens) | Passes of the equal split | Pass of 8 tokens or less | Logits |
|---|---|---|---:|---:|---|---|
| 1 | WikiText-2 | 38, 66, 102 | 800 | 3 | No | Last token of each suffix |
| 2 | WikiText-2 | 29, 31, 90 | 1500 | 3 | Yes (4 tokens) | Last token of each suffix |
| 3 | WikiText-2 | 77, 130 | 40 | 2 | No | Last token of each suffix |
| 4 | WikiText-2 | 52, 118 | 120 | 2 | No | Last token of each suffix |
| 5 | WikiText-2 | 4, 21, 9, 55, 26 | 500 | 5 | No | Last token of each suffix |
| 6 | Text of the states | 35, 60, 111 | 1800 | 3 | No | All 206 tokens |
| 7 | WikiText-2 | 33, 36, 81 | 250 | 3 | Yes (6 tokens) | All 150 tokens |
| 8 | WikiText-2 | 61, 64, 99 | 3000 | 3 | Yes (6 tokens) | Last token of each suffix |

A pass of 8 tokens or less uses MMVQ in the equal split. A longer pass uses MMQ. Ragged batching and a decode of one sequence use MMQ for all tokens.

## Limits

### Reference and limits

The two builds go against the same reference: a decode of one sequence at a time. It is the correct result for a request. The equal split of production is not the same as it. A test that compares ragged batching with the equal split is incorrect. Two changes of the same dimension from the reference are 2 times as far from each other in a KL divergence or a NMSE. One change is 1 time as far.

We accept build B when `d(ragged)` is not more than `d(equal)`, with a limit of 1.5 times for the sum of the 8 inputs. The test has only 8 inputs, thus it has noise. The limit of 1.5 times lets the noise of the sum in. The limit of 3 times for one input lets the noise of one input in. It shows an input that the sum does not show.

We write `d(equal)` for a quantity of the equal split against the reference. We write `d(ragged)` for the same quantity of ragged batching against the reference. The test program `test-ragged-suffix` gives the two for each input.

### Limits for the decision

These limits give the decision. For a KL divergence and a NMSE, the value of an input has three parts. They are its mean KL divergence, its largest KL divergence, and its largest NMSE in the layers and the sequences. For the top token, the value of an input is the number of rows with a different top token.

| Test | Limit |
|---|---|
| Logits and layer states of the 8 inputs: mean KL divergence, largest KL divergence, rows with a different top token, NMSE of the conv state, NMSE of the SSM state | For each quantity, the sum of `d(ragged)` for the 8 inputs is 1.5 times the sum of `d(equal)` or less. For each input and each quantity, `d(ragged)` is 3 times `d(equal)` or less. When `d(equal)` is 0, the limit of the input is 0. |
| Mean KL divergence of ragged batching against the reference | 10 % or less of the KL budget of the text, for each input. The limit is 0.002783 for WikiText-2 and 0.005759 for the text of the states. |
| Answers of the benchmark, 100 decisions: the largest and the mean difference of the probability. Build B with `-np 3` against build A with `-np 1`. | Each value is 1.5 times or less the same value of build A with `-np 3` against build A with `-np 1`. |
| Answers, top answer | No decision has a different top answer in build B with `-np 3` against build A with `-np 1`. The pair of build A with `-np 3` and build A with `-np 1` can have the decision. We write each decision. |
| Answers, `self_repeat` of arm B | True |
| The 52-decision set, build B against build A | The same number of correct decisions. We write each difference of an item. |
| Guards of the benchmark | All true in all runs |

The difference of the probability of a decision is the largest value of the difference of the probability in the answer keys of the decision. This is the value that `answers_compare.py` uses. The largest difference is the largest value in the 100 decisions. The mean difference is the mean value of the 100 decisions. With `-np 1`, the questions of a request run one at a time. This is the decode of one sequence at a time.

These tests also give the decision. They are the same as in the first test.

| Test | Limit |
|---|---|
| `test-backend-ops`, `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`, against the CPU | NMSE of 5e-4 or less. All cases are in the tolerance. |
| `test-backend-ops`, the fusion cases of change 0005, the `KQ_MASK` cases of change 0004, and the fusion of `GATED_DELTA_NET_RAGGED` with `CPY` | All cases are in the tolerance, as on build A. |
| `test-backend-ops`, 187 operations of the graph of mix8 | The same result for each test as build A. |
| A second run of ragged batching in the test of the logits | The same logits and the same layer states bit for bit |
| Ragged batching decodes the suffixes in one pass. The equal split has more than one pass. | True for each input |
| Answers, `self_repeat` of arm A | True |
| Answers, build B with `-np 1` against build A with `-np 1` | The same bit for bit. Ragged batching does not apply with one slot. |
| `compute-sanitizer --tool memcheck` on one request of each type, build B | 0 errors |
| Ubatches with ragged batching in the phase `seq` | More than 0 |

### Information only

We also give the first rules of this test. They do not give the decision. For each input and each of the 5 quantities, we give the ratio of ragged batching against the equal split to `d(equal)`. We write if the ratio is 1 or less, or 2 or less.

## Predicted result

The prediction of the logits starts from the 4 inputs of the first test. For these inputs, the limits for the decision are true. The ratio of the sums is 0.79 for the mean KL divergence and 0.41 for the largest KL divergence. It is 0.29 for the NMSE of the conv state and 0.43 for the NMSE of the SSM state. The number of rows with a different top token is 4 for the equal split and 4 for ragged batching.

The largest ratio of one input is 2.0. It is the NMSE of the conv state for the lengths 51 and 120. [INFERENCE: ragged batching and the decode of one sequence use MMQ for all tokens, thus ragged batching is near the decode of one sequence.]

We predict that all limits of the logits are true, with a probability of 0.65. The risk is the limit of one input. An input with a small number of rows has a top-token count of 0 or 1, and the limit is then 0 or 3. The limit of the mean KL divergence is true with a probability of 0.85. The first test has a value of 3.9e-3 against the limit of 5.8e-3.

The prediction of the answers: build A with `-np 3` against build A with `-np 1` gives a largest difference of about 0.02. We get this value from the first test, where ragged batching against the equal split gave 0.0224. Build B with `-np 3` against build A with `-np 1` gives about 0.02 to 0.03. The limit of the answers is true with a probability of 0.7.

The prediction of the speed of arm B against arm A (`build:stack-fbfd9cb`) starts from the first test. In the first test, arm B against `build:lead-30489` gave -3.32 % for `lat_ms`. Arm A of the second test is faster than build `lead-30489`. It has about 223 ms in the measurement window of `dmk/stack` (one run). The same gain in milliseconds gives a larger percent of this value.

Two effects decrease the gain. A pass with ragged batching does not use the fusion of change 0005 in the 24 layers. [INFERENCE: this adds about 0.3 ms for each pass with ragged batching, which is about 0.1 ms for each request.] Changes 0004 and 0007 decrease the time of each pass. A pass that ragged batching removes then has a smaller time.

| Metric | Predicted change (%) | Range (%) |
|---|---:|---|
| `lat_ms` | -3.3 | -1.8 to -4.5 |
| `short_ms` | -3.5 | -1.5 to -5.5 |
| `medium_ms` | -5.0 | -3.5 to -6.5 |
| `long_ms` | -2.0 | -1.0 to -3.0 |
| Short request with two questions (client) | -8 | -5 to -10 |
| Short request with one question (client) | 0 | -1 to +1 |
| `rps8` | +5 | 0 to +8 |
| `llama bench` | 0 | -1 to +1 |

Confidence: 0.65. This is the probability that the gain of `lat_ms` is 2.0 % or more and the 95 % bootstrap interval does not include 0.

## Conditions that show an incorrect prediction

- The gain of `lat_ms` is less than 2.0 %, or its interval includes 0.
- The log of the server has no ubatch with ragged batching in the phase `seq`.
- Build B with `-np 1` gives answers that are not the same as the answers of build A with `-np 1`.
- A limit for the decision of the logits or the answers is not true. The result entry then gives the cause.
