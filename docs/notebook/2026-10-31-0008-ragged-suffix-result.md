# Change 0008: ragged batching of the suffixes (result)

Written 2026-10-31, after the measurement windows (07:51 to 08:43 UTC and the measurement window of the census).

## Task of this entry

This entry gives the result of the tests of change `0008` against the [prediction entry](2026-10-31-0008-ragged-suffix-prediction.md). The commit time of the prediction is 2026-10-31 07:02:58 UTC. This is before the first run of the build with the change. The [README of the data set](../../data/2026-10-31-0008-ragged-suffix/README.md) gives the raw files, the tables, and the conditions. The [change document](../changes/0008-ragged-suffix.md) gives a summary.

## Result in short

- Arm B (`build:0008-ragged-suffix`) is faster than arm A (`build:lead-30489`) in the private benchmark. `lat_ms` is 3.32 % lower (95 % interval: -4.18 % to -2.63 %). The prediction was -6.6 %.
- The gain is 39 % to 61 % of the predicted gain for each request type. The gain of `lat_ms` is 50.3 % of the predicted gain. The prediction is not correct if the gain is less than 3.3 %. The result is 0.02 points above that limit.
- `rps8` is 5.96 % higher (interval +5.60 % to +6.31 %). This is in the predicted range of 0 % to +6.6 %. `llama bench` changes by -0.08 % to -0.30 %. This is in the predicted limit of 1 %.
- The cause of the smaller gain is the number of ubatches. The census of one pass of the benchmark finds 20 ubatches with suffixes of different lengths in the phase `seq`. The prediction used 36 requests. It removed 1 pass for each short request with two questions and 2 passes for each request with three questions.
- Correctness: some tests that the prediction declared give `FAIL`. The mean KL divergence of the logits is more than its limit. The NMSE of the layer states and the largest probability difference of the answers are also more than their limits. The prediction limits were too small for a change of the contents of a batch. A control shows the same differences between the equal split and a decode of one sequence at a time.
- The other tests give a pass. They are the tests of the two new operations, the graph of 187 operations, `llama perplexity`, the guards, the 52-decision set, `compute-sanitizer` and `split_equal`.

## Change

The hybrid memory divides a batch with the equal split. All sequences of a ubatch have the same number of tokens. A request with 3 questions has 3 suffixes of different lengths. The equal split makes one pass for each different length. Each pass reads all weights.

Ragged batching puts the suffixes in one ubatch. Two new operations, `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`, do the work of the recurrent layers for sequences of different lengths. The [change document](../changes/0008-ragged-suffix.md) gives the conditions of the change.

## Equivalence class

We selected `re-associated` on 2026-10-31, before the first run. The tests of this equivalence class use a KL divergence at the level of noise. They do not use a result that is the same bit for bit. The answers of arm B are different from the answers of arm A by 0.0224 in probability at most.

## Prediction against result

| Item | Prediction | Result | Same as the prediction |
|---|---|---|---|
| `lat_ms` | -6.6 % | -3.32 % (interval -4.18 % to -2.63 %) | No. The gain is 50 % of the predicted gain. It is not less than 3.3 %. |
| `short_ms` | -10.1 % | -3.91 % (interval -5.06 % to -3.04 %) | No. The gain is 39 % of the predicted gain. |
| `medium_ms` | -9.1 % | -5.46 % (interval -6.08 % to -4.98 %) | No. The gain is 60 % of the predicted gain. |
| `long_ms` | -4.1 % | -2.07 % (interval -2.63 % to -1.67 %) | No. The gain is 50 % of the predicted gain. |
| `short2q` | -13.7 % | -8.37 % (interval -8.90 % to -7.94 %), from the client `latency_by_type.py` | No. The gain is 61 % of the predicted gain. |
| `short1q` | 0 % | +0.09 % (interval -0.62 % to +0.80 %), from the client `latency_by_type.py` | Yes |
| `rps8` | 0 % to +6.6 % | +5.96 % (interval +5.60 % to +6.31 %) | Yes |
| `llama bench` | A change of 1 % or less | -0.20 % (64 tokens), -0.30 % (256 tokens), -0.08 % (3,000 tokens) | Yes. The interval of 256 tokens is -0.44 % to -0.15 %. It does not include 0. |
| Ubatches with ragged batching in the phase `seq` | More than 0 | 20 in one pass | Yes |
| Confidence of 0.45 that the tests are in their tolerances and the gain is 50 % or more | | The gain is 50.3 % of the predicted gain. Some tests are not in their tolerances. | No |

Source: tables `speed-bench.tsv`, `speed-llama-bench.tsv`, `speed-type.tsv` and `ragged-ubatches.tsv` in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md).

The condition "the gain of `lat_ms` is less than 3.3 %, or its interval includes 0" is not true. The condition "a test gives a result that is not in its tolerance" is true. The section "Correctness" gives the cause.

## Speed test with the private benchmark

The test has 4 blocks. Each block has one run of each arm. The power limit was 200 W. We did not lock the clocks. The median SM clock of each run was 1,845 to 1,860 MHz.

| Metric | Arm A | Arm B | Difference (%) | 95 % interval (%) |
|---|---:|---:|---:|---|
| `lat_ms` (ms) | 235.894 | 228.058 | -3.317 | -4.176 to -2.628 |
| `short_ms` (ms) | 72.224 | 69.395 | -3.913 | -5.056 to -3.041 |
| `medium_ms` (ms) | 354.685 | 335.332 | -5.455 | -6.084 to -4.976 |
| `long_ms` (ms) | 771.258 | 755.304 | -2.067 | -2.630 to -1.667 |
| `rps8` (requests/s) | 4.280 | 4.535 | +5.957 | +5.604 to +6.310 |

Source: table `speed-bench.tsv`.

The difference of `lat_ms` in the four blocks is -2.45 %, -4.52 %, -3.13 % and -3.16 %. Arm B is faster than arm A in all 4 blocks. Arm A has 235.9 ms. Arm `L` in the [lead 30489](2026-10-31-lead-30489.md) has 236.5 ms.

## Speed for each request type

The benchmark gives the latency for each dimension of request. It does not give the short requests with one question and the short requests with two questions as different groups. We wrote the client `latency_by_type.py` to do this. It uses the method of the phase `seq`.

It has the same warm-up, the same sequence of the cases, and the same tags against the cache of the prompt. It has no phase `quality` before it. The test has 4 blocks.

| Request type | Cases | Arm A (ms) | Arm B (ms) | Difference (%) | 95 % interval (%) |
|---|---:|---:|---:|---:|---|
| Short, 1 question | 12 | 46.852 | 46.890 | +0.089 | -0.620 to +0.799 |
| Short, 2 questions | 20 | 85.903 | 78.710 | -8.371 | -8.904 to -7.936 |
| Medium request | 8 | 351.971 | 333.051 | -5.373 | -5.741 to -5.005 |
| Long | 8 | 768.208 | 748.626 | -2.548 | -2.779 to -2.317 |

Source: table `speed-type.tsv`.

The client and the benchmark agree for the medium requests and the long requests. They do not agree for the short requests. The numbers of the client give -6.7 % for all short requests. The benchmark gives -3.9 %. The runs of the client have no phase `quality` before the phase `seq`. We do not know if this is the cause.

## Speed test with `llama bench`

| Prompt | Arm A (tokens/s) | Arm B (tokens/s) | Difference (%) | 95 % interval (%) |
|---|---:|---:|---:|---|
| 64 tokens | 2,834.659 | 2,829.112 | -0.195 | -0.579 to +0.190 |
| 256 tokens | 4,309.650 | 4,296.765 | -0.299 | -0.437 to -0.149 |
| 3,000 tokens | 4,703.247 | 4,699.538 | -0.078 | -0.547 to +0.290 |

Source: table `speed-llama-bench.tsv`.

These runs have one sequence, and the change does not apply. The difference of 256 tokens is small, and its interval does not include 0. The prediction gave 1 % or less. We do not know the cause. In `llama bench`, arm B ran second in 3 of the 4 blocks.

## Where the change applies

The pass of the private benchmark with the log of the ubatches (`-lv 5`) shows each ubatch with suffixes of different lengths. The table gives the number of them for each phase.

| Phase | Requests | Ubatches with suffixes of different lengths | In one pass (ragged batching) | Not in one pass |
|---|---:|---:|---:|---:|
| `warmup` | 3 | 3 | 3 | 0 |
| `quality` | 48 | 26 | 26 | 0 |
| `seq` | 48 | 20 | 20 | 0 |
| `tput` | 144 | 127 | 126 | 1 |

Source: table `ragged-ubatches.tsv`.

The phase `seq` has 20 ubatches: 17 with 2 sequences and 3 with 3 sequences. The phase `tput` has 29 ubatches with 3 sequences. One ubatch in `tput` has a step of 152 tokens. This is more than the limit of 128, and the equal split does it.

The prediction used 36 requests in the phase `seq`: 20 requests with two questions and 16 requests with three questions. The prediction thus removed 52 passes in one pass of the phase `seq`. The change removes 23 passes at most.

The log of the server gives the cause for the requests with 3 questions. The first question and one other question run in the same pass. The third question starts when the first question is complete. The log line is `waiting for parent slot to complete`.

[INFERENCE: this is the work of the server for the requests that we read. We did not change it.] The third question then runs in a pass for one sequence. Ragged batching cannot attach it to the other two.

The gain of each pass that the change removes is more than the prediction. The change removes 7.84 ms for each request, or 376 ms in the phase `seq` of 48 requests. This is 16.4 ms for each of the 23 passes. The prediction used 10.4 ms. [INFERENCE: a pass with a small number of tokens uses more time than the mean pass. The build of the graph and the start of the kernels are in this time.]

## Correctness

The prediction entry declared the tolerances before the first run. The table gives each test, its tolerance, and its result. Each result has a source in the [data set](../../data/2026-10-31-0008-ragged-suffix/README.md).

| Test | Tolerance | Result |
|---|---|---|
| `test-backend-ops`, `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`, against the CPU | NMSE of 5e-4 or less. All cases are in the tolerance. | Pass. 18 of 18 cases, 5 of 5 cases, and 3 of 3 cases with the fusion into `CPY`. |
| `test-backend-ops`, 187 operations of the graph of mix8 | 181 of 187 pass. The same 6 give `FAIL`. | Pass. 181 of 187. The 6 are 5 `CPY` and 1 `GATED_DELTA_NET`, as on build D. |
| `test-backend-ops`, 390 built-in cases of the five operations of the layer | Not declared. All pass, as before the change. | Pass. 390 of 390. |
| Layer states of each question, ragged batching against the equal split, 24 layers | NMSE of 5e-4 or less for each layer | `FAIL`. The largest NMSE is 1.45e-2 (conv state) and 1.73e-3 (SSM state) for the lengths 43, 47 and 73. |
| Logits, mean KL divergence | 1e-5 or less | `FAIL` in the 4 cases: 3.9e-4, 7.3e-4, 1.2e-3 and 1.8e-3 |
| Logits, largest KL divergence | 1e-3 or less | Pass in 2 cases (6.8e-4 and 7.4e-4). `FAIL` in 2 cases (2.6e-3 and 3.5e-2). |
| Logits, top token | The same in all rows | Pass in 3 cases (3 of 3, 2 of 2, 5 of 5 rows). `FAIL` in the case with all 163 logits: 161 of 163. |
| `llama perplexity`, mean KL divergence of arm B against the base file of arm A | The KL floor plus 1e-5. The KL budget (0.027830 and 0.057587) is the maximum. | Pass. The mean is 0.000000 on the two texts, as the KL floor. The largest is 0.000058 and 0.000055, as the KL floor. |
| Answers, `self_repeat` | True for the two arms | Pass |
| Answers, largest probability difference between the arms | 0.01 or less | `FAIL`. 0.0224 on 100 decisions. 0.0416 on the 52-decision set. |
| Answers, top answer | No decision has a different top answer | Pass. 0 decisions in the two sets. |
| Guards of the benchmark | All true in all runs | Pass. All guards are true in 11 runs. The runs have 94 correct decisions of 100, as arm A. |
| The 52-decision set | 48 correct. The same 4 incorrect. | Pass. 48 of 52 on the two arms. The incorrect decisions are `triage00.category`, `triage07.category`, `triage07.needs_action` and `triage08.category`. |
| `compute-sanitizer --tool memcheck` on one request of each type | 0 errors | Pass. `ERROR SUMMARY: 0 errors`. The exit code of the server is 0. |
| Ubatches with ragged batching in the phase `seq` | More than 0 | Pass. 20. |
| `test-batch-alloc`, the unit test of `split_equal` | Not declared. All pass. | Pass. 58 tests and 436 checks. 0 tests give `FAIL`. |

Source: tables `ops-tests.tsv`, `suffix-compare.tsv`, `suffix-checks.tsv`, `kl-perplexity.tsv`, `quality-bench.tsv`, `check-runs.tsv`, `answers-compare-bench.json`, `decisions-52.tsv`, `answers-compare-decisions-52.json`, `memcheck.tsv`, `unit-tests.tsv` and `ragged-ubatches.tsv`.

The prediction declared the layer states as an inference: "the conv states are the same bit for bit". This is not true. The test shows that the layer states are not the same bit for bit. At layer 0, the states are the same to 1e-14 only in the case of the lengths 51 and 120.

### Cause of the results with `FAIL`

The prediction states that the result entry gives the cause of a result that is not in its tolerance. The test program `test-ragged-suffix` has two more decodes as a control. They are the equal split against a decode of one sequence at a time, and ragged batching against the same decode. The control shows how much the equal split changes the layer states and the logits when the contents of the batch are different.

| Case (lengths) | Mean KL divergence: ragged batching against the equal split | Equal split against one sequence at a time | Ragged batching against one sequence at a time |
|---|---:|---:|---:|
| 43, 47, 73 | 3.9e-4 | 1.0e-3 | 1.0e-3 |
| 51, 120 | 7.3e-4 | 6.6e-4 | 6.1e-4 |
| 2, 33, 2, 64, 31 | 1.2e-3 | 1.4e-3 | 2.1e-3 |
| 43, 47, 73, all 163 logits | 1.8e-3 | 6.5e-3 | 3.9e-3 |

Source: table `suffix-compare.tsv`.

The three columns are in the same range. The equal split is as far from the decode of one sequence at a time as ragged batching is from the equal split. The layer states give the same result. For the lengths 43, 47 and 73, the largest NMSE of the conv state is 2.8e-2 (equal split against one sequence at a time) and 3.2e-3 (ragged batching against one sequence at a time). The table `suffix-layers.tsv` has all layers.

The cause is the selection of the kernel. A pass with 8 tokens or less uses MMVQ. A longer pass uses MMQ. For the lengths 43, 47 and 73, the equal split makes a pass of 8 tokens (2 sequences with 4 tokens each). A pass with ragged batching and a decode of one sequence use MMQ for all tokens.

At layer 0, the NMSE of the conv state is 2.4e-8 for ragged batching against the equal split. It is 2.4e-8 for the equal split against one sequence at a time. For ragged batching against one sequence at a time, it is 1.0e-14.

The NMSE becomes larger with the layer. The maximum is at layers 13 and 14. [INFERENCE: the cause is the selection between MMVQ and MMQ. We did not run a test that selects one kernel.]

For the lengths 51 and 120, the last pass of the equal split has 69 tokens and uses MMQ. The three decodes then use the same kernel at layer 0. The NMSE of ragged batching against the equal split at layer 0 is 1.1e-14 (conv state) and 6.0e-12 (SSM state). This shows that the two new operations give the same values as the operations of the build before the change. The equal split is also not the same as the decode of one sequence at a time in the other cases.

The prediction limit of 1e-5 for the mean KL divergence was too small. The KL floor of the test is 0, because the same build gives the same result in each run. The prediction used 1e-5 for this case. The control shows that a change of the contents of a batch gives 4e-4 to 6.5e-3 in the same build.

The KL divergence of `llama perplexity` shows no change. The two texts use sequences of equal lengths, and the new code does not run for them. The test shows that the build with the change gives the same results as arm A when the new code does not run.

## Host load

The host has 12 CPU threads. The one-minute load average was 1.65 to 17.04 in the private benchmark and 13.33 to 17.04 in `llama bench`. Arm B ran first in 2 of the 4 blocks of the private benchmark. The sign of the difference of `lat_ms` is the same in all 4 blocks. The difference of the first block is -2.45 %, with a load of 1.65 to 3.17.

## Limits and next work

- **Gain.** The gain is 50 % of the gain that the prediction gave. The cause is the structure of the server. A request with 3 questions has two passes after the change, and not one.
- **Server.** If the third question starts together with the other two, one pass can hold the three suffixes of a medium request or a long request. This removes one more pass for 16 of the 48 requests. It is a change of the server code. We did not do it.
- **Two measurements for the short requests.** The client and the benchmark give -6.7 % and -3.9 %. We did not find the cause.
- **Result of the numbers.** The tests of the layer state, the logits and the answers are not in the declared limits. The change is not `exact`. The tests show that the result is in the range of the noise of the contents of a batch. The answers of 100 decisions have no different top answer. A decision on the limits is necessary before a merge.
- **Not in the tests.** Other models, speculative decoding (`n_rs_seq` more than 0), the other backends, and more than one GPU. The new operations have only a CPU version and a CUDA version. Other backends use the equal split.
- **Other changes.** Changes 0002 to 0005 change the same files. A merge can give different lines in `ggml-cuda.cu`, in `qwen35.cpp` and in the graph code.
- **Go back to the baseline.** The environment variable `LLAMA_RAGGED_SPLIT=0` selects the equal split without a new build.
