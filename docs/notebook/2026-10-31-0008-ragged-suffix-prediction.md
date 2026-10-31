# Change 0008: ragged batching of the suffixes (prediction)

Written 2026-10-31, before the first run of the build with the change.

## Task of this entry

This entry gives the prediction for change `0008`. It is target `J` of the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md). We commit it before the first run of the build with the change. We do not change it after that. The result goes into a different entry.

## Cause

The hybrid memory divides each batch with the equal split. In each ubatch, all sequences have the same number of tokens. A request with 3 questions has 3 suffixes of different lengths, for example 43, 47 and 73 tokens.

The equal split makes 3 passes for these suffixes. They have 3 sequences of 43 tokens, 2 sequences of 4 tokens, and 1 sequence of 26 tokens. A request with 2 questions of 51 and 120 tokens makes 2 passes. Each pass reads all weights again. A pass of 26 tokens or less uses about 11.5 ms.

## Change

Ragged batching puts the suffixes of all sequences in one ubatch. The lengths stay different. A medium request or a long request then uses one pass of 163 tokens. A `short2q` request uses one pass of 171 tokens.

The code is in the branch `dmk/0008-ragged-suffix` of the fork. It has these parts:

1. The function `split_equal` gets a limit for the step between the lengths. The hybrid memory sets it.
2. A ubatch uses ragged batching when all of these conditions apply. Each set has one sequence. The lengths are different. All tokens fit in the ubatch. The memory has no snapshots. Each step between the lengths, from 0 to the longest length, is 128 tokens or less.
3. The last condition keeps a long prefix adjacent to a short question in the equal split. The GDN kernel for ragged batching runs one token at a time. Its time is about 28 µs for each token.
4. The graph uses two new operations: `SSM_CONV_RAGGED` and `GATED_DELTA_NET_RAGGED`. They have a CPU version and a CUDA version. Only the Gated DeltaNet layers use them.
5. The CUDA kernel for the convolution does the same operations in the same sequence as the kernel before the change. The GDN kernel is a copy of the kernel for one token at a time, with a token range for each set. It does not use the path with tensor cores.
6. The memory tests the backend of each recurrent layer when it starts. If the backend does not know the two operations, the memory uses the equal split.
7. The environment variable `LLAMA_RAGGED_SPLIT=0` selects the equal split. All other models also use the equal split.

## Equivalence class

We select `re-associated` on 2026-10-31, before the first run.

The kernels for the convolution and the GDN do the same operations for each token. The change moves tokens into a larger pass. A pass of more than 8 tokens uses MMQ for the matrix multiplies. A pass of 8 tokens or less uses MMVQ. The flash attention kernel can also be different. The sums then occur in a different sequence, and the last bits of the results can be different.

## Arms and schedule

| Arm | Build | Meaning |
|---|---|---|
| A | `build:lead-30489` | The branch `dmk/stack`: `b12004` and change 0001. |
| B | `build:0008-ragged-suffix` | The branch `dmk/0008-ragged-suffix`. Ragged batching is on. |

The speed test has 4 blocks of the private benchmark with seed 80080. It has 4 blocks of `llama bench` with seed 80081. The commands are the commands of the [lead 30489](2026-10-31-lead-30489.md) with the new arms. A block has one run of each arm in a random sequence.

The estimate is the mean of the 4 ratios of arm B to arm A. The interval is the 95 % bootstrap interval with 10,000 samples. The tool is `harness/analysis/ab_stats.py`. The metrics are `lat_ms`, `short_ms`, `medium_ms`, `long_ms` and `rps8`. For `llama bench` they are `pp64_ts`, `pp256_ts` and `pp3000_ts`. We do not declare an equivalence margin.

## Predicted gain

The prediction starts from the numbers of phase 4 and has one change. Phase 4 counts the GDN time of the second step of `short2q` with the path with tensor cores. The new kernel uses the path for one token at a time. For 69 tokens, the profile gives 1.67 ms for the first path. The second path uses about 2.8 ms. We thus remove 1.1 ms from the gain of `short2q`.

| Request type | Cases | Phase-4 gain (ms) | Phase-4 gain (%) | Predicted gain (ms) | Predicted gain (%) |
|---|---:|---:|---:|---:|---:|
| `short1q` | 12 | 0 | 0 | 0 | 0 |
| `short2q` | 20 | 13.0 | 14.9 | 11.9 | 13.7 |
| `medium` | 8 | 32.9 | 9.1 | 32.9 | 9.1 |
| `long` | 8 | 32.9 | 4.1 | 32.9 | 4.1 |

Source: the table of the [ranked list](2026-10-31-phase4-ranked-targets.md). The 32 short cases have 52 decisions. [INFERENCE: this gives 12 cases with one question and 20 cases with two questions. The numbers also give the gain of `lat_ms` in phase 4.]

| Metric | Arm A (ms) | Predicted change (%) | Phase 4 (%) |
|---|---:|---:|---:|
| `short_ms` | 73.4 | -10.1 | -11.1 |
| `medium_ms` | 362.9 | -9.1 | -9.1 |
| `long_ms` | 791.4 | -4.1 | -4.1 |
| `lat_ms` | 241.1 | -6.6 | -6.8 |

Source: the values of build D in the [lead 30489](2026-10-31-lead-30489.md). We calculated the other columns.

- `rps8`: between 0 % and +6.6 %. In this phase, 8 requests run at the same time. A batch can then hold the work of many requests. The limit for the step keeps a long prefix in the equal split. [INFERENCE: we did not measure the batches of this phase.]
- `llama bench`: a change of 1 % or less, as noise. These runs have one sequence.
- Confidence: 0.45. This is the probability that the change passes its tests and gives 50 % or more of the gain. The gain of `lat_ms` is then 3.3 % or more.

## Tests and tolerances

We declare the tolerances before the first run.

| Test | Tolerance |
|---|---|
| `test-backend-ops` for the two new operations against the CPU. The cases have the lengths of the model and sets with 0 tokens. | NMSE of 5e-4 or less. All cases pass. |
| `test-backend-ops` for the graph of mix8, on the build with the change | 181 of 187 tests pass, as on build D. The same 6 tests give `FAIL`. |
| Layer state of each question after one pass with ragged batching, against the layer state after the equal split. It has the conv state and the SSM state of all 24 layers. | NMSE of 5e-4 or less for each layer. [INFERENCE: the conv states are the same bit for bit.] |
| Logits of the last token of each suffix. Pass with ragged batching against the equal split. The lengths are 43, 47 and 73, and 51 and 120. | The mean KL divergence is 1e-5 or less. The largest is 1e-3 or less. The top token is the same in all rows. |
| KL floor for the test above: arm A against arm A on the same text | The limits above apply when the mean KL floor is less than them. If it is more, the limit is 10 times the KL floor. |
| `llama perplexity` against the base file of arm A, on the two text sets of the method. The KL floor is the result of a second run of arm A against the same base file. | The mean KL divergence of arm B is not more than the KL floor plus 1e-5. The KL budget (0.027830 and 0.057587) is the maximum. |
| Answers of 4 runs of arm B against 4 runs of arm A | `answers_compare.py` gives `self_repeat` true for the two arms. The largest probability difference between the arms is 0.01 or less. No decision has a different top answer. |
| Guards of the benchmark | All guards are true in all runs. |
| The 52-decision set | 48 decisions are correct. The 4 incorrect decisions are the same. |
| `compute-sanitizer --tool memcheck` on one request with 3 questions | 0 errors. |
| Ubatches with ragged batching in the log of the server, in each phase of the benchmark | We write the number of them for each phase. The number 0 in the phase `seq` makes the prediction incorrect. |

The tests of the layer state and of the logits use the program `tests/test-ragged-suffix.cpp`. It decodes a prefix in one sequence. It copies the layer state to 3 sequences. It then decodes the 3 suffixes 4 times. The 4 runs use the equal split, the equal split again, ragged batching, and one sequence at a time.

## Conditions that show an incorrect prediction

- The gain of `lat_ms` is less than 3.3 %, or its interval includes 0.
- The log of the server shows no ubatch with ragged batching in the phase `seq`.
- A test above gives a result that is not in its tolerance. The result entry then gives the cause.
