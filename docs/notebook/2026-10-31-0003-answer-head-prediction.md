# 0003 Answer head: prediction

Written 2026-10-31, before the first run of the measurement window.

## Task of this entry

This entry gives the prediction for change 0003. The change calculates the output head for the answer tokens only. It is target `A` of the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md).

The method has a rule. We commit the prediction before the first run of the measurement. The commit time is the indication of the time. We do not change this file after the measurement starts. The result goes in a different entry.

## Change

The output head is the last matrix multiply of a pass. Its weight has 248,320 rows of 2,560 values in the type Q8_0. It reads 675 MB. A decision uses the logits of the label tokens only. The labels of one question are some tokens.

The change has two parts.

1. The model object `llama_context` has a list of tokens (`llama_set_logit_tokens`). If the list is not empty, the output head calculates the logits of these tokens only. For each output row it reads the rows of the weight of these tokens. It does not read the other rows. A row of the result has one value for each token of the list.
2. The server sets the list for a batch of decisions. The list has the labels of the decisions in the batch. The server reads the values of the labels from the short rows. A batch of a different type uses the full output head as before.

The result is also much smaller. The copy of the logits from the GPU to the host is 993,280 bytes for each output row before the change. After the change it has one value for each label.

The output head uses the operation `GGML_OP_MUL_MAT_ID` with one token. Each row of the weight is an expert with one row. The MMVQ kernel with one column does the work. The new function `ggml_mul_mat_rows` makes the nodes. The graph has the same nodes for 0 outputs and for 1 to 4 outputs. Thus the memory of the graph does not change when the number of outputs changes.

The change uses the small output head only if all of these conditions are true. If one is false, the graph keeps the full output head. It gets the values of the labels from the rows of the full result. These values are the same, but the head reads the full weight.

- The output head is one multiply of a weight and an F32 activation. The weight has the type Q4_0, Q4_1, Q5_0, Q5_1 or Q8_0. The head has no scale, no bias, no LoRA, no precision flag and no `hdmd` step.
- The weight is in a buffer of one CUDA device.
- The pass has a maximum of 4 output rows. The server of this project has 3 slots, thus it has a maximum of 3.
- The model object does not output embeddings and has no backend sampler.

The change also has a small repair in the launch code of the MMVQ kernel. If K is small, the kernel can use 4 rows for each block. With `ids`, the guard at the end of the rows compares with the distance between two columns of the result. It does not compare with the number of rows. A weight with one row for each expert has blocks with rows after its last row. These blocks wrote into the results of the next `ids` and read after the weight.

The repair does not use the launch with 4 rows if `ids` is set and `nwarps` does not divide the number of rows. The sums have the same sequence in the two launches. Thus the repair does not change a result. Other graphs of the model do not use `ids`. Before the repair, `test-mul-mat-rows` had 120 of 480 cases with different values. They were all cases with K = 1536 and the types Q4_0, Q4_1, Q5_0 and Q5_1.

## Equivalence class

The equivalence class is `exact`. The answers must be the same bit for bit as the answers of arm L. The section "Tests" gives the tests.

The cause of `exact` is this. The kernel with one column and the kernel with 2 to 4 columns use the same sequence of sums for each row. [INFERENCE: this comes from the code of `mul_mat_vec_q` in `mmvq.cu` for the architecture `sm_86`. The tests must show it.] If the answers are not the same bit for bit, the equivalence class becomes `re-associated`. Then we tell the project lead before the next step.

## Arms

| Arm | Build | Contents |
|---|---|---|
| L | `build:lead-30489` | The branch `dmk/stack`: build D and the commit 6b8b65e66 of pull request #30489. |
| `A3` | `build:0003-answer-head` | Arm L and change 0003. |

The two builds use the same builder image and the same flags (`sm_86` only). The model is `h2o-lightning-4b-rq-Q4_0-mix8.gguf`. The server flags are the flags of the baseline of phase 1: `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`.

## Schedule

One measurement window has two schedules. Each block has one run of each arm in a random sequence. Each schedule has a seed that gives the random sequence.

```
make_schedule.py --prefix 0003-bench --blocks 4 --seed 3001 --arm L=build:lead-30489 --arm A3=build:0003-answer-head --kind bench --args "--write-reference {name}"
make_schedule.py --prefix 0003-llb --blocks 4 --seed 3002 --arm L=build:lead-30489 --arm A3=build:0003-answer-head --kind tool --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The first schedule runs the private benchmark. Each run has the quality pass and the speed passes. The quality pass writes its answers with `--write-reference`. The second schedule runs `llama bench` with `-r 5` for each prompt length. The measurement window is 30 to 35 minutes long. The harness records the load of the host.

Before the measurement window, there is a set of correctness runs. The set has a different schedule file. It is not a speed test, and its runs give no speed result. It has these runs: two runs of the private benchmark, one for each arm, with the answers. Then one run of arm `A3` with 6 slots. Then one run of the server of arm `A3` in `compute-sanitizer`, with one request of each type.

## How we calculate the result

The script is `harness/analysis/ab_stats.py` with `--a L --b A3`. For each block and metric, the ratio is the value of arm `A3` divided by the value of arm L. The estimate is the mean of the 4 ratios, in %. The interval is the 95 % bootstrap interval of the 4 blocks, with 10,000 samples. A latency that is less is faster. A speed that is more is faster.

The metrics are `lat_ms`, `short_ms`, `medium_ms` and `long_ms`. `rps8` is a secondary metric. For `llama bench` the metrics are `pp64_ts`, `pp256_ts` and `pp3000_ts`.

The script `harness/analysis/answers_compare.py` compares the 4 answer files of arm L with the 4 answer files of arm `A3`.

## Prediction

### Gain of one pass with outputs

| Item | Time (ms) |
|---|---:|
| Output head kernel (MMVQ, Q8_0, 675 MB), profile of phase 3 | -1.64 |
| Copy of the logits to the host: 993,280 bytes at 6.7 GB/s | -0.15 |
| New work: copy of the list to the GPU, 4 small kernels | +0.04 |
| Gain of one pass with outputs | -1.75 |

The number of phase 4 is 1.6 ms for each pass with an output. We change it to 1.75 ms. There are two causes.

- The number of phase 4 is the time of the output head kernel from the ablation. It does not include the copy of the logits. After the change, the copy has about 40 bytes. The copy does not run in parallel with other work, because the server must have the logits at the end of the pass.
- The new nodes add a small time. Each of the 4 small kernels uses 2 µs to 5 µs. The copy of the list to the GPU uses 10 µs to 30 µs. [INFERENCE: we did not measure these.]

The rate of 6.7 GB/s is the rate of the copies to the GPU in the profile of phase 3. The slot of the GPU runs at PCIe 3.0 x8. [INFERENCE: the rate to the host is the same.]

### Gain for each request type

A request has 1, 2, 3 and 3 passes with outputs for `short1q`, `short2q`, `medium` and `long`. The prefix passes have no output.

| Request type | Passes with outputs | Gain (ms) | Gain (%) of the request |
|---|---:|---:|---:|
| `short1q` | 1 | -1.75 | -3.7 |
| `short2q` | 2 | -3.50 | -4.0 |
| `medium` | 3 | -5.25 | -1.5 |
| `long` | 3 | -5.25 | -0.7 |

The request times for the values in % are 47.1, 87.1, 360.0 and 797.0 ms (phase 4).

### Gain for the metrics of the benchmark

The benchmark has 12 cases of `short1q`, 20 cases of `short2q`, 8 cases of `medium` and 8 cases of `long`. The mean values of arm L in the [lead 30489 test](2026-10-31-lead-30489.md) are the reference: `lat_ms` 236.5, `short_ms` 72.5, `medium_ms` 355.7 and `long_ms` 773.6 ms.

| Metric | Predicted change of arm `A3` against arm L (ms) | Predicted change (%) |
|---|---:|---:|
| `lat_ms` | -3.6 | -1.5 |
| `short_ms` | -2.8 | -3.9 |
| `medium_ms` | -5.3 | -1.5 |
| `long_ms` | -5.3 | -0.7 |
| `rps8` | | +0.5 to +1.5 |
| `pp64_ts`, `pp256_ts`, `pp3000_ts` | 0 | 0 |

`rps8` is a secondary metric. We have no time model for it. With 8 clients and 3 slots, some decisions can use one pass together. Then the pass reads the output head one time for them. The gain of one request is then less than the gain in the table above. [INFERENCE: the range is an estimate.]

`llama bench` does not set a list of tokens. It uses the full output head as before. The change must not change its speed. It is a control for the load of the host.

## Tolerances

- Answers: no tolerance. The answers must be the same bit for bit. `cross_identical_all` is true after `self_repeat_a` and `self_repeat_b` are true.
- `test-mul-mat-rows`: 0 values that are different, for the types and the numbers of columns that the graph uses.
- `test-backend-ops`: the new cases of `ggml_mul_mat_rows` against the CPU have an NMSE of 5e-4 or less. This is the limit of the tests of `MUL_MAT_ID`.
- The prediction holds if the estimate of `lat_ms` is -0.8 % or less, and the 95 % interval of `lat_ms` is less than 0. This is 50 % of the predicted gain or more. It is the same rule as the confidence of phase 4.
- A gain of -2.3 % or more for `lat_ms` is more than 150 % of the prediction. Then we look for a second cause.
- `llama bench`: the interval of each prompt length includes 0, or the estimate is between -1 % and +1 %. The lead 30489 test had a noise of about 1 % at 64 tokens.

## Tests

These tests run before the measurement window. The last test uses the runs of the measurement window.

1. `test-mul-mat-rows`. It compares `ggml_mul_mat_rows` with `ggml_mul_mat` on the GPU. It compares each value bit for bit. It uses the types Q4_0, Q4_1, Q5_0, Q5_1 and Q8_0, with 1 to 4 columns and 3 matrix dimensions. One run has the dimensions of the output head of the model.
2. `test-backend-ops`. The new cases of `ggml_mul_mat_rows` against the CPU.
3. `compute-sanitizer --tool memcheck` on the server of arm `A3` with one decision request of each type of question. This runs in the set of correctness runs.
4. The answers of the 4 runs of arm L and the 4 runs of arm `A3` in the measurement window (`answers_compare.py`). All 100 decisions.
5. A run with 6 slots in the set of correctness runs. This gives a pass with 5 or more outputs. Then the graph uses the full output head and gets the values of the labels from its rows. The answers must be the same as the answers of arm L.

## Causes of an incorrect prediction

- **The copy runs in parallel with other work.** Then the gain is 1.64 ms for each pass. The numbers above are 8 % too high.
- **The memory of the graph changes.** This occurs if the nodes of the graph change, or if a tensor is larger. The nodes of the new graph are the same for all numbers of outputs. The first graph with the new nodes can cause it. If it occurs in the runs, the time can be more than the gain. We count the lines `graph reallocation` in the log of the server.
- **The small kernel is slower than 5 µs.** One block reads one row of 2.7 KB with 4 warps. [INFERENCE: the latency limits it, and the time is 30 µs at the maximum.]
- **The load of the host.** The load was high in the lead 30489 test. Each block has the two arms, thus the load has the same effect on the two arms. The interval has 4 blocks only. It is too small to show all the noise.
