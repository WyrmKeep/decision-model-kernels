# Change 0004: attention mask on the GPU (prediction)

Written 2026-10-31, 05:24 UTC, before the first run of this change in the measurement window.

## Task of this entry

This entry gives the prediction for change 0004. The change makes the attention mask on the GPU. The method has a rule. We commit these items before the first run of the speed test. The items are the equivalence class, the arms, the schedule, the method of the result, the tolerances, and the predicted gain. We do not change this entry after that run.

The tests of the equivalence class run before the measurement window. They do not measure speed. The results entry gives their results.

## Cause

This cause comes from the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md), target `B`.

- The host writes the attention mask of each pass. The attention mask has `n_kv` values for each token. A value is an FP16 number.
- The host sends the attention mask to the GPU at 6.7 GB/s. The slot of the GPU has PCIe 3.0 x8.
- The attention mask of a large pass is 21 MB to 28 MB. The copy uses 3 ms to 4 ms. The GPU waits for the copy.
- The host also uses time to write the attention mask.

The next table gives the time of one request, from the [time model of phase 4](../../data/2026-10-31-phase4-model/README.md).

| Request type | Cases in the benchmark | Copies to the GPU (ms) | Host time of the attention mask (ms) |
|---|---:|---:|---:|
| `short1q` | 12 | 0.71 | 0.05 |
| `short2q` | 20 | 1.08 | 0.10 |
| `medium` | 8 | 7.54 | 2.90 |
| `long` | 8 | 21.44 | 5.80 |

Source: `raw/requests.tsv` in the [data set of the time model](../../data/2026-10-31-phase4-model/README.md).

## Change

The change has two parts.

1. The graph has one new input, `info`. It has 4 rows of 32-bit integers. The first `n_kv` columns hold the KV cells. The next columns hold the tokens of the pass.
   - A KV cell has its position, a set of sequences, and the 2D positions `x` and `y` of M-RoPE. The position is -1 for an empty KV cell. The set has one bit for each sequence of the pass. A pass can have 32 sequences or less.
   - A token has its position, the number of its sequence in the pass, `x`, and `y`.
   - The input has 16 bytes for each KV cell and each token. For `n_kv` = 6,912 and 2,048 tokens, it is 143 KB. The attention mask is 28 MB.
2. A new operation, `KQ_MASK`, writes the attention mask on the GPU from `info`. One thread writes 16 bytes of 4 rows. The CPU backend has the same operation, for the tests.

The attention mask has the same dimensions, the same type, and the same values as before. The type is F16 with flash attention. A value is 0 or `-inf`. The input `info` replaces the input of the attention mask. The number of input copies in a pass does not change.

The graph uses the new operation only if all these conditions are true. If one is not true, the host writes the attention mask as before.

- The device of the layers has the operation. A test at the start checks this, as for the other fused operations.
- The cells of the pass are in one group of sequences. This is true with `--kv-unified` or with one sequence.
- The model has no `SWA` and no `ALiBi`.
- The pass has 32 sequences or less. The KV cache is on the GPU.

The value `LLAMA_KQ_MASK_DEVICE=0` makes the host write the attention mask. The value `LLAMA_KQ_MASK_CHECK=1` makes the program compare each attention mask from the GPU with the attention mask from the host. The runs of the speed test do not use these values.

## Equivalence class

We select `exact`. The kernel compares integers. It does not calculate a new value. It writes the same two values as the host: 0 and `-inf`. The kernels for flash attention read the same attention mask.

The next table gives the tests of the equivalence class `exact`. The [method](../method.md) gives the first and the last test.

| Test | Result that is necessary |
|---|---|
| Answers | The answers of the 8 runs of the private benchmark are the same bit for bit. `self_repeat_a`, `self_repeat_b`, and `cross_identical_all` are true. `cross_max_abs_dp` is 0.0. No decision has a different top answer. All guards are true. |
| `test-backend-ops` | The result of each case of `KQ_MASK` is the same on the GPU backend and on the CPU backend. The error limit is 0. A case with one value `-inf` and one other value is an error. |
| Compare of the attention masks | With `LLAMA_KQ_MASK_CHECK=1`, each pass of the requests of the benchmark gives 0 different values. We compare the bytes of the attention mask from the GPU with the bytes of the attention mask from the host. |
| `compute-sanitizer` | The tool `memcheck` finds 0 errors for one request of each type. |

If one test is not true, the change is not `exact`. We then stop and speak to the owner of the project before we go on.

## Arms and schedule

| Arm | Build | Source |
|---|---|---|
| `D` | `build:lead-30489` | The branch `dmk/stack` of the fork, commit `6b8b65e66`. It is tag `b12004` plus change 0001. |
| `L` | `build:0004-gpu-mask` | The branch `dmk/0004-gpu-mask` of the fork. The results entry gives the commit. We do not change the code after the measurement window starts. |

Arm `D` is the build without the change. The two arms use the same image, the same model (mix8), the same GPU, and the default server flags: `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`. The measurement window sets the default power limit of 200 W and the default clocks.

The schedule has two parts, as in the [test of pull request 30489](2026-10-31-lead-30489.md). A seed sets the random sequence of the two arms in each block. Each run of the private benchmark starts a new server.

- The private benchmark: 4 blocks, seed 4001. Each run writes its answers with `--write-reference`.
- `llama bench`: 4 blocks, seed 4002. It uses the prompt lengths 64, 256, and 3,000 tokens. It measures each length 5 times.

```
python3 harness/make_schedule.py --prefix gpumask-bench --blocks 4 --seed 4001 \
    --arm D=build:lead-30489 --arm L=build:0004-gpu-mask --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix gpumask-llb --blocks 4 --seed 4002 \
    --arm D=build:lead-30489 --arm L=build:0004-gpu-mask --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

The two parts run in one measurement window. The harness records the load of the host before and after each run.

## How we calculate the result

For each block and each metric, the ratio is the value of arm `L` divided by the value of arm `D`. The estimate is the mean of the 4 ratios. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples and the default seed of `ab_stats.py`. The metrics of the private benchmark are `lat_ms`, `short_ms`, `medium_ms`, `long_ms`, and `rps8`. The metrics of `llama bench` are the prompt speeds at 64, 256, and 3,000 tokens.

We give no equivalence margin. The test looks for a gain. If an interval includes 0, we write this fact. We do not make a decision from the point estimate only.

The interval from 4 blocks is small, because it cannot be wider than the range of the 4 ratios. The [entry of pull request 30489](2026-10-31-lead-30489.md) gives the same limit.

## Prediction

The prediction starts from the numbers of phase 4. The phase-4 gain is the copy time plus the host time of the attention mask, less 1.7 % of the copy time. We lower it by the time of the new work. The new work has three parts.

- The GPU uses time to write the attention mask. The attention mask of a large pass is 28 MB. At about 330 GB/s, this is 85 µs.
- The host uses about 30 µs for each pass to write `info`. The copy of `info` uses about 15 µs for each pass.
- Each pass has one more kernel launch.

[INFERENCE: the values of the new work come from the number of bytes and from the rate of the DRAM of the GPU. We did not measure them before this entry.] The new work is 0.03 ms for each pass of a small request. It is 0.3 ms for a `medium` request and 0.4 ms for a `long` request.

| Request type | Gain in phase 4 (ms) | Predicted gain (ms) | Predicted gain (% of the time of arm `D`) |
|---|---:|---:|---:|
| `short1q` | 0.7 | 0.7 | 1.5 |
| `short2q` | 1.2 | 1.1 | 1.3 |
| `medium` | 10.3 | 10.0 | 2.8 |
| `long` | 26.9 | 26.5 | 3.4 |

The next table gives the same prediction for the metrics of the private benchmark. The mean of `short_ms` has the weights 12 for `short1q` and 20 for `short2q`. The mean of `lat_ms` uses the 48 cases.

| Metric | Predicted gain (ms) | Predicted change of arm `L` against arm `D` (%) | Range of the prediction (%) |
|---|---:|---:|---|
| `short_ms` | 0.95 | -1.3 | -0.3 to -2.3 |
| `medium_ms` | 10.0 | -2.8 | -1.8 to -3.8 |
| `long_ms` | 26.5 | -3.4 | -2.4 to -4.4 |
| `lat_ms` | 6.7 | -2.8 | -1.8 to -3.8 |

The percent of arm `D` uses the times of arm `D` that follow from the [test of pull request 30489](2026-10-31-lead-30489.md). Arm `D` is 1.2 % to 2.3 % faster than build `b12004` for these metrics. We calculated the times of arm `D` from the times of phase 4 and from these gains.

The type of work that the change removes is overhead: the copy and the host time. The change does not change the necessary work.

### `llama bench`

The attention mask of `llama bench` is small. A pass of 64 or 256 tokens has `n_kv` = 256 and an attention mask of 32 KB or 128 KB. The prompt of 3,000 tokens has 2 passes with attention masks of 8.4 MB and 5.9 MB. The copy of them uses 2.1 ms at 6.7 GB/s. The host time of the attention mask is about 2 ms. The new work is about 0.25 ms.

| Prompt length (tokens) | Predicted change of the prompt speed (%) | Range of the prediction (%) |
|---:|---:|---|
| 64 | 0.0 | -0.5 to +0.5 |
| 256 | 0.0 | -0.5 to +0.5 |
| 3,000 | +0.6 | +0.2 to +1.0 |

The intervals for 64 and 256 tokens can include 0. [INFERENCE: the noise of `llama bench` between blocks is 0.5 % to 1 % in the [test of pull request 30489](2026-10-31-lead-30489.md).]

### Rules for the result

- The result is as predicted if the point estimate of each metric is in its range.
- If a gain is less than 50 % of the prediction, or more than 150 % of the prediction, we find the cause before the next change. The [method](../method.md) has this rule.
- A result that is far from the prediction is not a cause to change the prediction. The results entry gives the two.

## Confidence and risks

The confidence of phase 4 is 0.70. We do not change it.

- The copy of the attention mask can be less than the model gives. The model uses the copy time from the profile of the server with 3 slots. If the copy is faster, the gain is less.
- The input `info` can use a copy that waits for the GPU, as the input of the attention mask did. The number of copies does not change. We predict that the time that the GPU waits does not change.
- The conditions of use can be not true for the requests of the benchmark. The server then writes the attention mask on the host, and the gain is 0. We check the log of the server for this. We also compare the attention masks.
- The host loop that writes `info` can use more than 30 µs for a large `n_kv`. This lowers the gain for `long`.

## Next steps

1. Run the tests of the equivalence class above on the build with the change.
2. Make the two schedule files with the commands above. Run them in one measurement window.
3. Write the data set, the results entry, and the change document `docs/changes/0004-gpu-mask.md`.
