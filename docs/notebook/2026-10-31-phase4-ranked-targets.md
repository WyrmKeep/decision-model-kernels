# Phase 4: ranked list of targets (prediction)

Written 2026-10-31, after measurement window 2 and before the first change to the code of a target.

## Task of this entry

This entry gives a ranked list of the changes that can make mix8 faster on the `RTX 3060 Ti`. It gives a prediction for each change before we write its code. The prediction is the gain of the change for each request type. Each target also has an equivalence class, an effort and a confidence. The budget of the project stops at this entry. The owner selects the next step from this list.

We do not change this entry after a measurement of a target starts. Each target that we do gets a prediction entry and a results entry.

## Inputs

| Input | Data set |
|---|---|
| Kernel time for each pass type, from the profiles of `nsys` | [2026-10-31-profiles](../../data/2026-10-31-profiles/README.md) |
| Speed of light for each kernel, from `ncu` | [2026-10-31-profiles](../../data/2026-10-31-profiles/README.md) |
| Maximum gain for each operation class, from the ablation | [2026-10-31-ablation](../../data/2026-10-31-ablation/README.md) |
| Gain of pull request 30489, from a measurement | [`2026-10-31-lead-30489`](../../data/2026-10-31-lead-30489/README.md) |
| The time model of this entry: input numbers, script and tables | [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md) |

The [phase 3 entry](2026-10-31-phase3-where-the-time-goes.md) gives the measurements. This entry uses them and adds the assumptions of each target.

## Request types

The benchmark has 48 cases. `lat_ms` is the mean time of the 48 cases. The time model of this entry uses four request types:

| Request type | Cases | Passes of one request |
|---|---:|---|
| `short1q` | 12 | One pass of about 158 tokens |
| `short2q` | 20 | A prefix of 87 tokens, then 2 suffix passes (2 × 51 tokens, then 69 tokens) |
| `medium` | 8 | A prefix of about 1,300 tokens, then 3 suffix passes (3 × 43, 2 × 4, 1 × 26 tokens) |
| `long` | 8 | Prefixes of 2,048 and about 1,173 tokens, then the same 3 suffix passes |

Source: [2026-10-31-profiles](../../data/2026-10-31-profiles/README.md), the table of the passes.

## Where the time goes

| Request type | Time of one request (ms) | Kernel time (ms) | Weight floor (ms) | Roofline floor (ms) | Ratio of time to roofline floor |
|---|---:|---:|---:|---:|---:|
| `short1q` | 47.1 | 42.4 | 7.7 | 19.2 | 2.45 |
| `short2q` | 87.1 | 85.1 | 20.7 | 39.8 | 2.19 |
| `medium` | 360.0 | 344.9 | 84.6 | 169.1 | 2.13 |
| `long` | 797.0 | 761.3 | 177.9 | 371.5 | 2.15 |

Source: `tables/floors.tsv` in [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md).

The facts for the ranked list:

1. The GPU runs a kernel during 90 % to 94 % of the time of a request. The host is not the limit. The idle time is 6 % to 10 % of the time of a request.
2. The MMQ kernels use about 63 % of the kernel time. For a large number of tokens they run at 35 % to 39 % of the nominal rate of the GPU for int8. cuBLASLt runs at about 69 % of that rate for GEMM with the same dimensions. But it does not apply the scales of groups of 32 values.
3. An MMQ thread block loads its tiles from global memory, then it waits at a barrier, then it does the MMA instructions. It does not load the next tile during the MMA work. Each SM holds only one thread block of 8 warps. Thus no other thread block uses the SM during the loads. The load instructions have about 28 % of the stall samples.
4. Each `IMMA` instruction of MMQ has 4 `I2FP`, 4 `FMUL` and 4 `FFMA` instructions after it. They apply the scales of the groups of 32 values. They are necessary for the result that we have at this time.
5. The MMVQ kernels for the suffix passes with 8 tokens or less read the weights at 90 % to 95 % of the DRAM bandwidth. They are at their limit.
6. The copies of the input to the GPU run at 6.7 GB/s, because the slot of the GPU runs at PCIe 3.0 x8. The attention mask of a large pass is 21 MB to 28 MB. The copy of it uses 3 ms to 4 ms, and the GPU waits for the copy.
7. `gdn_single` runs 32 thread blocks on 38 SMs. Each thread block does its chunks one after the other.
8. The concat before `ssm_conv` writes a new copy of the input of `ssm_conv`. It uses 12.9 ms in a pass of 2,048 tokens.

## Terms of the prediction

**Equivalence class.** `exact` means that the answers stay the same bit for bit as the answers of build D. The test is `answers_compare.py` on all 100 decisions, and `test-backend-ops` for each operation that the change touches. `re-associated` means that the sums occur in a different sequence. Then the answers can change by a small quantity. The test is the KL divergence against the KL floor of build D, plus the guards of the benchmark and the 52-decision set.

**Effort.** The work to write and test the change. `S` is 1 day or less, `M` is 2 to 5 days, `L` is 1 to 2 weeks and `XL` is more than 2 weeks.

**Confidence.** The probability that the change passes the test of its equivalence class and gives 50 % or more of the gain of its prediction. We give it before the test.

**Expected gain.** The gain of the prediction for `lat_ms`, multiplied by the confidence. It is not a measurement.

The script `make_tables.py` of the [data set of the time model](../../data/2026-10-31-phase4-model/README.md) calculates all numbers below. `raw/targets.tsv` gives the assumption of each target.

## Ranked list

The rank comes from the expected gain of `lat_ms`. Each cell gives the gain of the prediction for one request, in ms and in percent of that request type.

| Rank | Target | Equivalence class | Effort | Confidence | `short1q` | `short2q` | `medium` | `long` | `lat_ms` | Expected gain |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `G`. MMQ as a pipeline with `cp.async` tile loads | `exact` | `L` | 0.40 | 6.6 (14.0 %) | 12.4 (14.2 %) | 52.5 (14.6 %) | 120.8 (15.2 %) | 14.8 % | 5.9 % |
| 2 | `J`. Ragged batching for the suffixes of the questions | `re-associated` | `L` | 0.45 | 0 | 13.0 (14.9 %) | 32.9 (9.1 %) | 32.9 (4.1 %) | 6.8 % | 3.1 % |
| 3 | `N`. New MMQ kernels, after `G` | `re-associated` | `XL` | 0.20 | 5.1 (10.8 %) | 9.5 (11.0 %) | 36.6 (10.2 %) | 81.6 (10.2 %) | 10.4 % | 2.1 % |
| 4 | `B`. Attention mask on the GPU | `exact` | `M` | 0.70 | 0.7 (1.6 %) | 1.2 (1.3 %) | 10.3 (2.9 %) | 26.9 (3.4 %) | 2.9 % | 2.0 % |
| 5 | `E`. Pull request 30489: GLU fused into the quantization | `exact` | `S` | 0.95 | 0.6 (1.2 %) | 1.1 (1.2 %) | 7.2 (2.0 %) | 17.9 (2.2 %) | 2.0 % | 1.9 % |
| 6 | `C`. Concat folded into `ssm_conv` | `exact` | `M` | 0.70 | 0.9 (1.9 %) | 2.4 (2.7 %) | 9.8 (2.7 %) | 21.4 (2.7 %) | 2.7 % | 1.9 % |
| 7 | `F`. Fusions of RMS norm kernels, element-wise kernels and quantization kernels | `exact` | `M-L` | 0.50 | 1.4 (3.0 %) | 2.7 (3.1 %) | 12.8 (3.6 %) | 30.2 (3.8 %) | 3.6 % | 1.8 % |
| 8 | `H`. MMQ tile and occupancy adjustments for this GPU | `exact` or `re-associated` | `S-M` | 0.30 | 1.4 (3.0 %) | 2.7 (3.1 %) | 15.7 (4.4 %) | 36.8 (4.6 %) | 4.2 % | 1.3 % |
| 9 | `A`. Output head for the answer tokens only | `exact` | `M` | 0.85 | 1.6 (3.4 %) | 3.2 (3.7 %) | 4.9 (1.4 %) | 4.9 (0.6 %) | 1.4 % | 1.2 % |
| 10 | `D`. Value-split path of `gdn_single` | `exact` | `S` | 0.45 | 1.0 (2.2 %) | 0.6 (0.7 %) | 7.5 (2.1 %) | 19.5 (2.4 %) | 2.1 % | 0.9 % |
| 11 | `L`. Smaller KV window for flash attention (pull request 28495) | `exact` | `M` | 0.50 | 0.2 (0.4 %) | 0.8 (0.9 %) | 4.6 (1.3 %) | 10.0 (1.3 %) | 1.2 % | 0.6 % |
| 12 | `M`. One kernel for each layer (megakernel) | `re-associated` | `XL` | 0.10 | 2.2 (4.7 %) | 5.3 (6.1 %) | 15.1 (4.2 %) | 27.4 (3.4 %) | 4.1 % | 0.4 % |
| 13 | `O`. Cache of the prompt tokens | `exact` | `S-M` | 0.50 | 0.4 (0.9 %) | 0.9 (1.0 %) | 2.9 (0.8 %) | 5.7 (0.7 %) | 0.8 % | 0.4 % |
| 14 | `K`. CUDA graphs for passes with more than one token | `exact` | `M` | 0.30 | 0.7 (1.4 %) | 2.0 (2.2 %) | 3.4 (0.9 %) | 3.2 (0.4 %) | 0.9 % | 0.3 % |

Source: `tables/ranked.tsv` in [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md).

The gains do not add. Some targets change the same time:

- `G`, `H` and `N` change the same MMQ time. `H` is a short test of the cause that `G` removes. `N` can give a gain only after `G`.
- `J` removes 2 of the 3 output passes of `medium` and `long`. After `J`, `A` gives a gain for the output head of one pass only.
- `B` and `L` make the attention mask smaller or remove its copy.
- `E` is one of the fusions of `F`. The numbers of `F` do not include `E`.

## Recommended sequence

The ranked list has the sequence of the expected gain. The sequence below also uses the effort and the risk. Tests with a small effort and a high confidence come first.

1. **`E`.** The change is complete. We measured a gain of 1.94 % of `lat_ms` (95 % confidence interval 1.50 to 2.38). The answers are the same. Use it as the first change.
2. **`D` and `H`.** These targets are small changes of the code that selects a path, or of the tile adjustments. Each test also gives information about a cause in the list. `H` shows how much of the difference between the MMQ rate and the nominal rate comes from the missing overlap of loads and MMA work.
3. **`A`.** The best target for short requests with a high confidence. The `triage` use of the model sends short requests.
4. **`B` and `C`.** These changes are `exact` and have a high confidence. `B` has the largest gain for `long`. `C` has a gain for all types.
5. **`G`.** The largest target. Do it after `H`, because the result of `H` can change the method of `G`.
6. **`J`.** The largest target for `short2q`. `ssm_conv`, the GDN kernels and the code that divides the batch must accept sequences of different lengths.
7. **`F`, `L`, `K`, `O`.** Smaller targets. Do them if the rule to stop does not stop the project.
8. **`N` and `M`.** We do not recommend them at this time. The effort is very large and the confidence is low.

## Prediction of the sum

The sum uses targets `A`, `B`, `C`, `D`, `E`, `F`, `G` and `J`. It does not use `H`, because `H` changes the same MMQ time as `G`. It does not use `N`, because `N` comes after `G`.

| Case | `lat_ms` (ms) | Change |
|---|---:|---:|
| At this time (time model) | 240.9 | 0 |
| All 8 targets give the gains of their predictions | 153.7 | −36.2 % |
| Each gain multiplied by its confidence | 195.9 | −18.7 % |

Source: `tables/sum.tsv` in [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md).

These numbers are the sum of the gains. Thus they are approximate.

The time of a request is then 1.4 times (`long`) to 1.8 times (`short1q`) the roofline floor. Most of the remaining difference is in MMQ.

## Rule to stop the project

The project stops when the expected gain is less than 3 % for each request type. Three targets are above this limit for one or more request types:

| Target | Largest expected gain of one request type |
|---|---:|
| `J` | 6.7 % (`short2q`) |
| `G` | 6.1 % (`long`) |
| `A` | 3.2 % (`short2q`) |

Source: column `max_expected_pct` of `tables/ranked.tsv` in [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md).

Thus the rule to stop does not stop the project at this time.

## Targets

### `G`. MMQ as a pipeline with `cp.async` tile loads

- **Cause.** The loop of the MMQ kernel loads a tile, waits at a barrier, and then does the MMA work. One thread block of 8 warps fills an SM. Thus the load time and the MMA time add.
- **Change.** Load the next tile with `cp.async` while the warps do the MMA work on the tile that they use. Keep the 4-bit values of Q4_0 in shared memory, and make the int8 values from them in registers. This makes space for two or three steps of the pipeline.
- **Equivalence class.** `exact`, if the kernel divides `K` into the same parts and the sums of each output have the same sequence.
- **Prediction.** MMQ is 1.35 times faster at 1,025 tokens or more, 1.30 times at 65 to 256 tokens and 1.15 times at 9 to 64 tokens.
- **Risk.** Registers: the kernel uses 216 to 254 registers for each thread at this time. The instructions that make the int8 values add work to the MMA loop.

### `J`. Ragged batching for the suffixes of the questions

- **Cause.** The hybrid memory uses the equal split. In each ubatch all sequences have the same number of tokens. Thus 3 questions of 43, 47 and 73 tokens use 3 suffix passes. Each pass reads all weights again.
- **Change.** One pass with the 3 suffixes of different lengths. `ssm_conv`, the GDN kernels and the code that divides the batch must accept sequences of different lengths.
- **Equivalence class.** `re-associated`: the GDN kernel and the MMQ tile change for some tokens.
- **Prediction.** One pass of 163 tokens replaces 3 suffix passes of `medium` and `long`. One pass of 171 tokens replaces 2 suffix passes of `short2q`.
- **Risk.** Many kernels change. The code that divides the batch is in the hybrid memory of llama.cpp, not in the CUDA backend.

### `N`. New MMQ kernels, after `G`

- **Change.** Larger tiles for each warp, `ldmatrix` for the two operands, a better sequence of the scale instructions, and adjustments for this GPU only.
- **Prediction.** MMQ is 1.30 times faster than after `G`.
- **Risk.** The scales of the groups of values of Q4_0 and Q8_0 add 12 instructions to each `IMMA`. Thus MMQ cannot get the cuBLASLt rate.

### `B`. Attention mask on the GPU

- **Cause.** The host writes the attention mask (`n_kv` × tokens, FP16) and sends it to the GPU at 6.7 GB/s. The GPU waits for the copy.
- **Change.** Send the positions and the sequence of each KV cell to the GPU. They are small. Then a kernel writes the attention mask on the GPU.
- **Equivalence class.** `exact`: the attention mask has the same values.
- **Prediction.** It removes the time of the copies (0.7 ms to 21.4 ms for each request) and the host time of the attention mask (5.8 ms or less).

### `E`. Pull request 30489: GLU fused into the quantization

- **Result of the measurement.** `lat_ms` is 1.94 % less (95 % confidence interval 1.50 to 2.38). The answers are the same on all 100 decisions. The [entry about pull request 30489](2026-10-31-lead-30489.md) gives the data.
- **Prediction.** The same gain in production.

### `C`. Concat folded into `ssm_conv`

- **Cause.** The graph puts the conv state before the new input with a concat. Then `ssm_conv` reads the result.
- **Change.** `ssm_conv` reads the conv state and the new input directly. It also writes the new conv state.
- **Equivalence class.** `exact`: only the movement of data changes.
- **Prediction.** It removes 90 % of the time of the concat.

### `F`. Fusions of RMS norm kernels, element-wise kernels and quantization kernels

- **Change.** One kernel for an RMS norm, its weight and the quantization for the next MMQ. Also fusions of the element-wise kernels of the GDN layers.
- **Equivalence class.** `exact`, if each value has the same operations in the same sequence. A SASS check must make sure that the compiler does not put two operations in one `FFMA` instruction. The two operations multiply two numbers and add a third number.
- **Prediction.** It removes 35 % of the time of the quantization kernels, the RMS norm kernels and the element-wise kernels.

### `H`. MMQ tile and occupancy adjustments for this GPU

- **Change.** Other dimensions of the tile and other `__launch_bounds__`. Thus two thread blocks fit on one SM. Then the loads of one thread block occur during the MMA work of the other thread block.
- **Equivalence class.** `exact` for a matrix without stream-k. `re-associated` for a matrix with stream-k, because the parts of `K` change.
- **Prediction.** It removes 8 % of the MMQ time at 1,025 tokens or more and 5 % at less than 1,025 tokens.

### `A`. Output head for the answer tokens only

- **Cause.** The output head reads 675 MB of weights for each pass with an output. The decision uses only the logits of the label tokens.
- **Change.** Calculate the logits of the label tokens only.
- **Equivalence class.** `exact`: the logits of the label tokens come from the same multiply-add operations.
- **Prediction.** A gain of 1.6 ms for each pass with an output. The ablation of the output head gives the same number.

### `D`. Value-split path of `gdn_single`

- **Cause.** The code that selects the path uses the whole-head path because 32 heads × 2 is more than 38 SMs. Then 32 thread blocks run on 38 SMs, one chunk after the other.
- **Change.** Use the value-split path with 128 thread blocks on this GPU.
- **Equivalence class.** `exact` (we must make sure of it): each thread block calculates different value columns with the same operations.
- **Prediction.** `gdn_single` is 1.6 times faster.

### `L`. Smaller KV window for flash attention (pull request 28495)

- **Cause.** With `--kv-unified`, a slot behind other slots reads more KV cells than are necessary. The kernel for flash attention then uses 87 % more time.
- **Prediction.** It removes 30 % of the time of flash attention.

### `M`, `O` and `K`

- **`M`.** One kernel for each layer removes most launches and some of the movement of data in the memory. The GPU runs a kernel during 90 % or more of the time. Thus the gain is small for a very large effort.
- **`O`.** A cache of the tokens of the prompt prefix removes 70 % of the time to make the prompt and its tokens.
- **`K`.** CUDA graphs remove part of the idle time in the passes. The shape of a pass changes with each request. Thus a new graph can be necessary for each pass, and that can use more time than it removes.

## Causes of an incorrect prediction

- The time model uses the kernel time of the profiles of `nsys`. A kernel can be faster or slower when the kernel before it changes.
- The ratios of `G`, `H`, `D`, `F` and `L` come from the profiles, not from a test of the change.
- The `PCIe` speed is a fact of this host. If it is different, the gain of `B` is different.
- The time model does not include the time of the first request after a period with no request.
