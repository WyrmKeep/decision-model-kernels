# MMQ tile and occupancy for this GPU (target H): result of the sweep

Written 2026-10-31, after the measurement window `0006-sweep-r1` (05:40 to 05:53 UTC) and the measurement window `0006-sweep-r1b` (07:05 to 07:10 UTC).

## Task of this entry

This entry gives the result of the work on target `H` of the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md). Target `H` changes the tile and the occupancy of the MMQ kernel for the `RTX 3060 Ti`.

We ran a sweep with a micro-benchmark. We did not build a change of llama.cpp, and we did not run a test with two arms. We do not use target `H`. The section "Decision" gives the cause.

There is no prediction entry for the sweep. The prediction is the text for `H` in the phase-4 entry. The section "Prediction against result" gives it. The [README of the data set](../../data/2026-10-31-0006-mmq-tiles/README.md) gives the raw files, the tables, and the conditions of the runs.

The section "Effect on target `G`" gives the data of the sweep for target `G`.

## Result in short

- The sweep measured 142 tile adjustments of the MMQ kernel. It used 6 weight shapes of the model and 29 numbers of tokens, from 9 to 2,048. It did not use the two shapes with 32 rows, `ssm_alpha` and `ssm_beta`.
- The baseline reproduces the speed of the model. It gives 54.3 TOPS int8 for `ffn_up` at 2,048 tokens.
- At 256 tokens or more, the three large shapes have no adjustment that is more than 1.05 times faster than the baseline. These shapes are the two Q4_0 shapes and Q8_0 with 8,192 rows. The noise is 2 % to 3 %.
- The gains at 256 tokens or more are in the small Q8_0 shapes that run the fix-up kernel. The maximum is 1.44 times for Q8_0 with 1,024 rows. The two other small shapes get 1.10 times.
- At less than 256 tokens, 57 of 174 cells have an adjustment that is 10 % faster or more. In all of these cells, the baseline runs the fix-up kernel and the fast adjustment gives other bits. The equivalence class of these changes is `re-associated`, not `exact`.
- The `oracle` removes 6.4 %, 2.9 %, 1.3 %, and 0.6 % of the MMQ time of the request types `short1q`, `short2q`, `medium`, and `long`. This is 0.79 % of `lat_ms`. With the limit `guard3` it is 0.67 %.
- The fastest rows with the same bits as the baseline remove 0.21 % of `lat_ms`. With `guard3` they remove 0.11 %.
- The rules that the code can have do not give a gain. The rule "64-row tiles where the baseline runs the fix-up kernel" makes `lat_ms` 2.40 % slower. The same rule for Q8_0 makes it 0.43 % slower.
- The phase-4 prediction was 8 % of the MMQ time at 1,025 tokens or more. The `oracle` gives 0.54 % for the tokens from 1,025 to 2,048.
- More thread blocks on one SM do not give back load time at 1,173 tokens or more. The time is 1.09 to 1.27 times the baseline for two blocks of 8 warps. It is 1.04 to 1.17 times for three blocks of 4 warps.
- New estimate for target `G`: MMQ is 1.08 times faster at 1,025 tokens or more, in a range from 1.00 to 1.20. The phase-4 value was 1.35.

## Prediction against result

The phase-4 entry gives this prediction for `H`.

- **Equivalence class.** `exact` for a matrix without stream-k. `re-associated` for stream-k.
- **Change.** Other tile dimensions and other `__launch_bounds__`, to get two thread blocks on one SM.
- **Prediction.** The change removes 8 % of the MMQ time at 1,025 tokens or more, and 5 % at less. The confidence is 0.30.

The next table gives the time that a change removes from one request. The columns "Predicted" come from the phase-4 model. The columns "Oracle" are the fastest row for each cell. The columns "Same bits" are the fastest row with the same bits as the baseline. The weights come from the census.

| Request type | Predicted (ms) | Predicted (% of request) | Oracle (ms) | Oracle (% of request) | Same bits (ms) | Same bits (% of request) |
|---|---:|---:|---:|---:|---:|---:|
| `short1q` | 1.4 | 3.04 | 1.65 | 3.49 | 0.60 | 1.28 |
| `short2q` | 2.7 | 3.09 | 1.37 | 1.57 | 0.24 | 0.27 |
| `medium` | 15.7 | 4.37 | 2.68 | 0.74 | 0.86 | 0.24 |
| `long` | 36.8 | 4.62 | 2.89 | 0.36 | 0.65 | 0.08 |
| `lat_ms` | 10.2 | 4.25 | 1.91 | 0.79 | 0.50 | 0.21 |

Source: the tables `g-h-scenarios.tsv` and `rules.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The result is lower than the prediction for `short2q`, `medium`, and `long`. It is higher for `short1q`. The prediction for the large `N` was incorrect. The baseline runs one thread block for each tile with `J` of 128 for the large shapes, and no adjustment is faster at these `N`. The prediction for the small `N` is near the `oracle`, but only for rows with other bits.

## Method

The micro-benchmark `mmq_sweep` uses the launch function of the MMQ kernel of llama.cpp (`launch_mul_mat_q`). It does not run the model. The [README of the data set](../../data/2026-10-31-0006-mmq-tiles/README.md) gives the source and the build.

- **Tile adjustments.** Each adjustment sets the threads of a thread block and the thread blocks for each SM. It also sets the rows `I` and the tokens `J` of the tile. All adjustments use stream-k. There are 142 adjustments for the types Q4_0 and Q8_0.
- **Baseline adjustment.** It has 256 threads, 128 rows, and one thread block for each SM. `J` has the values from 8 to 128, in steps of 8.
- **Other adjustments.** 256 threads with two blocks for each SM. 128 threads with 64 rows and one, three, or four blocks for each SM. 512 threads with 256 rows.
- **Grid rules 0, 1, and 4.** Rule 0 is the rule of the baseline. It starts one thread block for each tile if the tiles use 90 % or more of the SMs. If not, it starts one block for each SM, and each block gets a part of `K`. Rule 1 starts one thread block for each tile. Rule 4 is rule 0, but it counts the SMs times the blocks for each SM.
- **Shapes.** Q4_0: 9,216 rows by 2,560 (`ffn_up`, `ffn_gate`) and 2,560 rows by 9,216 (`ffn_down`). Q8_0: 8,192 by 2,560, 4,096 by 2,560, 2,560 by 4,096, and 1,024 by 2,560.
- **Tokens.** `9 13 16 26 30 32 43 47 51 54 64 67 69 84 86 94 102 108 120 128 129 151 158 256 512 1024 1173 1288 2048`.
- **Data.** The weights and the activations are random. The program cycles through 48 MB of weights or more, and the L2 cache does not keep them.
- **Time.** One row is the lowest time of three runs. Each run starts the launch in a loop for about 8 ms, with 5 launches at the minimum.
- **Bits.** The program compares the output of each row with the output of the baseline adjustment, bit for bit. The column `ndiff` gives the number of values that are different.
- **Request time.** The model uses the passes of the 48 cases of the benchmark. For each pass, the model uses the cell with the nearest number of tokens.

The model has three limits. It does not use the 32-row shapes, which are 1.45 % of the MMQ time in the profile of phase 3. It counts the MMQ kernels only. It does not use the gaps between kernels.

## Results

### The baseline adjustment

The baseline gives 45.8 to 54.3 TOPS int8 at 2,048 tokens. The profile of phase 3 gives 35 % to 39 % of the int8 maximum (149.3 TOPS) for the production kernel with a large matrix. This is 52 to 58 TOPS.

| Shape | Tokens | Time (µs) | TOPS int8 | Fix-up kernel |
|---|---:|---:|---:|---|
| Q4_0 9,216 by 2,560 | 2,048 | 1,779.5 | 54.3 | no |
| Q4_0 2,560 by 9,216 | 2,048 | 1,804.9 | 53.5 | no |
| Q8_0 8,192 by 2,560 | 2,048 | 1,602.1 | 53.6 | no |
| Q8_0 1,024 by 2,560 | 2,048 | 234.6 | 45.8 | yes |

Source: the table `baseline.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

### Where an adjustment changes the time

The next table gives the MMQ time that a rule removes, as a part of the MMQ time of each group of passes. The groups use the number of tokens of the pass.

| Tokens of the pass | `oracle` | `oracle` with `guard3` | Same bits | Same bits with `guard3` |
|---|---:|---:|---:|---:|
| 9 to 64 | 4.30 % | 4.17 % | 1.46 % | 1.31 % |
| 65 to 256 | 3.34 % | 3.20 % | 0.60 % | 0.49 % |
| 257 to 1,024 | 0.24 % | 0.12 % | 0.05 % | 0.00 % |
| 1,025 to 2,048 | 0.54 % | 0.28 % | 0.19 % | 0.00 % |

The limit `guard3` uses a fast row only if it is 3 % faster than the baseline.

Source: the table `rules-by-n.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The passes from 1,025 to 2,048 tokens have 65 % of the MMQ time of the model. A gain there is only possible with an adjustment that is faster than the baseline at large `N`. No adjustment is faster, in the noise of the measurement.

- The noise of the same launch is 0.2 % to 0.9 % (median) and 1.8 % to 2.7 % (90th percentile). This is for 5 of the 6 shapes.
- The rows of Q8_0 with 8,192 rows in the first measurement window have more noise. The next section gives the cause. The tables use the second run for this shape.
- Each row of the `oracle` is the lowest of about 200 rows with noise. The `oracle` is thus faster than the true time of a rule. The limit `guard3` removes most of this effect.

### The equivalence class of the fast rows

- In the cells with a gain of 10 % or more, the fast row has other bits than the baseline. This is true for 57 of 57 cells. The baseline runs the fix-up kernel in each of these cells.
- A row has the same bits as the baseline only if it divides `K` as the baseline does. This is true if the two use one thread block for each tile. It is also true if the two use the same grid with the same tile.
- 11 of 174 cells have a row with the same bits and a gain of 3 % or more. No cell has a gain of 10 %.
- The Q4_0 shape with 9,216 rows has no cell with the fix-up kernel. Its best rows with the same bits are 1.09 times faster (`J` of 80 at 151 and 158 tokens) and 1.08 times faster (43 to 47 tokens, and 9 to 16 tokens).

Source: the tables `class.tsv` and `best-per-shape.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

A gain of more than 3 % thus has the equivalence class `re-associated`. The answers change at the level of one unit in the last position of a floating point number. The method has the test of KL divergence against the KL floor for this equivalence class.

### Gain for each request type

The next table gives the time that each rule removes from one request. A value with a minus sign shows a slower request.

| Rule | `short1q` (%) | `short2q` (%) | `medium` (%) | `long` (%) | `lat_ms` (%) |
|---|---:|---:|---:|---:|---:|
| `oracle` | 3.49 | 1.57 | 0.74 | 0.36 | 0.79 |
| `oracle` with `guard3` | 3.35 | 1.52 | 0.51 | 0.27 | 0.67 |
| Fastest row with the same bits | 1.28 | 0.27 | 0.24 | 0.08 | 0.21 |
| Fastest row with the same bits, `guard3` | 1.17 | 0.23 | 0.04 | 0.02 | 0.11 |
| 64-row tiles where the baseline runs the fix-up kernel, `J` of the baseline | -4.07 | -8.87 | -1.64 | -0.82 | -2.40 |
| The same rule with the best `J` | -1.32 | -6.45 | -0.89 | -0.42 | -1.49 |
| The same rule for Q8_0 only, `J` of the baseline | 0.27 | -2.01 | -0.30 | -0.12 | -0.43 |
| The same rule for Q8_0 only, with the best `J` | 1.32 | -0.97 | 0.02 | 0.04 | -0.06 |

The values in % are parts of the request times of the phase-4 model: 47.1, 87.1, 360.0, and 797.0 ms.

Source: the table `rules.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The first four rules must have one adjustment for each shape and each number of tokens. The code of llama.cpp has no such table. The last four rules are rules that the code can have.

### Cause of the slower time with the rules of the code

A tile with 64 rows and one thread block for each tile is faster for a short `K` and slower for a long `K`. The long `K` has more work for each tile. Stream-k divides `K` between thread blocks. This uses the SMs better.

| Operation | Shape | Time removed by the rule with `J` of the baseline |
|---|---|---:|
| `attn_gate` | Q8_0 4,096 by 2,560 | 4.74 % |
| `attn_k`, `attn_v` | Q8_0 1,024 by 2,560 | 1.75 % |
| `attn_qkv`, `attn_q` | Q8_0 8,192 by 2,560 | -0.59 % |
| `ssm_out`, `attn_output` | Q8_0 2,560 by 4,096 | -9.73 % |
| `ffn_down` | Q4_0 2,560 by 9,216 | -17.00 % |

Source: the table `rules-by-op.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

A rule that gives a gain must use the shape (`M` and `K`) and the number of tokens.

### Rows that other work changed

A task of a different agent ran on the GPU from 05:45:30 to 05:48:30 UTC, in the measurement window `0006-sweep-r1`. We found the overlap after the measurement window. It changed the rows of Q8_0 with 8,192 rows, and possibly the start of the rows with 4,096 rows.

We ran the two shapes again in the measurement window `0006-sweep-r1b`, with the same program and the same flags. The table `disturbed.tsv` compares the two measurement windows.

- For 4,096 by 2,560, the median ratio of the two measurement windows is 1.001 to 1.004 for each number of tokens. We can see no change in this shape.
- For 8,192 by 2,560, the 90th percentile of the ratio is 1.07 to 1.11 for 8 of the 29 numbers of tokens. The largest ratio is 1.15. The first measurement window is slower in these rows.

The tables use the rows of `0006-sweep-r1b` for these two shapes. The rows of `0006-sweep-r1` stay in `raw/`.

## Effect on target `G`

Target `G` is a pipeline for the tile loads with `cp.async`. The phase-4 prediction for `G` is that MMQ is 1.35 times faster at 1,025 tokens or more. This prediction uses the assumption that the load time and the MMA time add in one thread block. A pipeline can then remove a large part of the load time.

### Data of the sweep for `G`

The baseline has one thread block of 8 warps on each SM, with `J` of 128. It uses 216 or 254 registers and 57,856 bytes of shared memory. Three groups of adjustments let a second block use the SM while the first block loads. We compare them with the baseline in 13 cells. These cells have 1,173 tokens or more, and the two adjustments run one thread block for each tile.

| Adjustment | Warps for each SM | Time as a ratio of the baseline: lowest | Median | Highest |
|---|---:|---:|---:|---:|
| Two blocks of 4 warps, 64 rows, same `J` | 8 | 1.005 | 1.086 | 1.148 |
| Two blocks of 8 warps, 128 rows, `J` of 64 or less, 128 registers | 16 | 1.089 | 1.153 | 1.265 |
| Three blocks of 4 warps, 64 rows, `J` of 80 | 12 | 1.040 | 1.088 | 1.173 |

Source: the table `g-summary.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md). The table `g-test.tsv` has each cell.

In the 13 cells, no adjustment is faster than the baseline by more than the noise (2 % to 3 %).

The sweep has many `J` values for one thread block of 8 warps. The next table gives the time at 2,048 tokens, as a ratio of the time with `J` of 128.

| Shape | `J` | Shared memory (bytes) | Registers | Time as a ratio of `J` of 128 |
|---|---:|---:|---:|---:|
| Q4_0 9,216 by 2,560 | 64 | 48,384 | 156 | 1.247 |
| Q4_0 9,216 by 2,560 | 96 | 53,632 | 218 | 1.150 |
| Q4_0 9,216 by 2,560 | 128 | 57,856 | 216 | 1.000 |
| Q8_0 8,192 by 2,560 | 64 | 48,384 | 168 | 1.308 |
| Q8_0 8,192 by 2,560 | 96 | 53,632 | 228 | 1.170 |
| Q8_0 8,192 by 2,560 | 128 | 57,856 | 254 | 1.000 |

Source: the tables `j-ladder.tsv` and `j-fit.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The time has a part that gets smaller when `J` gets larger. This is the work of each column tile. We use `time = A + B / J` with the times at `J` of 64 and `J` of 128. Then `B / J` is 25 % of the time at `J` of 128 for Q4_0, and 31 % for Q8_0. If a change removes all of this work, MMQ can be 1.33 times (Q4_0) or 1.45 times (Q8_0) faster. This is a ceiling, and it uses two points only.

### Meaning of the data

1. Thread blocks that run at the same time on one SM do not remove time at large `N`. The first adjustment is the best test. It has the same number of warps and the same `J` as the baseline. The two blocks do not wait for each other at a barrier. The loads of one block can run during the MMA work of the other block. The time is 0.5 % to 15 % more than the baseline.
2. More warps make the time worse. 12 warps are 4 % to 17 % slower. 16 warps are 9 % to 27 % slower.
3. The three adjustments add other work. A smaller tile loads the same `y` data again for each block of 64 rows. It also does less MMA work for each load from shared memory. We did not measure this work apart from the gain. The sweep thus gives no number for the part of the time that is a load wait. It shows that more blocks on one SM do not remove this time.
4. The part that gets smaller when `J` gets larger has more than the load wait. It has the load of the weight tile, the work to change the 4-bit values to 8-bit values, the scales, and the barriers. The sweep does not divide this part. The phase-4 value of 1.35 is the same as the ceiling for Q4_0.

### Limits of the data for `G`

1. A pipeline in one thread block with `cp.async` is a different change. It keeps `J` of 128 and 8 warps. `cp.async` writes data into shared memory with no register, and it uses a smaller number of issue slots than a load and a copy. The sweep did not measure these effects.
2. A block of Q4_0 has 18 bytes and a block of Q8_0 has 34 bytes. Thus the code that loads the weight tile reads each 4 bytes as two values of 2 bytes (`get_int_b2`). For Q4_0, the code also changes the 4-bit values to 8-bit values with `__vsub4` before it writes them to shared memory.
3. [INFERENCE: `cp.async` copies data with no change, in parts of 4, 8, or 16 bytes. The blocks of Q4_0 and Q8_0 start at positions that do not agree with these parts. Thus `cp.async` cannot load the weight tile, but it can load the `y` tile. A load of the weight tile into registers before the MMA work is a different method. The kernel uses 216 or 254 of the 255 registers that one thread can have.]
4. The [phase-0 entry](2026-10-30-phase0-upstream-survey.md) lists pull request 28572 for a `cp.async` MMQ change. Upstream closed this pull request.
5. One block with `J` of 128 uses 57,856 bytes of shared memory. The limit for one block is 101,376 bytes (the value `sharedMemPerBlockOptin` of the GPU). Two full copies of the tile are more than this limit. The `J` values above show that a smaller `J` is slower.
6. The load time is not the only cause of the low speed of MMQ. In the phase-3 profile, the IMMA pipe is active for 35 % to 41 % of the time. The stall reason `math_pipe_throttle` is 27 % to 30 % in most kernels. A cuBLASLt int8 GEMM with the same dimensions runs at about 69 % of the int8 rate (about 103 TOPS). It does not apply the scales of MMQ. The fastest MMQ adjustment gets 54 TOPS.
7. [INFERENCE: after each `IMMA` instruction, the kernel runs 4 `I2FP`, 4 `FMUL`, and 4 `FFMA` instructions (phase 3). We use the value of 16 `I2FP` operations in each clock for an SM. Then the `I2FP` pipe can limit MMQ to about 50 % of the int8 maximum. A pipeline for the loads does not remove this limit. We did not measure it.]

### New estimate for `G`

We do not change the phase-4 entry. The next table gives the estimate that we use from this point. It uses the time model of phase 4.

| MMQ speed increase at | Phase 4 | New | Low | High |
|---|---:|---:|---:|---:|
| 1,025 tokens or more | 1.35 | 1.08 | 1.00 | 1.20 |
| 65 to 256 tokens | 1.30 | 1.10 | 1.05 | 1.15 |
| 9 to 64 tokens | 1.15 | 1.08 | 1.03 | 1.10 |
| Confidence | 0.40 | 0.25 | 0.25 | 0.25 |

The model gives these gains for `lat_ms`.

| Estimate | Gain of `lat_ms` (%) | Largest gain of one request type (%) | Largest expected gain of one request type (%) |
|---|---:|---:|---:|
| Phase 4 | 14.83 | 15.16 (`long`) | 6.07 |
| New | 4.70 | 5.61 (`short2q`) | 1.40 |
| Low | 0.81 | 2.94 (`short2q`) | 0.74 |
| High | 9.25 | 9.69 (`long`) | 2.42 |

The expected gain is the gain multiplied by the confidence. The rule to stop the project of phase 4 uses a limit of 3 % for the expected gain of one request type.

Source: the table `g-h-scenarios.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The cause of each value:

- **1,025 tokens or more.** The `J` values give a ceiling of 1.33 to 1.45. [INFERENCE: `cp.async` can load only the `y` tile.] Three adjustments that overlap the loads with the MMA work are slower, and the `I2FP` pipe is a possible second limit. The middle value is 1.08, and the low value is no gain. The high value of 1.20 is for a pipeline that removes about 60 % of the part that gets smaller when `J` gets larger.
- **65 to 256 tokens and 9 to 64 tokens.** These kernels run for 18 to 242 µs and have 10 to 36 steps in `K`. The fastest rows of the sweep show that the grid and the latency limit these launches. Other tiles remove 3 % to 4 % of the MMQ time. A pipeline can decrease the effect of latency in each block. We use values that are near this gain.
- **Confidence.** The sweep shows no gain for large `N` and adds no data for a gain. The confidence changes from 0.40 to 0.25.

With the new estimate, `G` is below the limit of the rule to stop the project: 1.40 % against 3 %. The high estimate is also below it, with 2.42 %. Phase 4 had `G` above the limit with 6.07 %.

A short test can give the ceiling of `G`. Run the MMA loop of the baseline kernel with the tile loads of the first step only. The loop then has no new loads. The time is the best time that a pipeline can get. The test has one small change in the code of the sweep, one build, and one measurement window.

## Decision

We do not use target `H`. We build no change, we run no test with two arms, and we open no pull request.

- No rule that the code of llama.cpp can have gives a gain. The rules of the code give -2.40 % to -0.06 % of `lat_ms`.
- The `oracle` gives 0.79 % of `lat_ms` with the noise, and 0.67 % with `guard3`. It has one row for each shape and each number of tokens. The test of lead 30489 had an interval of 0.44 % on each side of its result. We cannot keep a table of about 170 rows in the code for this gain.
- The cells with a large gain (10 % or more of the MMQ time) have the equivalence class `re-associated`. This equivalence class must have the test of KL divergence. It also adds a difference from upstream that we must keep.
- The rule to stop the project of phase 4 uses a limit of 3 % for the expected gain of one request type. Phase 4 gave `H` 1.3 % at the maximum. The sweep gives 1.0 % at the maximum: 3.49 % of the `short1q` request, multiplied by the confidence of 0.30.

## Limits

- The micro-benchmark uses random weights and random activations. [INFERENCE: the time of the MMA kernel does not change with the values. We did not check this.]
- The launches run one after the other. A request of the model has other kernels between the MMQ launches, and the state of the L2 cache is different.
- The request model does not include the 32-row shapes, the MMVQ kernels, or the gaps between kernels. It uses the cell with the nearest number of tokens.
- Each row is the lowest of three runs. The noise of 2 % to 3 % limits all results of 3 % or less.
- The sweep has no adjustment with 64 threads, none with a tile of 96 rows, and none for the 32-row shapes.
- The program has two more grid rules, 2 and 3. They start one block for each SM, and each block gets a part of `K`. We did not run them.
- The estimate for `G` is an estimate. No test measured a pipeline.

## Data

- [Data set 2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md): the raw files of the two measurement windows, the program, the tables, and the build.
- The branch `dmk/0006-mmq-sweep` of the fork of llama.cpp of this project has the code of the micro-benchmark. The commit is `df6730c95a37831f42c9660af91ec4a3e8b464a7`.
- [Phase-4 entry](2026-10-31-phase4-ranked-targets.md): the prediction for `H` and `G`.
