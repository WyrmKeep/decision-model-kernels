# MMQ tile load pipeline (target G): result of the micro-benchmark

Written 2026-10-31, after the measurement window `0009-pipef` (10:03 to 10:04 UTC). Five measurement windows between 08:32 and 09:22 UTC came before it.

## Task of this entry

This entry gives the result of the work on target `G` of the [ranked list of phase 4](2026-10-31-phase4-ranked-targets.md). Target `G` changes the MMQ kernel. The kernel loads the next tile while the warps work on the tile that they have.

The [entry of target H](2026-10-31-0006-mmq-tiles-result.md) set a limit for the first change. A change that gives Q4_0 less than 1.08 times at 1,025 tokens or more is below the limit. We made a pipeline for Q4_0 and measured it with a micro-benchmark. The pipeline is below the limit. We stopped the work there.

We did not run a test with two arms, a test of answers, or a test with `compute-sanitizer`. We do not open a pull request. There is no prediction entry, because no run with the model came after the micro-benchmark. The [README of the data set](../../data/2026-10-31-0009-mmq-pipeline/README.md) gives the raw files and the conditions of the runs.

## Result in short

- The last version uses `cp.async` to load the Q4_0 blocks and the activations of the next step into shared memory. The MMA work runs during these loads. The operations and their sequence do not change.
- The output has the same bits as the baseline. The column `ndiff` is 0 in all 62 cells of all 13 runs.
- The mean ratio is 1.042 at 1,025 tokens or more. The limit is 1.08. The mean is 1.027 for `ffn_up` and `ffn_gate`, and 1.056 for `ffn_down`. The largest ratio of a cell is 1.088.
- At 65 to 1,024 tokens the mean ratio is 1.08 to 1.09, with a range of 1.01 to 1.15. At 9 to 64 tokens the mean is 1.03, with a range of 0.84 to 1.13. The `ffn_up` shape is slower at 16 to 32 tokens.
- The pipeline removes 18 % of the ceiling `same_loads` at 1,025 tokens or more. It removes 33 % to 40 % at 65 to 1,024 tokens.
- Each request type has 0.95 % to 2.21 % less time if the pipeline gives the ratios of the micro-benchmark. The value for `lat_ms` is 1.65 %. The ceiling of the Q4_0 operations is 6.92 % of `lat_ms`.
- The largest value of a request type is 2.21 % (`short1q`). It is below the limit of 3 % of phase 4, before we multiply it by a confidence.
- The first versions were not faster. The version that loads only the weights is 0.93 times as fast. The loads of the activations give the gain.
- The shared memory of the last version is 94,720 bytes at `J` of 128. A Q8_0 pipeline with the same method uses 111,104 bytes, and the limit is 101,376 bytes. We did not make a pipeline for Q8_0.

## Prediction against result

Phase 4 gave this prediction for `G`.

- **Equivalence class.** `exact`.
- **Change.** Load the tile of the next step with `cp.async`. Do the MMA work during the load.
- **Prediction.** MMQ is 1.35 times faster at 1,025 tokens or more, and 1.15 times faster at 9 to 64 tokens. The confidence is 0.40.

The entry of target `H` found that the value of 1.35 is the ceiling `no_loads`. It uses the full load time as the gain. That entry measured the ceiling `same_loads` and set the confidence of Q4_0 to 0.42. The probability that the change works is 0.70, and the part of the ceiling is 0.60 with a range of 0.40 to 0.80.

The next table gives the result for the Q4_0 operations. The column "Entry of `H`" is the ceiling multiplied by the confidence of 0.42. This confidence includes the probability that the change works. The column "Pipeline" is the time that the last version removes if it runs in all cells. It is the result of a change that works.

| Request type | Ceiling (% of request) | Entry of `H` (% of request) | Pipeline (% of request) | Part of the ceiling (%) |
|---|---:|---:|---:|---:|
| `short1q` | 6.93 | 2.91 | 2.21 | 31.9 |
| `short2q` | 7.32 | 3.07 | 1.96 | 26.8 |
| `medium` | 6.56 | 2.76 | 0.95 | 14.4 |
| `long` | 6.97 | 2.93 | 1.84 | 26.4 |
| `lat_ms` | 6.92 | 2.91 | 1.65 | 23.9 |

Source: the table `projection.tsv` in [2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md) and the table `g-ceiling.tsv` in [2026-10-31-0006-mmq-tiles](../../data/2026-10-31-0006-mmq-tiles/README.md).

The result is lower than the entry of `H` for all request types, although the column "Pipeline" is for a change that works. The part of the ceiling is 0.24 for `lat_ms`. The entry of `H` used 0.60, with a range of 0.40 to 0.80. The value is below that range.

## Method

### The micro-benchmark

The program is the micro-benchmark of target `H`. It runs the MMQ kernel with random weights and random activations. It has the two Q4_0 shapes of the model: 9,216 by 2,560 (`ffn_up`, `ffn_gate`) and 2,560 by 9,216 (`ffn_down`). It uses 31 numbers of tokens from 9 to 2,048.

Each cell runs two variants. The baseline is the kernel of the build `dmk/stack` with the tile table of that build. The other variant is the same kernel with the pipeline. The two use the same `J`. The program compares the output of the pipeline with the output of the baseline, value by value. Each time is the lowest of five runs of a loop of 15 ms.

### The versions

We made six versions. The data set has the runs of all of them. The list shows three of them.

1. **Weights only.** `cp.async` loads the Q4_0 blocks of the next weight tile into a buffer in shared memory. The threads read the blocks from the buffer and make the int8 values with the operations of the baseline. The loads of the activations do not change.
2. **Weights and activations.** The same as version 1. `cp.async` also loads the next activation tile into two buffers, one for each part of the tile. Each step has four barriers. The kernel tests at run time that the data follows the 16-byte rule of `cp.async`.
3. **Three barriers.** The same as version 2, with one barrier less. The host uses the kernel without the pipeline when the data does not follow the 16-byte rule of `cp.async`. This is the last version.

The versions 2a and 2b change how many steps of the loops the compiler expands in version 2. The last version is the commit `4f7bc1fe74e61f79a4fd4f0643a14d8b111f0d31` of the branch `dmk/0009-mmq-pipeline`.

### Same operations as the baseline

The MMA work reads the same weight tile and the same activation tiles as the baseline. The code that makes the int8 values from the Q4_0 blocks has the same operations. The loop on `K` has the same steps in the same sequence. Only the time of the loads changes. The equivalence class of the change is `exact`.

The column `ndiff` of the micro-benchmark makes sure of this for 62 cells. The output of the pipeline has the same bits as the output of the baseline in each of them.

### The limit

The limit is the ratio 1.08 at 1,025 tokens or more. We use the mean of the ratios of the 10 cells with 1,025 tokens or more (5 numbers of tokens and 2 shapes).

## Results

### Versions against the limit

| Version | Name | Mean ratio at 1,025 tokens or more | Smallest and largest ratio at 1,025 tokens or more | Cells slower than the baseline (all cells) | Largest `ndiff` | At the limit |
|---|---|---:|---|---:|---:|---|
| 1 | weights only | 0.934 | 0.900 to 0.971 | 53 of 62 | 0 | no |
| 2 | weights and activations, 4 barriers | 1.035 | 0.981 to 1.101 | 19 of 62 | 0 | no |
| 2a | version 2, loops not expanded | 1.000 | 0.960 to 1.040 | 23 of 62 | 0 | no |
| 2b | version 2a, loops that make the int8 values expanded 2 times | 0.950 | 0.926 to 0.981 | 45 of 62 | 0 | no |
| 3 | 3 barriers, work copy | 1.041 | 0.994 to 1.087 | 8 of 62 | 0 | no |
| 3c | version 3, commit | 1.042 | 0.990 to 1.088 | 8 of 62 | 0 | no |

Source: the table `versions.tsv` in [2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md).

The column "Cells slower than the baseline" uses all 62 cells. The mean ratio and the smallest and largest ratio use the 10 cells with 1,025 tokens or more. Version 3 and version 3c have almost the same code. Their mean ratios are 1.041 and 1.042.

### The cells at the limit

| Shape | Tokens | `J` | Baseline (µs) | Pipeline (µs) | Ratio | Registers of the pipeline |
|---|---:|---:|---:|---:|---:|---:|
| `ffn_up`, `ffn_gate` | 1,025 | 120 | 967.2 | 967.9 | 0.999 | 255 |
| `ffn_up`, `ffn_gate` | 1,173 | 120 | 1,048.5 | 1,059.0 | 0.990 | 255 |
| `ffn_up`, `ffn_gate` | 1,288 | 120 | 1,150.6 | 1,135.0 | 1.014 | 255 |
| `ffn_up`, `ffn_gate` | 1,536 | 128 | 1,312.1 | 1,234.1 | 1.063 | 255 |
| `ffn_up`, `ffn_gate` | 2,048 | 128 | 1,759.9 | 1,644.4 | 1.070 | 255 |
| `ffn_down` | 1,025 | 120 | 960.3 | 926.9 | 1.036 | 255 |
| `ffn_down` | 1,173 | 120 | 1,079.0 | 1,058.0 | 1.020 | 255 |
| `ffn_down` | 1,288 | 120 | 1,160.0 | 1,088.4 | 1.066 | 255 |
| `ffn_down` | 1,536 | 128 | 1,370.2 | 1,259.9 | 1.088 | 255 |
| `ffn_down` | 2,048 | 128 | 1,753.5 | 1,634.3 | 1.073 | 255 |

Source: the tables `final-speedup.tsv` and `limit.tsv` in [2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md). The values are the mean of three runs.

The baseline uses 216 registers at `J` of 128 and 195 registers at `J` of 120. The pipeline uses 255 registers at the two `J` values. The compiler puts 16 bytes (`J` of 128) and 32 bytes (`J` of 120) in local memory.

The shared memory goes from 57,856 bytes to 94,720 bytes at `J` of 128. One thread block stays on each SM.

The six cells with `J` of 128 have ratios of 1.06 to 1.09. The four cells with `J` of 120 (1,025 to 1,288 tokens) have ratios of 0.99 to 1.07.

### The groups of numbers of tokens

The next table gives the part of the ceiling `same_loads` that the pipeline removes, for each group of numbers of tokens of a pass.

| Tokens of a pass | Cells | Mean ratio | Part of the ceiling removed (%) |
|---|---:|---:|---:|
| 9 to 64 | 22 | 1.029 | 3.9 |
| 65 to 256 | 26 | 1.080 | 32.8 |
| 257 to 1,024 | 4 | 1.089 | 40.1 |
| 1,025 to 2,048 | 6 | 1.042 | 17.8 |

Source: the tables `final-speedup.tsv` and `vs-ceiling-by-class.tsv` in [2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md). The group of 1,025 tokens or more has 10 cells in the mean ratio. It has 6 cells in the part of the ceiling, because the ablation runs have no cells at 1,025 and 1,536 tokens.

At 9 to 64 tokens the shapes are different. The `ffn_up` shape has a mean ratio of 0.979, and the `ffn_down` shape has 1.079. At 16 to 32 tokens the `ffn_up` shape has ratios of 0.84 to 0.92 and the `ffn_down` shape has 1.11 to 1.13.

At `J` of 16 and 32 the baseline runs two thread blocks on each SM, and the pipeline runs one. The `ffn_down` shape has the same change and does not become slower. We do not know the cause for `ffn_up`.

### Gain for each request type

The next table gives the time that the pipeline removes from the Q4_0 operations of each request type. The weights of the passes come from the census of the entry of target `H`.

| Request type | Q4_0 MMQ time (ms) | Removed (ms) | Removed (% of request) | Removed, only where faster (% of request) |
|---|---:|---:|---:|---:|
| `short1q` | 14.47 | 1.04 | 2.21 | 2.21 |
| `short2q` | 25.76 | 1.71 | 1.96 | 1.96 |
| `medium` | 122.93 | 3.40 | 0.95 | 1.13 |
| `long` | 279.50 | 14.67 | 1.84 | 1.93 |
| `lat_ms` | 81.42 | 3.98 | 1.65 | 1.75 |

Source: the table `projection.tsv` in [2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md).

The method gives the Q4_0 ceiling of the entry of `H` again: 6.93 %, 7.32 %, 6.56 %, and 6.97 % of the request types. This makes sure that the method of the table is correct.

A rule that uses the pipeline only in the cells where it is faster gives 1.75 % of `lat_ms`. The largest value of a request type is 2.21 %.

## Meaning of the result

### Cause for version 1

Version 1 loads only the weights with `cp.async`. [INFERENCE: In the baseline the loads of the weights and the loads of the activations are in progress at the same time. One barrier waits for the two. In version 1 the barrier waits for the activations, and the time does not become shorter.]

Version 1 also adds work. The threads read the blocks from shared memory and write the int8 values. The registers go from 216 to 240 at `J` of 128. The mean ratio is 0.934. Version 2 loads the activations too. Its mean ratio is 1.035.

### Data on the small gain

The ceiling `same_loads` is 1.235 times for `ffn_up` at 2,048 tokens. The pipeline gives 1.070 times. The data show these facts:

- **Local memory.** Local memory is not the cause. Version 2a has no local memory at `J` of 120 and 128, and its mean ratio is 1.000. Version 2 has 40 to 48 bytes and a mean ratio of 1.035.
- **Shared memory.** The pipeline has more work in shared memory than the baseline: it writes the blocks and reads them again.
- **The shape of the tile at `J` of 120.** The code uses 16 rows for each warp at `J` of 120 and 32 rows at `J` of 128. The cells with `J` of 120 have ratios of 0.99 to 1.07. The cells with `J` of 128 have 1.06 to 1.09.

[INFERENCE: The pipeline removes less at 1,025 tokens or more because the variant `same_loads` also removes the loads from VRAM and from the L2 cache. A pipeline gives an overlap of a load and the MMA work, but not more bandwidth. We did not measure this.]

### Effect on `N` and on Q8_0

The estimate for `lat_ms` is 1.65 % for Q4_0 with the pipeline. The Q8_0 operations have a ceiling of 5.46 % of `lat_ms`. [INFERENCE: if Q8_0 got the same part of its ceiling (0.24), it can remove 1.3 % of `lat_ms`.]

The method of the last version is too large for Q8_0. A row of 8 Q8_0 blocks has 272 bytes. The buffer of the blocks has 34,816 bytes at `J` of 128. With the second activation buffer of 18,432 bytes the total is 111,104 bytes, and the limit is 101,376 bytes. This buffer is possible only for `J` of 88 or less. The Q8_0 kernel also uses 254 registers.

Target `N` comes after `G`. This result is not a test of `N`.

## Decision

We do not use target `G` in this shape. We do not run a test with two arms, we do not run a test of answers, and we do not open a pull request.

- The mean ratio of 1.042 at 1,025 tokens or more is below the limit of 1.08. The largest ratio of a cell is 1.088.
- The value for each request type is 2.21 % at the maximum, with a confidence of 1.0. The limit of phase 4 is 3 % after the confidence.
- A test with the model can find at most 1.65 % of `lat_ms`, and the change is large. It adds 257 lines to `mmq.cuh` and `mmq-load-tiles.cuh`, removes 22 lines, and adds 36,864 bytes of shared memory for each thread block.

The branches `dmk/0009-mmq-pipeline` and `dmk/0009-mmq-pipeline-bench` of the fork have the code. They are for reference and not for a merge.

## Limits

- The micro-benchmark uses random weights and random activations. [INFERENCE: the time of the MMA kernel does not change with the values. We did not check this.]
- The launches run one after the other. A request of the model has other kernels between the MMQ launches, and the state of the L2 cache is different.
- The method for the request types uses the nearest of the 29 numbers of the ablation runs. It has no Q8_0, no MMVQ kernels, and no gaps between kernels.
- The ratios of the same cell in the three runs of the last version have a range of 0.4 % (median) and 4.9 % (maximum). At 1,025 tokens or more the maximum is 2.1 %. The mean ratio of 1.042 is 3.5 % below the limit of 1.08.
- The versions 1, 2, 2a, 2b, and 3 have no committed source. The data set has the raw files of their runs.
- We used only the column `ndiff` of the micro-benchmark as a test of the correctness. We did not run `test-backend-ops`, a test of answers, or `compute-sanitizer`.
- The micro-benchmark has no case with a number of weight rows that 128 does not divide. It has no case with data that does not follow the 16-byte rule of `cp.async`, and no case with `mul_mat_id`. The code of the kernel without the pipeline has no test.
- We did not try other alternatives. A pipeline for the loads from the L2 cache and a warp that only loads are two examples.

## Data

- [Data set 2026-10-31-0009-mmq-pipeline](../../data/2026-10-31-0009-mmq-pipeline/README.md): the raw files of the runs, the tables, and the build.
- The branch `dmk/0009-mmq-pipeline` of the fork of llama.cpp of this project has the change. The commit is `4f7bc1fe74e61f79a4fd4f0643a14d8b111f0d31`.
- The branch `dmk/0009-mmq-pipeline-bench` has the micro-benchmark with the variants of the pipeline. The commit is `251e766cff9a68c71a1e3c4cb7a2534256b74058`.
- [Entry of target H](2026-10-31-0006-mmq-tiles-result.md): the ceiling of `G` and the limit.
- [Phase-4 entry](2026-10-31-phase4-ranked-targets.md): the prediction for `G`.
