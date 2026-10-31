# Change 0002: value-split path of gdn_single (result)

Date: 2026-10-31. The measurement window was from 06:17 to 06:31 UTC. The tests of the paths were from 05:33 to 05:36 UTC. They ran with no GPU lock and measure no time. The tests of the builds of the branches were from 06:31 to 06:34 UTC, with the GPU lock.

## Task of this entry

This entry gives the result of target `D` of the [phase 4 entry](2026-10-31-phase4-ranked-targets.md). Target `D` was to make the `RTX 3060 Ti` use a value-split path of the kernel `gdn_single`. The phase 4 entry predicted that `gdn_single` becomes 1.6 times faster.

We measured the kernel time of each path before we made a change to the path selector. The value-split paths are slower than the whole-head path. Thus we did not make a speed test of two builds. There is no prediction entry for a speed test, because there is no speed test.

The [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) gives the raw files, the tables, and the conditions.

## Result in short

- The whole-head path is the fastest path for all 21 shapes. The shapes have 16 key heads, 32 value heads, and S=128, as in mix8. They have 1, 2, or 3 sequences and 65 to 2,048 tokens.
- The best value-split path is 1.15 to 1.31 times as slow as the whole-head path. For one sequence it is `<16,64,1>`, and it is 1.23 to 1.31 times as slow. The other value-split paths are as much as 3.15 times as slow.
- One sequence with 2,048 tokens uses 1,290 µs with the whole-head path and 1,696 µs with `<16,64,1>`.
- All 8 paths give the same bits. We compared 9 shapes. In each shape we compared the 7 value-split paths with the whole-head path. No value is different.
- The test `test-backend-ops` has 43 cases for `GATED_DELTA_NET`. All 43 cases agree with the CPU backend, for each of the 8 paths.
- The cause is the work of each thread block. The MMA instructions of the whole-head path fill 55 % of the time at the highest rate of the tensor core. Each slice does the work for the Gram matrix and for the inverse matrix again. The 6 SMs that the whole-head path does not use give a gain that is less than this work.
- We reject `D`. The time model of phase 4 has no gain from `D`. The expected gain of the total goes from 17.9 % to 16.9 % of `lat_ms`.
- The code change of the pull request is a set of perf cases for `test-backend-ops` and a comment. The program does not change.

## Prediction of phase 4

The phase 4 entry gives this text for target `D`.

- **Cause.** The path selector uses the whole-head path, because 32 heads × 2 is more than 38 SMs. Then 32 thread blocks run on 38 SMs, one chunk after the other.
- **Change.** Use the value-split path with 128 thread blocks on this GPU.
- **Equivalence class.** `exact`.
- **Prediction.** `gdn_single` becomes 1.6 times faster.
- **Confidence.** 0.45. The gain of `lat_ms` is 2.1 % (5.0 ms). The expected gain is 0.9 %.

The time model of phase 4 uses the ratio 1.6 as an assumption. It did not come from a test of the change. The phase 4 entry lists this in the section about the causes of an incorrect prediction.

## Test and paths

The kernel `gdn_single<C,V,BLOCKS>` has the template parameter `V`. It is the number of value columns that one thread block calculates. The whole-head path has `V=128` and 32 thread blocks for one sequence. A value-split path has `V=64`, `V=32`, or `V=16`. It has 2, 4, or 8 thread blocks for each head.

The build of the fork has 3 versions of the template for the CUDA backend: `<16,128,1>`, `<16,32,1>`, and `<16,32,2>`. The HIP backend uses `<16,64>`. We made a build for the test. An environment variable of this build sets the path. The build has the 8 paths of the next table.

| Path name | `V` | Thread blocks of 1 sequence | Thread blocks of each SM, 1 sequence | Minimum blocks of each SM (`BLOCKS`) |
|---|---:|---:|---:|---:|
| `128` | 128 | 32 | 0.84 | 1 |
| `64_1` | 64 | 64 | 1.68 | 1 |
| `64_2` | 64 | 64 | 1.68 | 2 |
| `32_1` | 32 | 128 | 3.37 | 1 |
| `32_2` | 32 | 128 | 3.37 | 2 |
| `16_1` | 16 | 256 | 6.74 | 1 |
| `16_2` | 16 | 256 | 6.74 | 2 |
| `16_3` | 16 | 256 | 6.74 | 3 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `perf-ratio-range.tsv`.

Two thread blocks of `<16,64>` cannot run on one SM at the same time. The shared memory of one thread block is 56 KB, and one SM has 100 KB. [INFERENCE: we calculated the shared memory from the structure in the source. We did not read it from the build.]

The test is `test-backend-ops perf -b CUDA0 -o GATED_DELTA_NET`. The program runs the operation many times and gives the time of one run. The test has 21 shapes: 3 numbers of sequences and 7 numbers of tokens. The 7 numbers of tokens are 65, 128, 256, 512, 1,024, 1,173, and 2,048. They are the same as the shapes of the prefix passes of the benchmark.

Each path ran 3 times. The sequence of the 24 runs is random. A seed sets it. The runs were in a measurement window, with the production server stopped. The power limit was 200 W. The clocks were the default clocks.

A first measurement window (05:11 to 05:28 UTC) had GPU programs of other agents with no GPU lock. Some times of that measurement window have a contended time. A contended time is more than 1.10 times the smallest time of the 3 runs.

We did not use that measurement window. We made the 24 runs again in a new measurement window with the GPU lock. No time of the new measurement window is a contended time. The data set README lists the GPU programs that ran in the first measurement window.

## Kernel time of each path

The tables give the median of 3 runs in µs. The number after each time is the time divided by the time of the whole-head path. The shapes have 16 key heads, 32 value heads, and S=128.

### One sequence

| Tokens | Whole-head path (µs) | `64_1` | `64_2` | `32_1` | `32_2` | `16_1` | `16_2` | `16_3` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 62 | 77 (1.23) | 82 (1.32) | 101 (1.62) | 78 (1.25) | 161 (2.60) | 109 (1.76) | 93 (1.50) |
| 128 | 92 | 117 (1.27) | 125 (1.36) | 163 (1.77) | 122 (1.32) | 256 (2.78) | 175 (1.90) | 145 (1.57) |
| 256 | 172 | 223 (1.30) | 239 (1.39) | 313 (1.82) | 233 (1.35) | 514 (2.98) | 342 (1.98) | 284 (1.65) |
| 512 | 331 | 433 (1.31) | 464 (1.40) | 612 (1.85) | 453 (1.37) | 1,021 (3.09) | 669 (2.02) | 558 (1.69) |
| 1,024 | 650 | 854 (1.31) | 917 (1.41) | 1,214 (1.87) | 898 (1.38) | 2,026 (3.12) | 1,326 (2.04) | 1,109 (1.71) |
| 1,173 | 752 | 986 (1.31) | 1,060 (1.41) | 1,398 (1.86) | 1,036 (1.38) | 2,340 (3.11) | 1,533 (2.04) | 1,281 (1.70) |
| 2,048 | 1,290 | 1,696 (1.31) | 1,822 (1.41) | 2,410 (1.87) | 1,790 (1.39) | 4,028 (3.12) | 2,638 (2.04) | 2,210 (1.71) |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `perf-summary.tsv`.

### Two sequences

| Tokens | Whole-head path (µs) | `64_1` | `64_2` | `32_1` | `32_2` | `16_1` | `16_2` | `16_3` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 121 | 149 (1.23) | 159 (1.32) | 178 (1.47) | 140 (1.15) | 308 (2.54) | 200 (1.66) | 166 (1.38) |
| 128 | 184 | 233 (1.27) | 249 (1.35) | 285 (1.55) | 221 (1.20) | 495 (2.69) | 321 (1.75) | 267 (1.45) |
| 256 | 344 | 443 (1.29) | 473 (1.38) | 549 (1.60) | 421 (1.22) | 1,024 (2.98) | 626 (1.82) | 520 (1.51) |
| 512 | 663 | 865 (1.31) | 926 (1.40) | 1,075 (1.62) | 826 (1.25) | 2,037 (3.07) | 1,236 (1.86) | 1,029 (1.55) |
| 1,024 | 1,305 | 1,710 (1.31) | 1,817 (1.39) | 2,138 (1.64) | 1,637 (1.26) | 4,052 (3.11) | 2,463 (1.89) | 2,055 (1.58) |
| 1,173 | 1,509 | 1,967 (1.30) | 2,126 (1.41) | 2,449 (1.62) | 1,894 (1.25) | 4,664 (3.09) | 2,872 (1.90) | 2,372 (1.57) |
| 2,048 | 2,567 | 3,392 (1.32) | 3,629 (1.41) | 4,220 (1.64) | 3,262 (1.27) | 8,075 (3.15) | 4,924 (1.92) | 4,106 (1.60) |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `perf-summary.tsv`.

### Three sequences

| Tokens | Whole-head path (µs) | `64_1` | `64_2` | `32_1` | `32_2` | `16_1` | `16_2` | `16_3` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 183 | 225 (1.23) | 240 (1.31) | 274 (1.50) | 211 (1.15) | 449 (2.45) | 298 (1.63) | 244 (1.33) |
| 128 | 276 | 348 (1.26) | 372 (1.35) | 445 (1.61) | 331 (1.20) | 726 (2.63) | 481 (1.74) | 392 (1.42) |
| 256 | 515 | 665 (1.29) | 713 (1.38) | 858 (1.67) | 639 (1.24) | 1,532 (2.97) | 946 (1.84) | 766 (1.49) |
| 512 | 993 | 1,303 (1.31) | 1,389 (1.40) | 1,690 (1.70) | 1,255 (1.26) | 3,052 (3.07) | 1,873 (1.89) | 1,525 (1.54) |
| 1,024 | 1,947 | 2,572 (1.32) | 2,751 (1.41) | 3,337 (1.71) | 2,490 (1.28) | 6,065 (3.12) | 3,716 (1.91) | 3,033 (1.56) |
| 1,173 | 2,254 | 2,965 (1.32) | 3,164 (1.40) | 3,846 (1.71) | 2,874 (1.27) | 7,015 (3.11) | 4,305 (1.91) | 3,508 (1.56) |
| 2,048 | 3,861 | 5,116 (1.32) | 5,454 (1.41) | 6,637 (1.72) | 4,963 (1.29) | 12,105 (3.13) | 7,418 (1.92) | 6,048 (1.57) |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `perf-summary.tsv`.

The table `perf-best-split.tsv` gives the best value-split path of each shape. It is `64_1` for one sequence and `32_2` for two or three sequences. Its ratio is 1.23 to 1.31 for one sequence and 1.15 to 1.29 for two or three sequences. The whole-head path is the fastest path in all 21 shapes.

For one path and one shape, the largest difference between the 3 runs is 0.6 % to 2.1 %. The table `perf-rep-spread.tsv` gives the value for each path.

## Cause

We did not make a profile of the value-split paths. The numbers below come from the source, from the measurements above, and from the profile of the whole-head path in phase 3. Each inference has the mark [INFERENCE].

### The tensor core limits the whole-head path

- The profile of phase 3 shows `gdn_single<16,128,1>` at 19.9 % of the DRAM limit. The largest stall reasons are `math_pipe_throttle` 36 %, `wait` 26 %, and `barrier` 12 %.
- The kernel uses 3 MMA instructions for each matrix multiply. This keeps each operand as two BF16 parts. One chunk of 16 tokens has 1,344 MMA instructions in one thread block of the whole-head path. We counted them from the loop limits in the source.
- The data set [`2026-10-30-card-limits`](../../data/2026-10-30-card-limits/README.md) measured 511 operations for each clock and each SM for BF16 with an FP32 accumulator. One `m16n8k16` instruction has 4,096 operations. Thus the highest rate is one instruction in 8.0 clocks for each SM.
- One sequence with 2,048 tokens has 128 chunks. The thread block runs 172,032 instructions in 1,290 µs. At the clock of 1,950 MHz this is 14.6 clocks for each instruction. The instructions fill 55 % of the time at the highest rate, on the 32 SMs that run a thread block.

### The time of the whole-head path follows the number of rounds

The whole-head path has one thread block for each head and each sequence. It uses 93 KB of shared memory. Thus one SM holds one thread block. The next table gives the time of the whole-head path divided by its time for one sequence.

| Sequences | Thread blocks | Rounds on 38 SMs | Ratio of the time, 65 tokens | Ratio of the time, 128 to 2,048 tokens |
|---:|---:|---:|---:|---|
| 1 | 32 | 1 | 1.000 | 1.000 |
| 2 | 64 | 2 | 1.950 | 1.990 to 2.008 |
| 3 | 96 | 3 | 2.954 | 2.993 to 3.004 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `perf-whole-scaling.tsv`. The ratios for 128 to 2,048 tokens are the smallest and the largest value of the 6 numbers of tokens.

The time is the number of rounds times the time of one thread block. The time of one thread block does not change with the number of thread blocks. With one sequence, 6 SMs have no work. They are 16 % of the SMs. They are the only gain of a value-split path that adds no work. The best time of such a path is 32/38 = 0.84 of the time of the whole-head path.

### More work in the slices

The next table gives the MMA instructions of one chunk for each path. A slice is a group of `V` value columns. Each slice calculates the Gram matrix `K K^T` and the Gram matrix `Q K^T` again, with the same operands.

| Path | `V` | Slices of a head | `Gram` | `Projection` | `Solve` | `Out` | `State` | Total of a head | Ratio to the whole-head path |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `128` | 128 | 1 | 96 | 768 | 48 | 48 | 384 | 1,344 | 1.000 |
| `64_1`, `64_2` | 64 | 2 | 192 | 768 | 48 | 48 | 384 | 1,440 | 1.071 |
| `32_1`, `32_2` | 32 | 4 | 384 | 768 | 48 | 48 | 384 | 1,632 | 1.214 |
| `16_1`, `16_2`, `16_3` | 16 | 8 | 768 | 768 | 48 | 48 | 384 | 2,016 | 1.500 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `mma-count.tsv`. The columns `Gram` to `State` give the sum of all slices of a head. The table of the data set gives the values for one thread block.

The MMA instructions are only a part of the work that a slice does again. Each slice also loads `Q` and `K` again. It also does the scan of the gates and the inverse matrix. Each step of the inverse matrix uses the result of the step before it. These parts are the same for all values of `V`. [INFERENCE: we did not measure the time of these parts.]

### The grids and the 38 SMs

The path `64_1` has 64 thread blocks. One SM holds one thread block. Thus 26 SMs run two thread blocks one after the other. The time of the path is 1.31 times the time of the whole-head path.

Each of the two rounds is 0.66 of the time of the whole-head path. A thread block has 0.54 of the MMA instructions of a thread block of the whole-head path (720 of 1,344). [INFERENCE: the difference of 0.12 is the work that each slice does again, and the time at the barriers.]

The path `32_2` has 128 thread blocks, and 76 of them can run at the same time on 38 SMs. Two thread blocks use the tensor core of one SM. The path is 1.15 to 1.39 times as slow as the whole-head path.

The paths with `V=16` are the slowest. They do the most work again.

## Equivalence class

The equivalence class of the change is `exact`, because the path selector keeps the whole-head path. For the candidate branch we made two tests.

### Bits of all paths

The test build runs all 8 paths on the same input in one program. The program copies the output of each path to the host and compares the bytes. The input is the input of a `test-backend-ops` case of `GATED_DELTA_NET` that the MMA path can run. The output has the attention output and the new state.

| Item | Result |
|---|---|
| Shapes | 9 shapes, with 16 to 64 value heads and 1 to 3 sequences |
| Tests in each shape | 7 value-split paths against the whole-head path |
| Values that the program compares in each shape | 395,264 to 26,726,400 |
| Values that are different | 0 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the tables `bitcompare.tsv` and `bitcompare-shapes.tsv`.

The thread blocks of different slices calculate different value columns with the same sequence of operations. No value column uses a different value column. This agrees with the result. [INFERENCE: the test has 9 shapes. It does not show that all inputs give the same bits.]

### The test of the operation

The command is `test-backend-ops test -b CUDA0 -o GATED_DELTA_NET`. It compares the CUDA backend with the CPU backend. We ran it one time for each path and one time with the path selector of the build.

| Path | Cases | Cases that agree with the CPU backend | Cases that do not agree |
|---|---:|---:|---:|
| Path selector of the build | 43 | 43 | 0 |
| `128`, `64_1`, `64_2`, `32_1`, `32_2`, `16_1`, `16_2`, `16_3` | 43 for each path | 43 for each path | 0 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `test-paths.tsv`.

## Builds of the branches

We made a build of each branch with the flags of the parent build. The parent build is the build of `dmk/stack` in the data set [`2026-10-31-lead-30489`](../../data/2026-10-31-lead-30489/README.md). We ran the test of the operation on each build. We compared the SASS of each build with the SASS of the parent build. We ran `compute-sanitizer` with its four tools on the build of the candidate branch.

| Build | Test of the operation | SASS compared with the parent build | `compute-sanitizer` |
|---|---|---|---|
| Pull request, `dmk/0002-gdn-value-split` | 43 of 43 cases agree | The same for all 6,766 names. No name is only in one build. | Not run. No kernel changes. |
| Candidate branch, `dmk/0002-gdn-split-candidate` | 43 of 43 cases agree | The same for 6,766 names. One new name: `gdn_single<16,64,1>`. | `memcheck`, `racecheck`, `initcheck`, and `synccheck`: 0 errors for each tool |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the tables `build-tests.tsv` and `sass-summary.tsv`.

The build of the pull request has the same SASS, the same instruction code, and the same `REG` and `STACK` values as the parent build. The program does not change.

The candidate branch adds one kernel to the library. It is `gdn_single<16,64,1>` with 3,216 instructions. All other kernels have the same SASS as the parent build.

The `compute-sanitizer` runs used the cases of `test-backend-ops` and not a request, because a request must have a model run in a measurement window. The candidate branch is not for a merge. A merge must have a test with a request.

## Effect on the total of phase 4

The time model of phase 4 uses 37.5 % of the time of `gdn_single` as the gain of `D`. This is 0.6 ms for a pass of 65 to 128 tokens and 1.0 ms for a pass of 129 to 256 tokens. It is 7.5 ms for a pass of 1,025 to 2,047 tokens and 12.0 ms for a pass of 2,048 tokens. The gains of the request types are 1.0 ms for `short1q`, 0.6 ms for `short2q`, 7.5 ms for `medium`, and 19.5 ms for `long`. All of these gains are 0 at this time.

We ran the script of the [time model](../../data/2026-10-31-phase4-model/README.md) in a work folder. We removed `D` from the list of targets and kept `A`, `B`, `C`, `E`, `F`, `G`, and `J`. We did not change the phase 4 entry or its data set.

| Total of `lat_ms` | With `D` (phase 4) | Without `D` |
|---|---:|---:|
| All targets give the full gain | 34.2 % (82.5 ms) | 31.8 % (76.7 ms) |
| Expected gain, with the confidence of each target | 17.9 % (43.1 ms) | 16.9 % (40.7 ms) |

Source: the table `sum.tsv` of the [data set of the time model](../../data/2026-10-31-phase4-model/README.md) and a run of its script without `D`. The run is not in the repository.

`D` gave 2.4 points of the full gain and 1.0 point of the expected gain.

## Estimate for the candidate branch

We did not make a speed test of the candidate branch. The next table is an estimate from the micro-benchmark. For each pass with `gdn_single`, we multiply the time of the pass in phase 3 by a ratio. The ratio is the ratio of the best value-split path for one sequence. It comes from the nearest shape. The shapes are 2,048, 1,173, 128, and 256 tokens.

| Request type | Time of `gdn_single` (ms) | Estimate of the difference (ms) | Estimate of the difference (%) |
|---|---:|---:|---:|
| `short1q` | 2.7 | 0.8 | 1.7 |
| `short2q` | 1.7 | 0.5 | 0.5 |
| `medium` | 19.9 | 6.2 | 1.7 |
| `long` | 52.1 | 16.3 | 2.0 |
| `lat_ms` | | 4.1 | 1.7 |

Source: [README of the data set](../../data/2026-10-31-0002-gdn-value-split/README.md) and the table `candidate-estimate.tsv`. [INFERENCE: we calculated this. No request ran with the candidate branch.]

The time model of phase 4 uses a rule. A change of the kernel time of one pass changes the time of the request by the same quantity. The measurements of phase 3 and of [pull request 30489](2026-10-31-lead-30489.md) agree with this model. We did not make a test of it for the candidate branch.

## Decision

We reject `D`. The value-split paths stay in the code for a different GPU and different shapes. The path selector keeps the whole-head path for a GPU with 38 SMs.

The branch `dmk/0002-gdn-value-split` of the fork has two commits. One commit adds the perf cases for the model shape to `test-backend-ops`. The other commit adds a comment at the path selector with the measurement. The program does not change. The branch `dmk/0002-gdn-split-candidate` is the candidate branch. We do not merge it.

## Limits and next work

- The test uses one GPU, one driver, and 3 runs for each path. The best path is different for a GPU with more SMs. The result does not apply to such a GPU.
- The test measures the kernel in a loop. It does not measure a request. The L2 cache is warm in this loop. The profile of phase 3 gave 740 µs for one launch in a long request. The test gives 752 µs for 1,173 tokens and 1,290 µs for 2,048 tokens.
- We did not make a profile of a value-split path. The part of the work that a slice does again, and that is not an MMA instruction, is an inference.
- A gain for `gdn_single` is possible with a change of the kernel. The MMA instructions fill 55 % of the time at the highest rate. The inverse matrix and the scan of the gates do not use the tensor core. A kernel with two thread blocks on one SM can use this time. Each thread block must use less than 50 KB of shared memory.
- The 6 SMs that have no work can give 16 % of the time of `gdn_single` at most. This is 1.2 % of a pass of 2,048 tokens. We calculated these values. We did not make a test.
- A change of the sequence of operations, for example 2 MMA instructions for each matrix multiply, makes the equivalence class `re-associated`. It is not part of this target.

## Data

- [`2026-10-31-0002-gdn-value-split`](../../data/2026-10-31-0002-gdn-value-split/README.md)
- [Change document 0002](../changes/0002-gdn-value-split.md)
