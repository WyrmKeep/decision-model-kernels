# Sweep of 142 MMQ tile adjustments (target H)

This data set has the raw files and the tables of one micro-benchmark. The micro-benchmark ran the MMQ kernel of llama.cpp with 142 tile adjustments. It used 6 weight shapes of the model and 29 numbers of tokens. It ran on one `RTX 3060 Ti` on 2026-10-31.

The data set also has the source of the micro-benchmark, the patch of llama.cpp, the build script, and the script that makes all tables. The [results entry](../../docs/notebook/2026-10-31-0006-mmq-tiles-result.md) gives the result and the decision.

## Question

The test answers these questions for target `H` of the [ranked list of phase 4](../../docs/notebook/2026-10-31-phase4-ranked-targets.md).

- Is a different tile or a different number of thread blocks for each SM faster than the baseline of the MMQ kernel for this GPU?
- Does a faster adjustment give the same bits as the baseline?
- How much time can a rule for the tile remove from the four request types?
- Do more thread blocks on one SM remove time at a large number of tokens? This question is for target `G`.

The test has no arm with a change of llama.cpp. It is a micro-benchmark with one program. The results entry gives the prediction and the decision.

## Dates

- First run: 2026-10-31, 05:40:22 UTC.
- Last run: 2026-10-31, 07:10:12 UTC.
- Measurement window `0006-sweep-r1`: 05:40:04 to 05:53:26 UTC. The run `0006-sweep-r1-q4` was from 05:40:22 to 05:44:52 UTC. The run `0006-sweep-r1-q8` was from 05:44:52 to 05:53:04 UTC.
- Measurement window `0006-sweep-r1b`: 07:05:38 to 07:10:29 UTC. The run `0006-sweep-r1b-q8` was from 07:05:54 to 07:10:12 UTC.
- List run: about 05:14 UTC. It lists the variants and runs a smoke test. The tables do not use it.

## Project commit

The commit that adds this data set also adds the results entry and the glossary rows. The commit before it is `31a5d54f528ee654d9fc4f7216c14e0373bbdbcc`.

## Build commits

The sweep program comes from the branch `dmk/0006-mmq-sweep` of the fork of llama.cpp of this project.

The commit is `df6730c95a37831f42c9660af91ec4a3e8b464a7`.

The commit before it is `6b8b65e6672f07805000d3c9683b65c72cc62303`. It is the branch `dmk/stack`: build b12004 and change 0001.

The files `src/mmq_sweep.cu`, `src/sweep_variant.cu`, `src/gen_variants.py`, and `src/build_sweep.sh` have the same bytes as the files in `tools/dmk-mmq-sweep/` of that commit.

The file `src/sweep.patch` is the difference between the commit and the commit before it in `ggml/src/ggml-cuda`. It adds the file `mmq-config-sweep.cuh`. It changes `mmq.cuh` to add the grid rules and the output of monitor values.

The build ran on the work copy of the commit, before we wrote the commit. The work copy had the same text as the commit.

## Build flags

The script `src/build_sweep.sh` builds one object file for each variant, the program `mmq_sweep.cu`, and the link. It runs in a build container and does not use a GPU. The program has the name `llama`, because the harness of the GPU host starts a program with this name. The flags of `nvcc` are:

```
nvcc -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_86,code=[sm_86] -use_fast_math -extended-lambda \
     -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -w -Xptxas -v
```

| Item | Value |
|---|---|
| Builder image | `localhost/dmk-builder:cuda12.8.1-gcc14`, image ID `54be7481bcfd4bd3085c5572a80f565384671b5aa2f5879ea4fb2a3c74f9c947`. We did not publish it. |
| Compilers | `nvcc` 12.8.93 and `g++-14` 14.2.0. |
| End of the build | 2026-10-31T05:14:12Z. The build made the objects in 125 s. |
| Variants that did not build | Two variants with 512 threads and `J` of 8 (`zero-sized variable y_df`). The script removes them. 142 variants stay. |
| Program | SHA-256 `367753501f81a1853e8655ff9b6078a376bae63f52c4c3593239c4aa18684ef2`, 22,081,304 bytes. The copy on the GPU host has the same SHA-256. |

`raw/build/MANIFEST-mmq-sweep-r1.txt` has the same facts. `raw/build/variants.tsv` and `raw/variants-list.tsv` list the variants with the registers, the shared memory, and the blocks for each SM.

## Image

No image of llama.cpp ran. The micro-benchmark is a program that does not use a server. The server of the decision model did not run on the GPU host during the runs.

## Model

No model ran. The weights and the activations are random numbers. The 6 shapes come from the "mix8" model. They are the rows `M` and the columns `K` of its weights.

| Type | Rows `M` | `K` | Operations of the model |
|---|---:|---:|---|
| Q4_0 | 9,216 | 2,560 | `ffn_up`, `ffn_gate` |
| Q4_0 | 2,560 | 9,216 | `ffn_down` |
| Q8_0 | 8,192 | 2,560 | `attn_qkv`, `attn_q` |
| Q8_0 | 4,096 | 2,560 | `attn_gate` |
| Q8_0 | 2,560 | 4,096 | `ssm_out`, `attn_output` |
| Q8_0 | 1,024 | 2,560 | `attn_k`, `attn_v` |

The two shapes with 32 rows (`ssm_alpha`, `ssm_beta`) are not in the sweep.

## Driver

The driver is 595.71.05.

## GPU

One `RTX 3060 Ti` with compute capability 8.6 and 38 SMs. The limit of shared memory for one thread block is 101,376 bytes. The first lines of each file `raw/*.out` have the values of the device.

## Power limit

200 W. The files `raw/*.meta` have the value.

## Clocks

The clocks have the default values. We did not lock them. The files `raw/*.smi` have the samples of the GPU during each run.

## Server flags

None. No server ran.

## Commands

The build command is:

```
bash src/build_sweep.sh <patched llama.cpp tree> r1 <output folder>
```

The two measurement windows ran the lines of `raw/0006-sweep-r1.txt` and `raw/0006-sweep-r1b.txt`. The harness of the GPU host started the program as `llama sweep ...`. The three runs used these flags:

```
sweep --shapes q4_0:9216:2560,q4_0:2560:9216 --ns <29 numbers> --policies 0,1,4 --reps-ms 8 --trials 3
sweep --shapes q8_0:8192:2560,q8_0:4096:2560,q8_0:2560:4096,q8_0:1024:2560 --ns <29 numbers> --policies 0,1,4 --reps-ms 8 --trials 3
sweep --shapes q8_0:8192:2560,q8_0:4096:2560 --ns <29 numbers> --policies 0,1,4 --reps-ms 8 --trials 3
```

The 29 numbers are `9 13 16 26 30 32 43 47 51 54 64 67 69 84 86 94 102 108 120 128 129 151 158 256 512 1024 1173 1288 2048`. The first line is the run `0006-sweep-r1-q4`. The second line is the run `0006-sweep-r1-q8`. The third line is the run `0006-sweep-r1b-q8`.

The flags of the program are:

- `--shapes`: the list `type:rows:K`.
- `--ns`: the numbers of tokens.
- `--policies`: the grid rules 0, 1, and 4. The results entry gives them.
- `--reps-ms`: the target time of one run of the loop, in ms. A run of the loop has 5 launches at the minimum.
- `--trials`: the number of runs of the loop for each row. The row has the lowest time.

The command to make all tables is:

```
python3 make_tables.py
```

The script uses the Python standard library only. It reads `raw/` and writes `tables/`.

## Arms

None. This is a micro-benchmark and not a test with arms. The 142 variants are the tile adjustments. The baseline is the row with `ref` of 1 and the grid rule 0.

## Raw files

| File | Data |
|---|---|
| `raw/0006-sweep-r1-q4.out`, `raw/0006-sweep-r1-q8.out` | The rows of the first measurement window. There is one row for each shape, number of tokens, variant, and grid rule. |
| `raw/0006-sweep-r1b-q8.out` | The rows of the second measurement window (two Q8_0 shapes). |
| `raw/*.err`, `raw/*.meta`, `raw/*.smi` | The error output, the start time, the power limit, the load of the host, and the samples of the GPU for each run. |
| `raw/0006-sweep-r1.txt`, `raw/0006-sweep-r1b.txt` | The lines of the schedule of the two measurement windows. |
| `raw/variants-list.tsv` | The output of `llama sweep --list`: the 142 variants with registers, shared memory, and blocks for each SM. |
| `raw/build/MANIFEST-mmq-sweep-r1.txt`, `raw/build/variants.tsv` | The facts of the build. |
| `raw/inputs/` | Copies of the census and of the model of phase 4: `census-cases.tsv`, `requests.tsv`, `pass_kernels.tsv`, and `targets.tsv`. They have the same bytes as the files of the data sets `2026-10-30-shape-census` and `2026-10-31-phase4-model`. |

The files in `raw/` are the files of the runs, with no change. The rows of the first measurement window stay in `raw/`, also the rows that the tables do not use.

### Columns of the `.out` files

| Column | Unit | Meaning |
|---|---|---|
| `type`, `M`, `K`, `N` | none | The type of the weights, the rows, `K`, and the number of tokens. |
| `variant` | none | The name of the variant. It has the threads, the blocks for each SM, `I`, `J`, and the flags. |
| `nt`, `occ_req`, `I`, `J` | none | The threads of a thread block, the thread blocks for each SM that the code sets, the rows, and the tokens of the tile. |
| `streamk`, `policy` | none | The stream-k flag (always 1) and the grid rule. |
| `regs`, `local_bytes`, `smem_dynamic`, `blocks_per_sm` | none, bytes | The registers, the local memory, the shared memory, and the blocks for each SM that the GPU lets run. |
| `ntiles`, `grid`, `fixup` | none | The tiles, the thread blocks of the launch, and 1 if the launch has a fix-up kernel. |
| `time_us` | µs | The lowest time of the runs of the loop. |
| `ndiff`, `maxabs` | none | The number of output values that are different from the output of the baseline, and the largest difference. |
| `ref` | none | 1 for the baseline variant. |

## Tables

`make_tables.py` makes all tables from `raw/`. The tables use the rows of the second measurement window for the two Q8_0 shapes with 8,192 and 4,096 rows. The weights of the request model are the passes of the 48 cases of the benchmark.

| Table | Data |
|---|---|
| `baseline.tsv` | The baseline row for each shape and number of tokens: the grid, the fix-up kernel, the time, and the TOPS int8. |
| `best-per-shape.tsv` | The fastest row, and the fastest row with the same bits as the baseline, for each shape and number of tokens. |
| `families.tsv` | The fastest row of each group of tile adjustments, for each shape and number of tokens. |
| `class.tsv` | For each shape: the cells with the fix-up kernel in the baseline, the cells with a gain of 10 %, and the cells with the same bits. |
| `g-test.tsv` | The test for target `G`: the baseline and three groups of adjustments with more than one thread block for each SM. |
| `g-summary.tsv` | The ratios of `g-test.tsv` for the cells with 1,173 tokens or more and no fix-up kernel. |
| `j-ladder.tsv` | The time for one thread block of 8 warps, for each `J`, at 1,173 and 2,048 tokens. |
| `j-fit.tsv` | The part of the time that gets smaller when `J` gets larger, at 2,048 tokens. |
| `weights.tsv` | The weights of the request model. |
| `rules-text.tsv` | The meaning of each rule. |
| `rules.tsv` | The time that each rule removes from the MMQ time and from the request time, for each request type and for `lat_ms`. |
| `rules-by-op.tsv` | The time that each rule removes from each operation. |
| `rules-by-n.tsv` | The time that each rule removes, for each group of numbers of tokens. |
| `g-h-scenarios.tsv` | The phase-4 prediction for `H` and `G`, and the new estimates for `G`. |
| `noise.tsv` | The difference between the same launches, for each measurement window and shape. |
| `disturbed.tsv` | The ratios of the two measurement windows for the two shapes that ran again. |

The next table gives the columns of the tables.

| Table | Column | Unit | Meaning |
|---|---|---|---|
| all tables with `type` | `M_rows`, `K`, `N_tokens` | none | The shape and the number of tokens. |
| `baseline.tsv` | `J_tile_tokens`, `grid_blocks`, `fixup`, `tiles`, `blocks_per_sm`, `registers` | none | The values of the baseline row. |
| `baseline.tsv` | `time_us`, `tops_int8` | µs, TOPS | The time and the int8 rate (2 times M times K times the number of tokens, divided by the time). |
| `best-per-shape.tsv` | `base_time_us`, `base_J`, `base_fixup` | µs, none | The time, `J`, and the flag for the fix-up kernel of the baseline. |
| `best-per-shape.tsv` | `best_variant`, `best_policy`, `best_time_us`, `best_speedup`, `best_ndiff` | none, µs | The fastest row. `best_speedup` is the baseline time divided by the time of the row. |
| `best-per-shape.tsv` | `exact_variant`, `exact_policy`, `exact_time_us`, `exact_speedup` | none, µs | The same values for the fastest row with `ndiff` of 0. |
| `families.tsv` | `family`, `J_of_best`, `policy_of_best` | none | The group (`nt<threads>_o<blocks>_I<rows>`) and the `J` and the grid rule of its fastest row. |
| `families.tsv` | `time_us`, `time_over_base`, `ndiff` | µs, none | The time, the ratio to the baseline, and the number of different values. |
| `class.tsv` | `N_values`, `N_values_where_base_runs_fixup`, `N_values_best_has_equal_bits`, `N_values_best_is_10pct_faster`, `of_these_with_equal_bits` | none | The numbers of cells of each type. |
| `g-test.tsv` | `test`, `variant`, `J_tile_tokens`, `blocks_per_sm`, `warps_per_sm` | none | The group, the row, and its values. |
| `g-test.tsv` | `time_us`, `time_over_base`, `fixup`, `ndiff` | µs, none | The time, the ratio to the baseline, the flag for the fix-up kernel, and the number of different values. |
| `g-summary.tsv` | `test`, `cells`, `min_time_over_base`, `median_time_over_base`, `max_time_over_base` | none | The group, the number of cells, and the lowest, median, and highest ratio. |
| `j-ladder.tsv` | `J_tile_tokens`, `registers`, `local_bytes`, `shared_memory_bytes`, `blocks_per_sm`, `grid_blocks`, `fixup` | none, bytes | The values of the row. |
| `j-ladder.tsv` | `time_us`, `time_over_base` | µs, none | The time and the ratio to the baseline. |
| `j-fit.tsv` | `time_J64_us`, `time_J128_us` | µs | The times at `J` of 64 and `J` of 128. |
| `j-fit.tsv` | `part_inverse_to_J_pct_of_time_J128`, `ceiling_if_part_removed` | %, none | The part `B / J` of `time = A + B / J` at `J` of 128, and 1 divided by (1 minus the part). |
| `weights.tsv` | `request_type`, `N_tokens`, `passes_in_a_run_of_the_benchmark` | none | The weight of each number of tokens for each request type. |
| `rules.tsv` | `rule`, `request_type` | none | The rule and the request type. The row `lat_ms` is the mean of the cases. |
| `rules.tsv` | `base_mmq_ms`, `rule_mmq_ms`, `saved_ms` | ms | The MMQ time of the baseline and of the rule in the request model, and the difference. |
| `rules.tsv` | `saved_pct_of_mmq`, `saved_pct_of_request`, `saved_ms_with_equal_bits` | %, ms | The difference as a part of the MMQ time and of the request time, and the part of it from rows with the same bits. |
| `rules-by-op.tsv` | `operation`, `launches_per_pass`, `base_ms_per_case`, `rule_ms_per_case`, `saved_pct_of_operation` | none, ms, % | The MMQ launches of one pass and the times of the baseline and of the rule. |
| `rules-by-n.tsv` | `tokens_of_pass`, `base_ms_per_case`, `rule_ms_per_case`, `saved_pct_of_class` | none, ms, % | The group of passes and the times of the baseline and of the rule. |
| `g-h-scenarios.tsv` | `scenario`, `confidence` | none | The estimate and its confidence. |
| `g-h-scenarios.tsv` | `<type>_saved_ms`, `<type>_saved_pct`, `lat_saved_ms`, `lat_saved_pct`, `max_expected_pct` | ms, % | The gain of each request type and of `lat_ms`, and the largest expected gain. |
| `noise.tsv` | `window`, `pairs`, `median_spread_pct`, `p90_spread_pct`, `max_spread_pct` | none, % | The difference between the rows of the same launch with two grid rules, and its percentiles. |
| `disturbed.tsv` | `rows_compared`, `median_r1_over_r1b`, `p90_r1_over_r1b`, `max_r1_over_r1b`, `base_row_r1_over_r1b` | none | The ratio of the times of the first measurement window to the times of the second measurement window. |

## Baseline and production speed

The baseline row is the row of the variant with 256 threads, 128 rows, and one block for each SM. It has the `J` that the host rule of llama.cpp gives. This is the `J` with the smallest number of column tiles, or the smaller `J` if two are equal. It has the grid rule 0.

The baseline reproduces the speed of the production kernel:

- Q4_0 9,216 by 2,560 at 2,048 tokens: 1,779.5 µs and 54.3 TOPS int8.
- The profile of phase 3 gives 35 % to 39 % of 149.3 TOPS for the production kernel at 1,173 to 2,048 tokens. This is 52 to 58 TOPS.

## Runs that other work changed

A task of a different agent used the GPU from 05:45:30 to 05:48:30 UTC. This was in the run `0006-sweep-r1-q8`. It ran at the same time as the shape Q8_0 8,192 by 2,560, and possibly the start of the shape Q8_0 4,096 by 2,560.

A different agent also started servers on the GPU until 05:40:18 UTC, before the first run.

Tasks of this work used the GPU before the rule for GPU use started. They ran from about 05:14:20 to 05:16:15 UTC: the list run, a smoke test with 2 shapes, and a smoke test with all shapes. They used about 60 s of kernel time. They ran at the same time as four runs of a different measurement window (`perf-r1-16_3`, `perf-r1-64_2`, `perf-r1-32_1`, and `perf-r1-128`, from 05:14:39 to 05:16:14 UTC). The tables do not use the smoke runs. `raw/variants-list.tsv` is the output of the list run.

The second measurement window `0006-sweep-r1b` ran the two shapes again, with the same program and the same flags. The table `disturbed.tsv` compares the two measurement windows.

- 4,096 by 2,560: the median ratio is 1.001 to 1.004. We can see no change in the first measurement window.
- 8,192 by 2,560: the 90th percentile of the ratio is 1.07 to 1.11 for 8 of the 29 numbers of tokens. The largest ratio is 1.15. The largest difference in a group of the same launch is 18.8 % in the first measurement window and 5.95 % in the second measurement window.

The tables use the rows of the second measurement window for these two shapes. The table `noise.tsv` has one row for each measurement window.

## Limits

- Each row is the lowest time of three runs of the loop. The difference between the same launches is 0.2 % to 0.9 % (median), 1.8 % to 2.7 % (90th percentile), and 8.5 % at the maximum.
- The two shapes with 32 rows are not in the sweep. They are about 1.45 % of the MMQ time.
- The tables use the nearest of the 29 numbers of the sweep for each number of tokens of the model.
- The rows of the oracle are the lowest of about 200 rows with noise. They are faster than the true time.
- The text at the start of `src/mmq_sweep.cu` lists the flag `--j-baseline-only`. The program does not have this flag. We did not change the source after the build.
- The data for the correctness of the output are the columns `ndiff` and `maxabs`. We did not run a test of answers, `test-backend-ops`, or `compute-sanitizer`, because no change of llama.cpp is in this test.

## Files not added

- The logs of the measurement windows. They have names of hosts and numbers of monitors.
- The program `llama` of the sweep (22 MB). Build it again with `src/build_sweep.sh`.
- The full log of the build (the `-Xptxas -v` output for each variant).
- The object files of the variants.
