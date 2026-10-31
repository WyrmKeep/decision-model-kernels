# ggml-backend: skip the uploads that no `COMPUTE` node reads

## Summary

The scheduler of ggml writes each input of a part of the graph to the backend of that part. For the model "mix8", 3 of the 10 uploads in each pass are inputs of only a branch that no backend calculates. The change records which copies a `COMPUTE` node reads. A `COMPUTE` node is a node with the flag `GGML_TENSOR_FLAG_COMPUTE`. The scheduler skips the other copies and their sync calls. Each pass then has 3 uploads and 8 sync calls less, and the answers stay the same.

## Equivalence class

`exact`. We selected the equivalence class on 2026-10-31, before the first run. The CPU backend and the CUDA backend skip the nodes without the flag `GGML_TENSOR_FLAG_COMPUTE`. Each `COMPUTE` node gets the same inputs as before. The [method](../method.md) gives the tests for this equivalence class.

## Results

The private benchmark has 4 blocks, and `llama bench` has 4 blocks. Each block has one run of each arm. The difference is the mean of the 4 ratios of the blocks. The interval is the 95 % bootstrap interval.

The power limit was 200 W. The median SM clock was 1,845 MHz in the runs of the private benchmark. It was 1,860 MHz to 1,868 MHz in the runs of `llama bench`.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| `lat_ms` (all 48 cases) | 234.47 | 232.27 | -0.93 | -2.05 to +0.12 | 4 | 200 | 1,845 |
| `short_ms` | 71.47 | 71.32 | -0.21 | -1.43 to +1.28 | 4 | 200 | 1,845 |
| `medium_ms` | 352.92 | 349.32 | -1.01 | -1.90 to -0.04 | 4 | 200 | 1,845 |
| `long_ms` | 768.79 | 758.99 | -1.27 | -2.00 to -0.55 | 4 | 200 | 1,845 |

The throughput `rps8` was 4.303 requests/s before and 4.337 requests/s after (+0.80 %, +0.51 % to +1.09 %).

| `llama bench` prompt | Before (tokens/s) | After (tokens/s) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| 64 tokens | 2,875.9 | 2,897.8 | +0.76 | +0.59 to +0.93 | 4 | 200 | 1,868 |
| 256 tokens | 4,330.2 | 4,391.3 | +1.41 | +1.19 to +1.60 | 4 | 200 | 1,868 |
| 3,000 tokens | 4,731.7 | 4,802.8 | +1.50 | +1.40 to +1.65 | 4 | 200 | 1,868 |

The host load was high in blocks 1 and 2 of the private benchmark (6.5 to 21.5). It was low in blocks 3 and 4 (3.3 to 6.5). In block 2, the difference of `lat_ms` is +0.51 %. The other 3 blocks give -2.50 %, -1.05 % and -0.69 %. The interval of `lat_ms` includes 0.

Source: [data set 2026-10-31-0007-input-copies](../../data/2026-10-31-0007-input-copies/README.md), tables `speed-bench.tsv`, `speed-llama-bench.tsv` and `speed-bench-blocks.tsv`.

## Prediction

We committed the [prediction](../notebook/2026-10-31-0007-input-copies-prediction.md) on 2026-10-31 at 05:32:15 UTC, before the first run of the new build in the test. The [result entry](../notebook/2026-10-31-0007-input-copies-result.md) compares it with the result.

| Metric | Predicted (%) | Result (%) | Prediction correct |
|---|---:|---:|---|
| `lat_ms` | -1.27 | -0.93 | No: the interval includes 0 |
| `short_ms` | -1.0 | -0.21 | No: not in the range, and the interval includes 0 |
| `medium_ms` | -1.28 | -1.01 | Yes |
| `long_ms` | -1.37 | -1.27 | Yes |
| `llama bench`, 64 tokens | +1.1 | +0.76 | Yes |
| `llama bench`, 256 tokens | +1.4 | +1.41 | Yes |
| `llama bench`, 3,000 tokens | +1.5 | +1.50 | Yes |

- The cause of the gain is the host time of the 3 uploads and of their sync calls. The time becomes larger with the number of tokens. The gain is thus larger for long requests.
- The short requests have a gain of less than 1 ms. The noise of 4 blocks is larger than this gain. The test cannot show it.
- The profile of the new build agrees with the mechanism. Each pass has 3 uploads and 8 sync calls less. The prediction had 7 sync calls less.

## Correctness

| Test | Threshold | Result |
|---|---|---|
| Answers of the private benchmark, 4 runs of each arm | `cross_identical_all` true, and `self_repeat` true for the two arms | Pass. The largest probability difference is 0.0. |
| SASS of `libggml-cuda.so` for `sm_86` | The same SASS for all names | Pass. 6,766 of 6,766 names have the same SASS and the same code of the instructions. |
| Uploads in a pass of the new build | 7 uploads in a pass that makes an answer | Pass. 7 uploads, and 14 sync calls in the upload phase. |
| `test-backend-ops` | Not applicable | The change has no operation and no kernel. No case tests the scheduler. |

## Reproduce

1. Get the fork and the commits: clone the fork of llama.cpp of this project, and select the branch `dmk/0007-input-copies`. The commit is `c3ec5653bda455b6d49976412ab5b7904f55fb13`. The commit before it is `6b8b65e6672f07805000d3c9683b65c72cc62303`, the build `L`.
2. Make the two builds with `harness/build/build.sh`. The `cmake` flags are in the [data README](../../data/2026-10-31-0007-input-copies/README.md).
3. Make the schedules:

   ```
   python3 harness/make_schedule.py --prefix c0007-bench --blocks 4 --seed 70071 \
       --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind bench \
       --args "--write-reference {name}"
   python3 harness/make_schedule.py --prefix c0007-llb --blocks 4 --seed 70072 \
       --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind tool \
       --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
   ```

4. Run each schedule with `harness/run_schedule.sh <schedule file> <results folder>`.
5. Make the tables: `python3 data/2026-10-31-0007-input-copies/make_tables.py`.

## Limits

- The change is in the scheduler that all backends use. A backend that does not skip the nodes without `GGML_TENSOR_FLAG_COMPUTE` can read an input that the scheduler did not write. This can give incorrect answers or damage the data of a tensor. We read the source of the CPU backend and the CUDA backend. We did not run a different backend.
- The measurement has 4 blocks, and two blocks had a high host load. The intervals of `lat_ms` and `short_ms` include 0.
- The other 14 to 18 sync calls of a pass stay. They use about 0.1 ms in each pass.
- The GPU kernels did not change. The test of the answers used one model and 100 decisions.
- To go back to the baseline, remove the commit `c3ec5653b`. The branch `dmk/stack` then has the code of the build `L`.

## Data

- [Data set 2026-10-31-0007-input-copies](../../data/2026-10-31-0007-input-copies/README.md)
- [Prediction entry](../notebook/2026-10-31-0007-input-copies-prediction.md)
- [Result entry](../notebook/2026-10-31-0007-input-copies-result.md)
