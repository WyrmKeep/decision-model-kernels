# Change 0002: perf cases for the GDN MMA path, and the measurement of the value-split path

## Summary

This change adds perf cases for the shape of the GDN MMA path to `test-backend-ops`. It adds a comment at the path selector `select_gdn_mma_path`. The comment records that all value-split paths of `gdn_single` are slower than the whole-head path on a GPU with 38 SMs. The program does not change. We do not use a value-split path on the `RTX 3060 Ti`.

## Equivalence class

`exact`. The change has no code that the program runs. The path selector keeps the whole-head path. We selected the equivalence class on 2026-10-31, before we made the measurement. All 8 paths of `gdn_single` give the same bits in the test. Thus a value-split path is also in the equivalence class `exact`.

## Results

The change does not change the kernel. The table gives the micro-benchmark that we made before we selected the path.

"Before" is the time of the whole-head path. It is also the time of the program with this change.

"After" is the time of the best value-split path of the shape. We do not use this path. The column "Range of the clean runs" has the range of the runs with no contended time. There are 3 runs, thus the table has no 95 % interval. The shapes have 16 key heads, 32 value heads, and S=128.

| Shape (sequences, tokens) | Before (µs) | After (µs) | Difference (%) | Range of the clean runs (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| 1, 65 | 62 | 77 (`<16,64,1>`) | +23.5 | +21.7 to +25.5 | 3 | 200 | default (1,950) |
| 1, 128 | 92 | 117 (`<16,64,1>`) | +27.5 | +26.8 to +27.7 | 3 | 200 | default (1,950) |
| 1, 256 | 172 | 223 (`<16,64,1>`) | +29.8 | +29.1 to +30.0 | 3 | 200 | default (1,950) |
| 1, 512 | 331 | 433 (`<16,64,1>`) | +31.1 | +30.5 to +31.2 | 3 | 200 | default (1,950) |
| 1, 1,024 | 650 | 854 (`<16,64,1>`) | +31.5 | +31.1 to +31.7 | 3 | 200 | default (1,950) |
| 1, 1,173 | 752 | 986 (`<16,64,1>`) | +31.1 | +31.0 to +31.2 | 3 | 200 | default (1,950) |
| 1, 2,048 | 1,290 | 1,696 (`<16,64,1>`) | +31.5 | +31.4 to +31.8 | 3 | 200 | default (1,950) |
| 2, 65 | 121 | 140 (`<16,32,2>`) | +15.5 | +15.3 to +15.6 | 3 | 200 | default (1,950) |
| 2, 128 | 184 | 221 (`<16,32,2>`) | +20.2 | +20.0 to +20.3 | 3 | 200 | default (1,950) |
| 2, 256 | 344 | 421 (`<16,32,2>`) | +22.5 | +22.4 to +22.8 | 3 | 200 | default (1,950) |
| 2, 512 | 663 | 826 (`<16,32,2>`) | +24.5 | +24.4 to +24.9 | 3 | 200 | default (1,950) |
| 2, 1,024 | 1,305 | 1,638 (`<16,32,2>`) | +25.5 | +25.4 to +26.0 | 3 | 200 | default (1,950) |
| 2, 1,173 | 1,510 | 1,894 (`<16,32,2>`) | +25.5 | +25.1 to +26.7 | 3 | 200 | default (1,950) |
| 2, 2,048 | 2,566 | 3,262 (`<16,32,2>`) | +27.1 | +26.8 to +27.5 | 3 | 200 | default (1,950) |
| 3, 65 | 183 | 211 (`<16,32,2>`) | +15.1 | +15.0 to +15.3 | 3 | 200 | default (1,950) |
| 3, 128 | 276 | 331 (`<16,32,2>`) | +19.9 | +19.8 to +20.1 | 3 | 200 | default (1,950) |
| 3, 256 | 515 | 639 (`<16,32,2>`) | +24.0 | +23.7 to +24.2 | 3 | 200 | default (1,950) |
| 3, 512 | 993 | 1,255 (`<16,32,2>`) | +26.4 | +26.3 to +26.9 | 3 | 200 | default (1,950) |
| 3, 1,024 | 1,947 | 2,490 (`<16,32,2>`) | +27.9 | +27.9 to +28.5 | 3 | 200 | default (1,950) |
| 3, 1,173 | 2,254 | 2,874 (`<16,32,2>`) | +27.5 | +27.4 to +28.1 | 3 | 200 | default (1,950) |
| 3, 2,048 | 3,861 | 4,963 (`<16,32,2>`) | +28.5 | +28.4 to +29.1 | 3 | 200 | default (1,950) |

Source: [data set `2026-10-31-0002-gdn-value-split`](../../data/2026-10-31-0002-gdn-value-split/README.md), table `perf-diff-interval.tsv`. The table `perf-summary.tsv` has all 8 paths.

The whole-head path is the fastest path for all 21 shapes. The best value-split path is `<16,64,1>` for one sequence and `<16,32,2>` for two or three sequences. The other paths are slower, as much as 3.15 times as slow as the whole-head path. The [results entry](../notebook/2026-10-31-0002-gdn-value-split-result.md) has all values and the cause.

## Prediction

The phase 4 entry [ranked targets](../notebook/2026-10-31-phase4-ranked-targets.md) predicted that `gdn_single` becomes 1.6 times faster with a value-split path (target `D`, confidence 0.45). The result is that it becomes 1.15 to 1.31 times as slow. We reject target `D`. The gain of `D` in the time model of phase 4 is 0. The expected gain of `lat_ms` goes from 17.9 % to 16.9 %.

The cause of the difference is the work in each thread block. The tensor core is the limit of the kernel.

A slice does the Gram matrix and the inverse matrix again. The number of MMA instructions becomes 7 % larger for `V=64` and 21 % larger for `V=32`. The 6 SMs that the whole-head path does not use are only 16 % of the SMs. The phase 4 ratio of 1.6 was an assumption. It did not come from a test.

## Correctness

| Test | Threshold | Result (`pass` or `fail`) |
|---|---|---|
| `test-backend-ops test -b CUDA0 -o GATED_DELTA_NET`, 8 paths that the test sets, and the path selector, on the test build | All 43 cases agree with the CPU backend, for each path | pass: 43 of 43 for each of the 9 runs |
| Bit compare of the 8 paths on 9 shapes | 0 values that are different | pass: 0 of 395,264 to 26,726,400 values |
| `test-backend-ops test -b CUDA0 -o GATED_DELTA_NET` on the build of this branch | All 43 cases agree with the CPU backend | pass: 43 of 43 |
| SASS of the build of this branch, compared with the parent build | 0 names with different SASS, different instruction code, or different `REG`, `STACK`, `SHARED`, `LOCAL`, or `CONSTANT` values | pass: 6,766 of 6,766 names are the same |
| `test-backend-ops test -b CUDA0 -o GATED_DELTA_NET` on the build of the candidate branch | All 43 cases agree with the CPU backend | pass: 43 of 43 |
| `compute-sanitizer` with `memcheck`, `racecheck`, `initcheck`, and `synccheck` on the same test, build of the candidate branch | 0 errors for each tool | pass: 0 errors for each tool |

Source: [data set `2026-10-31-0002-gdn-value-split`](../../data/2026-10-31-0002-gdn-value-split/README.md), tables `test-paths.tsv`, `bitcompare-shapes.tsv`, `build-tests.tsv`, and `sass-summary.tsv`.

## Reproduce

The start is the commit `6b8b65e6672f07805000d3c9683b65c72cc62303` of the branch `dmk/stack`. The branch `dmk/0002-gdn-value-split` has two commits after it: `4e116823a` and `35e3ad302`.

The flags for `cmake` are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

The command for the perf cases of this change is:

```
test-backend-ops perf -b CUDA0 -o GATED_DELTA_NET -p 'head_count=16,head_size=128'
```

On this branch, the command gives the time of the whole-head path. To get the time of the value-split paths, apply the file `raw/build/microbench.patch` of the data set. Then set `DMK_GDN_PATH` to `128`, `64_1`, `64_2`, `32_1`, `32_2`, `16_1`, `16_2`, or `16_3`. The patch is a test aid for the measurement. It is not part of this change.

The branch `dmk/0002-gdn-split-candidate` (commit `9b46881c4`) has the value-split path for a GPU with 38 SMs. We do not merge it.

## Limits

- The test uses one GPU (`RTX 3060 Ti`, 38 SMs), one driver (595.71.05), and 3 runs for each path. Four runs have contended times. The median of the 3 runs is a clean time for all shapes.
- The measurement is a kernel test. We did not make a speed test of two builds with requests. The estimate of the effect on the requests is an inference.
- We did not make a profile of a value-split path.
- The comment at the path selector has the numbers of the measurement. It does not apply to a GPU with more SMs.
- To go back to the baseline, remove the two commits. The program does not change, thus the baseline is the same program.

## Data

- [Data set `2026-10-31-0002-gdn-value-split`](../../data/2026-10-31-0002-gdn-value-split/README.md)
- [Results entry](../notebook/2026-10-31-0002-gdn-value-split-result.md)
- [Phase 4 entry](../notebook/2026-10-31-phase4-ranked-targets.md)
