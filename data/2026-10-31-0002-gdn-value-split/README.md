# Data set: change 0002, value-split path of gdn_single (2026-10-31)

This data set holds the raw files and the derived tables of the measurement of change 0002. The change is target `D` of the [phase 4 entry](../../docs/notebook/2026-10-31-phase4-ranked-targets.md). It makes the kernel `gdn_single` use a value-split path on the `RTX 3060 Ti`. The measurement is a micro-benchmark of the kernel. It gives the kernel time of each path, and it tests that all paths give the same bits.

## Question

1. Speed: Is a value-split path of `gdn_single` faster than the whole-head path for the shapes of the model "mix8" on the `RTX 3060 Ti`? The phase 4 entry predicted a ratio of 1.6 for the speed.
2. Bits: Do the 7 value-split paths give the same bits as the whole-head path? The equivalence class of the change is `exact`.
3. Test of the operation: Do all 8 paths pass the test `GATED_DELTA_NET` of `test-backend-ops`?
4. Builds of the branches: Does the build of the branch of the pull request give the same code and pass the same test? Does the build of the candidate branch pass the test?

The prediction is in the [phase 4 entry](../../docs/notebook/2026-10-31-phase4-ranked-targets.md). This data set has no speed test of two builds, thus the notebook has no prediction entry for this test. The [results entry](../../docs/notebook/2026-10-31-0002-gdn-value-split-result.md) gives the result and the cause.

## Dates

| Item | Value |
|---|---|
| First run of the micro-benchmark starts | 2026-10-31 06:18:04 UTC |
| Last run of the micro-benchmark starts | 2026-10-31 06:30:19 UTC |
| Measurement measurement window | 06:17 to 06:31 UTC |
| Tests of the 8 paths and test of the bits | 2026-10-31 05:33 to 05:36 UTC |
| Tests of the builds of the branches | 2026-10-31 06:31 to 06:34 UTC |

The micro-benchmark has 24 runs. They used about 13 minutes in a measurement window. The measurement window held the GPU lock. It stopped the server in production and started it again after the measurement window.

The tests of the builds of the branches held the GPU lock and ran with the server in production. The tests of the paths and the test of the bits ran with no GPU lock. None of these tests measure time.

The data set does not have the runs of a first measurement window of the micro-benchmark. Other GPU programs ran in that measurement window with no GPU lock. The section "Files not added" gives the times.

Source: the `start_utc` column of [runs.tsv](tables/runs.tsv), and the files `raw/test/*.meta`, `raw/bitcompare/bitcompare.meta`, and `raw/pr-build/*.meta`.

## Project commit

The runs did not use the scripts in the folder `harness/` of this repository. A small script started `test-backend-ops` in a container and wrote the `.meta` files. It ran in the script of the measurement window, which stops the server in production. The two scripts are not in this repository. The commit that adds this folder is the commit of the branch `notebook/0002-gdn-value-split` that has this README. It has the script `make_tables.py` and the tables.

## Build commits

The three builds are in the fork of llama.cpp of this project. All three use the commit `6b8b65e6672f07805000d3c9683b65c72cc62303` of the branch `dmk/stack` as the start. This commit is also the commit of arm `L` of the data set [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md).

| Build | Source | Commit |
|---|---|---|
| Test build (`microbench`) | The start, the difference of the commit `4e116823a` (perf cases), and the file `raw/build/microbench.patch` | The commit `6b8b65e6672f07805000d3c9683b65c72cc62303`, with changes in 2 files. The hash of `git diff HEAD` is `405085a1c94858afcc1995bae728eb06c1ff804b30df6aa4c8659f68f4fc5917`. |
| Build of the pull request | Branch `dmk/0002-gdn-value-split` | `35e3ad302d65140b46251716b8148216aafc9df3`. The source had 0 changes. |
| Build of the candidate branch | Branch `dmk/0002-gdn-split-candidate` | `9b46881c42961f115b5010c1fed24812fdcefb23`. The source had 0 changes. |

The file `raw/build/microbench.patch` has the SHA-256 value `8fa77ab15032009a950f0cba206726e50150a2b527cbfb878e73c28ddf088f62`. It adds two parts to the file `gated-delta-net-mma.cu`.

The environment variable `DMK_GDN_PATH` sets the path of `gdn_single` for each launch.

The values are `128`, `64_1`, `64_2`, `32_1`, `32_2`, `16_1`, `16_2`, and `16_3`. The first number is `V`. The second number is the minimum number of blocks of each SM.
The environment variable `DMK_GDN_CHECK=1` makes the program run all 8 paths on the same input. It compares each 4-byte value of the output of each path with the value of the path `128`.

The output has the attention output and the new state. It writes one line `GDN_CHECK ...` for each path to the standard error. In the line, `bytes_differ=N of M` gives the number of different values, and then the number of values that the program compared.

These two changes are not in the branch of the pull request and not in the candidate branch. They are a test aid for the measurement. The file `raw/build/MANIFEST-microbench.txt` gives the facts of the test build and the SHA-256 value of each file. The branch `dmk/0002-gdn-value-split` has two commits: `4e116823a` adds the perf cases, and `35e3ad302` adds a comment at the path selector. The branch `dmk/0002-gdn-split-candidate` has the commit `9b46881c4` after `4e116823a`.

We made the test build on 2026-10-31 at 05:09 UTC. We made the build of the pull request and the build of the candidate branch on 2026-10-31 at 05:46 UTC. The files `raw/build/MANIFEST-pr.txt` and `raw/build/MANIFEST-candidate.txt` give the facts of these builds.

## Build flags

We made the builds with `harness/build/build.sh`. The manifests show the same tools and the same `cmake` command.

| Tool | Version |
|---|---|
| Image for the build | `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`, digest `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` |
| `nvcc` | 12.8.93 |
| `gcc` | 14.2.0 |
| `cmake` | 3.28.3 |

The `cmake` flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

The flags are the flags of the builds in the data set [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). Each build used the object files of the build of that data set, and the compiler made only the files that changed again.

## Image

| Use | Image | Digest |
|---|---|---|
| Container of `test-backend-ops` | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` | `sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |

The container runs the file `/opt/dmk/test-backend-ops` of the build, with the libraries of the build in `/opt/dmk`. The image gives only the CUDA libraries and the system libraries.

## Model

The tests use no model file. `test-backend-ops` makes random input for each case. The shapes of the perf cases are the shapes of the GDN layers of the model "mix8": 16 key heads, 32 value heads, and S=128.

## Driver

The version of the driver is 595.71.05. The `.meta` files of the runs show it.

## GPU

The GPU is the `RTX 3060 Ti`. It has 38 SMs and the architecture `sm_86`.

## Power limit

The power limit was 200 W in all runs. The project did not change it.

## Clocks

We did not lock the clocks. The GPU used its default clocks. The column `sm_clock_mhz_start` of [runs.tsv](tables/runs.tsv) shows the SM clock at the start of each run. It is 1,665 to 1,965 MHz.

The first run (`16_3`, repetition 1) started at 1,665 MHz after the GPU had no work. Its `throttle_reasons_at_start` has the value `0x0000000000000001`. The run `16_2` of repetition 3 has the value `0x0000000000000024` at the start. All other runs have the value `0x0000000000000000`. The times of these two runs are in the range of the times of the other repetitions.

## Server flags

The tests use no server. The measurement window stopped the server in production and started it again.

## Host load

The one-minute load average of the host was 9.9 to 17.2 in the 24 runs. The columns `load1_before` and `load1_after` of [runs.tsv](tables/runs.tsv) show the load of each run. It is the load of the CPU of the host. Builds of other agents use it. The runs use the GPU only. No time of the 24 runs is a contended time.

## Commands

The measurement window started one run at a time. Each run used this command in a container. The value of `<path>` is one of the 8 path names. The list of the 24 runs is in [raw/perf/order.txt](raw/perf/order.txt). A seed set the sequence of the paths in each repetition.

```
docker run --rm --runtime=nvidia -e DMK_GDN_PATH=<path> -e LD_LIBRARY_PATH=/opt/dmk \
    -v <folder of the test build>/dist:/opt/dmk:ro --entrypoint /opt/dmk/test-backend-ops \
    ghcr.io/ggml-org/llama.cpp@sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352 \
    perf -b CUDA0 -o GATED_DELTA_NET -p 'head_count=16,head_size=128'
```

The tests of the operation used the same container, with the mode `test` and without the flag `-p`.

```
... test -b CUDA0 -o GATED_DELTA_NET
```

The test of the bits set `DMK_GDN_CHECK=1` and did not set `DMK_GDN_PATH`. It used the mode `test`. The tests of the builds of the branches used the same command for the mode `test`. The runs with `compute-sanitizer` used `/tools/compute-sanitizer-12.8.93/bin/compute-sanitizer --tool <tool> --error-exitcode 99` before the program. The tool is `memcheck`, `racecheck`, `initcheck`, or `synccheck`.

## Arms

This data set has no arms of a speed test. The micro-benchmark compares 8 paths of one build. The path name gives the value of `DMK_GDN_PATH`.

| Path name | `V` | Thread blocks of 1 sequence | Minimum blocks of each SM |
|---|---:|---:|---:|
| `128` | 128 | 32 | 1 |
| `64_1` | 64 | 64 | 1 |
| `64_2` | 64 | 64 | 2 |
| `32_1` | 32 | 128 | 1 |
| `32_2` | 32 | 128 | 2 |
| `16_1` | 16 | 256 | 1 |
| `16_2` | 16 | 256 | 2 |
| `16_3` | 16 | 256 | 3 |

The path `128` is the whole-head path. It is the path that the build of the pull request selects. The other 7 paths are value-split paths.

## Raw files

The folder `raw/` has 131 files. The files in `perf/`, `test/`, `bitcompare/`, and `build-tests/` are the output of the programs, with no change. A small script on the GPU host started the programs and wrote the `.meta` files. The script is not in this repository. The program `test-backend-ops` writes color codes in the lines of its result. The files keep these codes.

| Files | Number | Written by | Contents |
|---|---:|---|---|
| `perf/perf-r<N>-<path>.{out,err,meta}` | 72 | `test-backend-ops` and the script | The 24 runs of the micro-benchmark, with 3 files each. `<N>` is the repetition. The `.out` file has one line for each of the 21 shapes. The `.meta` file has the start time, the load, the driver, the power limit, the clock, the temperature, and the exit code. |
| `perf/order.txt` | 1 | The script | The sequence of the 24 runs. Each line has the repetition and the path name. |
| `test/test-<path>.{out,err,meta}` | 27 | `test-backend-ops` and the script | The test of the operation with the test build, for the path selector (`default`) and for the 8 paths |
| `bitcompare/bitcompare.{out,err,meta}` | 3 | The test build and the script | The test with `DMK_GDN_CHECK=1`. The `.err` file has the 72 lines `GDN_CHECK`. |
| `build-tests/0002-pr/test.{out,err,meta}` | 3 | `test-backend-ops` and the script | The test of the operation with the build of the pull request |
| `build-tests/0002-candidate/<tool>.{out,err,meta}` | 15 | `test-backend-ops`, `compute-sanitizer`, and the script | The test of the operation with the build of the candidate branch (`test`), and the runs with the four tools of `compute-sanitizer` |
| `sass-compare-pr/`, `sass-compare-candidate/` | 4 | `sass_compare.py` | For each build: `sass-compare.json` and `key-kernels.tsv`. The script compared the build with the parent build. The script used only the names with `gated_delta_net` or `gdn`. The two builds have no name with a different SASS, thus the folders have no `diff-<hash>.txt` file. |
| `build/MANIFEST-microbench.txt`, `build/MANIFEST-pr.txt`, `build/MANIFEST-candidate.txt` | 3 | `build.sh` | The facts of each build: the commit, the tools, the `cmake` command and the SHA-256 value of each file |
| `build/microbench.patch` | 1 | A person | The change of the test build that is not in the branches |
| `inputs/gdn-ms-per-pass.tsv`, `inputs/request-passes.tsv` | 2 | A person | Columns of the files `pass_kernels.tsv` and `requests.tsv` in `raw/` of the data set [`2026-10-31-phase4-model`](../2026-10-31-phase4-model/README.md). The script uses them for the estimate of the candidate branch. |

Each `.err` file starts with this text: `ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11`. The container set-up of the driver writes it. Then it has 3 lines of text from the start of the CUDA backend. The file `bitcompare.err` also has the 72 lines `GDN_CHECK`.

## Tables

The script [make_tables.py](make_tables.py) makes all tables in the folder `tables/` from the files in `raw/`. The tables are TSV files. The first line of a TSV file has the names of the columns. An empty field means that the table has no value there. The script uses Python 3 and no other library. It removes the color codes of the lines of `test-backend-ops`.

### runs.tsv

One row for each run of the micro-benchmark, in the sequence of the start time. It has 24 rows.

| Column | Unit | Meaning |
|---|---|---|
| `path` | name | The path name |
| `rep` | number | The repetition, 1 to 3 |
| `start_utc` | UTC | The start time of the run |
| `load1_before`, `load1_after` | number | The one-minute load average of the host before and after the run |
| `driver` | version | The version of the driver |
| `power_limit_w` | W | The power limit at the start |
| `sm_clock_mhz_start`, `sm_clock_mhz_end` | MHz | The SM clock at the start and at the end |
| `temperature_c_start`, `temperature_c_end` | °C | The temperature of the GPU at the start and at the end |
| `throttle_reasons_at_start` | bit mask | The cause of a lower clock at the start, from `nvidia-smi` |
| `power_draw_w_end` | W | The power at the end |
| `exit_code` | number | The exit code of `test-backend-ops` |
| `shapes` | number | The number of shapes in the output. It is 21. |
| `contended_shapes` | number | The number of shapes for which the run has a contended time |

### perf-long.tsv

One row for each time. It has 504 rows: 8 paths, 3 repetitions, and 21 shapes.

| Column | Unit | Meaning |
|---|---|---|
| `path`, `rep` | name, number | The path name and the repetition |
| `n_seqs`, `n_tokens` | number | The number of sequences and the number of tokens of the shape |
| `runs` | number | The number of runs of the operation that `test-backend-ops` made |
| `us_per_run` | µs | The time of one run of the operation |
| `kb_per_run` | kB | The memory that one run uses, from `test-backend-ops` |
| `gb_per_s` | GB/s | The rate that `test-backend-ops` calculates from the memory and the time |
| `contended` | true or false | `true` if the time is a contended time |

### perf-summary.tsv

One row for each path and shape. It has 168 rows.

| Column | Unit | Meaning |
|---|---|---|
| `n_seqs`, `n_tokens`, `path` | number, number, name | The shape and the path name |
| `reps` | number | The number of repetitions. It is 3. |
| `median_us`, `min_us`, `max_us` | µs | The median, the smallest, and the largest of the 3 times |
| `clean_reps` | number | The number of times with no contended time |
| `ratio_to_whole` | ratio | `median_us` divided by `median_us` of the path `128` for the same shape |

### perf-diff-interval.tsv

One row for each shape. It has 21 rows. It compares the whole-head path with the best value-split path of the shape.

| Column | Unit | Meaning |
|---|---|---|
| `n_seqs`, `n_tokens` | number | The shape |
| `whole_us` | µs | The median time of the path `128` |
| `best_split`, `best_split_us` | name, µs | The value-split path with the smallest median time, and its median time |
| `difference_pct` | % | The difference of the median times, in percent of `whole_us` |
| `low_pct`, `high_pct` | % | The smallest and the largest difference that the clean times give: the smallest time of the value-split path against the largest time of the path `128`, and the largest time of the value-split path against the smallest time of the path `128` |
| `clean_reps_whole`, `clean_reps_split` | number | The number of clean times of the two paths |

### perf-best-split.tsv

One row for each shape. It has 21 rows.

| Column | Unit | Meaning |
|---|---|---|
| `n_seqs`, `n_tokens` | number | The shape |
| `whole_us` | µs | The median time of the path `128` |
| `best_split`, `best_split_us` | name, µs | The value-split path with the smallest median time, and its median time |
| `ratio_to_whole` | ratio | `best_split_us` divided by `whole_us` |
| `whole_is_fastest` | true or false | `true` if the path `128` has a smaller median time than all 7 value-split paths |

### perf-ratio-range.tsv

One row for each value-split path. It has 7 rows.

| Column | Unit | Meaning |
|---|---|---|
| `path`, `slice_width` | name, number | The path name and `V` |
| `blocks_bound` | number | The minimum number of blocks of each SM that the path sets |
| `thread_blocks_1_seq` | number | The number of thread blocks for one sequence |
| `blocks_per_sm_1_seq` | number | `thread_blocks_1_seq` divided by the 38 SMs |
| `ratio_min`, `ratio_max` | ratio | The smallest and the largest `ratio_to_whole` of the path for the 21 shapes |
| `ratio_1_seq_2048` | ratio | `ratio_to_whole` for one sequence with 2,048 tokens |

### perf-rep-spread.tsv

One row for each path. It has 8 rows.

| Column | Unit | Meaning |
|---|---|---|
| `path` | name | The path name |
| `times` | number | The number of times of the path. It is 63. |
| `contended_times` | number | The number of contended times |
| `fewest_clean_reps` | number | The smallest number of clean times of one shape |
| `largest_spread_clean_pct` | % | The largest value of the largest clean time divided by the smallest clean time, minus 1, for the 21 shapes |

### perf-whole-scaling.tsv

One row for each number of tokens. It has 7 rows.

| Column | Unit | Meaning |
|---|---|---|
| `n_tokens` | number | The number of tokens |
| `ratio_1_seq`, `ratio_2_seq`, `ratio_3_seq` | ratio | The median time of the path `128` for 1, 2, and 3 sequences, divided by its median time for 1 sequence |

### mma-count.tsv

One row for each path. It has 8 rows. The script calculates the numbers from the loop limits of `gdn_single` for one chunk of 16 tokens and S=128. One matrix operation has 3 MMA instructions for each group of 16 numbers that it adds. No raw file has these numbers.

| Column | Unit | Meaning |
|---|---|---|
| `path`, `slice_width`, `slices_per_head` | name, number, number | The path name, `V`, and the number of slices of a head |
| `gram`, `projection`, `solve`, `out`, `state` | instructions | The MMA instructions of one thread block for the Gram matrix, for the terms with the state at the start, for the inverse matrix, for the output, and for the new state |
| `per_thread_block` | instructions | The sum of the five columns |
| `per_head` | instructions | `per_thread_block` times `slices_per_head` |
| `ratio_to_whole` | ratio | `per_head` divided by `per_head` of the path `128` |

### test-paths.tsv

One row for each run of the test of the operation on the test build. It has 9 rows: the path selector of the build, and the 8 paths.

| Column | Unit | Meaning |
|---|---|---|
| `path` | name | `default` for the path selector of the build, or the path name |
| `exit_code` | number | The exit code of `test-backend-ops` |
| `cases`, `cases_ok`, `cases_failed` | number | The number of cases of `GATED_DELTA_NET`, and the numbers with the result `OK` and with a different result |
| `backend_ok` | true or false | `true` if the output has the line `Backend CUDA0: OK` |

### bitcompare.tsv

One row for each shape and path. It has 72 rows.

| Column | Unit | Meaning |
|---|---|---|
| `heads` | number | The number of value heads |
| `n_seqs`, `n_tokens` | number | The number of sequences and the number of tokens |
| `path` | name | The path name. The path `128` is the reference. |
| `floats_differ` | number | The number of 4-byte values of the output that are different from the value of the path `128` |
| `floats_compared` | number | The number of 4-byte values that the program compared |

### bitcompare-shapes.tsv

One row for each shape. It has 9 rows.

| Column | Unit | Meaning |
|---|---|---|
| `heads`, `n_seqs`, `n_tokens` | number | The shape |
| `floats_compared` | number | The number of 4-byte values that the program compared for each path |
| `paths_compared` | number | The number of paths, with the reference. It is 8. |
| `floats_differ_in_all_paths` | number | The sum of `floats_differ` for the paths |

### build-tests.tsv

One row for each test of the builds of the branches. It has 6 rows.

| Column | Unit | Meaning |
|---|---|---|
| `build` | name | `0002-pr` for the build of the pull request, or `0002-candidate` for the build of the candidate branch |
| `sanitizer_tool` | name | The tool of `compute-sanitizer`, or `none` |
| `exit_code` | number | The exit code of the command |
| `cases`, `cases_ok`, `cases_failed` | number | The number of cases of `GATED_DELTA_NET`, and the numbers with the result `OK` and with a different result |
| `sanitizer_errors` | number | The number of errors that the tool reports. The field is empty if the run has no tool. |

### sass-summary.tsv

One row for each build of a branch. It has 2 rows. The parent build is the build of `dmk/stack`.

| Column | Unit | Meaning |
|---|---|---|
| `build` | name | The build of the branch |
| `cubins_per_side` | number | The number of cubins in each build |
| `names_in_both` | number | The number of kernel names that are in the two builds |
| `names_only_parent`, `names_only_build` | number | The number of names that only one build has |
| `names_sass_identical`, `names_sass_different` | number | The number of names with the same SASS text, and with different SASS text |
| `names_encoding_identical`, `names_encoding_different` | number | The number of names with the same code of the instructions, and with different code |
| `names_resusage_different` | number | The number of names with different values of `REG`, `STACK`, `SHARED`, `LOCAL` or `CONSTANT` |
| `instructions_parent`, `instructions_build` | instructions | The number of SASS instructions of each build |
| `names_only_in_build` | text | The names that only the build has |

### candidate-estimate.tsv

One row for each request type, and one row for `lat_ms`. It has 5 rows. The table is an estimate, and it is not a measurement. The method is in the [results entry](../../docs/notebook/2026-10-31-0002-gdn-value-split-result.md).

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | name | The request type, or `lat_ms` |
| `cases` | number | The number of cases of the type in the benchmark |
| `wall_ms` | ms | The time of the request from the phase 4 model. For `lat_ms`, it is the mean of the cases. |
| `gdn_single_ms` | ms | The sum of the times of the passes with `gdn_single` in the request |
| `extra_ms_estimate` | ms | The estimate of the longer time of the candidate branch |
| `extra_pct_estimate` | % | `extra_ms_estimate` in percent of `wall_ms` |

## Steps to make the tables again

1. Run `python3 data/2026-10-31-0002-gdn-value-split/make_tables.py` in the repository folder.
2. Compare the new tables with the tables in the repository.

The script gives the same tables each time.

## Files not added

| Files | Cause |
|---|---|
| The 24 runs of an first measurement window of the micro-benchmark (2026-10-31, 05:11 to 05:28 UTC), and one run of a first try in the same measurement window | Other GPU programs ran with no GPU lock in that measurement window. Their times (UTC) are 05:11:03 to 05:12:03, 05:13:51 to 05:14:28, 05:14:20 to 05:16:15, 05:26:10 to 05:26:17, and 05:26:36 to 05:27:52. The runs `16_3`, `64_2`, `32_1`, and `128` of repetition 1, and the run `16_1` of repetition 3, are in these times. The first try stopped after one run, because a command read the list of runs. We made all 24 runs again in the measurement window of this data set. The medians of the first measurement window are different from the medians of this data set by 0.06 % (median) and by 2.0 % (largest). |
| The first runs of the tests of the 8 paths, of the test of the bits, and of a speed run with the server in production (2026-10-31, 05:07 to 05:11 UTC) | They ran with no GPU lock. The files `test/*` and `bitcompare/*` are from a second run (05:33 to 05:35 UTC). That run also ran with no GPU lock. The tests do not measure time, and the two runs had the same results. A run with the GPU lock did not start for more than 45 minutes, because other programs held the GPU lock. We stopped it. |
| The scripts of the measurement window and the script of the runs | They hold names of hosts and of monitors. The `.meta` files and `order.txt` have the facts of the runs. |
| The cubins, the libraries, and the files `*-resusage.txt` of the SASS compare | They are binary files or large files. The manifests give the SHA-256 value of each library. |
| The run of the script of the phase 4 model without target `D` | It ran in a work folder. Its output is not a raw file of this data set. The [results entry](../../docs/notebook/2026-10-31-0002-gdn-value-split-result.md) gives the totals. |

## Files that the scan of the folder finds

The scan `tools/publish/scan-dir.sh` finds no item in this folder.
