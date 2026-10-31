# Micro-benchmark of the Q4_0 tile load pipeline of the MMQ kernel (target G)

This data set has the raw files and the tables of one micro-benchmark. The micro-benchmark ran the MMQ kernel of llama.cpp for Q4_0 with and without a pipeline for the tile loads. It used the two Q4_0 weight shapes of the model and 31 numbers of tokens. It ran on one `RTX 3060 Ti` on 2026-10-31.

The data set also has the source of the change, the source of the micro-benchmark, and the script that makes all tables. The [results entry](../../docs/notebook/2026-10-31-0009-mmq-pipeline-result.md) gives the result and the decision.

## Question

The test answers these questions for target `G` of the [ranked list of phase 4](../../docs/notebook/2026-10-31-phase4-ranked-targets.md).

- Is Q4_0 1.08 times faster or more at 1,025 tokens or more with a pipeline for the tile loads? This is the limit that the [entry of the ceiling](../../docs/notebook/2026-10-31-0006-mmq-tiles-result.md) set for the first change.
- Does the pipeline give the same bits as the kernel without it?
- How much time does the pipeline remove from the four request types?

The test has no arm with a server. It is a micro-benchmark with one program. The results entry gives the decision.

## Dates

- First run: 2026-10-31, 08:32:20 UTC.
- Last run: 2026-10-31, 10:04:17 UTC.
- Measurement window `0009-pipe-r1`: 08:32:04 to 08:33:04 UTC. The runs `0009-pipe-r1-q4` and `0009-pipe-r2-q4` started at 08:32:20 and 08:32:33 UTC.
- Measurement window `0009-pipe2-r1`: 09:06:05 to 09:07:05 UTC. The runs `0009-pipe2-r1-q4` and `0009-pipe2-r2-q4` started at 09:06:20 and 09:06:33 UTC.
- Measurement window `0009-pipe34`: 09:10:48 to 09:12:17 UTC. The four runs `0009-pipe3-r1-q4`, `0009-pipe4-r1-q4`, `0009-pipe3-r2-q4`, and `0009-pipe4-r2-q4` started at 09:11:04, 09:11:18, 09:11:31, and 09:11:45 UTC.
- Measurement window `0009-pipe5`: 09:21:02 to 09:22:01 UTC. The runs `0009-pipe5-r1-q4` and `0009-pipe5-r2-q4` started at 09:21:18 and 09:21:31 UTC.
- Measurement window `0009-pipef`: 10:03:33 to 10:04:49 UTC. The runs `0009-pipef-r1-q4`, `0009-pipef-r2-q4`, and `0009-pipef-r3-q4` started at 10:03:50, 10:04:03, and 10:04:17 UTC.

## Project commit

The commit that adds this data set also adds the results entry. The commit before it is `af6a63564d8ebcc8a02808792d18c3995fe3fbde`.

## Build commits

The change is on the branch `dmk/0009-mmq-pipeline` of the fork of llama.cpp of this project. The commit is `4f7bc1fe74e61f79a4fd4f0643a14d8b111f0d31`. The commit before it is `5d61f3ca38a7353fffceaedc616837671466bf91`. It is the branch `dmk/stack` with build b12004 and the changes 0001, 0007, 0002 (test cases), and 0004.

The micro-benchmark is on the branch `dmk/0009-mmq-pipeline-bench`. The commit is `251e766cff9a68c71a1e3c4cb7a2534256b74058`. The commit before it is the commit of the change. The branch is a tool and is not for a merge.

The files `src/mmq_sweep.cu`, `src/sweep_variant.cu`, `src/gen_variants.py`, and `src/build_sweep.sh` have the same bytes as the files in `tools/dmk-mmq-sweep/` of that commit. The file `src/pipeline.patch` is the change: the difference between the two commits of `dmk/0009-mmq-pipeline`. The file `src/sweep.patch` is the difference between the commit of the change and the commit of the micro-benchmark in `ggml/src/ggml-cuda`.

We did not commit the work copies of the versions 1, 2, 2a, 2b, and 3, and `src/` has no source for them. The results entry gives the versions. Version 3 has the code of the commit with four temporary macros: `MMQ_PIPE_UW`, `MMQ_PIPE_UQ`, `MMQ_PIPE_UD`, and `MMQ_PIPE_UY`. They set how many steps of the four copy loops the compiler expands. The default of 16 gives the loops of the commit. The versions 2a and 2b use these macros with `-DMMQ_PIPE_UW=1 -DMMQ_PIPE_UY=1` and with `-DMMQ_PIPE_UW=1 -DMMQ_PIPE_UY=1 -DMMQ_PIPE_UQ=2 -DMMQ_PIPE_UD=2`.

The programs of version 3 and of the commit have 6,382,288 bytes each and different SHA-256 values. We built the program `0009-pipef` from a work copy with the same text as the commit of the micro-benchmark. We wrote the commit after the build.

## Build flags

The script `src/build_sweep.sh` builds one object file for each variant, the program `mmq_sweep.cu`, and the link. It runs in a build container and does not use a GPU. The program has the name `llama`, because the harness of the GPU host starts a program with this name. The flags of `nvcc` are:

```
nvcc -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_86,code=[sm_86] -use_fast_math -extended-lambda \
     -DGGML_CUDA_USE_GRAPHS -DGGML_SCHED_MAX_COPIES=4 -D_GNU_SOURCE -D_XOPEN_SOURCE=600 -w -Xptxas -v
```

The set `pipe` has 32 variants. For each `J` from 8 to 128 in steps of 8 it has one baseline variant and one variant with the suffix `_abl1`. With the environment variable `SWEEP_BASE_NOPIPE=1` the script adds `-DGGML_CUDA_MMQ_NO_WEIGHT_PIPE` to each baseline variant. This flag turns the pipeline off in the kernel and in the code of the host. The variant `_abl1` has the pipeline.

| Item | Value |
|---|---|
| Image of the build container | `localhost/dmk-builder:cuda12.8.1-gcc14`, image ID `54be7481bcfd4bd3085c5572a80f565384671b5aa2f5879ea4fb2a3c74f9c947`. We did not publish it. |
| Compilers | `nvcc` 12.8.93 and `g++-14` 14.2.0. |
| Variants | 32 for each program. All variants built. |

The next table gives the end of each build and the SHA-256 of each program. The build of the objects had a time of 23 s to 38 s.

| Version | Program | End of the build (UTC) | SHA-256 |
|---|---|---|---|
| 1 | `mmq-pipe-bench` | 08:26:45 | `f29b239c7dbbcfc9e4ee363c68b1ad95ab32e43f50ad95c0c9bd77b6581c3ffc` |
| 2 | `mmq-pipe-bench2` | 08:42:15 | `4e6d3d1d51ea7755a6e077917b3709d70a9115d4d13bec6cd540a65a812e12f6` |
| 2a | `mmq-pipe-bench3` | 09:10:16 | `420daea87bf3ba0cb49bee28d5c512ae5d70c2eca5735fec0a681da9e99fce9c` |
| 2b | `mmq-pipe-bench4` | 09:10:44 | `ac96d3041e7d4d94492e25292370c639e816b45cb2bc9e3524fb13ddb726b18b` |
| 3 | `mmq-pipe-bench5` | 09:21:01 | `5199035f244a2feb34c37f32c4c6965028dba4a87a273c45be7ce2d127251f4f` |
| 3c | `mmq-pipe-final` | 09:24:52 | `f56ca03ec27ca762c2f1785ac6ba115de88711ba3b7da764bfa13b34fddaef3c` |

The copy of each program on the GPU host has the same SHA-256. The file `raw/build/variants-pipe.tsv` lists the 32 variants.

## Image

No image of llama.cpp ran. The micro-benchmark is a program that does not use a server.

## Model

No model ran. The weights and the activations are random numbers. The 2 shapes come from the "mix8" model. They are the rows `M` and the columns `K` of its Q4_0 weights.

| Type | Rows `M` | `K` | Operations of the model |
|---|---:|---:|---|
| Q4_0 | 9,216 | 2,560 | `ffn_up`, `ffn_gate` |
| Q4_0 | 2,560 | 9,216 | `ffn_down` |

The Q8_0 shapes are not in the test. The change has no pipeline for Q8_0.

## Driver

The driver is 595.71.05.

## GPU

One `RTX 3060 Ti` with compute capability 8.6 and 38 SMs. The limit of shared memory for one thread block is 101,376 bytes. The first lines of each file `raw/*.err` have the values of the device.

## Power limit

200 W. The files `raw/*.meta` have the value.

## Clocks

The clocks have the default values. We did not lock them. The files `raw/*.smi` have the samples of the GPU during each run.

## Server flags

None. No server ran.

## Commands

The build command is:

```
SWEEP_BASE_NOPIPE=1 bash src/build_sweep.sh <patched llama.cpp tree> pipe <output folder>
```

The measurement windows ran the lines of `raw/0009-pipe*-r*.txt`. The harness of the GPU host started the program as `llama sweep ...`. Each run used this command:

```
sweep --shapes q4_0:9216:2560,q4_0:2560:9216 --ns 9,13,16,26,30,32,43,47,51,54,64,67,69,84,86,94,102,108,120,128,129,151,158,256,512,1024,1025,1173,1288,1536,2048 --policies 0 --variants _abl1 --j-baseline-only 1 --reps-ms 15 --trials 5
```

The flag `--variants _abl1` selects the variants with the pipeline, and the baseline of each cell always runs first. The flag `--j-baseline-only 1` runs a variant only at the `J` that the baseline of the build uses for the cell. The program compares the output of each variant with the output of the baseline of the cell. The column `ndiff` is the number of values that are different.

Run this command to make the tables:

```
python3 make_tables.py
```

## Arms

The test has no arms. Each cell runs the kernel without the pipeline (baseline) and the kernel with the pipeline in the same program. The ratio of a cell is the time of the baseline divided by the time of the pipeline.

## Raw files

Each run has four files with the same name and the suffixes `.out`, `.err`, `.meta`, and `.smi`. The file `.out` has the table of the run. The file `.err` has the device line. The file `.meta` has the command, the date, and the exit code. The file `.smi` has the samples of the GPU. Each schedule has one file `raw/0009-pipe*-r*.txt` with the command lines.

| Version | Description | Runs |
|---|---|---|
| 1 | Weights only. `cp.async` loads the Q4_0 blocks of the weight tile into shared memory. The threads make the int8 values from them. The activation loads do not change. | `0009-pipe-r1`, `0009-pipe-r2` |
| 2 | Version 1 and `cp.async` for the activations, in two buffers. Four barriers in each step. A test at run time that the data follows the 16-byte rule of `cp.async`. | `0009-pipe2-r1`, `0009-pipe2-r2` |
| 2a | Version 2 with the copy loops not expanded. | `0009-pipe3-r1`, `0009-pipe3-r2` |
| 2b | Version 2a with the loops that make the int8 values expanded 2 times. | `0009-pipe4-r1`, `0009-pipe4-r2` |
| 3 | Version 2 with three barriers in each step. The host uses the kernel without the pipeline when the data does not follow the 16-byte rule. | `0009-pipe5-r1`, `0009-pipe5-r2` |
| 3c | Version 3 of the commit. This is the last version. | `0009-pipef-r1`, `0009-pipef-r2`, `0009-pipef-r3` |

The file `raw/build/variants-pipe.tsv` lists the variants.

## Tables

| Table | Data |
|---|---|
| `versions.tsv` | For each version: the mean, the smallest, and the largest ratio at 1,025 tokens or more, the cells that are slower than the baseline, `ndiff`, and the result against the limit of 1.08. |
| `limit.tsv` | The ratio of each cell at 1,025 tokens or more for the last version, and the mean for each shape. |
| `final-speedup.tsv` | The time of the baseline and of the pipeline, the ratio, the registers, and the shared memory for each shape and number of tokens of the last version. |
| `by-version.tsv` | The ratio of each version for each shape and number of tokens. |
| `registers.tsv` | The registers, the local memory, and the shared memory of the baseline and of the pipeline for `J` of 16, 64, 120, and 128. |
| `vs-ceiling.tsv` | The time of the pipeline against the time of the variant `same_loads` of the [ablation runs](../2026-10-31-0006-mmq-tiles/README.md), for each cell. |
| `vs-ceiling-by-class.tsv` | The part of the ceiling that the pipeline removes for each group of numbers of tokens. |
| `projection.tsv` | The time that the pipeline removes from the Q4_0 operations of each request type. It uses the census of the [data set of target H](../2026-10-31-0006-mmq-tiles/README.md). |

The script `make_tables.py` makes all tables from `raw/*.out`. The table `vs-ceiling.tsv` also reads `raw/0006-abl-r3-q4.out` of the data set 2026-10-31-0006-mmq-tiles. The table `projection.tsv` also reads its files `raw/inputs/requests.tsv` and `tables/weights.tsv`.

## Limits

- Each row is the lowest time of five runs of the loop. The three runs of the last version give ratios of one cell with a range of 0.4 % (median), 1.8 % (90th percentile), and 4.9 % (maximum). At 1,025 tokens or more the maximum is 2.1 %.
- The micro-benchmark uses random weights and random activations. [INFERENCE: the time of the MMA kernel does not change with the values. We did not check this.]
- The launches run one after the other. A request of the model has other kernels between the MMQ launches, and the state of the L2 cache is different.
- The table `projection.tsv` uses the nearest of the 29 numbers of the ablation runs for each number of tokens of a pass. It has no Q8_0 and no kernels other than MMQ.
- The data for the correctness of the output are the column `ndiff` of each run. The columns have 0 in all rows of all runs. We did not run `test-backend-ops`, a test of answers, or `compute-sanitizer`, because we do not use the change.
- The versions 1, 2, 2a, 2b, and 3 have no committed source. See the section "Build commits".

## Files not added

- The logs of the measurement windows. They have names of hosts and numbers of monitors.
- The programs `llama` (6 MB to 7 MB each). Build them again with `src/build_sweep.sh`.
- The full log of each build (the `-Xptxas -v` output for each variant) and the object files.
