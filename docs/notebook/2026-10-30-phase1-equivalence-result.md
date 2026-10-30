# Phase 1: test of the build from source (result)

Written 2026-10-30, after the measurement window (22:07 to 22:51 UTC).

## Task of this entry

This entry gives the result of the test that the [prediction entry](2026-10-30-phase1-equivalence-plan.md) declares. The test compares the build from source (arm D) with the stock build (arm S). We did not change the prediction entry after the measurement. The [README of the data set](../../data/2026-10-30-window1/README.md) gives the raw files and the tables.

## Result in short

Build D is in all limits of the prediction entry. The point estimate of the difference of `lat_ms` is -0.40 % (D against S), in the ±0.5 % of the prediction. All 95 % confidence intervals are in ±2 %. The answers of the two arms are the same bit for bit. Build D is the baseline build for all subsequent work.

## Conditions

| Item | Value |
|---|---|
| Measurement window | 2026-10-30, 22:07 to 22:51 UTC. The GPU did the work of one test at a time. The server in production was off. |
| GPU | `RTX 3060 Ti` |
| Driver | 595.71.05 |
| Power limit | 200 W |
| Clocks | Not locked. The GPU used its default clocks. We did not change the clocks or the power limit. |
| Arm S, image | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004@sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352` |
| Arm D, source | Tag `b12004`, commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. No file of the source had a change. |
| Arm D, build date | 2026-10-30, 21:50 UTC |
| Arm D, image for the build | `nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66` |
| Arm D, compilers | `nvcc` 12.8.93, `gcc-14` 14.2.0, CMake 3.28.3 |
| Arm D, CMake flags | The code block below this table |
| Differences from the upstream flags | Five flags: `-DCMAKE_CUDA_ARCHITECTURES=86-real`, `-DLLAMA_BUILD_TESTS=ON`, `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON`, `-DLLAMA_USE_PREBUILT_UI=OFF` and `-DGGML_CCACHE=OFF` |
| How arm D runs | In the image of arm S. The build is in the folder `/opt/dmk` of the container. |
| Server flags, the two arms | `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots` |
| Model | The "mix8" GGUF file `h2o-lightning-4b-rq-Q4_0-mix8.gguf` |
| Benchmark | The private benchmark with 100 decisions. The client did one warm-up, one quality pass, 5 timed passes, and one throughput pass with 8 requests at the same time. |
| Blocks and seeds | Benchmark: 6 blocks, seed 20261030. `llama bench`: 4 blocks, seed 20261031. Each block has one S run and one D run in a random sequence. |
| Servers | Each run of the benchmark started a new server. |
| Host CPU load | The 1-minute load average was 12.57 to 19.53 in blocks 1 and 2 of the benchmark. It fell from 17.84 to 2.91 in block 3. It was 2.13 to 2.84 in blocks 4 to 6. |
| SASS, before the measurement window | The `sm_86` SASS of D was the same as the SASS in the image of S for all 6,754 kernel names. |
| `test-backend-ops`, before the measurement window | `MUL_MAT` on D: 2,490 of 2,490 tests are true. |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `runs.tsv`. The two checks before the measurement window have no table.

Arm D used these CMake flags. They are the same as the flags of `.devops/cuda.Dockerfile` of the tag `b12004`, with the five differences in the table.

```
-DCMAKE_BUILD_TYPE=Release
-DGGML_NATIVE=OFF
-DGGML_CUDA=ON
-DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON
-DLLAMA_BUILD_TESTS=ON
-DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF
-DGGML_CCACHE=OFF
```

## Speed test with the private benchmark

Each block has one S run and one D run. The estimate is the mean of the 6 block ratios, in %. A block ratio is the value of D divided by the value of S, minus 1. The interval is the 95 % bootstrap interval of the blocks with 10,000 samples.

A latency that is smaller is faster. A rate that is larger is faster. For the benchmark, the limit of the prediction entry uses `lat_ms` only. The tables also give the other metrics, but they are not part of the limit.

| Metric | S mean (ms) | D mean (ms) | D against S (%) | 95 % interval, low end (%) | 95 % interval, high end (%) | In ±2 % |
|---|---:|---:|---:|---:|---:|---|
| `lat_ms` | 239.239 | 238.285 | -0.397 | -0.759 | -0.091 | Yes |
| `short_ms` | 72.309 | 72.079 | -0.314 | -0.669 | -0.002 | Yes |
| `medium_ms` | 360.319 | 359.317 | -0.276 | -0.561 | -0.041 | Yes |
| `long_ms` | 786.093 | 784.452 | -0.208 | -0.460 | -0.038 | Yes |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `speed-bench.tsv`.

| Metric | S mean (requests/s) | D mean (requests/s) | D against S (%) | 95 % interval, low end (%) | 95 % interval, high end (%) | In ±2 % |
|---|---:|---:|---:|---:|---:|---|
| `rps8` | 4.218 | 4.220 | +0.051 | -0.002 | +0.131 | Yes |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `speed-bench.tsv`.

The next table gives each run in time sequence. The load is the 1-minute load average of the host before and after the run.

| Start (UTC) | Run | Arm | `lat_ms` (ms) | `short_ms` (ms) | `medium_ms` (ms) | `long_ms` (ms) | `rps8` (requests/s) | Load before | Load after |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 22:07:26 | `b1-D` | D | 238.9 | 72.9 | 360.7 | 787.1 | 4.193 | 14.14 | 12.57 |
| 22:09:23 | `b1-S` | S | 241.7 | 73.6 | 364.1 | 793.4 | 4.183 | 12.57 | 19.53 |
| 22:11:16 | `b2-S` | S | 242.2 | 74.0 | 365.0 | 792.1 | 4.184 | 19.53 | 17.05 |
| 22:13:09 | `b2-D` | D | 241.6 | 73.8 | 364.2 | 791.3 | 4.185 | 17.05 | 15.04 |
| 22:15:02 | `b3-S` | S | 239.2 | 71.9 | 359.4 | 784.5 | 4.234 | 17.84 | 6.81 |
| 22:16:53 | `b3-D` | D | 237.4 | 71.5 | 357.8 | 782.5 | 4.234 | 6.81 | 2.91 |
| 22:18:42 | `b4-D` | D | 237.2 | 71.5 | 357.5 | 781.8 | 4.235 | 2.84 | 2.28 |
| 22:20:31 | `b4-S` | S | 237.6 | 71.5 | 357.9 | 782.3 | 4.236 | 2.28 | 2.13 |
| 22:22:20 | `b5-S` | S | 237.3 | 71.4 | 357.9 | 782.0 | 4.234 | 2.13 | 2.67 |
| 22:24:10 | `b5-D` | D | 237.3 | 71.4 | 357.7 | 782.2 | 4.235 | 2.67 | 2.58 |
| 22:25:59 | `b6-S` | S | 237.5 | 71.3 | 357.7 | 782.4 | 4.235 | 2.58 | 2.63 |
| 22:27:48 | `b6-D` | D | 237.3 | 71.4 | 358.0 | 781.8 | 4.237 | 2.63 | 2.74 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `runs.tsv`. The source table has more decimal places.

## Speed test with `llama bench`

The test has 4 blocks with seed 20261031. Each block has one S run and one D run. Each run measures each prompt length 5 times. This is the command. In the command, `mix8` is the path of the model file.

```
llama bench -m mix8 -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json
```

The load average of the host was 2.08 to 2.62 during these runs. A speed in tokens/s that is larger is faster.

| Prompt length (tokens) | S mean (tokens/s) | D mean (tokens/s) | D against S (%) | 95 % interval, low end (%) | 95 % interval, high end (%) | In ±2 % |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 2833.748 | 2840.474 | +0.238 | -0.113 | +0.543 | Yes |
| 256 | 4231.033 | 4231.444 | +0.010 | -0.066 | +0.085 | Yes |
| 3,000 | 4615.892 | 4618.268 | +0.052 | -0.090 | +0.194 | Yes |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `speed-llama-bench.tsv`. The load comes from the table `runs.tsv`.

## Host load and its effect

Different work used the host in blocks 1 to 3 of the benchmark. The load changed fast in these blocks. The next table gives the ratio of `lat_ms` of each block. It also gives the load of the S run and of the D run.

| Block | First run | S load before | S load after | D load before | D load after | D against S, `lat_ms` (%) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | D | 12.57 | 19.53 | 14.14 | 12.57 | -1.162 |
| 2 | S | 19.53 | 17.05 | 17.05 | 15.04 | -0.261 |
| 3 | S | 17.84 | 6.81 | 6.81 | 2.91 | -0.756 |
| 4 | D | 2.28 | 2.13 | 2.84 | 2.28 | -0.148 |
| 5 | S | 2.13 | 2.67 | 2.67 | 2.58 | -0.004 |
| 6 | S | 2.58 | 2.63 | 2.63 | 2.74 | -0.049 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `speed-bench-blocks.tsv`. The ratios use the values of `lat_ms` with all decimal places, not the values of the table of runs above.

### Facts

- The load was 12.57 to 19.53 in blocks 1 and 2. It fell from 17.84 to 2.91 in block 3. It was 2.13 to 2.84 in blocks 4 to 6.
- In each of blocks 1 to 3, the load after the S run is higher than the load after the D run.
- The mean ratio of blocks 1 to 3 is -0.726 %. The mean ratio of blocks 4 to 6 is -0.067 %. We calculated the two means from the table above.
- The three ratios of blocks 4 to 6 are -0.148 %, -0.004 % and -0.049 %.
- `lat_ms` was 241.6 to 242.2 ms in the runs `b1-S`, `b2-S` and `b2-D`. The load was 12.57 to 19.53.
- `lat_ms` was 237.2 to 237.6 ms in blocks 4 to 6. The load was 2.13 to 2.84.
- The difference between these two groups is about 2 %.
- The runs `b1-D` and `b3-S` have `lat_ms` of 238.9 ms and 239.2 ms. Their load was 12.57 to 14.14 and 6.81 to 17.84.

### Meaning of the facts

- The two builds have equal speed in the equivalence margin of ±2 %. The test shows this.
- The 95 % interval of `lat_ms` does not include 0. This is because of blocks 1 to 3, where the load was high or changed fast. [INFERENCE: we did not calculate the interval for blocks 4 to 6 only.]
- A high load on the host CPU makes each request about 2 % slower. [INFERENCE: no test changed only the load.]
- Then part of each request is work of the host CPU. [INFERENCE: we did not measure the work of the host CPU.]
- The noise between two runs on a host with a low load is about 0.1 %. [INFERENCE: the data has only 3 blocks with a low load.]

### Changes to the method

1. Record the host load for each run. The harness does this at this time.
2. Use a host with a low load for each run, or use more blocks.
3. Compare a block only when the load is the same before and after the two runs. [INFERENCE: this rule is a proposal and has no test.]
4. Keep the equivalence margin of ±2 %.

## Answers

Each run of the private benchmark wrote its answers with the flag `--write-reference`. The tool `harness/analysis/answers_compare.py` compared the answers of the 12 runs.

| Test | Result |
|---|---|
| Test of arm S | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Test of arm D | The 6 runs give the same answers bit for bit. The largest probability difference is 0.0. |
| Test of two arms | The answers of S and D are the same bit for bit. The largest probability difference is 0.0. No decision has a different top answer. |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `answers-compare-bench.json`.

All 12 runs give these quality numbers.

- `correct` is 94 of 100 decisions.
- `agree` is 97 of 100 decisions. In these decisions, the top answer is the top answer of the `Q8_0` reference.
- `mean_dp`, the mean probability difference from the `Q8_0` reference, is 0.013322.
- All guards are true.
- The number of errors is 0.

## KL divergence and the KL floor

We use `llama perplexity` to measure the KL divergence. The run on arm S writes a base file with the flag `--kl-divergence-base`. A run with the flag `--kl-divergence` compares its probabilities with the base file. The test uses 20 chunks of 512 tokens, the flags `-b 2048 -ub 2048` and the flag `-fa auto`.

The base file keeps the log-probabilities of each token as 16-bit integers, with one scale for each token. [INFERENCE: we read this in the function `log_softmax` of `tools/perplexity/perplexity.cpp` in llama.cpp at b12004.] A run of the same build against this base file thus gives a KL divergence that is small and not 0. The rows of the second run of S in the next table show this. The largest KL divergence is not 0, and the same top token is less than 100 %. This is the KL floor.

All 4 runs in the table compare with the base file that arm S wrote.

| Text | Arm | Mean KL divergence | Largest KL divergence | 99.9 % percentile of the KL divergence | Same top token (%) | Error of the same top token (%) | Perplexity of the run | Perplexity of the base file |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| WikiText-2 test | S, second run | 0.000000 | 0.000058 | 0.000038 | 99.980 | 0.020 | 14.894258 | 14.894249 |
| WikiText-2 test | D | 0.000000 | 0.000058 | 0.000038 | 99.980 | 0.020 | 14.894258 | 14.894249 |
| Benchmark states (the 16 `medium` and `long` states) | S, second run | 0.000000 | 0.000055 | 0.000043 | 99.922 | 0.039 | 9.177134 | 9.177135 |
| Benchmark states (the 16 `medium` and `long` states) | D | 0.000000 | 0.000055 | 0.000043 | 99.922 | 0.039 | 9.177134 | 9.177135 |

Source: [README of the data set](../../data/2026-10-30-window1/README.md) and the table `kld.tsv`.

D gives the same numbers as the second run of S in all decimal places of the table. The result is in the three limits of the prediction entry.

| Limit | Value | In the limit |
|---|---|---|
| Mean KL divergence | 1e-5 or less | Yes, on the two texts |
| Largest KL divergence | 1e-3 or less | Yes, on the two texts |
| Same top token | 99.9 % or more | Yes, on the two texts |

The KL floor of the same top token on the benchmark states is 99.922 %. The limit is 99.9 %. The difference is 0.022 %. The error of the KL floor is 0.039 %. The limit is too near the KL floor.

The next prediction entries must compare with the KL floor of the same text. They must not use one number of 99.9 % for all texts.

## Prediction against result

The prediction entry declares these items before the measurement. The speed of the two arms is the same, and the point estimate is in ±0.5 %. The answers are the same bit for bit, if the `sm_86` SASS of the two builds is the same. If the SASS is different, the answers can have small differences in the tolerances of the prediction entry. The limit of the speed is a 95 % interval in ±2 % for `lat_ms` and for each prompt length of `llama bench`.

| Item | Prediction | Measurement | Same as the prediction |
|---|---|---|---|
| Speed, point estimate of `lat_ms` | In ±0.5 % | -0.40 % | Yes |
| Speed, limit | The 95 % interval is in ±2 % for `lat_ms` and for each prompt length of `llama bench` | The 95 % interval is in ±2 % for `lat_ms` and for the 3 prompt lengths | Yes |
| Answers, test of one arm | The same bit for bit | The same bit for bit | Yes |
| Answers, test of two arms | The same bit for bit, if the SASS is the same | The same bit for bit | Yes |
| SASS of `sm_86` | The answers are the same bit for bit, if the SASS is the same | The SASS is the same for all 6,754 kernel names. We checked this before the measurement window. | Yes |
| Tolerances for different answers | The tolerances apply only if the answers are different | Not used, because the answers are the same. The three limits of the KL divergence are true. | Yes |
| Differences of the two builds | Only the list of GPU architectures is different | Five flags are different: `-DCMAKE_CUDA_ARCHITECTURES`, `-DLLAMA_BUILD_TESTS`, `-DCMAKE_BUILD_RPATH_USE_ORIGIN`, `-DLLAMA_USE_PREBUILT_UI` and `-DGGML_CCACHE`. The SASS is the same. | Not fully |

Source: the [prediction entry](2026-10-30-phase1-equivalence-plan.md) and the tables in this entry.

## Result and next work

Build D is equivalent to the stock build in speed and in answers, in the limits of the prediction entry. Build D is the baseline build for all subsequent work. The next work is the [entry of phase 2](2026-10-30-phase2-correctness-baseline.md).
