# Phase 3: where the time of a request goes (result)

Written 2026-10-31, after the second measurement window (00:02 to 01:14 UTC).

## Task of this entry

This entry records where the time of one request goes on build D with the model mix8. It has no prediction entry. It gives measurements and numbers. It does not rank the targets. Phase 4 ranks the targets.

The entry has five parts: the census, the decomposition, the profiles of Nsight Systems, the profiles of Nsight Compute, and two leads. The [README of the census](../../data/2026-10-30-shape-census/README.md) and the [README of the profiles](../../data/2026-10-31-profiles/README.md) give the raw files, the tables, and the conditions. The profile files of Nsight Systems and Nsight Compute are not in the repository. The README of the profiles gives the cause.

A request has one state and one to three questions. The name of a request type shows the length of the state and the number of questions. The type `short1q` has one question and `short2q` has two questions. The types `medium3q` and `long3q` have three questions. The runs used the 48 cases of the benchmark: 12 `short1q`, 20 `short2q`, 8 `medium3q`, and 8 `long3q`. The server had the production flags `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified`.

## Result in short

- The four request types use 46.8, 89.8, 368, and 807 ms of wall time. They have 1, 3, 4, and 5 passes. A pass reads 3.34 GB of weights, or 2.62 GB when it has no output head.
- The GPU time of a `short1q` request is between 35.9 and 45.5 ms. The weight-read floor of its one pass is 7.44 ms. The GPU span of 45.5 ms is 6.1 times the floor. The measurement of 36.7 ms before phase 3 is in this range.
- The kernels of the `short1q` pass use 42.4 ms. The idle time of the GPU in the pass is 1.0 ms. The kernels are slower than the floors. The launches are not the cause.
- The GPU busy time is 90 to 94 % of the wall time of a request. The GPU limits all four request types. In the large types, 86 to 89 % of the submit time is time in sync calls.
- The first kernel starts 3 to 33 ms after the start of the request. The uploads run at 6.7 GB/s. They use 0.7 to 21.3 ms of a request.
- MMQ is 60 to 70 % of the kernel time of a pass with more than 8 tokens. At 1,173 tokens or more, MMQ is at 32.5 to 39.2 % of the int8 peak. Each SM runs one thread block.
- MMVQ is at 90 to 95 % of the DRAM limit.
- The kernel `gdn_single` runs 32 thread blocks on 38 SMs. It is at 20 % of the DRAM limit.
- For the lead #28495, the KV window changes the time of the flash-attention kernel by 87 % with `--kv-unified`. It changes the GPU busy time of the request by 3.4 %.
- For the lead #30779, no request has more passes than the census gives. The 119 requests with one more use of `llama_decode` are not slower.

## Census: the work of one pass

The census gives the work of each pass in numbers. It uses the shapes of the model and the rules of the code. It has no time measurement.

The model has 32 layers: 24 Gated DeltaNet layers and 8 attention layers. A pass with the output head reads 3.34 GB of weights. A pass without the output head reads 2.62 GB. The time of the two reads at 448 GB/s is 7.44 ms and 5.85 ms. The table `census-passes.tsv` of the data set has these values.

The column `Passes` gives the passes of the case. `P` is a prefix pass and the number after it is the number of tokens. `S3x43` is a suffix pass of 3 sequences with 43 tokens each. `F` is a pass of one full prompt.

| Request type | Case | Passes | Passes in the request | Tokens in all passes | Weights read (GB) | Weight-read floor (ms) | Floor of weights or int8 (ms) | Roofline floor (ms) | MMQ launches | MMVQ launches |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short-1q` | `check03` | `F158` | 1 | 158 | 3.335 | 7.444 | 7.644 | 19.073 | 245 | 3 |
| `short-2q` | `triage09` | `P86+S2x51+S1x69` | 3 | 257 | 9.290 | 20.737 | 20.737 | 39.760 | 735 | 6 |
| `medium-3q` | `medium01-memory` | `P1288+S3x43+S2x4+S1x26` | 4 | 1451 | 12.625 | 28.181 | 84.569 | 170.954 | 735 | 254 |
| `long-3q` | `long00-storage` | `P2048+P1173+S3x43+S2x4+S1x26` | 5 | 3384 | 15.245 | 34.029 | 177.970 | 383.300 | 980 | 254 |

Source: [README of the census](../../data/2026-10-30-shape-census/README.md) and the table `request-floors.tsv`.

The census gives the launches of each kernel class in all 72 passes of the three profiles of Nsight Systems of 2026-10-30. It also gives each launch of MMQ and MMVQ in the 37 passes that have a launch list. It gives the pass sequence of 20 of 20 requests.

The census gives the KV window correctly for 33 of 72 passes. The KV window in the profiles is wider than the KV window of the census in the other 39 passes. The lead #28495 below uses this fact. The [README of the census](../../data/2026-10-30-shape-census/README.md) shows the tables of this check.

## Decomposition of the request time

A build of the branch `dmk/ablate-timing` has the flag `DMK_TIMING`. It writes the time of each step of a request to the server log. The client writes the start and the end of the request. The two clocks are the same.

The run has 348 requests and no error. The build with `DMK_TIMING` is 0.1 to 0.4 % faster than the reference build. The time code does not make the requests slower.

| Request type | Requests in each arm | Build with `DMK_TIMING` (ms) | Reference build (ms) | Difference (%) |
|---|---:|---:|---:|---:|
| `short1q` | 84 | 47.20 | 47.36 | -0.35 |
| `short2q` | 100 | 87.79 | 88.15 | -0.41 |
| `medium3q` | 76 | 366.58 | 367.18 | -0.17 |
| `long3q` | 76 | 808.53 | 809.37 | -0.10 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/instrumentation-cost.tsv`.

The next table divides the wall time of the phase `unique` (30 requests of each type). The sum of the four steps is the wall time. The step before submit has HTTP, the prompt, the start of the task, and the host work to prepare the passes. The submit step has the host time to send the passes to the GPU.

The step other steps has the host work between the passes and after the last pass. The GPU span is the time from the start of the first pass to the end of the last GPU wait. It is the maximum of the GPU time. The pass floor is the sum of the larger floor (weights or int8) of the passes. The last column is the GPU span divided by the pass floor.

| Request type | Requests | Wall time (ms) | Before submit (ms) | Submit (ms) | GPU wait (ms) | Other steps (ms) | GPU span (ms) | Pass floor (ms) | Roofline floor (ms) | GPU span divided by pass floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | 30 | 46.8 | 1.9 | 8.7 | 35.9 | 0.4 | 45.5 | 7.7 | 19.2 | 5.90 |
| `short2q` | 30 | 89.8 | 4.8 | 63.3 | 20.9 | 0.9 | 87.6 | 20.7 | 39.8 | 4.22 |
| `medium3q` | 30 | 368.4 | 12.5 | 329.2 | 25.2 | 1.5 | 362.3 | 84.6 | 169.1 | 4.28 |
| `long3q` | 30 | 807.3 | 26.4 | 751.4 | 26.2 | 3.2 | 796.4 | 177.9 | 371.5 | 4.48 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/request-time-decomposition.tsv`.

### The request `short1q` against the weight-read floor and the time before phase 3

The next table gives the steps of a `short1q` request.

| Step | Mean (ms) | Sum to this step (ms) | Part of the wall time (%) |
|---|---:|---:|---:|
| Weight-read floor of one pass | 7.44 | 7.44 | 15.9 |
| HTTP and client time | 0.44 | 7.89 | 0.9 |
| Prompt build and use of the tokenizer | 0.59 | 8.48 | 1.3 |
| Start of the task and start of decode | 0.24 | 8.72 | 0.5 |
| Graph build and submit | 9.55 | 18.27 | 20.4 |
| End of the request after the last wait | 0.08 | 18.36 | 0.2 |
| GPU wait above the floor | 28.49 | 46.85 | 60.8 |
| Wall time | 46.85 | 46.85 | 100.0 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/short1q-waterfall.tsv`.

| Quantity | Unit | Value |
|---|---|---:|
| weight-read floor of a pass with the output head | ms | 7.44 |
| int8 floor of the pass | ms | 7.64 |
| roofline floor of the pass | ms | 19.07 |
| measurement of one short pass before phase 3 | ms | 36.70 |
| wall time, mean | ms | 46.85 |
| GPU span, mean (maximum of the GPU time) | ms | 45.49 |
| GPU wait, mean (minimum of the GPU time) | ms | 35.93 |
| GPU span divided by the weight-read floor | ratio | 6.11 |
| wall time divided by the weight-read floor | ratio | 6.29 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/short1q-reference.tsv`.

The measurement of 36.7 ms before phase 3 was one short pass with no profile. It is between the GPU wait of 35.9 ms and the GPU span of 45.5 ms of this run. The wall time of this run is 46.8 ms. It is 10.1 ms more than 36.7 ms. HTTP, the prompt, the start of the task, and the end of the request use 1.4 ms. They are not in the GPU span.

The GPU span is 38.0 ms more than the weight-read floor. The decomposition cannot divide this time into the time of the kernels and the time of the launches. The data of Nsight Systems below divide it. The kernels use 42.4 ms and the GPU has 1.0 ms of idle time in the pass. The time of the kernels is 2.2 times the roofline floor of 19.07 ms.

### Correction of the text `host-bound`

The script of the decomposition wrote the text `host-bound` for `medium3q` and `long3q`. It gave the launch loop as the cause of the GPU span. This text is incorrect. The submit step is 91 % of the GPU span for `medium3q` and 94 % for `long3q`.

The GPU busy time is 92.8 % and 93.7 % of the wall time of these types. The decomposition cannot see the GPU. Nsight Systems shows that the decode thread waits in sync calls for most of the submit time. The part [Where the CPU waits](#where-the-cpu-waits) shows the numbers.

### The warm path

A second request with the same case is faster than the first request for `short1q`, `medium3q`, and `long3q`. The tokens and the passes are the same. The change is in the submit time. For `short1q` the GPU wait is longer by the same time.

| Request type | Case | Pairs | First request (ms) | Second request (ms) | Change of wall time (ms) | Change of submit (ms) | Change of GPU wait (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `short1q` | `check03` | 8 | 46.6 | 45.9 | -0.68 | -4.46 | 4.82 |
| `short1q` | `check04` | 7 | 50.3 | 49.8 | -0.56 | -5.25 | 5.72 |
| `short2q` | `triage09` | 8 | 89.8 | 90.3 | 0.47 | 0.49 | -0.05 |
| `short2q` | `triage10` | 7 | 87.7 | 88.1 | 0.40 | 0.43 | 0.02 |
| `medium3q` | `medium01-memory` | 8 | 371.6 | 364.6 | -7.05 | -6.64 | -0.34 |
| `medium3q` | `medium02-network` | 7 | 375.4 | 368.2 | -7.19 | -6.48 | -0.45 |
| `long3q` | `long00-storage` | 8 | 821.7 | 781.6 | -40.18 | -37.30 | -1.03 |
| `long3q` | `long01-memory` | 7 | 853.2 | 811.3 | -41.91 | -39.65 | -0.68 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/warm-path.tsv`.

[INFERENCE] The data do not show the cause of the faster second request.

## Nsight Systems: the GPU is active

The profiles show the full server for the four request types (the run `all48` has the 48 cases). The command used `nsys profile --trace=cuda,osrt,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`. Nsight Systems adds 0.3 to 1.6 % to the time of a request. The table compares pairs of the same case with and without Nsight Systems.

| Request type | Pairs | Without Nsight Systems (ms) | With Nsight Systems (ms) | Difference (ms) | Difference (%) | 95 % interval, plus or minus (ms) |
|---|---:|---:|---:|---:|---:|---:|
| `short1q` | 22 | 46.7 | 47.1 | 0.45 | 0.95 | 0.17 |
| `short2q` | 30 | 84.8 | 86.2 | 1.35 | 1.60 | 0.33 |
| `medium3q` | 18 | 358.5 | 360.5 | 1.97 | 0.55 | 2.59 |
| `long3q` | 18 | 795.8 | 798.5 | 2.68 | 0.34 | 3.93 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-overhead.tsv`.

### Where the wall time goes

| Request type | Requests | Passes in a request | Wall time (ms) | Time before the first kernel (ms) | GPU busy time (ms) | Idle time in the passes (ms) | Idle time between passes (ms) | Time after the last kernel (ms) | GPU busy time (% of the wall time) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | 12 | 1 | 47.1 | 3.2 | 42.4 | 1.0 | 0.0 | 0.5 | 90.0 |
| `short2q` | 20 | 3 | 87.1 | 3.5 | 78.6 | 2.8 | 1.6 | 0.7 | 90.2 |
| `medium3q` | 8 | 4 | 359.9 | 17.3 | 333.8 | 4.8 | 3.2 | 0.7 | 92.8 |
| `long3q` | 8 | 5 | 795.1 | 33.3 | 745.0 | 4.5 | 11.5 | 0.8 | 93.7 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-request-time.tsv`.

The GPU busy time is 90.0, 90.2, 92.8, and 93.7 % of the wall time. The idle time in the passes is 0.6 to 3.2 % of the wall time. The largest part of the idle time is the request-start gap.

### The gap before a pass

| Request type | Position of the gap | Passes | Gap, mean (ms) | GPU done to the end of the sync call (ms) | No CUDA call (ms) | No CUDA call, system wait (ms) | No CUDA call, CPU work (ms) | CUDA calls (ms) | First launch to first kernel (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | `request-start` | 12 | 3.85 | 0.16 | 2.84 | 1.17 | 1.68 | 0.82 | 0.024 |
| `short2q` | `request-start` | 20 | 4.29 | 0.16 | 3.60 | 1.82 | 1.77 | 0.51 | 0.023 |
| `short2q` | `prefix-small->suffix` | 20 | 0.84 | 0.01 | 0.22 | 0.00 | 0.22 | 0.59 | 0.021 |
| `short2q` | `suffix->suffix` | 20 | 0.74 | 0.17 | 0.22 | 0.00 | 0.22 | 0.32 | 0.021 |
| `medium3q` | `request-start` | 8 | 18.08 | 0.16 | 11.05 | 4.57 | 6.49 | 6.83 | 0.026 |
| `medium3q` | `prefix-small->suffix` | 8 | 0.87 | 0.02 | 0.25 | 0.00 | 0.25 | 0.58 | 0.026 |
| `medium3q` | `suffix->suffix` | 16 | 1.18 | 0.17 | 0.68 | 0.01 | 0.67 | 0.31 | 0.021 |
| `long3q` | `request-start` | 8 | 34.21 | 0.17 | 21.02 | 8.18 | 12.84 | 13.00 | 0.027 |
| `long3q` | `prefix-large->prefix-small` | 8 | 7.78 | 0.02 | 0.27 | 0.00 | 0.27 | 7.46 | 0.026 |
| `long3q` | `prefix-small->suffix` | 8 | 0.92 | 0.02 | 0.25 | 0.00 | 0.25 | 0.62 | 0.026 |
| `long3q` | `suffix->suffix` | 16 | 1.38 | 0.17 | 0.83 | 0.01 | 0.82 | 0.37 | 0.019 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-gap-before-pass.tsv`.

The request-start gap of a `long3q` request is 34.2 ms. CUDA calls use 13.0 ms of it. The other 21.0 ms have no CUDA call. In these 21.0 ms, system waits use 8.2 ms and CPU work uses 12.8 ms.

The CUDA calls at the start of the request are the uploads. The gap between the large prefix pass and the small prefix pass is 7.8 ms. The uploads use 7.5 ms of it.

### The uploads

| Request type | Requests | Uploads in a request | Data uploaded in a request (MB) | GPU time of the uploads (ms) | Rate (GB/s) |
|---|---:|---:|---:|---:|---:|
| `short1q` | 12 | 10.0 | 4.7 | 0.71 | 6.65 |
| `short2q` | 20 | 29.0 | 7.4 | 1.12 | 6.57 |
| `medium3q` | 8 | 39.0 | 50.9 | 7.60 | 6.70 |
| `long3q` | 8 | 48.0 | 142.7 | 21.26 | 6.72 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-uploads.tsv`.

The uploads run at 6.7 GB/s. [INFERENCE] This rate fits PCIe 3.0 with a link width of 8. The data set does not record the width of the link.

### Where the CPU waits

| Request type | Requests | Launches in a request | Time in the launches (ms) | Sync calls in a request | Time in the sync calls (ms) | Time in all CUDA calls (ms) | Full command buffers in a request | Time of the full command buffers (ms) | Largest launch queue |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | 12 | 1296 | 5.5 | 24.0 | 35.7 | 41.2 | 13.7 | 0.5 | 1020 |
| `short2q` | 20 | 3829 | 15.1 | 68.0 | 56.4 | 71.7 | 0.0 | 0.0 | 963 |
| `medium3q` | 8 | 4883 | 46.9 | 91.6 | 283.9 | 331.2 | 117.8 | 27.8 | 1020 |
| `long3q` | 8 | 6014 | 85.0 | 112.6 | 662.1 | 747.5 | 184.8 | 61.4 | 1020 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-host-cuda-calls.tsv`.

The table gives the CUDA calls of the decode thread as the mean of one request. For `medium3q` and `long3q`, the time of all CUDA calls (331 ms and 748 ms) is near the submit time of the decomposition (329 ms and 751 ms). For these two types, 86 to 89 % of this time is in sync calls. A `medium3q` request has 92 sync calls and a `long3q` request has 113. This is about 23 for each pass. The launches use 47 ms and 85 ms of a request.

The CPU thus waits in the sync calls while the GPU works. The launch queue is full (1,020 launches) in all types but `short2q`. The full command buffers use 28 ms of a `medium3q` request and 61 ms of a `long3q` request. [INFERENCE] The data do not show the source of the sync calls.

### Kernel time of each pass type

| Pass type | Tokens in the pass | Passes | Launches in a pass | All kernels (ms) | `MMQ` (ms) | `GDN` (ms) | `glu` (ms) | `FA` (ms) | `quantize` (ms) | `copy` (ms) | `rms_norm` (ms) | `ssm_conv` (ms) | `elementwise` (ms) | `MMVQ` (ms) | Other kernels (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefix-large` | 2048 | 18 | 1166 | 416.4 | 263.5 | 32.1 | 24.8 | 18.0 | 23.0 | 16.4 | 16.6 | 10.8 | 9.9 | 0.0 | 1.2 |
| `prefix-small` | 65-128 | 30 | 1285 | 25.0 | 17.4 | 1.7 | 1.0 | 0.4 | 1.0 | 1.1 | 0.8 | 0.7 | 0.7 | 0.0 | 0.1 |
| `prefix-small` | 257-1024 | 6 | 1173 | 218.4 | 136.7 | 14.7 | 11.2 | 19.4 | 10.7 | 7.8 | 7.6 | 5.2 | 4.4 | 0.0 | 0.6 |
| `prefix-small` | 1025-2047 | 30 | 1168 | 268.8 | 172.2 | 19.9 | 15.4 | 12.2 | 14.5 | 10.3 | 10.4 | 7.1 | 6.2 | 0.0 | 0.8 |
| `suffix` | N<=8 | 22 | 1065 | 14.0 | 0.0 | 0.4 | 0.2 | 0.4 | 0.4 | 1.3 | 0.3 | 0.1 | 0.2 | 10.5 | 0.0 |
| `suffix` | 9-32 | 40 | 1248 | 16.5 | 10.6 | 0.9 | 0.3 | 0.5 | 0.5 | 1.0 | 0.4 | 0.2 | 0.2 | 1.8 | 0.0 |
| `suffix` | 33-64 | 14 | 1296 | 21.0 | 12.6 | 1.8 | 0.5 | 0.9 | 0.7 | 1.1 | 0.6 | 0.4 | 0.6 | 1.8 | 0.0 |
| `suffix` | 65-128 | 81 | 1281 | 30.0 | 18.2 | 2.8 | 1.1 | 1.1 | 1.1 | 1.7 | 0.9 | 0.7 | 0.6 | 1.8 | 0.1 |
| `suffix` | 129-256 | 11 | 1296 | 45.7 | 28.2 | 4.4 | 1.6 | 2.1 | 1.5 | 2.7 | 1.2 | 1.2 | 0.8 | 1.8 | 0.1 |
| `single` | 129-256 | 22 | 1296 | 42.4 | 28.7 | 2.7 | 1.9 | 0.6 | 1.8 | 1.5 | 1.4 | 1.0 | 0.9 | 1.8 | 0.1 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/nsys-kernel-ms-per-pass.tsv`.

The table has all profiles together. A pass has 1,065 to 1,296 launches. MMQ is 60 to 70 % of the kernel time of a pass with more than 8 tokens. A pass with at most 8 tokens uses MMVQ for the weights. The output head (MMVQ, Q8_0) uses 1.64 ms in each pass that has outputs. The kernel `concat_non_cont` has 24 launches in each pass.

## Nsight Compute: how near each kernel is to its limit

The runs used `ncu` with the server as the target. Each run profiles a group of launches that starts after a number of other launches. The list of a run gives the kernels that the run must profile. The table has 7 aligned runs, 1 shifted run, and 5 misaligned runs.

| Run | Launches in the list | Launches that agree with the list | Move of the start (launches) | Result |
|---|---:|---:|---:|---|
| `smoke-a` | 6 | 2 | 0 | `MISALIGNED` |
| `smoke-b` | 6 | 6 | 0 | `ALIGNED` |
| `core-long-p0` | 13 | 1 | 1 | `MISALIGNED` |
| `core-long-p1` | 12 | 12 | 0 | `ALIGNED` |
| `core-long-p2` | 17 | 16 | -16 | `SHIFTED` |
| `core-mv` | 11 | 0 | 0 | `MISALIGNED` |
| `core-short-p0` | 14 | 14 | 0 | `ALIGNED` |
| `core-short-p1` | 14 | 14 | 0 | `ALIGNED` |
| `extra-long-att` | 13 | 1 | 3 | `MISALIGNED` |
| `extra-long-ew` | 9 | 0 | 0 | `MISALIGNED` |
| `extra-long-full` | 8 | 8 | 0 | `ALIGNED` |
| `extra-long-p4` | 14 | 14 | 0 | `ALIGNED` |
| `extra-med-p0` | 10 | 10 | 0 | `ALIGNED` |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-run-alignment.tsv`.

The five misaligned runs have complete profiles. The kernel names give the kernel classes of their kernels. These kernel classes have no census floor. [INFERENCE] Kernels of the start of the server before the first request cause the misaligned runs. The run `smoke-a` shows such kernels.

The run `core-long-p2` is a shifted run. Its profiled launches start 16 launches before the first launch of its list. Its kernel classes are correct.

### MMQ with many tokens

| Tokens | Operation | Weight type | Launches in a pass | Time of the kernel (µs) | Part of the kernel time of the pass (%) | Int8 peak (%) | IMMA pipe active (%) | DRAM, % of 422 GB/s | Largest stall reasons |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 1173 | `attn_qkv` | `Q8_0` | 24 | 939.4 | 8.8 | 35.1 | 37.9 | 31.0 | `math_pipe_throttle 27%; mio_throttle 16%; long_scoreboard 16%` |
| 1173 | `attn_gate(z)` | `Q8_0` | 24 | 498.1 | 4.7 | 33.1 | 37.8 | 30.4 | `math_pipe_throttle 27%; long_scoreboard 16%; mio_throttle 16%` |
| 1173 | `ssm_out` | `Q8_0` | 24 | 506.4 | 4.7 | 32.5 | 35.1 | 85.2 | `math_pipe_throttle 23%; long_scoreboard 20%; lg_throttle 17%` |
| 1173 | `ffn_up` | `Q4_0` | 31 | 1017.4 | 12.3 | 36.4 | 39.4 | 28.8 | `math_pipe_throttle 29%; wait 19%; mio_throttle 16%` |
| 1173 | `ffn_gate` | `Q4_0` | 31 | 1013.5 | 12.2 | 36.6 | 39.4 | 28.9 | `math_pipe_throttle 29%; wait 19%; mio_throttle 16%` |
| 1173 | `ffn_down` | `Q4_0` | 31 | 1019.2 | 12.3 | 36.4 | 39.3 | 78.3 | `math_pipe_throttle 29%; wait 18%; long_scoreboard 17%` |
| 1288 | `attn_qkv` | `Q8_0` | 24 | 1041.6 | 9.2 | 34.7 | 38.4 | 32.2 | `math_pipe_throttle 27%; mio_throttle 16%; wait 15%` |
| 1288 | `attn_gate(z)` | `Q8_0` | 24 | 553.1 | 4.9 | 32.7 | 38.0 | 31.4 | `math_pipe_throttle 27%; mio_throttle 16%; long_scoreboard 16%` |
| 1288 | `ssm_out` | `Q8_0` | 24 | 528.1 | 4.7 | 34.3 | 38.5 | 28.2 | `math_pipe_throttle 27%; long_scoreboard 15%; wait 15%` |
| 1288 | `ffn_up` | `Q4_0` | 31 | 1126.0 | 12.8 | 36.2 | 39.5 | 30.0 | `math_pipe_throttle 30%; wait 19%; mio_throttle 16%` |
| 1288 | `ffn_gate` | `Q4_0` | 31 | 1119.8 | 12.8 | 36.4 | 39.6 | 30.1 | `math_pipe_throttle 30%; wait 19%; mio_throttle 16%` |
| 1288 | `ffn_down` | `Q4_0` | 31 | 1121.9 | 12.8 | 36.3 | 40.8 | 23.2 | `math_pipe_throttle 30%; wait 19%; mio_throttle 14%` |
| 2048 | `attn_qkv` | `Q8_0` | 24 | 1495.2 | 8.6 | 38.5 | 38.3 | 40.7 | `math_pipe_throttle 27%; mio_throttle 16%; wait 15%` |
| 2048 | `attn_gate(z)` | `Q8_0` | 24 | 776.3 | 4.5 | 37.1 | 38.2 | 40.3 | `math_pipe_throttle 27%; mio_throttle 16%; long_scoreboard 16%` |
| 2048 | `ssm_out` | `Q8_0` | 24 | 787.4 | 4.6 | 36.5 | 38.5 | 36.5 | `math_pipe_throttle 27%; wait 15%; mio_throttle 15%` |
| 2048 | `ffn_up` | `Q4_0` | 31 | 1649.5 | 12.3 | 39.2 | 39.2 | 39.6 | `math_pipe_throttle 29%; wait 19%; mio_throttle 16%` |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-mmq-large-n.tsv`.

With 1,173 to 2,048 tokens, the MMQ kernels are at 32.5 to 39.2 % of the int8 peak. The scripts use 149.3 TOP/s as the int8 peak and 422 GB/s as the DRAM limit. The IMMA pipe is active for 35 to 41 % of the time. Two rows have a high DRAM value (85 % and 78 %). [INFERENCE] The data do not show the cause. The kernels `ffn_up`, `ffn_gate`, `ffn_down`, and `attn_qkv` use the largest parts of the pass.

| Kernel | Grid | Block | Registers for each thread | Shared memory (B) | Thread blocks of an SM, limit of registers | Thread blocks of an SM, limit of shared memory | Thread blocks of an SM, limit of warps | Thread blocks of an SM, limit of blocks | Warps active (%) | Issue slots active (%) | IMMA pipe active (%) | SM clock (MHz) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Q8_0 attn_qkv` | `1024x1x1` | `32x8x1` | 254 | 57856 | 1 | 1 | 6 | 16 | 16.7 | 41.3 | 38.3 | 1946 |
| `Q8_0 ssm_beta` | `38x1x1` | `32x8x1` | 254 | 57856 | 1 | 1 | 6 | 16 | 17.1 | 34.2 | 27.9 | 1925 |
| `Q8_0 ssm_alpha` | `38x1x1` | `32x8x1` | 254 | 57856 | 1 | 1 | 6 | 16 | 16.9 | 34.1 | 27.7 | 1925 |
| `Q8_0 attn_gate(z)` | `512x1x1` | `32x8x1` | 254 | 57856 | 1 | 1 | 6 | 16 | 16.7 | 41.0 | 38.2 | 1941 |
| `Q8_0 ssm_out` | `320x1x1` | `32x8x1` | 254 | 57856 | 1 | 1 | 6 | 16 | 16.6 | 40.9 | 38.5 | 1942 |
| `Q4_0 ffn_up` | `1152x1x1` | `32x8x1` | 216 | 57856 | 1 | 1 | 6 | 16 | 16.6 | 46.4 | 39.2 | 1964 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-mmq-resources.tsv`.

Each MMQ kernel uses 216 or 254 registers for each thread and 57,856 B of shared memory. The registers and the shared memory limit an SM to one thread block of 8 warps. This is 16.7 % of the warp slots. No other thread block can run on the SM while the thread block waits for loads. The issue slots are active for 34 to 46 % of the time.

The loop of the MMQ kernel is in `ggml/src/ggml-cuda/mmq.cuh`, lines 954 to 987. It loads a tile from the memory of the GPU through the registers to the shared memory. Then it runs `__syncthreads`, `vec_dot`, and `__syncthreads`. The code has no `cp.async` and no second buffer. [INFERENCE] With one thread block on each SM, the loads are not at the same time as the matrix work.

### The instructions of one MMQ kernel

The report `extra-long-full` has the data of the SASS lines of one `ffn_up` kernel with the weight type Q4_0 and 2,048 tokens.

| Opcode | SASS lines | Stall samples (%) | Instructions run (10^6) | Instructions for each IMMA |
|---|---:|---:|---:|---:|
| `IMMA` | 256 | 23.4 | 11.80 | 1.00 |
| `I2FP` | 1024 | 12.5 | 47.19 | 4.00 |
| `LDS` | 408 | 11.6 | 16.07 | 1.36 |
| `LDG` | 149 | 10.1 | 6.64 | 0.56 |
| `PRMT` | 32 | 7.2 | 1.47 | 0.12 |
| `STS` | 147 | 7.0 | 6.64 | 0.56 |
| `FFMA` | 1024 | 6.6 | 47.19 | 4.00 |
| `FMUL` | 1024 | 6.5 | 47.19 | 4.00 |
| `LOP3` | 231 | 3.3 | 9.11 | 0.77 |
| `HADD2` | 264 | 2.5 | 12.17 | 1.03 |
| `all opcodes` | 5856 | 100.0 | 223.08 | 18.91 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-instruction-mix.tsv`.

The IMMA instructions have 23.4 % of the stall samples. The instructions `I2FP`, `FMUL`, and `FFMA` run four times for each IMMA instruction. Together they are 63 % of the instructions that the kernel runs. [INFERENCE] The line `sum += C.x[l]*dA*dB` in `mmq-vec-dot.cuh` (lines 265 to 273) can cause the number four. It runs for each result value of an IMMA instruction. The data show the numbers, not the cause.

### MMQ with a small number of tokens

| Tokens | Tile columns | Operation | Weight type | Time of the kernel (µs) | Limit | Of the limit (%) | IMMA pipe active (%) | Int8 peak (%) |
|---:|---:|---|---|---:|---|---:|---:|---:|
| 26 | 32 | `attn_qkv` | `Q8_0` | 68.4 | DRAM | 80.4 | 14.2 | 10.7 |
| 26 | 32 | `ssm_out` | `Q8_0` | 40.9 | DRAM | 66.8 | 12.6 | 8.9 |
| 26 | 32 | `ffn_up` | `Q4_0` | 57.3 | DRAM | 59.1 | 19.9 | 14.3 |
| 26 | 32 | `ffn_down` | `Q4_0` | 64.3 | DRAM | 50.9 | 16.6 | 12.8 |
| 86 | 96 | `attn_qkv` | `Q8_0` | 93.8 | DRAM | 64.0 | 30.0 | 25.8 |
| 86 | 96 | `ssm_out` | `Q8_0` | 56.6 | DRAM | 52.0 | 26.6 | 21.4 |
| 86 | 96 | `ffn_up` | `Q4_0` | 98.8 | DRAM | 40.0 | 33.0 | 27.5 |
| 86 | 96 | `ffn_down` | `Q4_0` | 92.9 | DRAM | 38.4 | 33.6 | 29.3 |
| 102 | 112 | `attn_qkv` | `Q8_0` | 102.4 | DRAM | 60.0 | 32.1 | 28.0 |
| 102 | 112 | `ssm_out` | `Q8_0` | 59.8 | DRAM | 50.2 | 29.9 | 24.0 |
| 102 | 112 | `ffn_up` | `Q4_0` | 110.5 | DRAM | 37.2 | 34.7 | 29.2 |
| 102 | 112 | `ffn_down` | `Q4_0` | 105.0 | DRAM | 34.7 | 35.1 | 30.7 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-mmq-small-n.tsv`.

With 26 to 102 tokens, the DRAM limit is the limit of the kernel. The kernels are at 35 to 80 % of the DRAM limit and at 9 to 31 % of the int8 peak.

### MMVQ, `gdn_single`, `concat_non_cont`, and small kernels

| Run | Type of the run | Kernel | Time of the kernel (µs) | Limit | Of the limit (%) | DRAM (GB/s) | DRAM, % of 422 GB/s | Largest stall reasons |
|---|---|---|---:|---|---:|---:|---:|---|
| `core-mv` | `MISALIGNED` | `mul_mat_vec_q grid=(4096,1,1)` | 59.1 | DRAM | 94.5 | 398.7 | 94.5 | `long_scoreboard 74%; barrier 17%; wait 3%` |
| `core-mv` | `MISALIGNED` | `mul_mat_vec_q grid=(16,1,1)` | 3.6 | launch | 28.0 | 28.5 | 6.8 | `long_scoreboard 48%; imc_miss 26%; barrier 11%` |
| `core-mv` | `MISALIGNED` | `mul_mat_vec_q grid=(2048,1,1)` | 32.5 | DRAM | 90.3 | 380.9 | 90.3 | `long_scoreboard 71%; barrier 17%; wait 3%` |
| `core-mv` | `MISALIGNED` | `mul_mat_vec_q grid=(1280,1,1)` | 35.3 | DRAM | 90.4 | 381.2 | 90.3 | `long_scoreboard 74%; lg_throttle 9%; barrier 8%` |
| `core-mv` | `MISALIGNED` | `mul_mat_vec_q grid=(4608,1,1)` | 37.9 | DRAM | 91.0 | 384.1 | 91.0 | `long_scoreboard 46%; barrier 35%; wait 6%` |
| `core-long-p2` | `SHIFTED` | `gdn_single grid=(32,1,1)` | 740.3 | DRAM | 19.9 | 84.0 | 19.9 | `math_pipe_throttle 36%; wait 26%; barrier 12%` |
| `core-long-p0` | `MISALIGNED` | `concat_non_cont grid=(8192,1,1)` | 18.3 | launch | 6.4 | 19.8 | 4.7 | `long_scoreboard 41%; wait 23%; short_scoreboard 14%` |
| `core-long-p0` | `MISALIGNED` | `ssm_conv_f32 grid=(1,64,1)` | 2.8 | launch | 36.0 | 112.9 | 26.8 | `long_scoreboard 52%; imc_miss 32%; wait 8%` |
| `extra-long-att` | `MISALIGNED` | `flash_attn_ext_f16 grid=(16,1,1)` | 11.2 | DRAM | 27.7 | 117.0 | 27.7 | `no_instruction 44%; long_scoreboard 13%; imc_miss 13%` |
| `extra-long-ew` | `MISALIGNED` | `rms_norm_f32 grid=(2,1,1)` | 3.3 | launch | 30.7 | 11.8 | 2.8 | `barrier 27%; long_scoreboard 24%; short_scoreboard 14%` |
| `extra-long-ew` | `MISALIGNED` | `rms_norm_f32 grid=(16,2,1)` | 2.1 | launch | 47.6 | 10.0 | 2.4 | `imc_miss 46%; long_scoreboard 29%; wait 9%` |
| `extra-long-ew` | `MISALIGNED` | `rms_norm_f32 grid=(32,2,1)` | 2.4 | launch | 42.2 | 16.9 | 4.0 | `long_scoreboard 48%; imc_miss 25%; wait 10%` |
| `extra-long-ew` | `MISALIGNED` | `unary_gated_op_kernel grid=(32,1,1)` | 2.1 | launch | 47.3 | 35.3 | 8.4 | `imc_miss 43%; long_scoreboard 35%; wait 10%` |
| `extra-long-ew` | `MISALIGNED` | `unary_gated_op_kernel grid=(72,1,1)` | 2.3 | launch | 42.8 | 72.6 | 17.2 | `long_scoreboard 39%; imc_miss 38%; wait 9%` |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/ncu-other-kernels.tsv`.

The MMVQ kernels with a large grid are at 90 to 95 % of the DRAM limit. They are at the limit. The kernel `gdn_single` has a grid of 32 thread blocks on 38 SMs. It runs for 740 µs and is at 20 % of the DRAM limit.

The rows of `core-mv`, `core-long-p0`, `core-long-p2`, `extra-long-att`, and `extra-long-ew` come from misaligned runs and from the shifted run. The profiled launches are not the launches of the list of the run. The kernel name and the grid name each kernel. These rows have no model floor. Their limit is the DRAM time of the bytes that Nsight Compute measured, or the launch floor. The column of the limit then shows the DRAM rate as a part of 422 GB/s.

The code of `gated-delta-net-mma.cu` (lines 403 to 416) selects the path for the full head in one case. In this case, 2 times the number of heads times the number of sequences is more than the number of SMs. Here the result is 64 and the number of SMs is 38.

The kernel `concat_non_cont` runs for 18 µs in each launch and is at 5 % of the DRAM limit. The small kernels of the other kernel classes run for 2 to 3 µs. The launch limits them.

## The two leads

### #28495: the KV window

In the lead, the flash-attention kernel reads a KV window that is wider than the cells of the sequence. The test sends one long request (phase T1). Then it sends the same request again on a different task slot (phases T2 to T4). The table compares the passes with the same position in the sequence of passes and with the same number of tokens.

| Cache mode | Phase | Passes compared | Flash attention (ms) | Flash attention, same passes of T1 (ms) | Ratio of flash attention | GPU busy time (ms) | GPU busy time, same passes of T1 (ms) | Ratio of the GPU busy time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `kv-unified` | `T2` | 3 of 5 | 49.78 | 22.16 | 2.246 | 710.6 | 679.2 | 1.046 |
| `kv-unified` | `T3` | 2 of 5 | 25.53 | 21.54 | 1.186 | 669.4 | 661.9 | 1.011 |
| `kv-unified` | `T4` | 5 of 5 | 53.24 | 24.34 | 2.187 | 764.2 | 731.3 | 1.045 |
| `no-kv-unified` | `T2` | 3 of 5 | 20.06 | 19.90 | 1.008 | 681.1 | 676.8 | 1.006 |
| `no-kv-unified` | `T3` | 2 of 5 | 19.47 | 19.31 | 1.008 | 663.7 | 659.6 | 1.006 |
| `no-kv-unified` | `T4` | 5 of 5 | 22.16 | 21.99 | 1.008 | 732.7 | 728.7 | 1.006 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/lead-28495-nsys-phases.tsv`.

With `--kv-unified`, the flash-attention time of the phases T2 to T4 is 1.19 to 2.25 times the time of T1. The mean change is 87 %. With `--no-kv-unified` it is 0.8 %. The GPU busy time changes by 3.4 % with `--kv-unified` and by 0.6 % with `--no-kv-unified`.

The runs with the `DMK_TIMING` build (no profiler) give an effect of 4.2 % on the time of the pass with `--kv-unified`. They give 0.2 % with `--no-kv-unified`. The threshold of the script is 4.0 %. The value of 4.2 % is above the threshold. The value of 0.2 % is below it. The [report of the lead](../../data/2026-10-31-profiles/reports/leads-kvwin/report.txt) has these values.

### #30779: a child slot with a lower number than the parent slot

In the lead, a child task on a task slot with a lower number than the parent slot gets more passes. The test compares the passes of all 372 requests of the runs with `DMK_TIMING` with two models. The census model for `--kv-unified` and the task slots of the request gives the passes. The fixed-order model has the parent task on the lowest task slot and all child tasks in one batch.

| Request type | Requests | Requests with the passes of the census model | Requests with more passes than the census model | Requests with a child slot with a lower number than the parent slot | Requests with more uses of `llama_decode` than the fixed-order model | Requests with other pass lengths than the fixed-order model |
|---|---:|---:|---:|---:|---:|---:|
| `short1q` | 89 | 89 | 0 | 0 | 0 | 0 |
| `short2q` | 105 | 105 | 0 | 0 | 0 | 0 |
| `medium3q` | 79 | 79 | 0 | 53 | 53 | 53 |
| `long3q` | 99 | 99 | 0 | 66 | 66 | 66 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/lead-30779-passes.tsv`.

| Request type | Child slot with a lower number than the parent slot | Requests | Request time, mean (ms) | Uses of `llama_decode`, mean |
|---|---|---:|---:|---:|
| `medium3q` | yes | 53 | 364.29 | 3.00 |
| `medium3q` | no | 26 | 370.39 | 2.00 |
| `long3q` | yes | 66 | 800.74 | 4.00 |
| `long3q` | no | 33 | 815.88 | 3.00 |

Source: [README of the profiles](../../data/2026-10-31-profiles/README.md) and the table `summary/lead-30779-cost.tsv`.

No request has more passes than the census model gives. 119 requests (53 `medium3q` and 66 `long3q`) have a child slot with a lower number than the parent slot. They make one more use of `llama_decode` and have other pass lengths than the fixed-order model. The census model gives their passes. They are not slower.

The request time is 6.1 ms (`medium3q`) and 15.1 ms (`long3q`) less. The roofline floor of the passes is 0.9 ms and 5.4 ms less. After this difference the request time is 5.2 ms and 9.7 ms less. The [report of the lead](../../data/2026-10-31-profiles/reports/leads-childpass/report.txt) has these values. The data do not show a time decrease from a change of the sequence of the tasks.

## Limits of this entry

- It does not show the time of each of the 1,300 launches of a pass. The Nsight Compute runs profile 6 to 17 launches each.
- It does not give a cause for the sync calls, for the high DRAM value of two MMQ kernels, or for the faster second request.
- It does not rank the targets.
