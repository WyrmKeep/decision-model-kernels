# Phase 3: caps of the parts of the GPU work (result)

Written 2026-10-31, after the measurement window (00:02 to 01:14 UTC).

## Task of this entry

This entry gives the result of the first ablation. The ablation build skips the work of one part of the model in each run. The time of the run then shows how much time that part uses. The [README of the data set](../../data/2026-10-31-ablation/README.md) gives the raw files, the tables and the conditions. The [ablation README](../../harness/ablate/README.md) gives the meaning of each arm.

This entry has no prediction entry and it makes no prediction. A cap is the largest time that a faster part can remove from a request. A cap is not a prediction of the time that a change removes. The ablation runs give the time only. The answers of these runs are not correct.

## Result in short

- The matrix multiplies and the quantize kernels have a cap of 68.6 %, 67.9 % and 66.7 %. This is for a short request, a medium request and a long request (arm `matmul_q`). The caps are 65.1 %, 63.2 % and 61.7 % if the quantize kernels stay (arm `matmul`).
- The nodes of the recurrent layers have a cap of 9.0 %, 9.4 % and 9.3 % (arm `gdn_conv_l2`).
- The other arms have a cap of 7.3 % or less. The largest are `gdn` (7.1 % to 7.3 %, a part of `gdn_conv_l2`), `quantize` (3.3 % to 7.2 %), `glu` (3.5 % to 6.1 %), `elementwise` (3.2 % to 5.5 %) and `copies` (4.0 % to 4.2 %).
- The arm `all` has a latency of 7.6 ms, 25.2 ms and 53.2 ms. That is 10.7 %, 7.0 % and 6.8 % of the latency of `base`.
- The caps do not add up. The sum of the caps of eleven arms is 94 %, 102 % and 103 %. The cap of `all` is 89 %, 93 % and 93 %.
- The arm `ref` has a drift of 0.3 % to 0.4 % in stage 1. The pass spread of `base` is 0.70 %, 2.04 % and 2.52 %. Each arm has one run. A cap below the pass spread can be noise.
- The fusions of b12004 remove time. With `GGML_CUDA_DISABLE_FUSION=1` the latency is 3.5 %, 5.6 % and 6.0 % higher.
- The CUDA backend does not use CUDA graphs for these requests. It captured 2 graphs in 741 passes and used none of them again.
- The table `DMK_OPS` has no rows. The cause is an error in the build. The commit `de091da69` repairs it. We have no GPU test of the repair.

## Method in short

The client sends the 48 cases of the private benchmark to the server of the arm. The cases are 32 short requests, 8 medium requests and 8 long requests. A run has 5 timed passes and 6 requests before them. Each arm has one run in stage 1. The arm `ref` has three runs: at the start, in the middle and at the end.

The arm `ref` is the baseline build `b12004-sm86`. All other arms are the ablation build. The arm `base` is the ablation build with no environment variable. The cap of an arm is the latency of `base` minus the latency of the arm. The [data set](../../data/2026-10-31-ablation/README.md) gives the build commits, the flags and the files.

`llama bench` ran a second test with prompts of 8, 32, 160, 1100 and 3200 tokens and the flags `-b 2048 -ub 2048`. It has no server and no client. It shows the caps without the work of the server. The five lengths are in four of the five ranges of ubatch length that the server used. The table [timing-ubatch.tsv](../../data/2026-10-31-ablation/tables/timing-ubatch.tsv) gives the ranges.

## Noise: drift and pass spread

The three runs of the arm `ref` are the first run, the run in the middle and the last run of stage 1. The first run and the last run of stage 1 started 27 minutes apart.

| Request | Start (ms) | Middle (ms) | End (ms) | Drift (%) |
|---|---:|---:|---:|---:|
| `short` | 71.3 | 71.5 | 71.3 | 0.30 |
| `medium` | 356.3 | 357.6 | 357.1 | 0.37 |
| `long` | 781.4 | 784.5 | 782.9 | 0.40 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [drift-ref.tsv](../../data/2026-10-31-ablation/tables/drift-ref.tsv).

The drift is the highest value minus the lowest value, in % of the mean of the three values. The arm `ref` is the only arm with more than one run. The drift of `ref` is thus not a limit of the noise of the other arms.

The pass spread of `base` is 0.70 % for short requests, 2.04 % for medium requests and 2.52 % for long requests. The table [pass-spread.tsv](../../data/2026-10-31-ablation/tables/pass-spread.tsv) has the pass spread of each arm.

The arm `ref` is 0.2 % faster than `base` for the three types of request (0.2 ms, 0.8 ms and 1.7 ms). The drift of `ref` is larger than this difference. The data do not show an effect of the host code of the ablation build. The difference between `timing` and `base` is 0.2 % or less for the three types of request.

## Caps for each part and each request

The cap is the latency of `base` minus the latency of the arm. The latency of `base` is 71.6 ms for short requests, 357.8 ms for medium requests and 784.7 ms for long requests. The mark `*` shows a cap that is smaller than the pass spread of `base` for the same type of request. A cap with this mark can be noise.

| Arm | Skipped work | `short` (ms) | `short` (%) | `medium` (ms) | `medium` (%) | `long` (ms) | `long` (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| `all` | All nodes | 63.9 | 89.3 | 332.6 | 93.0 | 731.5 | 93.2 |
| `matmul_q` | All matrix kernels and all quantize kernels | 49.1 | 68.6 | 242.8 | 67.9 | 523.0 | 66.7 |
| `matmul` | All matrix kernels. The quantize kernels stay. | 46.6 | 65.1 | 226.1 | 63.2 | 484.5 | 61.7 |
| `mmq_quantize` | The MMQ kernels and the quantize kernels | 46.2 | 64.6 | 233.0 | 65.1 | 512.1 | 65.3 |
| `mmq` | The MMQ kernels | 43.7 | 61.1 | 216.0 | 60.4 | 473.4 | 60.3 |
| `mmvq` | The MMVQ kernels | 3.0 | 4.2 | 10.7 | 3.0 | 11.9 | 1.5* |
| `quantize` | The quantize kernels | 2.3 | 3.3 | 23.9 | 6.7 | 56.5 | 7.2 |
| `head` | The output matrix multiply and its quantize kernel | 2.9 | 4.0 | 5.5 | 1.5* | 5.8 | 0.7* |
| `gdn_conv_l2` | The nodes `GATED_DELTA_NET`, `SSM_CONV` and `L2_NORM` | 6.5 | 9.0 | 33.6 | 9.4 | 73.0 | 9.3 |
| `gdn` | The node `GATED_DELTA_NET` | 5.1 | 7.1 | 26.1 | 7.3 | 56.0 | 7.1 |
| `ssm_conv` | The node `SSM_CONV` | 1.6 | 2.3 | 8.6 | 2.4 | 18.1 | 2.3* |
| `l2norm` | The node `L2_NORM` | 0.2 | 0.3* | 1.3 | 0.4* | 2.2 | 0.3* |
| `glu` | The `GLU` nodes and the fused kernels `silu(z)*x`, `sigmoid(g)*x` and `softplus*x` | 2.5 | 3.5 | 20.5 | 5.7 | 48.1 | 6.1 |
| `elementwise` | The nodes `ADD`, `MUL`, `SCALE` and the other nodes of this type | 2.3 | 3.2 | 18.8 | 5.3 | 43.0 | 5.5 |
| `copies` | The nodes `CPY`, `SET_ROWS`, `CONCAT` and `GET_ROWS` | 3.0 | 4.2 | 15.1 | 4.2 | 31.4 | 4.0 |
| `cpy` | The nodes `CPY`, `DUP` and `CONT` | 0.5 | 0.7* | 2.9 | 0.8* | 6.8 | 0.9* |
| `set_rows` | The node `SET_ROWS` | 0.0 | 0.1* | 0.5 | 0.1* | 1.6 | 0.2* |
| `concat` | The node `CONCAT` | 1.7 | 2.4 | 10.4 | 2.9 | 22.2 | 2.8 |
| `get_rows` | The node `GET_ROWS` | 1.0 | 1.4 | 3.1 | 0.9* | 4.8 | 0.6* |
| `fattn` | The flash attention node | 1.3 | 1.8 | 9.4 | 2.6 | 34.3 | 4.4 |
| `norm` | The nodes `RMS_NORM` and `NORM` | 2.1 | 2.9 | 12.5 | 3.5 | 27.2 | 3.5 |
| `rope` | The node `ROPE` | 0.3 | 0.4* | 1.8 | 0.5* | 4.0 | 0.5* |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [caps-request.tsv](../../data/2026-10-31-ablation/tables/caps-request.tsv). The value in % is the cap in ms divided by the latency of `base`.

The table shows:

- The matrix kernels have a cap of 60 % to 65 % in the three types of request. The cap of `mmq` is 61.1 %, 60.4 % and 60.3 %.
- The quantize kernels have a larger cap for long requests (7.2 %) than for short requests (3.3 %). The cap of `matmul_q` is not the sum of the caps of `matmul` and `quantize`. For medium requests the sum is 69.9 % and the cap of `matmul_q` is 67.9 %.
- The cap of `gdn_conv_l2` is smaller than the sum of the caps of `gdn`, `ssm_conv` and `l2norm`. The values are 9.0 % against 9.7 %, 9.4 % against 10.1 % and 9.3 % against 9.7 %.
- The cap of `head` decreases with the length of the request (4.0 %, 1.5 %, 0.7 %). The cap of `fattn` increases (1.8 %, 2.6 %, 4.4 %). [INFERENCE: the work of the output head follows the number of output rows, not the number of tokens. This data set does not count the rows.]
- The cap of `mmvq` is 4.2 % for short requests and 1.5 % for long requests.
- The cap of `glu` has the SwiGLU of the feed-forward layers. We read in the code of b12004 that no fusion has the SwiGLU in a pass of more than 8 tokens. The cap of `mmq` has no SwiGLU step.
- The cap of `head` is in the caps of `mmq` and `mmvq`. Its quantize kernel is in the cap of `quantize`. Do not add it to them.

The caps do not add up. The next table gives the sum of the caps of eleven arms: `matmul`, `quantize`, `gdn`, `ssm_conv`, `l2norm`, `glu`, `elementwise`, `copies`, `fattn`, `norm` and `rope`. These arms skip almost different nodes. A fused kernel can be in two items, for example `rms_norm+mul+rope`. The arms `cpy`, `set_rows`, `concat`, `get_rows`, `mmq`, `mmvq` and `head` are in them.

| Request | Sum of the eleven caps (%) | Sum of the eleven caps (ms) | Cap of `all` (%) | Cap of `all` (ms) |
|---|---:|---:|---:|---:|
| `short` | 94.1 | 67.4 | 89.3 | 63.9 |
| `medium` | 101.8 | 364.2 | 93.0 | 332.6 |
| `long` | 102.7 | 805.4 | 93.2 | 731.5 |

Source: the values of the arms in the table [caps-request.tsv](../../data/2026-10-31-ablation/tables/caps-request.tsv). We calculated the sums.

The sum is larger than the cap of `all` in the three types of request. A cap thus has more than the work of its part. We have no test of the cause. [INFERENCE: one cause is the clock of the GPU. The next table shows that the clock is higher in the arms with a large cap. A second cause can be that a skipped node also removes the time between two kernels.]

## Clock and power in the arms with a large cap

An arm that skips many nodes gives less work to the GPU. The median power of the GPU is lower in these arms, and the median SM clock is higher.

| Arms | Median SM clock (MHz) | Median power (W) |
|---|---:|---:|
| `base` | 1845 | 194.6 |
| `mmq`, `matmul`, `matmul_q`, `mmq_quantize` | 1950 | 141 to 152 |
| `elementwise` | 1935 | 178 |
| `quantize` | 1905 | 191 |
| `glu` | 1875 | 193 |
| `all`, `all_timing` | 1755, 1770 | 46 to 47 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the columns `sm_clock_mhz_median` and `power_draw_w_median` of [runs.tsv](../../data/2026-10-31-ablation/tables/runs.tsv).

The clock of the matrix arms is 105 MHz (5.7 %) higher than the clock of `base`. [INFERENCE: the kernels that stay in these arms run faster, and a part of the caps of these arms comes from the higher clock. This data set cannot show how large that part is.] The medians include samples with no load at the start and at the end of the run.

## Caps in `llama bench`

The first table gives the cap in ms for one prompt of the length in the column. The first row is the time of `base`.

| Arm | 8 tokens | 32 tokens | 160 tokens | 1100 tokens | 3200 tokens |
|---|---:|---:|---:|---:|---:|
| `base` (time in ms) | 15.0 | 17.7 | 43.1 | 237.5 | 678.3 |
| `all` | 14.7 | 17.2 | 41.8 | 227.6 | 641.6 |
| `matmul_q` | 12.3 | 13.1 | 31.3 | 165.2 | 458.6 |
| `matmul` | 11.8 | 12.6 | 29.8 | 152.8 | 422.4 |
| `mmq` | 0.0 | 10.8 | 28.1 | 151.0 | 419.2 |
| `mmvq` | 11.7 | 1.7 | 1.8 | 1.5 | 3.9 |
| `head` | 1.6 | 1.7 | 1.7 | 1.4 | 3.8 |
| `quantize` | 0.5 | 0.3 | 2.1 | 18.2 | 54.9 |
| `gdn` | 0.4 | 1.3 | 2.7 | 15.8 | 46.9 |
| `ssm_conv` | 0.0 | 0.2 | 0.7 | 5.7 | 14.4 |
| `glu` | 0.1 | 0.3 | 1.9 | 15.0 | 44.0 |
| `elementwise` | 0.6 | 0.5 | 1.2 | 10.7 | 34.1 |
| `copies` | 0.7 | 0.8 | 1.6 | 9.3 | 27.1 |
| `fattn` | 0.1 | 0.3 | 0.6 | 3.0 | 19.4 |
| `norm` | 0.4 | 0.4 | 1.2 | 8.7 | 23.7 |
| `ref` | 0.1 | -0.1 | 0.0 | -0.4 | 0.0 |
| `timing` | 0.0 | 0.0 | 0.1 | 0.3 | 0.9 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [caps-llama-bench.tsv](../../data/2026-10-31-ablation/tables/caps-llama-bench.tsv).

The next table gives the same caps in % of the time of `base`.

| Arm | 8 tokens | 32 tokens | 160 tokens | 1100 tokens | 3200 tokens |
|---|---:|---:|---:|---:|---:|
| `all` | 97.8 | 97.2 | 96.8 | 95.9 | 94.6 |
| `matmul_q` | 81.8 | 73.9 | 72.6 | 69.5 | 67.6 |
| `matmul` | 79.0 | 70.9 | 69.1 | 64.4 | 62.3 |
| `mmq` | -0.1 | 61.1 | 65.0 | 63.6 | 61.8 |
| `mmvq` | 78.4 | 9.8 | 4.1 | 0.6 | 0.6 |
| `head` | 11.0 | 9.3 | 3.9 | 0.6 | 0.6 |
| `quantize` | 3.2 | 1.8 | 4.8 | 7.7 | 8.1 |
| `gdn` | 3.0 | 7.5 | 6.2 | 6.7 | 6.9 |
| `glu` | 0.5 | 1.6 | 4.4 | 6.3 | 6.5 |
| `elementwise` | 4.2 | 2.8 | 2.9 | 4.5 | 5.0 |
| `copies` | 4.6 | 4.6 | 3.7 | 3.9 | 4.0 |
| `norm` | 3.0 | 2.3 | 2.9 | 3.6 | 3.5 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [caps-llama-bench.tsv](../../data/2026-10-31-ablation/tables/caps-llama-bench.tsv).

The tables show:

- The prompt of 8 tokens uses the MMVQ kernels. The cap of `mmq` is −0.1 % and the cap of `mmvq` is 78.4 %.
- The prompt of 32 tokens uses the MMQ kernels for most matrix multiplies. The cap of `mmq` is 61.1 % and the cap of `mmvq` is 9.8 %. The switch point of this build is thus between 8 tokens and 32 tokens.
- From 160 tokens, the cap of `mmvq` is near the cap of `head`. [INFERENCE: the remaining MMVQ launches are the output head and, in the last layer, the fused launch of `ffn_gate`, `ffn_up` and the SwiGLU.]
- The caps of `matmul` and `mmq` for 1100 tokens and 3200 tokens are 62 % to 64 %. The caps for long requests of the server are 60 % to 62 %.
- The caps of `gdn`, `glu`, `quantize` and `elementwise` increase with the length of the prompt. For 3200 tokens they are 6.9 %, 6.5 %, 8.1 % and 5.0 %.
- The cap of `ref` is between −0.4 ms and 0.1 ms.

## The latency that stays when all nodes are skipped

The arm `all` skips all nodes of all graphs. The server builds the prompt, builds the graph, gives the memory for the buffers, sets the inputs and sends the answer. The table gives the latency of the arm.

| Request | Latency of `all` (ms) | Part of the latency of `base` (%) |
|---|---:|---:|
| `short` | 7.6 | 10.7 |
| `medium` | 25.2 | 7.0 |
| `long` | 53.2 | 6.8 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [caps-request.tsv](../../data/2026-10-31-ablation/tables/caps-request.tsv).

In `llama bench`, the time of the arm `all` for the five prompts is 0.33 ms, 0.49 ms, 1.4 ms, 9.9 ms and 36.7 ms. The arm `all_timing` shows the host time of the ubatches. The table gives the median of each host step of one ubatch, for each range of the number of tokens.

| Tokens of the ubatch | Ubatches | `build_us` | `alloc_us` | `inputs_us` | `submit_us` | `total_us` |
|---|---:|---:|---:|---:|---:|---:|
| 1 to 8 | 48 | 458 | 392 | 11 | 621 | 1510 |
| 9 to 32 | 91 | 410 | 385 | 10 | 844 | 1646 |
| 33 to 128 | 390 | 499 | 420 | 18 | 1623 | 2590 |
| 129 to 512 | 89 | 523 | 428 | 43 | 2127 | 3156 |
| 513 to 2048 | 123 | 583 | 451 | 2812 | 9880 | 13803 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [timing-ubatch.tsv](../../data/2026-10-31-ablation/tables/timing-ubatch.tsv), rows of the run `abl-s1-b1-all_timing`. The unit is µs. The table also has the step `apply_us`. The sum of the medians of the steps is not the median of the total.

In the same run, the median request used 9.6 ms in the server. The median time of the model was 8.4 ms, and the build of the prompt was 1.2 ms. The server builds the graph again for each ubatch. Only 2.1 % of the ubatches of 1 to 8 tokens used the graph of the previous pass again. No larger ubatch did. The table [timing-summary.tsv](../../data/2026-10-31-ablation/tables/timing-summary.tsv) has the values.

A change that makes kernels faster cannot give a latency below the latency of `all`. This is a limit and not a prediction.

## The effect of the fusions: `nofusion`

The arm `nofusion` sets `GGML_CUDA_DISABLE_FUSION=1`. This environment variable turns off all fusions of the CUDA backend of b12004. The arm skips no work. It shows how much time the fusions of this build remove.

| Request | Latency of `nofusion` minus latency of `base` (ms) | The same, in % of `base` |
|---|---:|---:|
| `short` | 2.5 | 3.5 |
| `medium` | 20.0 | 5.6 |
| `long` | 46.9 | 6.0 |

Source: [README of the data set](../../data/2026-10-31-ablation/README.md) and the table [caps-request.tsv](../../data/2026-10-31-ablation/tables/caps-request.tsv). The caps in the table are less than 0.

The range of the 5 pass means of `nofusion` has no overlap with the range of `base` in the three types of request. The difference is thus larger than the pass spread of one run. It is the result of one run for each arm.

The arm `nofusion_glu` skips the `GLU` nodes with all fusions off. Its baseline is `nofusion`. It is 1.9 ms (2.5 %), 14.1 ms (3.7 %) and 33.5 ms (4.0 %) faster than `nofusion`. We calculated these values from the table [caps-request.tsv](../../data/2026-10-31-ablation/tables/caps-request.tsv).

With all fusions on, the cap of `glu` is larger: 2.5 ms, 20.5 ms and 48.1 ms. [INFERENCE: with all fusions off, the fused kernels of the item `glu` are not in the graph. Their nodes, for example `UNARY` and `MUL`, are not in the item `glu`.]

## CUDA graphs are not used: `nographs`

The arm `nographs` sets `GGML_CUDA_DISABLE_GRAPHS=1`. The cap is 0.6 ms (0.8 %) for short requests, 1.2 ms (0.3 %) for medium requests and 1.8 ms (0.2 %) for long requests. The caps for medium requests and long requests are smaller than the pass spread of `base`. For short requests the range of the pass means of `nographs` has no overlap with the range of `base`.

The arms `timing` and `all_timing` show the mode of each graph that the CUDA backend ran. The run `timing` has 741 passes. 739 passes ran in the mode `direct`, 2 passes in the mode `graph_capture` and none in the mode `graph_replay`. The run `all_timing` has the same numbers. The table [timing-summary.tsv](../../data/2026-10-31-ablation/tables/timing-summary.tsv) has them.

The function `ggml_backend_cuda_graph_compute` starts a CUDA graph only after two passes in a row that have the same node properties. A change of the properties starts the test again. [INFERENCE: two passes in a row have different numbers of tokens, thus different node properties. The backend captures a graph 2 times and does not use it again. This data set has no test of the cause.]

The arm `nographs` is not slower than `base`. The data show no time that CUDA graphs remove for these requests in this build.

## The error in the table `DMK_OPS`

The arms `sync` and `sync_all` set `DMK_ABLATE_SYNC=1`. The build then waits for the GPU after each node and adds the time of each node to a table. The server writes the table as lines `DMK_OPS` in its log. The three runs with this environment variable have 0 lines `DMK_OPS`. The table [ops-lines.tsv](../../data/2026-10-31-ablation/tables/ops-lines.tsv) shows it. The data set has no time for each node.

The cause is in the code of the build `aaf86b052`, in `ggml/src/ggml-cuda/ggml-cuda.cu`.

- The build writes the table when no node runs for one second. A function reads the value `dmk_ops_last_ns` for this check. It writes the table only if the value is not 0.
- No code wrote this value. It stayed 0, and the check did not write the table.
- The build also writes the table when the program stops with its exit code. The harness removes the container with `docker rm -f`. [INFERENCE: this kills the server, and the exit code does not run.]

The repair is one line in the function `dmk_sync_account`: `dmk_ops_last_ns.store(dmk_now_ns());`. It is the commit `de091da699f7f63c739e82aea03aa2fa9f550834` on the branch `dmk/ablate-timing`. The runs of this data set used the build `aaf86b052`, without the repair. We did not build the repair. We have no test of it on a GPU. [INFERENCE: the repair makes the check write the table.]

## Limits

- Each arm of stage 1 has one run. The drift of the arm `ref` and the pass spread are the only estimates of the noise.
- The caps do not add up, and the clock changes between arms. A cap is the largest time of a part. It is not a prediction of the time that a change removes.
- The schedule of stage 2 (more blocks of some arms) did not run.
- The table `DMK_OPS` has no rows. The data set has no time for each node. The arms give the time of a group of nodes only.
- The caps in `llama bench` use one prompt length at a time. They do not show the different lengths in one request.
- The ablation arms give no correct answers. The client does not read them.
