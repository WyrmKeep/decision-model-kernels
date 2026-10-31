# Ablation

An ablation measures the time that one part of the GPU work uses in a request. The ablation build skips the work of the part that you select. The time of the request then decreases by the time of that part, or by less. The result is the largest decrease that a faster kernel can give.

**The ablation runs give the time only. The answers of an ablation run are not correct. Do not use them for a quality number.**

The [harness](../README.md) runs the tests. The [data set of the first ablation](../../data/2026-10-31-ablation/README.md) has the results. The [notebook entry](../../docs/notebook/2026-10-31-phase3-ablation-caps.md) shows them.

## Contents

| Path | Contents |
|---|---|
| `make_ablate_runs.py` | Makes the schedules and the files of the arms. |
| `arms/` | One `.env` file and one `.sh` file for each arm, and `SHA256SUMS`. `make_ablate_runs.py` writes them. |
| `extra/ablate/ablate_bench.py` | The client that sends the requests. It does not read the answers. |
| `summarise_ablate.py` | Reads the results of the runs and makes the tables. |
| `sass_check.sh` | Compares the SASS of the ablation build with the SASS of the baseline build. |

## The ablation build

The ablation build is the branch `dmk/ablate-timing` of the fork of llama.cpp. It has three commits on the tag `b12004` (commit `3c809f443`). A fourth commit repairs the table `DMK_OPS`. The first measurement used the build without the fourth commit. The section The table `DMK_OPS` gives the cause.

| Commit | Change |
|---|---|
| `c3747185f` | The environment variables `DMK_ABLATE`, `DMK_ABLATE_SYNC` and `DMK_ABLATE_SYNC_DUMP` in the CUDA backend. |
| `5d35fda2c` | The lines `DMK_TIMING` for each ubatch, each decode and each time that the host waits for the GPU. |
| `aaf86b052` | The line `DMK_TIMING request` in the server. With `DMK_ABLATE`, the server writes 0 for an answer value that is not a number. |
| `de091da69` | The table `DMK_OPS` can write its lines while the server runs. |

The build reads each environment variable one time, when the CUDA backend starts. If you set no environment variable, each skip is one load and one branch in the code of the host. The code of the GPU does not change.

We compared the SASS of `sm_86` of the build `aaf86b052` with the SASS of the baseline build `b12004-sm86`. The two builds have the same SASS for all 6,754 kernel names. They have the same instruction code, and the same values of `REG`, `STACK`, `SHARED`, `LOCAL` and `CONSTANT`. Each side has 146 cubins and 6,154,400 instructions. The table `sass.tsv` of the [data set](../../data/2026-10-31-ablation/README.md) shows the numbers.

The code of the host is different. The arm `base` against the arm `ref` measures the effect of this difference.

## Parts that the ablation skips

A skipped part makes no launch of its kernels. The graph, the buffers, the fusion decisions and the launches of all other parts stay the same. The CUDA backend skips a node of the graph before it launches the node. For a matrix multiply, the backend skips the matrix kernels and keeps the quantize kernel. An item can also skip the quantize kernel.

Set `DMK_ABLATE` to a list of items. Put `,` or a space between two items. If the build does not know an item, it stops with an error.

| Item | Skipped work |
|---|---|
| `mmq` | All launches of `mul_mat_q`: the matrix kernel and the second kernel of the stream-k method. The quantize kernel stays. |
| `mmvq` | All launches of `mul_mat_vec_q`. The quantize kernel stays. |
| `mmvf`, `mmf`, `cublas` | The kernels `mul_mat_vec_f` and `mul_mat_f`, and cuBLAS. This model does not launch them. |
| `quantize` | The two kernels that quantize activations: `quantize_mmq_q8_1` in front of MMQ and `quantize_q8_1` in front of MMVQ. `quantize_mmq` and `quantize_mmvq` select one of them. |
| `matmul` | All matrix multiplies: `mmq`, `mmvq`, `mmvf`, `mmf`, `cublas` and `mmid`. The quantize kernels stay. |
| `head` | The output matrix multiply with its quantize kernel. It is the only matrix with 65,536 rows or more. The matrix `token_embd.weight` has 248,320 rows. |
| `fattn` | The node of flash attention. |
| `gdn` | The node `GATED_DELTA_NET` of the recurrent layers, and the variant that writes the state copies. |
| `glu` | The `GLU` nodes that no fusion includes, and the fused kernels `silu(z)*x`, `sigmoid(g)*x` and `softplus*x`. In each pass of more than 8 tokens, the SwiGLU of each feed-forward layer is a `GLU` node that no fusion includes. |
| `norm` | `RMS_NORM`, `NORM`, and the fused kernels `rms_norm+mul`, `rms_norm+add` and `rms_norm+scale`. |
| `l2norm` | `L2_NORM`. |
| `cpy`, `set_rows`, `concat`, `get_rows` | The nodes `CPY`, `DUP`, `CONT` and `SET`. The node `SET_ROWS`, which writes the KV cache. The node `CONCAT`. The node `GET_ROWS`. The item `copies` is the four items together. |
| `ssm_conv` | The node `SSM_CONV` and the fused kernel `conv+add+silu`. |
| `elementwise` | The nodes `ADD`, `SUB`, `MUL`, `DIV`, `SCALE`, `CLAMP`, `SQR`, `REPEAT` and `UNARY`, the other nodes of this type, and their fused kernels. |
| `rope` | The node `ROPE` and the fused kernel `rope+view+set_rows`. A fused kernel `rms_norm+mul+rope` counts for `norm` and for `rope`. |
| `mmid`, `other` | The node `MUL_MAT_ID`, which this model does not have. Each node that no other item names: `SOFT_MAX`, `SUM_ROWS`, `CUMSUM`, `TRI`, `SOLVE_TRI` and more. |
| `all` | All items. |

## The arms

The file `arms/<arm>.env` has the environment variables of the arm, for the flag `--env-file` of Docker. The file `arms/<arm>.sh` sets the same environment variables and starts a command. A `tool` line uses the file `.sh`. The arm `ref` has no file. It is the baseline build with no environment variable.

| Arm | Environment variables | Meaning |
|---|---|---|
| `ref` | None | The baseline build `b12004-sm86`. |
| `base` | None | The ablation build with no environment variable. |
| `timing` | `DMK_TIMING=1` | The lines `DMK_TIMING` on the standard error. |
| `all` | `DMK_ABLATE=all` | Skips all nodes. The time of the host stays. |
| `all_timing` | `DMK_ABLATE=all`, `DMK_TIMING=1` | The same, with the lines `DMK_TIMING`. |
| `mmq`, `mmvq`, `quantize`, `matmul`, `head`, `fattn`, `gdn`, `glu`, `norm`, `ssm_conv`, `elementwise`, `copies`, `l2norm`, `cpy`, `set_rows`, `concat`, `get_rows`, `rope` | `DMK_ABLATE=<arm>` | Skips the part with the same name. |
| `matmul_q` | `DMK_ABLATE=matmul,quantize` | Skips all matrix multiplies and all quantize kernels. |
| `mmq_quantize` | `DMK_ABLATE=mmq,quantize` | Skips the MMQ kernels and the quantize kernels. |
| `gdn_conv_l2` | `DMK_ABLATE=gdn,ssm_conv,l2norm` | Skips the nodes `GATED_DELTA_NET`, `SSM_CONV` and `L2_NORM`. The matrix multiplies stay. |
| `nofusion` | `GGML_CUDA_DISABLE_FUSION=1` | It is not an ablation. This environment variable of b12004 turns off all fusions. |
| `nofusion_glu` | `GGML_CUDA_DISABLE_FUSION=1`, `DMK_ABLATE=glu` | Turns off all fusions and skips the GLU nodes. Compare it with `nofusion`, not with `base`. |
| `nographs` | `GGML_CUDA_DISABLE_GRAPHS=1` | It is not an ablation. This environment variable of b12004 turns off the CUDA graphs. |
| `sync` | `DMK_ABLATE_SYNC=1` | One stream synchronization after each node. The server writes the table `DMK_OPS`. |
| `sync_all` | `DMK_ABLATE=all`, `DMK_ABLATE_SYNC=1` | The same, with all nodes skipped. It gives the time of one node with no launch. |

## The cap of an arm

The cap of an arm is the time of the arm `base` minus the time of the arm. It is in ms, and in % of the time of `base`. The cap is the largest time that a faster part can remove from a request. It is not a prediction of the time that a change removes.

- A faster kernel cannot remove more time than the cap. The cap is the time that you remove if the part uses no time.
- A skipped node also removes the time between a kernel and the next kernel. The cap has this time.
- The caps do not add up. The GPU does some work at the same time, and the clocks change. Use the arms that put items together (`matmul_q`, `mmq_quantize`, `copies`, `gdn_conv_l2`) for sums.
- A skipped part gives less work to the GPU. The power can decrease and the clock can increase. Compare the `.smi` files of the runs before you use a small cap.
- The next kernel reads data that a skipped kernel did not write. The time of the next kernel can change because of this. [INFERENCE: the data set has no test of this effect.]
- The arm `head` skips the quantize kernel of the output matrix. The arms `mmq` and `mmvq` skip the matrix kernel of the output matrix, but not its quantize kernel. The output matrix is thus in the caps of `mmq` and `mmvq`. Do not add the cap of `head` to them.
- The baseline of the arm `nofusion_glu` is the arm `nofusion`.

In b12004, no fusion includes the feed-forward layer of this model in a pass of more than 8 tokens. The matrix multiplies `ffn_gate` and `ffn_up` are two MMQ launches. The SwiGLU is a `GLU` node. The cap of `glu` thus has the SwiGLU of the feed-forward layers. The cap of `mmq` has no SwiGLU step.

Only the last layer, on the output rows, has one fused MMVQ launch for `ffn_gate`, `ffn_up` and the SwiGLU. The cap of `mmvq` has it.

## Other environment variables

| Environment variable | Effect |
|---|---|
| `DMK_ABLATE_SYNC=1` | The build launches each node directly, with no CUDA graph, because a CUDA graph cannot hold a stream synchronization. It does a stream synchronization after each node. It adds the time of each node to a table. |
| `DMK_ABLATE_SYNC_DUMP=N` | The build also writes the table and makes it empty after each `N` graphs. |
| `DMK_TIMING=1` | The build writes the lines of the table below on the standard error. Each line starts with `DMK_TIMING`. `t=` is the time of `steady_clock` in seconds. Each `*_us` is in microseconds. |

| Line | When | Fields |
|---|---|---|
| `ubatch` | Each ubatch | `id n_tokens n_seqs n_seq_tokens n_outputs n_nodes reuse apply_us build_us alloc_us inputs_us submit_us total_us`. `build_us` and `alloc_us` are 0 if the pass uses the graph of the previous pass again. `submit_us` is the time of the host for the step that calculates the graph. It does not have the time of the GPU. |
| `cuda_compute` | Each time the CUDA backend runs a graph | `n_nodes mode host_us`. `mode` is `direct`, `graph_capture` or `graph_replay`. |
| `decode` | Each `llama_decode` | `n_tokens n_ubatches init_us total_us` |
| `sync` | Each time the host waits for the GPU | `passes tokens wait_us since_first_submit_us since_last_submit_us` |
| `request` | Each request of `/v1/systemone` | `n_questions n_groups parse_us build_us wait_us format_us total_us` |

## The table DMK_OPS

With `DMK_ABLATE_SYNC=1` the build writes lines of this shape:

```
DMK_OPS<TAB>bucket<TAB>label<TAB>n<TAB>total_us<TAB>avg_us
```

`bucket` is the number of tokens of the pass, in a group (`T1`, `T2-8`, `T9-32`, `T33-128`, `T129-512` or `T513-2048`). `label` has the names of the operations of the node. A fused node has `+` between the names. The first matrix multiply of a node also has the type, the number of rows, the number of columns and the kernel.

The first measurement gave no `DMK_OPS` line. The cause is in the build `aaf86b052`. The build writes the table when no node ran for one second. It reads the time of the last node for this check. No code wrote that time, and its value stayed 0. The check did not find a time without a node.

The build also writes the table when the program stops with its exit code. The harness removes the container with `docker rm -f`. [INFERENCE: this kills the server, and the server cannot write the table.]

The commit `de091da69` writes the time of the last node when a node adds its time to the table. The first measurement did not use this commit. We have no test of it. [INFERENCE: it repairs the table. It is one line of code, and no GPU ran it.]

## Steps to make a measurement

1. Build the ablation build with the [build script](../README.md#build-the-programs-of-an-arm). Put the output in `builds/ablate-timing` on the GPU host.
2. Put a copy of the folder `arms/` in `builds/ablate-timing/dist/abl/` on the GPU host. The container of an arm sees it at `/opt/dmk/abl`.
3. Make the schedules with `make_ablate_runs.py`. Set `DMK_REMOTE_DIR` to the work folder of the GPU host, or use `--remote-dir`.
4. Set `DMK_EXTRA_DIR` in the env file to the folder `harness/ablate/extra`.
5. Run each schedule with `run_schedule.sh` in a measurement window.
6. Make the tables with `summarise_ablate.py`.

```
python3 harness/ablate/make_ablate_runs.py --out schedules/ablate --remote-dir /path/to/work
rsync -a harness/ablate/arms/ gpu-host:/path/to/work/builds/ablate-timing/dist/abl/
export DMK_ENV=/path/to/harness.env
harness/run_schedule.sh schedules/ablate/sched/abl-smoke.txt results/abl-smoke
harness/run_schedule.sh schedules/ablate/sched/abl-s1.txt results/abl-s1
python3 harness/ablate/summarise_ablate.py results/abl-smoke results/abl-s1
```

`make_ablate_runs.py` writes the files of the arms to `harness/ablate/arms` and the schedules to `<out>/sched`. The table gives its flags.

| Flag | Default | Meaning |
|---|---|---|
| `--out` | None | The folder for the schedules. This flag is necessary. |
| `--remote-dir` | `DMK_REMOTE_DIR` | The work folder of the GPU host. The lines of the server arms have it in the path of the `.env` file. |
| `--build` | `ablate-timing` | The name of the ablation build. |
| `--ref-build` | `b12004-sm86` | The name of the baseline build. |
| `--model` | `h2o-lightning-4b-rq-Q4_0-mix8.gguf` | The model file in `DMK_MODELS`. |
| `--arms-dir` | `harness/ablate/arms` | The folder for the files of the arms. |
| `--passes` | 5 | The number of timed passes of the client in stage 1 and stage 2. |
| `--blocks-core` | 2 | The number of blocks of stage 2. |
| `--llb-reps` | 5 | The number of times that each `llama bench` test runs. |
| `--seed` | 20261101 | The seed of the random sequence. |

The script writes five schedules.

| Schedule | Runs | Contents |
|---|---:|---|
| `abl-smoke.txt` | 5 | A check before the long schedules. One pass each for `base`, `all`, `all_timing` and `sync`, and a `llama bench` with `mmq`. |
| `abl-s1.txt` | 31 | Stage 1. Each arm one time, in a random sequence, with 5 passes. The arm `ref` is also first, in the middle and last, to show the drift. |
| `abl-s2.txt` | 36 | Stage 2. The control arms and the primary arms, 2 more blocks. The first measurement did not run it. |
| `abl-ops.txt` | 2 | The arms `sync` and `sync_all`, with 2 passes. The client waits 3 s after the last request. |
| `abl-llb.txt` | 18 | `llama bench` with prompts of 8, 32, 160, 1100 and 3200 tokens. One run for each control arm and primary arm. |

A run of a server arm is a `client` line. It has the override `DMK_DOCKER_EXTRA=--env-file=<path of the .env file>`. The client sends the 48 cases of the case file of the benchmark, one request at a time. The first line of each request is different from the first line of each other request. A slot of the server then cannot use the data of a previous request. The sequence of the cases changes in each pass.

A run of a tool arm is a `tool` line with the override `DMK_WRAP=/opt/dmk/abl/<arm>.sh`. The wrapper sets the environment variables and starts `llama bench`. The prompt of 8 tokens is equivalent to the question of a child task. The prompts of 32 and 160 tokens are equivalent to a short parent task. The prompts of 1100 and 3200 tokens are equivalent to a medium request and a long request.

The client `ablate_bench.py` has these flags.

| Flag | Default | Meaning |
|---|---|---|
| `--cases` | `/bench/cases.json` | The case file in the container. |
| `--size` | `all` | The requests to send: `short`, `medium`, `long` or `all`. |
| `--passes` | 5 | The number of timed passes. |
| `--warmup` | 6 | The number of requests before the timed passes. |
| `--linger-s` | 0 | The seconds to wait after the last request. Use it with `DMK_ABLATE_SYNC=1`. The server then writes the table before the harness removes the container. |
| `--out` | None | The result file. The harness gives `/runs/{name}.json`. |
| `--host`, `--port` | `127.0.0.1`, `18080` | The host and the port of the server. |

## Make the tables

`summarise_ablate.py` reads the results folders of `run_schedule.sh`. The name of a run gives its arm.

```
python3 harness/ablate/summarise_ablate.py [--tsv-dir DIR] RESULTS_FOLDER [RESULTS_FOLDER ...]
```

Without `--tsv-dir` the script writes the tables to the standard output. With `--tsv-dir` it writes one TSV file for each table. The first lines of `summarise_ablate.py` name the tables. The [data set](../../data/2026-10-31-ablation/README.md) gives the columns of each table.

## Compare the SASS

`sass_check.sh` compares the SASS of two builds in the builder image. It uses `harness/build/sass_compare.py`, and it uses `podman`.

```
harness/ablate/sass_check.sh OUT_FOLDER BASELINE/libggml-cuda.so NEW/libggml-cuda.so
```

The result is `OUT_FOLDER/cmp/sass-compare.json`. The two builds have the same SASS when `names_sass_different`, `names_encoding_different`, `names_only_A`, `names_only_B` and `names_resusage_different` are all 0. The check runs for about 1 minute.

## Limits

- The ablation runs give the time only. The answers are not correct. The client does not read them.
- Each arm of stage 1 has one run. One run gives no estimate of the noise. The range of the timed passes of the run is the only estimate. The arm `ref` shows the drift of the host.
- With `DMK_ABLATE` set, the server writes 0 for an answer value that is not a number. Without it, the server does not change. [INFERENCE: the kernels have no control flow that uses a value from a skipped kernel. We did not check each kernel.]
- An arm that stops with an error shows it in the file `<run>.client.out` and in the server log. The first measurement had no such arm.
- The table `DMK_OPS` of the first measurement has no rows. The section The table DMK_OPS gives the cause.
