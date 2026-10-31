# 0007 Uploads of the scheduler: prediction

Written 2026-10-31. No run of the change has a time measurement at this time. The commit time of this file gives the indication.

## Task of this entry

Target `P` of the [ranked list](2026-10-31-phase4-ranked-targets.md) is "Remove the stream synchronizations in a pass". The ranked list tells that we did not find the code that starts them. This entry gives the code. It gives the change, the equivalence class, the arms, the ratios, the limits, and the prediction. It also tells where the prediction is different from the numbers of phase 4, and the cause.

## Where the stream synchronizations come from

All file names and line numbers are for the tag `b12004`. The lead build has the same lines in `ggml-backend.cpp`, `llama-context.cpp`, `llama-graph.cpp` and `server-context.cpp`. The file `ggml-cuda.cu` has other line numbers in the lead build.

| Sync call | Where | Number in one pass |
|---|---|---:|
| `cudaStreamSynchronize` on the CUDA stream of the backend, before an upload of an input | `ggml_backend_sched_copy_input`, `ggml/src/ggml-backend.cpp:2094` for an input with the flag `INPUT`, and `:2104` for the other inputs. The function `ggml_backend_cuda_synchronize` (`ggml/src/ggml-cuda/ggml-cuda.cu:2842`) makes the sync call. | 12 |
| `cudaMemcpyAsync` and `cudaStreamSynchronize` on the CUDA stream of the thread, for each upload | `ggml_backend_tensor_copy` (`ggml-backend.cpp:832`), from `ggml-backend.cpp:2096` and `:2121`. It uses `ggml_backend_cuda_buffer_set_tensor` (`ggml-cuda.cu:828` to `:834`). | 9 or 10 |
| `cudaStreamSynchronize` at the end of a decision | `llama_get_logits_ith` (`src/llama-context.cpp:4292`) uses `llama_context::synchronize` (`:888`). The server starts it in `send_decision` (`tools/server/server-context.cpp:2527`). | 1 for each decision |

The function `llama_context::process_ubatch` (`src/llama-context.cpp:1525`) starts `graph_compute` at line 1588. The function `graph_compute` starts `ggml_backend_sched_graph_compute_async` at line 2775. That function starts `ggml_backend_sched_compute_splits` (`ggml-backend.cpp:2125`). The loop at `ggml-backend.cpp:2148` to `:2157` makes the upload of each input of each part of the graph. The scheduler does this before it starts the part on its backend.

### Data

- We ran Nsight Systems with the list of the functions above each CUDA call (`--cudabacktrace=sync:0,memory:0`) on the lead build. The run has 14 passes of requests of the type `short2q`. The table above gives the functions above each sync call. 168 sync calls of the first row (12 in each pass) and 136 sync calls of the second row (9.7 in each pass) are in the passes. 8 sync calls of the third row and 12 other sync calls of `ggml_backend_sched_synchronize` are between the passes.
- The list of functions and the profiles of phase 3 give the same result. A pass has 21 to 25 sync calls. No sync call is between two kernels of a pass. All sync calls are before the first kernel of the pass, in the uploads of the inputs.

### Time that the sync calls use

The ranked list tells that the GPU waits 40 µs to 50 µs after each stream synchronization. The profiles do not show this.

- A pass has 1,200 to 1,270 gaps between two kernels. Almost all gaps are 1 µs to 2 µs. A pass has 2 or 3 gaps of more than 8 µs, at the start of the pass. These gaps use 0.02 ms to 0.07 ms in a pass.
- The idle time of 0.9 ms in a pass is the sum of the small gaps. It does not come from the sync calls.
- The phase of the uploads in a pass of `short1q` (151 tokens) uses 0.82 ms. The copy of the data on PCIe uses 0.70 ms of it, at 6.7 GB/s. The CUDA calls use about 0.12 ms.

The sync calls are thus the procedure of an upload that makes the host wait. The uploads use most of the time of the phase, because of the copy on PCIe. A change that makes the uploads not wait gives 0.1 ms to 0.15 ms in a pass at most. This is 0.3 % of `short1q` and 0.05 % of `long`. We do not measure that change. The gain is too small for the test.

## Finding: 3 of the 10 uploads of a pass are not necessary

A pass uploads 10 inputs to the GPU. The first 4 uploads have these numbers of bytes, in this sequence, for a pass of `n` tokens:

| Upload | Bytes | Flag `INPUT` |
|---:|---|---|
| 1 | `n` × 10,240 | Yes |
| 2 | `n` × 10,240 | No |
| 3 | `n` × 8 | Yes |
| 4 | `n` × 10,240 | No |

[INFERENCE: we did not write the names of the inputs. The bytes, the flags, and the sequence agree with the graph of `llm_graph_context::build_inp_embd` (`src/llama-graph.cpp:2657` to `:2776`), as follows.]

- The graph has a `mixed` path for a batch with tokens and embeddings together. The path has the inputs `inp_mixed_embd` (`n` × 2,560 numbers of 4 bytes) and `inp_mixed_slots` (`n` numbers of 8 bytes). It also has the rows of `tok_embd` for the tokens of the path. `ggml_get_rows` on the CPU gives these rows (`n` × 2,560 numbers of 4 bytes).
- `ggml_build_forward_select` puts the nodes of all paths in the graph. It sets the flag `GGML_TENSOR_FLAG_COMPUTE` on the nodes of the path that it selects (`ggml/src/ggml.c:7340`). The backends do not calculate the nodes with no flag.
- This model gets tokens only. The graph selects the path of the tokens. The nodes `dup` and `set_rows` of the `mixed` path have no flag. The scheduler puts them on the GPU. It uploads their 3 inputs to the GPU.

Upload 4 is the result of `get_rows` for the path of the tokens. The first layer reads it.

The uploads 1 to 3 are not necessary. They have 2 × `n` × 10,240 + `n` × 8 bytes, and 7 of the 22 sync calls of a pass. The copy of the 2 large uploads on PCIe uses 3.05 µs for each token.

## Change

The scheduler of ggml records, for each upload of an input, if a node with the flag `GGML_TENSOR_FLAG_COMPUTE` reads it. It does not make an upload that no such node reads. A part of the graph with no upload to make waits for the part before it. This is the same procedure as for a part with no inputs.

The commit is `c3ec5653b` on the branch `dmk/0007-input-copies` of the fork. It changes the file `ggml/src/ggml-backend.cpp` only (26 lines). No kernel and no operation changes.

The number of uploads in a pass goes from 10 to 7. The number of sync calls in a pass goes from 21 to 25, to 14 to 18.

## Equivalence class

We select the equivalence class `exact`. The scheduler uploads the same data to the same positions for all nodes that the backends calculate. The 3 uploads that it does not make have inputs of nodes that the backends do not calculate. [INFERENCE: we read this in the code of the backends. The CPU backend (`ggml-cpu.c:3156`) and the CUDA backend (`ggml-cuda.cu:4905`) skip the nodes with no flag.]

The method gives these tests for the equivalence class `exact`:

- The answers of the private benchmark are the same bit for bit as the answers of arm L. The tool `answers_compare.py` must give `cross_identical_all` true, and `self_repeat` true for the two arms.
- The change has no operation and no kernel. `test-backend-ops` has no operation for this change. The SASS of `libggml-cuda.so` does not change. We do not run `compute-sanitizer`.
- A check of the mechanism: a profile of the new build must show 7 uploads and 14 to 18 sync calls in a pass. The profile is not a run for the time measurement.

## Prediction

The prediction comes from the Nsight Systems profiles of phase 3 (build D), 12 requests of each type. For each pass we measured one time. It starts at the end of the first sync call of the upload phase. It stops at the end of the sync call after upload 3.

The new build does not use this host time. The GPU runs no kernel in this time. The time becomes larger with the number of tokens: about 60 µs + 3.05 µs for each token.

| Request type | Cases | Passes (tokens) | Removed uploads (MB) | Predicted gain (ms) | Phase 4 (ms) |
|---|---:|---|---:|---:|---:|
| `short1q` | 12 | 1 (151) | 3.1 | 0.51 | 0.8 |
| `short2q` | 20 | 3 (87, 102 and 69) | 4.6 | 0.85 | 2.2 |
| `medium` | 8 | 4 (1,300, 129, 8 and 26) | 28.8 | 4.55 | 3.8 |
| `long` | 8 | 5 (2,048, 1,173, 129, 8 and 26) | 69.1 | 10.58 | 3.6 |

Source: the profiles `p3-nsys-short1q`, `p3-nsys-short2q`, `p3-nsys-medium3q`, and `p3-nsys-long3q` of the [phase 3 data](../../data/2026-10-31-profiles/README.md). [INFERENCE: the profiles are not in this repository. We made the sums.]

The next table gives the predicted ratio of arm I to arm L. A number less than 0 is a faster build. The reference is the mean of arm L in the [lead entry](2026-10-31-lead-30489.md): 236.5 ms for `lat_ms`, 72.5 ms for `short_ms`, 355.7 ms for `medium_ms`, and 773.6 ms for `long_ms`.

| Metric | Predicted change (%) | Range that agrees with the prediction (%) |
|---|---:|---|
| `lat_ms` | -1.27 | -0.9 to -1.7 |
| `short_ms` | -1.0 | -0.6 to -1.5 |
| `medium_ms` | -1.28 | -0.9 to -1.7 |
| `long_ms` | -1.37 | -0.9 to -1.8 |
| `llama bench`, 64 tokens | +1.1 | +0.6 to +1.7 |
| `llama bench`, 256 tokens | +1.4 | +0.9 to +2.0 |
| `llama bench`, 3,000 tokens | +1.5 | +1.0 to +2.0 |

The gain of `lat_ms` is 3.0 ms. The weights of the types are 12, 20, 8 and 8 cases of 48. The numbers of `llama bench` come from the time that we remove from one pass and the pass times of arm L in the lead entry. The time is 0.26 ms at 64 tokens, 0.84 ms at 256 tokens, and 6.3 ms plus 3.0 ms for the two passes of 3,000 tokens.

### Where this prediction is different from phase 4

- Phase 4 gave a time of 40 µs to 50 µs after each of 21 to 25 stream synchronizations, in which the GPU waits. The profiles show that all sync calls are before the first kernel. The time in which the GPU waits in a pass is the time of the copy on PCIe. The gain of phase 4 for the small requests was too large (0.8 ms and 2.2 ms against 0.51 ms and 0.85 ms).
- Phase 4 had no cause that becomes larger with the number of tokens. The 3 uploads that we remove become larger with it. The gain for `medium` and `long` is larger (4.55 ms and 10.58 ms against 3.8 ms and 3.6 ms).
- The total is almost the same: 1.27 % of `lat_ms` against 1.0 %. The cause is different.
- We do not make the change `P` of phase 4. We remove the uploads that no node reads. The other sync calls stay. The section "Time that the sync calls use" gives the cause.

### Results that show an incorrect prediction

- The gain of `lat_ms` is less than 0.9 %. The host time that we remove then does not change the time of the request. A cause can be that the GPU runs kernels while the host makes the uploads. The profiles show no such time.
- The answers are not the same. A node that the backend calculates then reads one of the 3 uploads. The equivalence class is not `exact`. We stop and examine the change.
- The profile of the new build has more than 7 uploads in a pass. The 3 inputs are then not the inputs that we think.

## Arms and schedule

| Arm | Build | Source |
|---|---|---|
| `L` | `build:lead-30489` | Branch `dmk/stack` of the fork, commit `6b8b65e66`. It is the build before the change. |
| `I` | `build:0007-input-copies` | Branch `dmk/0007-input-copies`, commit `c3ec5653b`. |

The measurement window has two schedules. The harness makes them with `harness/make_schedule.py`. The seed gives the random sequence of the arms in each block.

```
python3 harness/make_schedule.py --prefix c0007-bench --blocks 4 --seed 70071 --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind bench --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix c0007-llb --blocks 4 --seed 70072 --arm L=build:lead-30489 --arm I=build:0007-input-copies --kind tool --args "bench -m /models/h2o-lightning-4b-rq-Q4_0-mix8.gguf -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

| Block | Sequence of the private benchmark | Sequence of `llama bench` |
|---:|---|---|
| 1 | I, L | L, I |
| 2 | I, L | L, I |
| 3 | L, I | L, I |
| 4 | L, I | I, L |

Each run of the private benchmark writes its answers (`--write-reference`). The 16 runs of the [lead test](2026-10-31-lead-30489.md) used about 16 minutes. We think that the new test uses the same time. The harness keeps the power limit and the clocks of the other measurement windows. It writes the load of the host before and after each run.

## Ratios

- The ratio of a block is the value of arm I divided by the value of arm L, minus 1. The estimate is the mean of the 4 ratios. The interval is the 95 % bootstrap interval, with 10,000 samples. The tool is `harness/analysis/ab_stats.py` with `--a L --b I`.
- The metrics are `lat_ms`, `short_ms`, `medium_ms`, `long_ms` and `rps8` of the private benchmark, and the prompt speed of `llama bench` at 64, 256 and 3,000 tokens.
- The prediction is correct for a metric if the estimate is in the range of the table above and the interval does not include 0.
- The interval of 4 blocks is small. The [lead entry](2026-10-31-lead-30489.md) gives the cause. We write this in the result.
- The tool `harness/analysis/answers_compare.py` compares the 8 answer files of the private benchmark.

## Limits of this prediction

- The time that we remove comes from one source: the host time of 3 uploads. The prediction does not include a change in the work or in the cache of the CPU.
- The upload times come from build D with Nsight Systems. The lead build has other kernels, but the same scheduler and the same uploads. The reference of the ratios is arm L, which is about 2 % faster than build D.
- We did not run the new build before this entry. The check of the mechanism comes after the entry.
