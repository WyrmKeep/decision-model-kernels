# Kernels

This document gives facts about the kernels of llama.cpp that the project can change. At this time, the project has no change to a kernel. A change that touches a kernel adds one section to this document. The section gives the task of the kernel, its file, and its limits.

## Kernels on the path of a request

The table shows the kernels that a request uses in tag `b12004`. The facts come from the [inspection of the upstream code](notebook/2026-10-30-phase0-upstream-survey.md). The paths are below `ggml/src/ggml-cuda/` unless the column gives a different path.

| Kernel | Task | File | Facts |
|---|---|---|---|
| Output head | Calculates the logits of all 248,320 rows of the output head | `src/models/qwen35.cpp` (line 233) | The server reads only the logits of the answer tokens. |
| MMQ | Multiplies quantized model weights with int8 numbers | `mmq.cuh`, `mmq-config-ampere.cuh` | Q4_0 and Q8_0 on Ampere use 256 threads, occupancy 1, `I=128`, and stream-k. `J` goes from 8 to 128 in steps of 8. MMQ has no `cp.async` code. |
| MMVQ | Multiplies quantized model weights with a batch of a small number of columns | `mmvq.cu` | The maximum is `MMVQ_MAX_BATCH_SIZE`, with 8 columns on `sm_86`. |
| Chunked kernel | Does prefill for a Gated DeltaNet layer | `gated-delta-net-mma.cu` | It runs only with 64 or more tokens in each sequence. |
| Autoregressive kernel | Does one token at a time for a Gated DeltaNet layer | `gated_delta_net.cu` | It runs for all other batches. |
| Flash attention | Does attention for the 8 attention layers | `fattn-common.cuh` | The kernel finds the last KV cell that it must read from the attention mask only if `Q->ne[1] >= 1024` or `Q->ne[3] > 1`. |

Source: [inspection of the upstream code](notebook/2026-10-30-phase0-upstream-survey.md), sections 3.a, 3.d, 3.e, 3.h, and 5.

## Facts for a change

- The MMQ fusion of the `gate`, `up`, and GLU steps is for the types Q4_K, Q5_K, Q6_K, Q8_0, and IQ4_XS. Q4_0 is not one of them. The feed-forward layers of the production model use Q4_0 [INFERENCE: the type comes from the commands that made the model file].
- The chunked kernel is for a scalar gate and a head dimension of 128. It must have `H_k == 16` and `H` of 16, 32, 48, or 64. It must have 64 or more tokens in each sequence.
- A ubatch of the hybrid model has sequences of equal length. The shortest sequence sets the length.
- The code has no limit on the number of tokens of a batch for CUDA graphs. The code captures a graph only after two graph runs in sequence that have the same node properties.
- With a unified KV cache, `Q->ne[3]` is 1. A short batch then does not find the last KV cell that it must read. Each sequence reads KV cells of other sequences. Only 8 of the 32 layers use attention. We did not measure the effect on this model [INFERENCE].
- The model code makes one tensor from the conv state and the new input with `ggml_concat`. It does this before `ssm_conv`. CUDA fuses `SSM_CONV`, `ADD`, and `SILU` into one kernel. It does not fuse the concat.

## Measured facts (phase 3)

Phase 3 measured the kernels on build `D` with the model mix8 on the `RTX 3060 Ti`. The [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md) has the method and the full tables. The scripts use 149.3 TOPS as the int8 peak and 422 GB/s as the DRAM limit.

| Kernel | Measure | Value | Unit | Source |
|---|---|---:|---|---|
| MMQ, 1,173 to 2,048 tokens | Part of the int8 peak | 32.5 to 39.2 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 1,173 to 2,048 tokens | IMMA pipe active | 35 to 41 | % of the time | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 1,173 to 2,048 tokens | Issue slots active | 34 to 46 | % of the time | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 1,173 to 2,048 tokens | Registers for each thread | 216 or 254 | registers | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 1,173 to 2,048 tokens | Shared memory for each thread block | 57,856 | B | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 1,173 to 2,048 tokens | Thread blocks of 8 warps on each SM | 1 | thread block | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| MMQ, 26 to 102 tokens | Part of the DRAM limit | 35 to 80 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-a-small-number-of-tokens) |
| MMQ, 26 to 102 tokens | Part of the int8 peak | 9 to 31 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-a-small-number-of-tokens) |
| MMQ, passes of more than 8 tokens | Part of the kernel time of the pass | 60 to 70 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#kernel-time-of-each-pass-type) |
| MMVQ | Part of the DRAM limit | 90 to 95 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| `gdn_single` | Thread blocks of the launch, on 38 SMs | 32 | thread blocks | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| `gdn_single` | Time of one launch | 740.3 | µs | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| `gdn_single` | Part of the DRAM limit | 19.9 | % | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| `concat_non_cont` | Time of one launch | 18.3 | µs | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| `concat_non_cont` | Launches in each pass | 24 | launches | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#kernel-time-of-each-pass-type) |
| Flash attention | Time of the profiled launch | 11.2 | µs | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |
| Output head | Time in each pass that has outputs | 1.64 | ms | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#kernel-time-of-each-pass-type) |
| Small kernels (`rms_norm`, `unary_gated_op`, `ssm_conv`) | Time of one launch | 2 to 3 | µs | [Entry](notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |

Source: the tables `ncu-mmq-large-n.tsv`, `ncu-mmq-small-n.tsv`, `ncu-mmq-resources.tsv`, `ncu-other-kernels.tsv`, and `nsys-kernel-ms-per-pass.tsv` of the [data set of the profiles](../data/2026-10-31-profiles/README.md). The kernel `gdn_single` and the small kernels come from runs with the result `MISALIGNED` or `SHIFTED`. Thus their launches are not the launches of the list of the run.

Facts about the structure of the kernels:

- The loop of MMQ is in `mmq.cuh`, lines 954 to 987. It loads a tile through the registers to the shared memory. Then it runs `__syncthreads`, `vec_dot`, and `__syncthreads`. The code has no `cp.async` and no second buffer. The loads do not run at the same time as the matrix work [INFERENCE].
- In one `ffn_up` kernel with 2,048 tokens, each IMMA instruction has 4 `I2FP`, 4 `FMUL` and 4 `FFMA` instructions. Together they are 63 % of the instructions that the kernel runs. The line `sum += C.x[l]*dA*dB` in `mmq-vec-dot.cuh` (lines 265 to 273) can cause this [INFERENCE].
- The code in `gated-delta-net-mma.cu` (lines 403 to 416) selects the path for the full head in one case. The case is when 2 times the number of heads times the number of sequences is more than the number of SMs. Here the result is 64 and the number of SMs is 38.
- With `--kv-unified`, the flash attention time of the same request on a different slot was 1.19 to 2.25 times the time of the first request. Without `--kv-unified` it was 1.008 times. This is the lead #28495. The [entry](notebook/2026-10-31-phase3-where-the-time-goes.md#28495-the-kv-window) has the table.
- With `GGML_CUDA_DISABLE_FUSION=1`, the latency was 3.5 %, 5.6 % and 6.0 % higher for a short request, a medium request and a long request. The fusions of `b12004` thus remove time. The [entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md#the-effect-of-the-fusions-nofusion) has the table.
- The CUDA backend ran 741 passes. 739 passes ran in the mode `direct`, 2 passes captured a graph, and no pass used a graph again. The [entry of the ablation](notebook/2026-10-31-phase3-ablation-caps.md#cuda-graphs-are-not-used-nographs) gives the numbers.

The [ablation](notebook/2026-10-31-phase3-ablation-caps.md#caps-for-each-part-and-each-request) gives the cap of the MMQ kernels as 60.3 % to 61.1 % of the latency of a request. The [long version](story.md) puts these facts in the sequence of the work.

## Related documents

- [Model](model.md): the layers, the quantization, and the server.
- [Method](method.md): how we measure a change and the tests of its correctness.
- [Glossary](glossary.md): the terms of the project.
