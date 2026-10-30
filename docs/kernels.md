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

## Related documents

- [Model](model.md): the layers, the quantization, and the server.
- [Method](method.md): how we measure a change and the tests of its correctness.
- [Glossary](glossary.md): the terms of the project.
