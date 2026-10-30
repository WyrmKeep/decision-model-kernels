# Upstream inspection of llama.cpp for the decision-model kernel project

Date of the inspection: 2026-10-30. Repository: ggml-org/llama.cpp. The access was read-only. The inspection wrote no comment, no reaction, and no issue.

The baseline of the inspection is the build in production. It is the release with the tag b12004. Its commit is [`3c809f4`](https://github.com/ggml-org/llama.cpp/commit/3c809f44383bcebc318b8cddd8c6d11aaa861e9b).

The newest release has the tag **b12017**. It is a `pre-release`. Its release time is 2026-10-30T20:19:54Z. The tag points to commit [`5047748f1`](https://github.com/ggml-org/llama.cpp/commit/5047748f114564fcf271e6a648f60e2d2eae6264). This commit is also the newest commit of the `master` branch at the time of the inspection.

## 1. Read this first

- The `master` branch has 13 more commits than the tag b12004. None of them changes the CUDA code, the code that divides batches, the hybrid memory code, or the decision server.
- One commit touches a target path (section 4 gives the paths): [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc) ([#31368](https://github.com/ggml-org/llama.cpp/pull/31368)). It limits the rollback of recurrent layers to the snapshots of the last ubatch.
- That commit has no effect on the speed of this project [INFERENCE: the server does not use speculative decoding, thus it does not use rollback].
- No target has the result "Done in `master`". With that result, a rebase gets the change. A rebase from b12004 to b12017 adds no change that helps targets `a` to `j`.
- All merged changes that this document names are in b12004. The result "Done in b12004" marks these cases.

### 1.1 Upstream work that is done or in progress

- **Target `h`, done:** the chunked kernel of Gated DeltaNet for prefill on CUDA is in b12004 ([#29353](https://github.com/ggml-org/llama.cpp/pull/29353), [`62df1815b`](https://github.com/ggml-org/llama.cpp/commit/62df1815b)). Do not build it again.
- **Target `c`, done for decode:** CUDA graphs in chunks are in b12004 ([#30804](https://github.com/ggml-org/llama.cpp/pull/30804), [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d)).
- **Target `d`, done in part:** the MMQ fusion of the `gate` matrix, the `up` matrix and GLU is in b12004 ([#29948](https://github.com/ggml-org/llama.cpp/pull/29948), [`c49edf096`](https://github.com/ggml-org/llama.cpp/commit/c49edf096)). It does **not** include Q4_0, the type of the feed-forward layers in the model in production.
- **Target `d`, in progress:** [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) does the fusion of GLU with the quantization of the activation in MMQ, for all MMQ types that use Q8_1. The project can cherry-pick it.
- **Target `a`, in progress for drafts only:** [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) and [#29143](https://github.com/ggml-org/llama.cpp/pull/29143) add a smaller output head for drafts.
- **Target `e`, done:** the MMQ tile tables for each architecture and stream-k are in b12004 ([#24127](https://github.com/ggml-org/llama.cpp/pull/24127), [#22298](https://github.com/ggml-org/llama.cpp/pull/22298), [#30553](https://github.com/ggml-org/llama.cpp/pull/30553)).
- **Target `i`, in progress:** [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) removes a decode pass that is not necessary. The pass is in the method of `/v1/systemone` that uses a shared prefix in slots.
- **Target `f`, be careful:** the `cp.async` MMQ change ([#28572](https://github.com/ggml-org/llama.cpp/pull/28572)) closed after a reviewer wrote a comment. The reviewer will do different work on the kernel, and the change does not agree with it.

## 2. Summary table

Meaning of the results:

- **Done in b12004:** the change is in the tag in production. A rebase adds no change. Other persons did this work before this project.
- **In progress upstream:** an open item is in the upstream repository. Monitor the item, or cherry-pick it.
- **Rejected upstream:** a person closed an item for the change. The item tells the cause if it has one.
- **No item found:** the queries found no item for the target.

| Target | Result | Primary items | Effect on the project |
|---|---|---|---|
| `a`. Answer-token-only output head | In progress upstream, for drafts only. No item found for the answer tokens of decisions. | [#31031](https://github.com/ggml-org/llama.cpp/pull/31031), [#29143](https://github.com/ggml-org/llama.cpp/pull/29143), [#25187](https://github.com/ggml-org/llama.cpp/issues/25187), [#30445](https://github.com/ggml-org/llama.cpp/pull/30445) | The build in production calculates all 248,320 rows of the output head. The project must write the code for a head for answer tokens only. [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) gives an example that the project can follow. |
| `b`. Ragged batching for hybrid memory | No item found. | [#14217](https://github.com/ggml-org/llama.cpp/pull/14217), [#25278](https://github.com/ggml-org/llama.cpp/pull/25278), [#15636](https://github.com/ggml-org/llama.cpp/pull/15636) | `split_equal` continues to be for sequences of equal length only. Ragged batching is a new area for work. No upstream work is available for a rebase. |
| `c`. CUDA graphs for prompt batches | Decode graphs: done in b12004. Prompt batches: no item found. Graph selection by shape: in progress. | [#30804](https://github.com/ggml-org/llama.cpp/pull/30804), [#31004](https://github.com/ggml-org/llama.cpp/pull/31004), [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) | b12004 captures a graph only after it sees the same graph two times, one after the other. A prompt batch has a different shape in most passes. Then b12004 does not capture a graph for it [INFERENCE]. Monitor [#31004](https://github.com/ggml-org/llama.cpp/pull/31004). |
| `d`. Fusion of operations that wait for memory | Most of the work is in b12004. Fusion for Q4_0 feed-forward layers is missing. GLU in the quantization of the activation is in progress. Fusion of the concat for the conv state: no item found. | [#29948](https://github.com/ggml-org/llama.cpp/pull/29948), [#30489](https://github.com/ggml-org/llama.cpp/pull/30489), [#22710](https://github.com/ggml-org/llama.cpp/pull/22710), [#25930](https://github.com/ggml-org/llama.cpp/pull/25930) | The project can cherry-pick [#30489](https://github.com/ggml-org/llama.cpp/pull/30489). An extension of [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) to Q4_0 is new work. Two items for the fusion of RMS norm with the quantization of Q8_1 closed. |
| `e`. Adjustment of MMQ tiles for `sm_86` | Done in b12004 (tile tables and stream-k). A change of the Q4_0 tiles for small batches closed before a reviewer looked at it. | [#24127](https://github.com/ggml-org/llama.cpp/pull/24127), [#30553](https://github.com/ggml-org/llama.cpp/pull/30553), [#28204](https://github.com/ggml-org/llama.cpp/pull/28204), [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) | The adjustment is a change in `mmq-config-ampere.cuh`. No upstream adjustment for `sm_86` is in progress. |
| `f`. New MMQ kernels or other int8 matrix kernels | No item found for Ampere. Blackwell and Hopper have items that test new methods. One item closed. | [#28572](https://github.com/ggml-org/llama.cpp/pull/28572), [#26704](https://github.com/ggml-org/llama.cpp/pull/26704), [#28514](https://github.com/ggml-org/llama.cpp/discussions/28514) | An MMQ change with `cp.async` for `sm_86` is new work. A reviewer will do different work on this kernel. The two changes can have an overlap. |
| `g`. Megakernel or persistent kernel | No item found. | [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331), [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | No upstream work is available for this target. CUDA graphs in chunks decrease part of the launch overhead. |
| `h`. CUDA kernels for Gated DeltaNet | Done in b12004 (the chunked kernel for prefill). No item found for short prompts or for prompts with different lengths. | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353), [#30087](https://github.com/ggml-org/llama.cpp/pull/30087), [#20537](https://github.com/ggml-org/llama.cpp/pull/20537), [#22587](https://github.com/ggml-org/llama.cpp/pull/22587) | The chunked kernel can run only when a sequence has 64 tokens or more. Shorter rows use the autoregressive kernel. |
| `i`. Server API for decisions | Primary API: done in b12004. Speed repairs: in progress upstream. | [#29818](https://github.com/ggml-org/llama.cpp/pull/29818), [#30361](https://github.com/ggml-org/llama.cpp/pull/30361), [#30567](https://github.com/ggml-org/llama.cpp/pull/30567), [#30779](https://github.com/ggml-org/llama.cpp/pull/30779), [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) removes a decode pass that is not necessary after [#30567](https://github.com/ggml-org/llama.cpp/pull/30567). The project can cherry-pick it. |
| `j`. Copy of the layer state in hybrid memory | In progress upstream, in parts. No item found for the time of `seq_cp` only. | [#28841](https://github.com/ggml-org/llama.cpp/issues/28841), [#29856](https://github.com/ggml-org/llama.cpp/pull/29856), [#30137](https://github.com/ggml-org/llama.cpp/pull/30137), [#30683](https://github.com/ggml-org/llama.cpp/pull/30683), [#31391](https://github.com/ggml-org/llama.cpp/pull/31391) | In b12004, `seq_cp` for a recurrent layer makes no copy of tensor data. The gather of the layer state runs in the graph on each pass. |

## 3. Information for each target

Each table of items has one row for each item. A row shows the number of the issue or the pull request and its title as GitHub shows it. It also shows the condition, the dates, and information about the item.

The tables give each title as GitHub shows it, in code, with no change. GitHub makes some long titles shorter and ends them with `…`. A title is not STE text. It can have semicolons, contractions, or words that are not in the STE dictionary.

A merged item shows its commit hash. A range of dates shows the date when the item opened and the date when it merged or closed. The date of a merged item comes from its commit on the `master` branch. Each merged item in these tables is in b12004. A table tells you when this is not true.

### 3.a Answer-token-only output head

**Question.** Can the model calculate only the logits of the answer tokens and not the other rows of the vocabulary?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) | `Trimmed draft vocab support (1/2) - reduced draft LM head from a user supplie…` | Open | 2026-10-23 to open | Adds a smaller output head for drafts. The head keeps only the rows of the vocabulary that a list gives. It puts the result in a logit tensor with the full dimension of the vocabulary. This is the nearest upstream method. It is for drafts only. |
| [#29143](https://github.com/ggml-org/llama.cpp/pull/29143) | `feat(qwen35): support d2t draft-vocabulary trim for MTP speculative decoding` | Open | 2026-09-19 to open | Cuts the draft vocabulary for Qwen3.5 MTP. The shape of the head becomes the dimension of the embedding by the number of tokens in the draft vocabulary. It is for drafts only. |
| [#25187](https://github.com/ggml-org/llama.cpp/issues/25187) | `Research: FR-Spec-style draft-vocab trimming for native MTP speculative decod…` | Open | 2026-07-01 to open | This issue wants a draft vocabulary with a small number of tokens. The `stale` bot closed it, and it is open again. A maintainer wrote that upstream will add it subsequently. A person who wrote a comment measured the Q8_0 head with 248,320 rows at 675 MB. |
| [#26226](https://github.com/ggml-org/llama.cpp/pull/26226) | `core : support output vocab size distinct from embedding vocab size` | Open | 2026-07-28 to open | A change that lets the output vocabulary have a different number of tokens from the embedding vocabulary. It has the label `stale`. The work is not complete. |
| [#26224](https://github.com/ggml-org/llama.cpp/issues/26224) | `core: support output vocab size distinct from embedding vocab size` | Closed | 2026-07-28 to 2026-09-21 | The issue behind #26226. It closed with no merged change. |
| [#30445](https://github.com/ggml-org/llama.cpp/pull/30445) | `graph: skip the LM head for pooled embeddings` | Merged [`cb55ffe6e`](https://github.com/ggml-org/llama.cpp/commit/cb55ffe6e) | 2026-10-13 to 2026-10-13 | The graph has no output head when the program uses pooling. This shows that upstream accepts a graph without the output head. It is for pooling only. |
| [#30265](https://github.com/ggml-org/llama.cpp/pull/30265) | `server : add logit_gate to /v1/completions and /v1/chat/completions` | Open | 2026-10-10 to open | Adds `logit_gate` to `/v1/completions` and `/v1/chat/completions`. It reads all logits on the host. It then applies softmax to the logits of the tokens in `logit_gate`. It does not make the output head smaller. |
| [#29022](https://github.com/ggml-org/llama.cpp/issues/29022) | `Feature Request: Fast Tool Gating & Single-Pass Selection via Prefill Logit S…` | Open | 2026-09-17 to open | Wants the logits of some tokens after prefill. A client can then use them to select a tool in one pass. It wants the same result as target `a`. It does not want a change of the output head. |
| [#30323](https://github.com/ggml-org/llama.cpp/issues/30323) | `perf: pooled embeddings context copies full-vocab logits to host on every dec…` | Open | 2026-10-11 to open | Reports that the program moves the logits of the full vocabulary to the host when it uses pooling. It shows the time that full rows of logits use. |
| [#29487](https://github.com/ggml-org/llama.cpp/issues/29487) | `Misc. bug: Backend Top-K output buffer is still allocated at full vocabulary…` | Open | 2026-09-26 to open | Reports that the memory for the output of the backend `top-k` step continues to have the full width of the vocabulary. |
| [#27050](https://github.com/ggml-org/llama.cpp/issues/27050) | `Server: backend sampling (-bs) gives +48% throughput at 32 slots (706 -> 1046…` | Closed | 2026-08-14 to 2026-09-28 | Measures the time that the copy of all logits to the host uses when the server has many slots. Backend sampling gave +48 percent throughput at 32 slots. |

**Changes in `master` since b12004.** No commit changes the output head or the path of the logits. The only change of the graph is [`fb5997274`](https://github.com/ggml-org/llama.cpp/commit/fb5997274). The constant values of the KV cache rotation move to the GPU.

**Code in b12004.**

- The graph of `qwen35` makes the head with `build_lora_mm(model.output, ...)` for all rows. See [`src/models/qwen35.cpp:233`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/qwen35.cpp#L233).
- The decision server reads the logits with `llama_get_logits_ith`. It reads the logit of each answer token at the position of that token. See [`tools/server/server-context.cpp:2527-2535`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L2527-L2535).
- Each answer of H2O-Lightning is one token. See [`tools/server/server-decision.cpp:178-193`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-decision.cpp#L178-L193).
- The Q8_0 `output.weight` has 248,320 rows of 2,560 values. It uses about 675 MB of memory. This number is from a comment on [#25187](https://github.com/ggml-org/llama.cpp/issues/25187).
- [INFERENCE] The largest bandwidth of the `RTX 3060 Ti` is 448 GB/s. To read 675 MB, the GPU uses 1.5 ms or more in each pass that has output rows.

**Result.** In progress upstream, for drafts only. No item found for an output head for the answer tokens of decisions.

**Effect on the project.** The method in [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) puts the rows of the tokens in a list into a small head. It then puts the result into a full tensor with the value `-inf` in the other positions.
A version for decisions can give only the values of the answer tokens, with no full tensor [INFERENCE: this version has no test].
[#30445](https://github.com/ggml-org/llama.cpp/pull/30445) shows that upstream accepts a graph without the output head for pooling. Monitor [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) because it touches `llama-context.cpp`, `llama-graph.cpp` and `llama.h`.
### 3.b Ragged batching for hybrid memory

**Question.** Can one ubatch hold some sequences of different lengths for a hybrid model?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#14217](https://github.com/ggml-org/llama.cpp/pull/14217) | `ubatch : new splitting logic` | Merged [`4c9fdfbe1`](https://github.com/ggml-org/llama.cpp/commit/4c9fdfbe1) | 2025-06-16 to 2025-06-20 | The pull request adds the code that divides a batch into ubatches. The first commit of `split_equal` in `llama-batch.cpp` is in this pull request. |
| [#14511](https://github.com/ggml-org/llama.cpp/pull/14511) | `batch : add optional for sequential equal split` | Merged [`67d1ef23c`](https://github.com/ggml-org/llama.cpp/commit/67d1ef23c) | 2025-07-03 to 2025-07-04 | The pull request adds the optional sequential equal split. |
| [#16442](https://github.com/ggml-org/llama.cpp/pull/16442) | `memory : use sequential equal splits for recurrent modules` | Merged [`0123ff38f`](https://github.com/ggml-org/llama.cpp/commit/0123ff38f) | 2025-10-06 to 2025-10-07 | The pull request makes the memory code for recurrent layers use sequential equal splits. |
| [#25278](https://github.com/ggml-org/llama.cpp/pull/25278) | `llama-batch: enable parallel sequences for partial rollback` | Merged [`230ea9d21`](https://github.com/ggml-org/llama.cpp/commit/230ea9d21) | 2026-07-03 to 2026-07-08 | The pull request adds `n_keep_tail` to `split_equal` for a rollback of some of the tokens. It is the newest change to the code that divides batches. The rule that all sequences have the same length stays. |
| [#25004](https://github.com/ggml-org/llama.cpp/pull/25004) | `recurrent : support equal splits for recurrent-state rollback` | Open | 2026-06-25 to open | The pull request adds equal splits for the rollback of the layer state. #25278 replaces a part of it. |
| [#15636](https://github.com/ggml-org/llama.cpp/pull/15636) | `` batch : add `pad_equal` [RFC] `` | Open, as a draft pull request | 2025-08-28 to open | The pull request adds padding to the sequences to give them the same length. The author wants comments before the work is complete. The work has not continued for a long time. It is the only proposal for padding that we found. |
| [#30402](https://github.com/ggml-org/llama.cpp/issues/30402) | `Eval bug: vllm-d3 answers change when a shorter request shares the batch (spl…` | Open | 2026-10-12 to open | The issue reports that the answers of a decision model with non-causal attention change when a shorter request is in the same batch. `split_equal` divides the prompts. |
| [#30408](https://github.com/ggml-org/llama.cpp/pull/30408) | `model: fix vllm d3 multi seq case` | Merged [`cbe28bb3c`](https://github.com/ggml-org/llama.cpp/commit/cbe28bb3c) | 2026-10-12 to 2026-10-13 | The pull request repairs a related case. In this case a hybrid model with non-causal attention has more than one sequence. The case is behind #30402. The pull request changes the batch code and the memory code. |
| [#25840](https://github.com/ggml-org/llama.cpp/issues/25840) | `Eval bug: server -np 3 decode drops ~4.4x (166 -> 38 t/s) whenever the two bu…` | Closed | 2026-07-17 to 2026-09-03 | With `-np 3`, the default is the split KV cache. If a slot has no request between slots that have a request, the server uses two ubatches in each step. The flag `--kv-unified` prevents the problem. The flags in production include `--kv-unified`. |
| [#28495](https://github.com/ggml-org/llama.cpp/issues/28495) | `Eval bug: with -np 2 and --kv-unified, prompt processing drops 42-54% from th…` | Open | 2026-09-06 to open | The issue reports that the speed of prefill is 42 to 54 percent lower on HIP with `-np 2 --kv-unified`. A person who has a V100 GPU saw the same on CUDA. The flash attention kernels ignore only the last part of the attention mask. This is important for the flags in production. |
| [#20006](https://github.com/ggml-org/llama.cpp/issues/20006) | `Eval bug: Qwen3.5-27B batched performance much lower than expected` | Closed | 2026-03-01 to 2026-04-27 | The issue reports a slow speed for batches of Qwen3.5 on Vulkan. A person who wrote a comment found that the gather and copy operations for the layer state continue to use much time after a repair. |

**Changes in `master` since b12004.** No commit changes `src/llama-batch.cpp` or `src/llama-memory-hybrid.cpp`.

**Code in b12004.**

- The hybrid memory uses `balloc.split_equal(...)`. See [`src/llama-memory-hybrid.cpp:90`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-memory-hybrid.cpp#L90).
- `split_equal` does the equal split. In each ubatch all sequences have the same number of tokens. The shortest sequence sets the number. See [`src/llama-batch.cpp:619-712`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-batch.cpp#L619-L712).
- The GDN MMA kernel runs only when a sequence has 64 or more tokens. If not, the autoregressive kernel runs. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391).
- The grid of the kernel has one thread block for each head and each sequence. Thus each sequence has the same length. The kernel has no structure for the lengths of sequences, for example `cu_seqlens`.

**Result.** We found no item that is a proposal for ragged batching with recurrent layers. The only proposal for padding is [#15636](https://github.com/ggml-org/llama.cpp/pull/15636). The work has not continued for a long time.

**Effect on the project.** All work on ragged batching is new work. The proposal for padding shows an alternative with less work. The flag `--kv-unified` is in production. It prevents the `-np 3` problem of [#25840](https://github.com/ggml-org/llama.cpp/issues/25840), in which a slot has no request between slots that have a request.

### 3.c CUDA graphs for prompt batches

**Question.** Does upstream decrease the launch overhead for prompt batches, for example with CUDA graphs or `GGML_CUDA_GRAPH_OPT`?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | `CUDA: chunk graphs to amortize graph launch` | Merged [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d) | 2026-10-19 to 2026-10-24 | The pull request divides CUDA graphs into chunks. This divides the time of one launch between many kernels. The speed of decode increases by 1 to 2 percent. This change is in b12004. |
| [#28549](https://github.com/ggml-org/llama.cpp/pull/28549) | `Enable CUDA graph for MTP draft` | Merged [`2f3fd0252`](https://github.com/ggml-org/llama.cpp/commit/2f3fd0252) | 2026-09-07 to 2026-09-16 | The pull request lets the program use a CUDA graph for the MTP draft. It uses an arena that only the draft uses, and stable cache keys. |
| [#26802](https://github.com/ggml-org/llama.cpp/pull/26802) | `CUDA: only disable CUDA graphs when mul_mat_id actually needs a stream sync` | Merged [`ebb546b7e`](https://github.com/ggml-org/llama.cpp/commit/ebb546b7e) | 2026-08-09 to 2026-08-11 | The pull request makes the program stop CUDA graphs only when `mul_mat_id` must have a stream synchronization. |
| [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) | `CUDA: Recapture graph after memory pool flush` | Merged [`13786fbb3`](https://github.com/ggml-org/llama.cpp/commit/13786fbb3) | 2026-08-08 to 2026-10-20 | The pull request captures the graph again after the program flushes the memory pool. |
| [#25749](https://github.com/ggml-org/llama.cpp/pull/25749) | `Enable CUDA graphs on Volta+Turing` | Merged [`3f08ef2c5`](https://github.com/ggml-org/llama.cpp/commit/3f08ef2c5) | 2026-07-16 to 2026-07-16 | The pull request lets the program use CUDA graphs on Volta and Turing. |
| [#16991](https://github.com/ggml-org/llama.cpp/pull/16991) | `CUDA: add stream-based concurrency` | Merged [`c7af376c2`](https://github.com/ggml-org/llama.cpp/commit/c7af376c2) | 2025-11-04 to 2025-11-30 | The pull request lets the program run kernels on different CUDA streams at the same time. It is the source of `GGML_CUDA_GRAPH_OPT`. |
| [#22522](https://github.com/ggml-org/llama.cpp/pull/22522) | `Programmatic Dependent Launch (PDL) for more performance on newer NVIDIA GPUs…` | Merged [`e94722822`](https://github.com/ggml-org/llama.cpp/commit/e94722822) | 2026-04-29 to 2026-05-20 | The pull request adds programmatic dependent launch. It helps only a GPU of the Hopper type or of a newer type. The speed does not increase on Ampere. |
| [#31004](https://github.com/ggml-org/llama.cpp/pull/31004) | `Implement graph shape keying to prevent graph thrashing` | Open, as a draft pull request | 2026-10-22 to open | The pull request puts the shape of a graph in the cache key of the CUDA graph. Graphs with different shapes then do not replace each other in the cache. A reviewer wants the repair at the llama.cpp level. The author agreed to examine it. |
| [#28666](https://github.com/ggml-org/llama.cpp/pull/28666) | `ggml-cuda: add first-node extents to the CUDA graph cache key` | Closed, not merged | 2026-09-09 to 2026-09-09 | The pull request tried to add the dimensions of the first node to the cache key of the graph. It comes before #31004. The author closed it. #31004 continues the work. |
| [#28652](https://github.com/ggml-org/llama.cpp/issues/28652) | `Misc. bug: CUDA: alternating variants sharing one graph cache key cause warmu…` | Closed | 2026-09-09 to 2026-10-25 | The issue reports that different graphs with the same cache key replace each other again and again in the cache of CUDA graphs. |
| [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) | `Feature Request: CUDA graphs for multi-slot decode via shape-stable padded u` | Closed | 2026-08-13 to 2026-10-06 | The issue is a proposal for CUDA graphs in decode with more than one slot, with ubatches that have padding. A test on a 4090 GPU showed that the speed increases by 2 to 3 percent. |
| [#21897](https://github.com/ggml-org/llama.cpp/pull/21897) | `ggml-cuda: enable concurrent streams for linear attention` | Open | 2026-04-14 to open | The pull request runs the kernels of linear attention on parallel CUDA streams with `GGML_CUDA_GRAPH_OPT=1`. A subsequent comment reports that the speed increases only in decode. |
| [#29796](https://github.com/ggml-org/llama.cpp/pull/29796) | `cuda: skip redundant stream synchronization` | Open, as a draft pull request | 2026-10-01 to open | The pull request removes a stream synchronization that is not necessary. |
| [#16548](https://github.com/ggml-org/llama.cpp/pull/16548) | `Implement and use cuda graph plans` | Open | 2025-10-13 to open | The pull request is a proposal for graph plans for CUDA graphs. The work has not continued for a long time. |
| [#11867](https://github.com/ggml-org/llama.cpp/pull/11867) | `Overlap CUDA graph building and processing to minimize GPU idle time and impr…` | Open | 2025-02-14 to open | The pull request is a proposal to make a graph while the GPU does the work of a different graph. The work has not continued for a long time. |

**Changes in `master` since b12004.** No commit changes `ggml/src/ggml-cuda`.

**Code in b12004.**

- The code has no condition on the number of tokens in the batch for CUDA graphs. The GPU type must be Volta or a newer type. See [`ggml/src/ggml-cuda/ggml-cuda.cu:4983-4996`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L4983-L4996).
- The program captures a graph only after two runs in sequence in which the properties of the nodes do not change. If a node changes, the program must wait for two new runs. See [`ggml/src/ggml-cuda/ggml-cuda.cu:5016-5020`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L5016-L5020).
- The check for a graph change is in [`ggml/src/ggml-cuda/ggml-cuda.cu:2898-2940`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L2898-L2940). The cache key of the graph is the first node of the graph.
- llama.cpp uses the graph of the previous ubatch again only if the new ubatch has the same properties. See [`src/llama-context.cpp:1539`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-context.cpp#L1539).
- `GGML_CUDA_GRAPH_OPT=1` lets the program use more than one CUDA stream when three nodes use the output of `attn_norm`. It counts only nodes with one row or less. Thus it has an effect only on decode.

**Result.** The release b12004 has CUDA graphs for decode. The work on the shape in the cache key is in progress ([#31004](https://github.com/ggml-org/llama.cpp/pull/31004)). We found no item for CUDA graphs for prompt batches.

**Effect on the project.** [INFERENCE] A prompt batch that changes shape does not frequently give two runs in sequence with the same nodes. Thus the program does not frequently use graphs for it.

A prompt shape with padding, or with a small number of shapes, is new work. [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) is a proposal for padding in decode. A test on a 4090 GPU showed that the speed increases by only 2 to 3 percent.

Monitor [#31004](https://github.com/ggml-org/llama.cpp/pull/31004). A reviewer wrote that the program must use an arena for each shape at the llama.cpp level.

### 3.d Fusion of operations that wait for memory

**Question.** Does upstream fuse `rms_norm` with `mul`, SwiGLU, activation quantization, add with `rms_norm`, or the concat of the conv state with `ssm_conv`?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) | `CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up rows` | Merged [`c49edf096`](https://github.com/ggml-org/llama.cpp/commit/c49edf096) | 2026-10-04 to 2026-10-25 | The pull request fuses the `gate` and `up` operations and GLU of the feed-forward layer in MMQ. Q4_0 is not in the list of types of the fusion. See `mmq.cuh:437-440`. |
| [#31188](https://github.com/ggml-org/llama.cpp/pull/31188) | `CUDA: fuse MoE gate/up matmuls and GLU in MMQ` | Merged [`f5222f25c`](https://github.com/ggml-org/llama.cpp/commit/f5222f25c) | 2026-10-26 to 2026-10-28 | The pull request makes the same MMQ fusion for the `gate` and `up` operations of MoE layers. |
| [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) | `ggml-cuda: fuse GLU into the MMQ Q8_1 activation quantization` | Open | 2026-10-14 to open | The pull request fuses GLU into the quantization of the activation in MMQ. The speed of prefill increases by 1.0 to 1.8 percent on a 4090 GPU with Q4_K models. It is one commit. A rebase on `master` is from 2026-10-30. |
| [#22478](https://github.com/ggml-org/llama.cpp/pull/22478) | `CUDA: fuse SSM_CONV + ADD(bias) + SILU` | Merged [`098705a29`](https://github.com/ggml-org/llama.cpp/commit/098705a29) | 2026-04-28 to 2026-04-30 | The pull request fuses `SSM_CONV`, the operation that adds a bias, and SiLU in CUDA. |
| [#20128](https://github.com/ggml-org/llama.cpp/pull/20128) | `CUDA: use shared mem for ssm_conv` | Merged [`1e38a7a6f`](https://github.com/ggml-org/llama.cpp/commit/1e38a7a6f) | 2026-03-05 to 2026-03-06 | The pull request uses shared memory in the CUDA kernel for `ssm_conv`. |
| [#15631](https://github.com/ggml-org/llama.cpp/pull/15631) | `CUDA: fuse adds, fuse add with rms norm` | Merged [`009b709d6`](https://github.com/ggml-org/llama.cpp/commit/009b709d6) | 2025-08-28 to 2025-08-29 | The pull request fuses `ADD` with RMS norm. |
| [#26767](https://github.com/ggml-org/llama.cpp/pull/26767) | `CUDA: fuse rms_norm + mul + rope (+ view + set_rows)` | Merged [`687e77892`](https://github.com/ggml-org/llama.cpp/commit/687e77892) | 2026-08-08 to 2026-08-09 | The pull request fuses RMS norm, `mul` and RoPE. |
| [#29393](https://github.com/ggml-org/llama.cpp/pull/29393) | `CUDA: fuse RMS_NORM + SCALE into one kernel` | Merged [`1ab7e5ad2`](https://github.com/ggml-org/llama.cpp/commit/1ab7e5ad2) | 2026-09-24 to 2026-09-25 | The pull request fuses RMS norm and `scale`. |
| [#25441](https://github.com/ggml-org/llama.cpp/pull/25441) | `CUDA: dedup MoE gate/up activation quantization` | Merged [`5839ba352`](https://github.com/ggml-org/llama.cpp/commit/5839ba352) | 2026-07-08 to 2026-07-16 | The pull request quantizes the activation only one time for the `gate` and `up` operations of MoE layers. |
| [#23940](https://github.com/ggml-org/llama.cpp/pull/23940) | `Remove redundant CUDA copies after gated_delta_net.` | Merged [`5a460dea9`](https://github.com/ggml-org/llama.cpp/commit/5a460dea9) | 2026-05-31 to 2026-07-03 | The pull request removes copies after the GDN operation. |
| [#29807](https://github.com/ggml-org/llama.cpp/pull/29807) | `CUDA: Remove redundant CUDA copies after SSM_SCAN` | Merged [`50e3e3e48`](https://github.com/ggml-org/llama.cpp/commit/50e3e3e48) | 2026-10-01 to 2026-10-09 | The pull request removes copies after `SSM_SCAN`. |
| [#22710](https://github.com/ggml-org/llama.cpp/pull/22710) | `Fuse rms_norm, mul, quantize_q8_1` | Closed, not merged | 2026-05-05 to 2026-08-10 | The pull request fuses RMS norm, `mul` and the quantization to Q8_1. The author closed it. Maintainers wrote that the change has many parts that connect with each other. |
| [#25930](https://github.com/ggml-org/llama.cpp/pull/25930) | `CUDA: fuse q8_1 quantization into the producer rms_norm` | Closed, not merged | 2026-07-20 to 2026-07-21 | The pull request fuses the Q8_1 quantization into the RMS norm that makes the data for it. A maintainer closed it. The maintainer wrote that the work is not careful and that a better method is necessary. |
| [#29187](https://github.com/ggml-org/llama.cpp/pull/29187) | `CUDA: Fuse GDN alpha/beta projections` | Open, as a draft pull request | 2026-09-20 to open | The pull request fuses the two operations that multiply by the model weights `alpha` and `beta` of GDN. After #29633 the speed increases by only 2 to 4 percent. This is less than before. The work stops at this time. |
| [#20520](https://github.com/ggml-org/llama.cpp/pull/20520) | `CUDA: add new op template rms_norm_f32_vec4 for float4 vectorized load/store` | Open | 2026-03-13 to open | The pull request vectorizes the RMS norm kernel for reads and writes. It has no effect on the speed of the full server. |
| [#26468](https://github.com/ggml-org/llama.cpp/pull/26468) | `ggml : fuse soft_max sweeps into fewer passes` | Open | 2026-08-02 to open | The pull request fuses scans of the data in softmax to decrease the number of scans. |
| [#26486](https://github.com/ggml-org/llama.cpp/pull/26486) | `ggml : vectorize rms_norm reduce and fuse the scale write` | Open, as a draft pull request | 2026-08-03 to open | The pull request vectorizes the sum in the RMS norm kernel. |
| [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | `rfc: Memory eliding fusions` | Open, as a draft pull request | 2026-09-21 to open | The author wants comments on fusions that remove memory reads and writes. |
| [#15602](https://github.com/ggml-org/llama.cpp/issues/15602) | `Feature Request: Repeated Unecessary Activation Quantization Ops` | Open | 2025-08-26 to open | The issue is a proposal to prevent the quantization of the same activation again and again. |
| [#31179](https://github.com/ggml-org/llama.cpp/pull/31179) | `ggml-cpu : rewrite concat as one row kernel for all types` | Merged [`4d71b42fd`](https://github.com/ggml-org/llama.cpp/commit/4d71b42fd) | 2026-10-26 to 2026-10-26 | The pull request makes the concat of the CPU one kernel for rows. It is for the CPU only. |
| [#30149](https://github.com/ggml-org/llama.cpp/pull/30149) | `vulkan: add tiled concat for transposed inputs` | Closed, not merged | 2026-10-08 to 2026-10-19 | The pull request adds a concat with tiles on Vulkan for inputs that the program transposes. It is not for CUDA. |
| [#28303](https://github.com/ggml-org/llama.cpp/pull/28303) | `HIP : add shared-memory tiled transpose fast path for F32 concat on RDNA3.5` | Open | 2026-09-03 to open | The pull request adds a transpose with tiles on HIP for an F32 concat on RDNA3.5. It is not for CUDA. |

**Changes in `master` since b12004.** No commit changes `norm.cu`, `quantize.cu` or `ggml-cuda.cu`.

**Code in b12004.**

- The MMQ fusion of the `gate` and `up` operations and GLU has a list of types. The list has only Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS. See [`ggml/src/ggml-cuda/mmq.cuh:437-440`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L437-L440). Q4_0 is missing.
- The feed-forward layers of the model in production use Q4_0. They are not MoE layers. Thus the model uses the kernels without fusion for them. [INFERENCE: the type is from the method that made the model file. This inspection did not read the GGUF file.]
- [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) fuses GLU into the Q8_1 quantization of the activation for the `down` operation. Its description tells that all MMQ types with Q8_1 can use it. The speed of prefill increases by 1.0 to 1.8 percent on a 4090 GPU with Q4_K models. We did not measure Q4_0 on `sm_86`.
- The concat of the conv state is `ggml_concat(conv_states, qkv_mixed, 0)`. See [`src/models/delta-net-base.cpp:469`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/delta-net-base.cpp#L469). The operation `ssm_conv` follows in [`src/models/qwen35.cpp:403`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/qwen35.cpp#L403).
- CUDA has the code `concat_non_cont`. See [`ggml/src/ggml-cuda/concat.cu:85`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/concat.cu#L85). CUDA fuses `SSM_CONV`, `ADD` and SiLU ([#22478](https://github.com/ggml-org/llama.cpp/pull/22478)), but not the concat.

**Result.** Most of the fusions are in b12004. The fusion of GLU into the quantization ([#30489](https://github.com/ggml-org/llama.cpp/pull/30489)) is in progress. Two pull requests ([#22710](https://github.com/ggml-org/llama.cpp/pull/22710), [#25930](https://github.com/ggml-org/llama.cpp/pull/25930)) are proposals for the fusion of RMS norm with the quantization. Upstream did not merge them. We found no item for the fusion of the concat of the conv state in CUDA.

**Effect on the project.** Cherry-pick [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) and measure it on Q4_0. The code that lets the [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) fusion read Q4_0 weights is new work.

A reviewer wrote that the fusion of RMS norm with the quantization is not new. The reviewer also wrote that it must have a better method. A new pull request must show how the backend changes the graph.

Only CPU, HIP, Vulkan and Hexagon have items for the concat of the conv state (for example [#31179](https://github.com/ggml-org/llama.cpp/pull/31179)).
### 3.e Adjustment of MMQ tiles for `sm_86`

**Question.** Does upstream tune the MMQ tiles, `mmq_x`, `mmq_y` or stream-k for `sm_86` with `Q4_0` or `Q8_0`?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#24127](https://github.com/ggml-org/llama.cpp/pull/24127) | `CUDA: refactor MMQ kernel configuration` | Merged [`6eddde06a`](https://github.com/ggml-org/llama.cpp/commit/6eddde06a) | 2026-06-04 to 2026-07-13 | The pull request moves the tile dimensions of MMQ into tables for each architecture. The Ampere tables for `Q4_0` and `Q8_0` are in `mmq-config-ampere.cuh`. |
| [#22298](https://github.com/ggml-org/llama.cpp/pull/22298) | `CUDA: reduce MMQ stream-k overhead` | Merged [`9725a313b`](https://github.com/ggml-org/llama.cpp/commit/9725a313b) | 2026-04-23 to 2026-04-25 | The pull request decreases the overhead of stream-k in MMQ. |
| [#30553](https://github.com/ggml-org/llama.cpp/pull/30553) | `cuda: cap MMQ Stream-K blocks at available work` | Merged [`4c1004c49`](https://github.com/ggml-org/llama.cpp/commit/4c1004c49) | 2026-10-14 to 2026-10-19 | The pull request limits the number of stream-k thread blocks to the work that is available. |
| [#26079](https://github.com/ggml-org/llama.cpp/pull/26079) | `CUDA: adding switch points per HW and quant type to tune the mvq->MMQ decode…` | Merged [`2b5621094`](https://github.com/ggml-org/llama.cpp/commit/2b5621094) | 2026-07-24 to 2026-08-20 | The pull request sets the switch points from MMVQ to MMQ for each GPU and each type of quantization. |
| [#28285](https://github.com/ggml-org/llama.cpp/pull/28285) | `tune MMVQ to MMQ crossover for SM87` | Merged [`8c1a25166`](https://github.com/ggml-org/llama.cpp/commit/8c1a25166) | 2026-09-03 to 2026-09-03 | The pull request tunes the switch point from MMVQ to MMQ for `SM87`. |
| [#28912](https://github.com/ggml-org/llama.cpp/pull/28912) | `tune MMVQ to MMQ crossover for SM70 (Volta)` | Merged [`68d9053af`](https://github.com/ggml-org/llama.cpp/commit/68d9053af) | 2026-09-14 to 2026-09-21 | The pull request tunes the switch point from MMVQ to MMQ for `SM70`. |
| [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) | `Feature Request: Add an sm_86 entry to the MMVQ cutoff table (Q4_0 crosses at…` | Open | 2026-08-31 to open | The issue wants an `sm_86` entry in the table of switch points. It shows that the best switch point changes with the number of SMs and not with the compute capability. |
| [#28204](https://github.com/ggml-org/llama.cpp/pull/28204) | `cuda : MMQ q4_0 small-batch tile config for Ampere+` | Closed, not merged | 2026-09-01 to 2026-09-10 | The pull request adds a tile for small batches of `Q4_0` on Ampere and on newer types of GPU. The author closed it before a reviewer read it. |
| [#29330](https://github.com/ggml-org/llama.cpp/pull/29330) | `cuda: optimize MMQ configuration for prefill phase with larger tile sizes` | Closed, not merged | 2026-09-23 to 2026-09-23 | The pull request uses larger tiles for prefill. A person closed it because it did not follow the template. |
| [#21849](https://github.com/ggml-org/llama.cpp/pull/21849) | `ggml-cuda: make mul_mat_q tile selection tunable per-type and per-arch` | Open | 2026-04-13 to open | The pull request lets a person select the tile for each type and each architecture. The pull request #24127 replaces it. |
| [#30493](https://github.com/ggml-org/llama.cpp/pull/30493) | `ggml-cuda: vectorize the MMQ activation tile copy to int2` | Open, as a draft pull request | 2026-10-14 to open | The pull request copies the activation tile of MMQ in `int2` units. The speed decreases by a maximum of 8 percent on a 4090 GPU. |
| [#30235](https://github.com/ggml-org/llama.cpp/issues/30235) | `CUDA: Pascal DP4A MMQ config uses I=64 for Q8_0, I=128 gives +12% pp512 on sm…` | Open | 2026-10-09 to open | The issue reports that `I=128` for `Q8_0` gives 12 percent more speed on a Pascal GPU. |
| [#30797](https://github.com/ggml-org/llama.cpp/pull/30797) | `ggml-cuda: use MMA for Q5_K, Q4_K, Q4_1, Q5_1, IQ1_S and Q2_K J128 minimum co…` | Open | 2026-10-19 to open | The pull request adds an MMA path for more quantization types. `Q4_0` is not one of the types. |

**Changes in `master` since b12004.** No commit touches the files `mmq*`.

**Code in b12004.**

- The tile tables are in `mmq-config-ampere.cuh`, a file of 383 lines. The pull request [#24127](https://github.com/ggml-org/llama.cpp/pull/24127) changed the names `mmq_x` and `mmq_y` to `J` and `I`.
- The table macro is `CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)`. See [`ggml/src/ggml-cuda/mmq.cuh:208`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L208).
- `Q4_0` and `Q8_0` on Ampere use 256 threads, occupancy 1 and `I` of 128. `J` goes from 8 to 128 in steps of 8. Stream-k is on.
- MMQ reads the A tiles with `ldmatrix` but does not use `cp.async`. Only the flash-attention code includes `cp-async.cuh`.
- On `sm_86`, MMVQ is for batches with a maximum of 8 columns (`MMVQ_MAX_BATCH_SIZE`). The table of switch points has no `sm_86` entry. See the issue [#28090](https://github.com/ggml-org/llama.cpp/issues/28090).

**Result.** Done in b12004 for the tile tables and stream-k. The pull request [#28204](https://github.com/ggml-org/llama.cpp/pull/28204) adds a small-batch tile for `Q4_0`. Its author closed it before a reviewer read it.

**Effect on the project.** To tune for `sm_86`, you change the Ampere table. No open item in upstream does the same work.

The information in the issue [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) shows that the best switch point changes with the GPU and its clock. The `RTX 3060 Ti` has a power limit. Its best switch point can be the same as the one of the A10 with a clock set to one value. Because of this, a test of many switch points on this GPU is a good next step.

### 3.f New MMQ kernels or other int8 matrix kernels

**Question.** Does upstream have an item for new MMQ kernels, or for an int8 matrix kernel that uses CUTLASS, cuBLASLt, `ldmatrix` or `cp.async`?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#28572](https://github.com/ggml-org/llama.cpp/pull/28572) | `ggml-cuda: pipeline NVFP4 MMQ tile loads with cp.async and TMA on Blackwell (…` | Closed, not merged | 2026-09-07 to 2026-09-17 | The pull request reads `NVFP4` tiles with `cp.async` and TMA in a pipeline. It gives 14 percent more speed on Blackwell. A maintainer closed it and wrote that the maintainer wants to make a different change to the kernel. |
| [#26704](https://github.com/ggml-org/llama.cpp/pull/26704) | `CUDA: Add experimental SM120 CUTLASS MoE prefill for MXFP4 and NVFP4` | Open | 2026-08-07 to open | The pull request adds a prefill path with CUTLASS for `MXFP4` and `NVFP4` on `SM120`. The path is not stable. It is only for Blackwell. |
| [#28514](https://github.com/ggml-org/llama.cpp/discussions/28514) | `CUDA: NVFP4 prefill on Blackwell leaves ~1.45x performance on the table. Comp…` | Open | 2026-09-06 to open | The discussion includes code that works as a first test. `NVFP4` prefill on Blackwell can be about 1.45 times faster. |
| [#24777](https://github.com/ggml-org/llama.cpp/pull/24777) | `cuda: opt-in Hopper (sm_90a) wgmma path for Q1_0 prefill` | Closed, not merged | 2026-06-18 to 2026-06-19 | The pull request adds a `wgmma` path on Hopper for `Q1_0` prefill. It is not for Ampere. |
| [#29699](https://github.com/ggml-org/llama.cpp/pull/29699) | `` musa: opt-in muDNN INT8 `MUL_MAT`, a work-gated decode path, and vectorized `… `` | Closed, not merged | 2026-09-30 to 2026-09-30 | The pull request adds an INT8 path with muDNN for MUSA. The path is not for an NVIDIA GPU. |
| [#7860](https://github.com/ggml-org/llama.cpp/pull/7860) | `CUDA: int8 tensor cores for MMQ (q4_K, q5_K, q6_K)` | Merged [`bdcb8f422`](https://github.com/ggml-org/llama.cpp/commit/bdcb8f422) | 2024-06-10 to 2024-06-11 | The pull request added int8 tensor cores for MMQ. The MMQ kernel in b12004 comes from it. |

**Changes in `master` since b12004.** No commit touches the files `mmq*` or the code that uses cuBLAS.

**Code in b12004.**

- The CUDA backend uses the cuBLASLt library in its build but has no int8 use of cuBLASLt. The cuBLAS operations are GemmEx and its variants for batches. See [`ggml/src/ggml-cuda/ggml-cuda.cu:1697-1750`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L1697-L1750).
- No source file includes a file from CUTLASS.
- Only `fattn-mma-f16.cuh` and `cp-async.cuh` have `cp.async`.

**Result.** No item found for Ampere or for `Q4_0`. Upstream rejected the `cp.async` change for Blackwell ([#28572](https://github.com/ggml-org/llama.cpp/pull/28572)). A reviewer wrote that it adds too much technical debt and that the reviewer wants to make a different change to the kernel.
The CUTLASS pull request [#26704](https://github.com/ggml-org/llama.cpp/pull/26704) is open. It is only for `SM120`.

**Effect on the project.** A `cp.async` pipeline for MMQ on `sm_86` is new work. A maintainer of the kernel wants to change it. Because of this, a local patch can have an overlap with subsequent work in upstream. Keep the patch small. Change only the MMQ files.

A reviewer told the CUTLASS author to remove the `MXFP4` code and the `async` load of repacked tensors. Reviewers can reject large structural changes.

### 3.g Megakernel or persistent kernel

**Question.** Does upstream have a megakernel or a persistent kernel for CUDA decode or prefill?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331) | `Experiment: tile-level task DAG executor for batch=1 CPU decode (+13.7% tg on…` | Open | 2026-08-18 to open | The discussion gives a test of a method that divides CPU decode into tile tasks. A task starts when all tasks before it have completed. It gives 13.7 percent more speed on one Apple device. It is not for CUDA. |
| [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | `CUDA: chunk graphs to amortize graph launch` | Merged [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d) | 2026-10-19 to 2026-10-24 | The pull request divides each CUDA graph into chunks. It is the CUDA work that is nearest to a megakernel. It decreases the time between launches and is not a megakernel. |
| [#22522](https://github.com/ggml-org/llama.cpp/pull/22522) | `Programmatic Dependent Launch (PDL) for more performance on newer NVIDIA GPUs…` | Merged [`e94722822`](https://github.com/ggml-org/llama.cpp/commit/e94722822) | 2026-04-29 to 2026-05-20 | The pull request adds programmatic dependent launch. It is only for Hopper and for newer types of GPU. |
| [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | `rfc: Memory eliding fusions` | Open, as a draft pull request | 2026-09-21 to open | The pull request is for comments from other persons on fusions that remove memory writes and reads. It is not for a persistent kernel. |
| [#29187](https://github.com/ggml-org/llama.cpp/pull/29187) | `CUDA: Fuse GDN alpha/beta projections` | Open, as a draft pull request | 2026-09-20 to open | A reviewer wrote that the author can use NVRTC to make the fusion automatically. Some short comments followed about the work that the change must do. |

**Changes in `master` since b12004.** No commit touches CUDA code.

**Result.** No item found. Thirteen queries were about megakernels, persistent kernels and kernels for a full model layer. They are `g01` to `g08` in `queries.tsv` and `G01` to `G05` in `queries2.tsv`. None of them gave a CUDA item for a megakernel or a persistent kernel. The items that the word `persistent` found are about caches of MoE experts and prompt caches.

The items in the table have only a loose relation to a megakernel. The discussion [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331) is about the CPU. The pull request [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) decreases the time between launches with chunks of CUDA graphs and is not a megakernel.

**Effect on the project.** There is no previous work that this project can use, and no previous work that has an overlap with it. Chunks of CUDA graphs remove a part of the launch overhead in decode. b12004 has them.

### 3.h CUDA kernels for Gated DeltaNet

**Question.** Which CUDA work does upstream have for Gated DeltaNet in prefill, and for prompts that are short or have different lengths?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#19504](https://github.com/ggml-org/llama.cpp/pull/19504) | `ggml: add GATED_DELTA_NET op` | Merged [`c5a778891`](https://github.com/ggml-org/llama.cpp/commit/c5a778891) | 2026-02-11 to 2026-03-07 | The pull request added the `GATED_DELTA_NET` operation. |
| [#20340](https://github.com/ggml-org/llama.cpp/pull/20340) | `llama : enable chunked fused GDN path` | Merged [`d28961d81`](https://github.com/ggml-org/llama.cpp/commit/d28961d81) | 2026-03-10 to 2026-03-11 | The pull request turned on the chunked kernel with fusion in the graph. |
| [#20537](https://github.com/ggml-org/llama.cpp/pull/20537) | `CUDA: GDN hide memory latency` | Merged [`34818ea6c`](https://github.com/ggml-org/llama.cpp/commit/34818ea6c) | 2026-03-14 to 2026-03-16 | The pull request lets the CUDA GDN kernel do other work while it waits for memory. |
| [#30087](https://github.com/ggml-org/llama.cpp/pull/30087) | `ggml-cuda: assign four GDN state columns per warp` | Merged [`24e41838e`](https://github.com/ggml-org/llama.cpp/commit/24e41838e) | 2026-10-07 to 2026-10-08 | The pull request gives four columns of the layer state to each warp. |
| [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) | `cuda, hip: GDN chunked kernel` | Merged [`62df1815b`](https://github.com/ggml-org/llama.cpp/commit/62df1815b) | 2026-09-24 to 2026-10-14 | The pull request adds the chunked kernel for CUDA in `gated-delta-net-mma.cu`. The prefill speed is 3 to 6 percent higher on a dense model and 11 to 21 percent higher on an MoE model. |
| [#23940](https://github.com/ggml-org/llama.cpp/pull/23940) | `Remove redundant CUDA copies after gated_delta_net.` | Merged [`5a460dea9`](https://github.com/ggml-org/llama.cpp/commit/5a460dea9) | 2026-05-31 to 2026-07-03 | The pull request removes the copies after the GDN operation. |
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | `CUDA: Support of GDN chunked kernel for prefill` | Closed, not merged | 2026-07-22 to 2026-10-06 | This is a previous chunked kernel. The pull request #29353 replaces it. |
| [#24561](https://github.com/ggml-org/llama.cpp/pull/24561) | `CUDA/HIP: chunked MFMA prefill kernel for GATED_DELTA_NET (CDNA)` | Closed, not merged | 2026-06-13 to 2026-07-10 | This is a chunked kernel for CDNA that uses `MFMA`. The author closed it. It stays closed if no person agrees to keep it. |
| [#22587](https://github.com/ggml-org/llama.cpp/pull/22587) | `CUDA: row-per-warp kernel for GATED_DELTA_NET` | Open | 2026-05-01 to open | This is a decode kernel that gives one row to each warp. A reviewer wants the decode kernel to stay basic. |
| [#21897](https://github.com/ggml-org/llama.cpp/pull/21897) | `ggml-cuda: enable concurrent streams for linear attention` | Open | 2026-04-14 to open | The pull request puts the kernels of the recurrent layers in different CUDA streams. The GPU can do them at the same time. |
| [#18102](https://github.com/ggml-org/llama.cpp/pull/18102) | `ggml-cuda: Delta-Net linear attention for Qwen3-Next` | Open | 2025-12-16 to open | The pull request has a DeltaNet kernel. No person has continued the work. |
| [#25162](https://github.com/ggml-org/llama.cpp/issues/25162) | `Performance regression on Turing GPUs after 9e58d4d69 — Qwen35 SSM kernels af…` | Closed | 2026-06-30 to 2026-08-03 | The issue reports a slower speed on a Turing GPU after a GDN change. |

**Changes in `master` since b12004.** No commit touches the files `gated*`.

**Code in b12004.**

- The pull request [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) added `gated-delta-net-mma.cu`. The kernel runs only for a scalar gate and a head dimension of 128. It also runs only when `H_k == 16`, `H` is 16, 32, 48 or 64, and each sequence has 64 tokens or more. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391) and [`ggml/src/ggml-cuda/gated_delta_net.cu:276`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated_delta_net.cu#L276).
- The same condition is true only when parameter 0 of the operation is less than the token count. That parameter sets how many snapshots the operation keeps.
- The pull request reports that the prefill speed is 3 to 6 percent higher on a 27B dense model. It is 11 to 21 percent higher on a 35B MoE model.
- The file `gated_delta_net.cu` keeps the autoregressive kernel and a `TODO` for a faster prefill kernel. See [`ggml/src/ggml-cuda/gated_delta_net.cu:192`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated_delta_net.cu#L192).
- The commits that tune the kernel for decode are [`34818ea6c`](https://github.com/ggml-org/llama.cpp/commit/34818ea6c) ([#20537](https://github.com/ggml-org/llama.cpp/pull/20537)) and [`24e41838e`](https://github.com/ggml-org/llama.cpp/commit/24e41838e) ([#30087](https://github.com/ggml-org/llama.cpp/pull/30087)).

**Result.** Done in b12004 for the chunked kernel for prefill. No item found for prompts with different lengths or for short prompts.

**Effect on the project.** Do not write the chunked kernel again. Count how many prompts of decisions have less than 64 tokens in each sequence. Those rows use the autoregressive kernel, which is slower [INFERENCE].
A reviewer wants the autoregressive kernel to stay basic ([#22587](https://github.com/ggml-org/llama.cpp/pull/22587)). Reviewers can reject a large change of the decode kernel.
### 3.i Server API for decisions

**Question.** Does a change of the `/v1/systemone` API or of its server code have an effect on the speed?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#29818](https://github.com/ggml-org/llama.cpp/pull/29818) | `llama, server: add /v1/systemone API (models: laya, julia-1, lev, openjev, ke…` | Merged [`a4cb4c61f`](https://github.com/ggml-org/llama.cpp/commit/a4cb4c61f) | 2026-10-01 to 2026-10-02 | Adds `/v1/systemone` and five decision models. |
| [#30361](https://github.com/ggml-org/llama.cpp/pull/30361) | `model : add H2O-Lightning-4B decision model` | Merged [`7e3eeee04`](https://github.com/ggml-org/llama.cpp/commit/7e3eeee04) | 2026-10-12 to 2026-10-20 | Adds the H2O-Lightning-4B decision type. |
| [#30567](https://github.com/ggml-org/llama.cpp/pull/30567) | `server: prefer empty slots and keep states overwritten by child tasks` | Merged [`b0d239164`](https://github.com/ggml-org/llama.cpp/commit/b0d239164) | 2026-10-15 to 2026-10-15 | Child tasks get empty slots first. When a child task uses a slot that has data, the server keeps that data in the prompt cache. |
| [#30498](https://github.com/ggml-org/llama.cpp/pull/30498) | `server: save only restorable slot states to the prompt cache` | Merged [`2639a3b4e`](https://github.com/ggml-org/llama.cpp/commit/2639a3b4e) | 2026-10-14 to 2026-10-15 | Keeps only slot data in the prompt cache that the server can put back into a slot. |
| [#29856](https://github.com/ggml-org/llama.cpp/pull/29856) | `graph: fix CI realloc abort by gathering the recurrent states once` | Merged [`436f6f89e`](https://github.com/ggml-org/llama.cpp/commit/436f6f89e) | 2026-10-02 to 2026-10-03 | Gathers the layer states one time in `build_rs`. The decode speed does not change. |
| [#30408](https://github.com/ggml-org/llama.cpp/pull/30408) | `model: fix vllm d3 multi seq case` | Merged [`cbe28bb3c`](https://github.com/ggml-org/llama.cpp/commit/cbe28bb3c) | 2026-10-12 to 2026-10-13 | Repairs the work of a hybrid model with non-causal attention and more than one sequence. |
| [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) | `server: visit parent slots before child slots` | Open | 2026-10-18 to open | The server goes through the slots of parent tasks before the slots of child tasks. Without this change, a child task with a slot number lower than the slot number of its parent task reads the tokens after the shared prefix in a third pass. |
| [#30710](https://github.com/ggml-org/llama.cpp/pull/30710) | `server: cap shared decision prefixes at the first output` | Open | 2026-10-17 to open | Stops the shared prefix at the first output position. This repairs a 500 error when two questions have the same tokens in their possible answers. |
| [#30482](https://github.com/ggml-org/llama.cpp/issues/30482) | `Misc. bug: /v1/systemone returns 500 "the question and its options must fit i…` | Open | 2026-10-13 to open | The issue behind #30710. `/v1/systemone` gives the error 500 when two questions have a shared prefix. |
| [#30682](https://github.com/ggml-org/llama.cpp/pull/30682) | `feat: Do not disable prompt caching for text + decision models` | Open, as a draft pull request | 2026-10-16 to open | Keeps the prompt cache on for text models plus decision models. |
| [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | `server : share prompt prefix cells between slots (--slot-attach)` | Open | 2026-10-22 to open | Lets slots use the same KV cells for a prompt prefix, with `--slot-attach`. |
| [#30683](https://github.com/ggml-org/llama.cpp/pull/30683) | `llama-server: Avoids extra copying of existing checkpoints` | Open | 2026-10-16 to open | Uses the checkpoints that the server has and makes no new copy of them. |
| [#29918](https://github.com/ggml-org/llama.cpp/pull/29918) | `server : add --cache-reuse-hybrid for recurrent/hybrid models` | Open, as a draft pull request | 2026-10-03 to open | Adds `--cache-reuse-hybrid` for hybrid models and for models with recurrent layers. |
| [#26204](https://github.com/ggml-org/llama.cpp/pull/26204) | `server : add /slots endpoint action=clone_to (KV clone between slots)` | Open | 2026-07-27 to open | Adds `action=clone_to` to the `/slots` API. It makes a copy of the KV cache of one slot in a different slot. |
| [#30780](https://github.com/ggml-org/llama.cpp/discussions/30780) | `server : reuse image features within a SystemOne request` | Open | 2026-10-18 to open | A discussion about how one decision request can use the same visual data again. |
| [#30073](https://github.com/ggml-org/llama.cpp/issues/30073) | `Eval bug: decision models fail on large input size` | Open | 2026-10-07 to open | Decision models give an error with large input. |
| [#30064](https://github.com/ggml-org/llama.cpp/issues/30064) | `Eval bug: Clef-GGUF Q8_0 /v1/systemone probabilities collapse toward uniform,…` | Closed | 2026-10-06 to 2026-10-07 | With `Q8_0`, the probabilities of the decisions became almost equal for all answers. With `BF16`, the probabilities stayed correct. |
| [#29902](https://github.com/ggml-org/llama.cpp/issues/29902) | `Misc. bug: llama-server with laya model aborts when the questions of a batch…` | Closed | 2026-10-03 to 2026-10-03 | The server stopped with a fault when the questions of a batch had more tokens than the value of `-ub`. |

**Changes in `master` since b12004.** No commit touches `tools/server/server-decision.cpp` or `tools/server/server-context.cpp`.

**Code in b12004.**

- `server_decision_group_tasks` puts the questions in groups. A group has a maximum of one task for each slot. A slot is the part of the server that holds the data of one task. The first task of a group is the parent task. It runs the model on the shared prefix. See [`tools/server/server-decision.cpp:1753-1783`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-decision.cpp#L1753-L1783).
- The child tasks wait for the parent task. The parent task then makes a copy of the data of its slot for each child task. See [`tools/server/server-context.cpp:4360-4372`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L4360-L4372) and [`tools/server/server-context.cpp:4785-4789`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L4785-L4789).
- The functions that make the copy use `mem.seq_cp` and make a copy of the prompt. See [`tools/server/server-context.cpp:845-864`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L845-L864).
- The description of [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) tells that, since [#30567](https://github.com/ggml-org/llama.cpp/pull/30567), a child task can get a slot number lower than the slot number of its parent task. The server goes through the slots from the lowest number to the highest number. The child task is then not in one batch. The model reads the tokens after the shared prefix in a third pass.
- The pull request [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) with the repair is open. Its test shows that tasks wait 5 times across two requests on `master` and 4 times with the repair.
- The API first came in the release b11361, as a comment on [#29818](https://github.com/ggml-org/llama.cpp/pull/29818) tells. The H2O-Lightning decision type came in [`7e3eeee04`](https://github.com/ggml-org/llama.cpp/commit/7e3eeee04).

**Result.** API: done in b12004. Speed repairs: in progress upstream. No item is about how the server reads the logits of the answer tokens.

**Effect on the project.** Cherry-pick [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) if a child task in production has a lower slot number than its parent task. First check this with a log of the slot numbers that the server gives to the tasks.
The items about the prompt cache ([#30682](https://github.com/ggml-org/llama.cpp/pull/30682), [#30970](https://github.com/ggml-org/llama.cpp/pull/30970), [#30683](https://github.com/ggml-org/llama.cpp/pull/30683), [#29918](https://github.com/ggml-org/llama.cpp/pull/29918)) have an effect when prompts have the same prefix again and again. Only monitor them.

### 3.j Copy of the layer state in hybrid memory

**Question.** Which items of the upstream project are about the time of the copy and of the gather of the layer state in hybrid models?

| Item | Title | Condition | Dates | Information |
|---|---|---|---|---|
| [#29856](https://github.com/ggml-org/llama.cpp/pull/29856) | `graph: fix CI realloc abort by gathering the recurrent states once` | Merged [`436f6f89e`](https://github.com/ggml-org/llama.cpp/commit/436f6f89e) | 2026-10-02 to 2026-10-03 | Gathers the layer states one time. Repairs a problem with the memory that the program keeps for the graph. |
| [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) | `Feature Request: skip redundant recurrent-state gather when n_rs == 1 && n_se…` | Open | 2026-09-13 to open | The issue wants to remove the 96 gathers of the layer state that are not necessary when one sequence has one memory cell. The measurement is 2.2 percent of the decode time on Vulkan. |
| [#29121](https://github.com/ggml-org/llama.cpp/pull/29121) | `llama : update the GDN recurrent state in place on the CPU (#28841)` | Open | 2026-09-19 to open | Changes the layer state of Gated DeltaNet in the same memory on the CPU. |
| [#27991](https://github.com/ggml-org/llama.cpp/pull/27991) | `kv cache : optimize restoring non-contiguous cells` | Merged [`2d8d612e4`](https://github.com/ggml-org/llama.cpp/commit/2d8d612e4) | 2026-08-29 to 2026-08-31 | Makes it faster to put back KV cells that are not adjacent. |
| [#26004](https://github.com/ggml-org/llama.cpp/pull/26004) | `server : preserve context checkpoints across slot save/restore` | Merged [`033df86b6`](https://github.com/ggml-org/llama.cpp/commit/033df86b6) | 2026-07-22 to 2026-10-08 | Keeps the checkpoints during the `save` and the `restore` of a slot. |
| [#20087](https://github.com/ggml-org/llama.cpp/pull/20087) | ``Hybrid model cache: add `--checkpoint-every-nb` `` | Merged [`f5ddcd169`](https://github.com/ggml-org/llama.cpp/commit/f5ddcd169) | 2026-03-03 to 2026-03-06 | Adds `--checkpoint-every-nb` for hybrid models. |
| [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) | `llama-server: Use reference-counted checkpoints instead of deep-copies` | Open | 2026-10-08 to open | Uses a reference count for each checkpoint and makes no full copy of it. |
| [#30683](https://github.com/ggml-org/llama.cpp/pull/30683) | `llama-server: Avoids extra copying of existing checkpoints` | Open | 2026-10-16 to open | Uses the checkpoints that the server has and makes no new copy of them. |
| [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | `server : share prompt prefix cells between slots (--slot-attach)` | Open | 2026-10-22 to open | Lets slots use the same KV cells for a prefix. |
| [#29918](https://github.com/ggml-org/llama.cpp/pull/29918) | `server : add --cache-reuse-hybrid for recurrent/hybrid models` | Open, as a draft pull request | 2026-10-03 to open | Adds `--cache-reuse-hybrid`. |
| [#26204](https://github.com/ggml-org/llama.cpp/pull/26204) | `server : add /slots endpoint action=clone_to (KV clone between slots)` | Open | 2026-07-27 to open | Adds `action=clone_to` to the `/slots` API. It makes a copy of the KV cache of one slot in a different slot. |
| [#24785](https://github.com/ggml-org/llama.cpp/pull/24785) | `server: add recurrent state shrink/expand for prompt cache (#22746)` | Open | 2026-06-18 to open | Makes the layer state smaller and larger for the prompt cache. |
| [#30252](https://github.com/ggml-org/llama.cpp/issues/30252) | `server : saving idle slots to the prompt cache stalls all requests for second…` | Open | 2026-10-09 to open | Requests wait while the server keeps slots without a task in the prompt cache, when KV cells of one slot are between KV cells of other slots. |
| [#31391](https://github.com/ggml-org/llama.cpp/pull/31391) | `cuda : gather batched tensor reads into mapped host memory (PR to master)` | Open | 2026-10-29 to open | Gathers a group of tensor reads into mapped host memory. In one test, the time to keep a slot without a task decreases from 140 to 7 ms. |
| [#29324](https://github.com/ggml-org/llama.cpp/issues/29324) | `Misc. bug: Misc. bug: --cache-ram -1 is not "no limit": the token cap stays a…` | Open | 2026-09-23 to open | With hybrid models, `--cache-ram -1` lets the memory of the host become larger. |
| [#28118](https://github.com/ggml-org/llama.cpp/pull/28118) | `server : keep speculative recurrent-state checkpoints on-device` | Open, as a draft pull request | 2026-08-31 to open | Keeps the checkpoints for speculative decoding on the GPU. |
| [#31368](https://github.com/ggml-org/llama.cpp/pull/31368) | `memory: refuse rollbacks past the snapshots of the last ubatch` | Merged [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc), after b12004 | 2026-10-29 to 2026-10-30 | A commit after b12004. It limits a rollback to the snapshots of the last ubatch. |

**Changes in `master` since b12004.** The only commit is [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc). It adds the value `rs_max` to each memory cell and changes `seq_rm` in `src/llama-memory-recurrent.cpp`. It adds no copy of a tensor.

**Code in b12004.**

- The function `seq_cp` for recurrent layers links the sequence that gets the data to the memory cell of the source sequence. It makes no copy of tensor data. See [`src/llama-memory-recurrent.cpp:262-297`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-memory-recurrent.cpp#L262-L297).
- The graph moves the data. Each `build_rs` does one `ggml_get_rows` on the layer states and writes the other layer states back. See [`src/llama-graph.cpp:3937-3949`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-graph.cpp#L3937-L3949).
- Qwen3.5 uses `build_rs` two times for each linear layer. One time is for the conv state and one time is for the SSM state. [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) counts 96 gathers for each graph for a model with 48 linear layers.
- [INFERENCE] The slot of a child task makes the copy of the layer state in its first pass. This occurs because the child task and the parent task use the same memory cell.
- [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) measured the gathers as 2.2 percent of the decode time on Vulkan with one sequence. The measurement used a commit before [#29856](https://github.com/ggml-org/llama.cpp/pull/29856). This text does not give a measurement of the time for CUDA with more than one sequence.
- A comment on [#29324](https://github.com/ggml-org/llama.cpp/issues/29324) calculated about 3 MiB of SSM state for each layer of a 27B model with 48 value heads. This text does not give a measurement of the number of bytes of the layer state of the model in production.

**Result.** In progress upstream, for some parts. No item found that decreases the time that `seq_cp` or the gather of each pass uses for batches with more than one sequence.

**Effect on the project.** Measure the gather kernels and the copy kernels for `-np 3` before other work. The upstream items are about the prompt cache, which lets the server use data again, and about the memory of the host. Thus a repair of the gather on the GPU is new work.

## 4. Commits since b12004

Command: `git log --format='%h %cd %s' --date=short b12004..origin/master`. It shows 13 commits. The newest commit has the tag b12017.

Command for the target paths. A target path is a path of code that a target can change:

`git log --format='%h %cd %s' --date=short b12004..origin/master -- 'ggml/src/ggml-cuda/mmq*' 'ggml/src/ggml-cuda/gated*' 'ggml/src/ggml-cuda/fattn*' 'ggml/src/ggml-cuda/ggml-cuda.cu' 'ggml/src/ggml-cuda/norm.cu' 'ggml/src/ggml-cuda/quantize.cu' 'src/llama-batch.cpp' 'src/llama-memory-*.cpp' 'tools/server/server-decision.cpp'`.

The result is one commit.

| Commit | Date | First line of the commit message | In a target path | Effect on targets `a` to `j` |
|---|---|---|---|---|
| [`5047748f1`](https://github.com/ggml-org/llama.cpp/commit/5047748f1) | 2026-10-30 | `rpc : delete fs_path_to_utf8() (#31424)` | No | None. It does not change a target path. |
| [`792fcf4e7`](https://github.com/ggml-org/llama.cpp/commit/792fcf4e7) | 2026-10-30 | `opencl: add initial iq1 support (#31341)` | No | None. The change is in OpenCL code, not in a target path. |
| [`a34de8618`](https://github.com/ggml-org/llama.cpp/commit/a34de8618) | 2026-10-30 | `common : delete fs_path_to_utf8() (#31423)` | No | None. It does not change a target path. |
| [`aa588a346`](https://github.com/ggml-org/llama.cpp/commit/aa588a346) | 2026-10-30 | `common : drop the Windows argv reconstruction (#31417)` | No | None. It does not change a target path. |
| [`889f2ec96`](https://github.com/ggml-org/llama.cpp/commit/889f2ec96) | 2026-10-30 | `quant : allow quantization for LFM3-Audio codec decoder (#31386)` | No | None. It does not change a target path. |
| [`1ab23c33b`](https://github.com/ggml-org/llama.cpp/commit/1ab23c33b) | 2026-10-30 | `vendor : update cpp-httplib to 0.64.0 (#31400)` | No | None. It does not change a target path. |
| [`27b8464e8`](https://github.com/ggml-org/llama.cpp/commit/27b8464e8) | 2026-10-30 | `ci : add missing ccache-clear to windows behavior check (#31413)` | No | None. It does not change a target path. |
| [`d63fbfe79`](https://github.com/ggml-org/llama.cpp/commit/d63fbfe79) | 2026-10-30 | `cmake : embed an app manifest in all Windows exes (#31329)` | No | None. It does not change a target path. |
| [`e216f4ec8`](https://github.com/ggml-org/llama.cpp/commit/e216f4ec8) | 2026-10-30 | `ci : extract most apple jobs into daily workflow (#31408)` | No | None. It does not change a target path. |
| [`344846b4f`](https://github.com/ggml-org/llama.cpp/commit/344846b4f) | 2026-10-30 | `gemma3n, gemma4 : create token_embd before the tied output (#31394)` | No | None. The commit changes how the program loads the `gemma3n` and `gemma4` models. It does not change a target path. |
| [`fb5997274`](https://github.com/ggml-org/llama.cpp/commit/fb5997274) | 2026-10-30 | `kv-cache : keep rotation constants on the GPU (#31073)` | No | None. The constant values of the KV cache rotation move to the GPU. The commit changes `llama-graph.cpp`, which is not a target path. |
| [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc) | 2026-10-30 | `memory: refuse rollbacks past the snapshots of the last ubatch (#31368)` | Yes: `src/llama-memory-recurrent.cpp` | We think that the effect is none. The commit limits a rollback of a recurrent layer to the snapshots of the last ubatch. The server does not use rollback [INFERENCE]. |
| [`e0f142d5d`](https://github.com/ggml-org/llama.cpp/commit/e0f142d5d) | 2026-10-29 | `opencl: add initial q2_0 support (#31340)` | No | None. The change is in OpenCL code, not in a target path. |

The file `commits-since-b12004.tsv` has the full list with the names of the files.

Commands for the folders:

- The command `git log b12004..origin/master -- ggml/src/ggml-cuda` shows no commit.
- The same command for `tools/server` shows two commits. They change `server-models.cpp` and `server-tools.cpp`.
- The same command for `src/llama-context.cpp` shows no commit.

## 5. Other points with an effect on the production flags

1. **Flash attention and the attention mask with a unified KV cache (open, [#28495](https://github.com/ggml-org/llama.cpp/issues/28495)).**
   - The CUDA code for flash attention reads the attention mask to find the last KV cell that it must use. It does this only when `Q->ne[1] >= 1024` or `Q->ne[3] > 1`. See [`ggml/src/ggml-cuda/fattn-common.cuh:1117`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/fattn-common.cuh#L1117).
   - A unified KV cache gives `Q->ne[3] == 1`. Then a short batch does not read the attention mask. Each slot then reads the KV cells of other slots.
   - The persons who wrote in the issue measured 42 to 54 percent lower prefill speed on HIP. A person with a V100 measured about 46 to 49 percent with the data of a second slot in the KV cache.
   - The production flags use `-np 3 --kv-unified`. Only 8 of the 32 model layers are attention layers. A measurement must show the effect on this model [INFERENCE].
2. **The sequence of slots in the group of decisions (open, [#30779](https://github.com/ggml-org/llama.cpp/pull/30779)).** See section 3.i. The problem starts after [#30567](https://github.com/ggml-org/llama.cpp/pull/30567). That pull request is in b12004.
3. **Slots that are not adjacent, with the split KV cache (closed, [#25840](https://github.com/ggml-org/llama.cpp/issues/25840)).** The split KV cache is the default. With it, slots in use cannot be in one ubatch if they are not adjacent, for example slots 0 and 2. The production flag `--kv-unified` prevents this.
4. **List of types for MMQ fusion.** Q4_0 is not in the list at [`ggml/src/ggml-cuda/mmq.cuh:437-440`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L437-L440). The feed-forward layers of the production model use Q4_0 [INFERENCE: from the instructions for the quantization of the model].
5. **Token limit of Gated DeltaNet.** The kernel for chunks works only with 64 or more tokens for each sequence. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391).
6. **Interval before the first CUDA graph.** The program can capture a CUDA graph only after two passes in sequence that have the same graph. See [`ggml/src/ggml-cuda/ggml-cuda.cu:5016-5020`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L5016-L5020).
7. **No `cp.async` in MMQ.** Only the files for flash attention use `cp-async.cuh`.
8. **An item about the quality of the decision model.** The issue [#30064](https://github.com/ggml-org/llama.cpp/issues/30064) shows that with Q8_0 the decision probabilities go near the same value for all answers. With BF16 the probabilities stay correct. The issue is not open. Make sure that the mix8 GGUF does not show the same effect before you give a speed result.

## 6. Method and limits

The method has five steps:

1. Write the queries in `queries.tsv` (151 queries) and `queries2.tsv` (49 queries). Each query has an identification, the letter of a target and a text.
2. Run `upstream_search.py`. It uses the GitHub GraphQL API through the `gh` command. The script sends each query for issues, for pull requests and for discussions of the repository.
3. Get the top 30 results for each type and each query. The script writes `search-raw.json`, `search-results.tsv`, `search2-raw.json` and `search2-results.tsv`.
4. Read the issue, the files of the pull request and the comments of 169 items with the REST API. The JSON data as GitHub sent it is in `items/`.
5. Read the source files of b12004 that this text names. Read the commits of `master` in a blobless clone.

Numbers:

- The 200 queries gave 7,871 rows. The rows have 2,322 different items: 1,113 pull requests, 891 issues and 318 discussions.
- The first 151 queries gave 6,956 rows with 2,211 different items. The 49 queries of the second set are for missing information. They added 111 new items. None of the new items changes a result.
- The date of a merged pull request is the commit date on `master`. The dates of other items come from GitHub.

Limits:

- GitHub finds items that have the words of the query. It does not find items by meaning. A query cannot find an item that has unusual words. For a general query, the limit of 30 results can remove items that are not new.
- We did not read all code changes. We read the titles, the descriptions, the files and the comments of the items that this text names.
- The result "No item found" means that the queries found no item. It does not show that no person works in a branch that we cannot read, or in a fork.
- The condition of each item is the condition on 2026-10-30. An open item can change.
- A sentence with the mark [INFERENCE] gives information from the code and not from a run. We did not run a benchmark for this work.
- The JSON files have the names of the persons who wrote the items. This document does not name these persons.

## 7. Files

These files are not part of this repository.

- `upstream-survey.md`: the source of this document.
- `queries.tsv`, `queries2.tsv`: the 200 queries.
- `upstream_search.py`: the script that sends the queries.
- `search-raw.json`, `search-results.tsv`: the output of the first 151 queries.
- `search2-raw.json`, `search2-results.tsv`: the output of the 49 queries of the second set.
- `items/`: one JSON file for each item that we read with the REST API.
- `commits-since-b12004.tsv`: the 13 commits after b12004 with the names of their files.
