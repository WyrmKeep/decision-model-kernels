# Upstream survey of llama.cpp for the decision-model kernel project

Survey date: 2026-10-30. Repository: ggml-org/llama.cpp. Access was read-only. No comment, reaction, or issue was written.

Production baseline: release tag b12004, commit [`3c809f4`](https://github.com/ggml-org/llama.cpp/commit/3c809f44383bcebc318b8cddd8c6d11aaa861e9b).
Newest release tag: **b12017**, a pre-release published 2026-10-30T20:19:54Z. It points at commit [`5047748f1`](https://github.com/ggml-org/llama.cpp/commit/5047748f114564fcf271e6a648f60e2d2eae6264), which is also the head of master at survey time.

## 1. Read this first

- Master is 13 commits ahead of b12004. None of them changes CUDA code, the batch splitter, the hybrid memory class, or the decision server.
- One commit touches a listed path: [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc) ([#31368](https://github.com/ggml-org/llama.cpp/pull/31368)). It limits recurrent rollback to the snapshots of the last micro-batch.
- That commit has no speed effect for this project [INFERENCE: the server does not use speculative decoding, so rollback is not used].
- No target has the verdict "already in master (a rebase gains it)". A rebase from b12004 to b12017 gains nothing for targets a to j.
- Every merged change named in this survey is already inside b12004. The verdict "already in b12004" marks these cases.

### 1.1 Work that upstream already did or is doing

- **Target h, done:** the chunked GDN prefill kernel for CUDA is in b12004 ([#29353](https://github.com/ggml-org/llama.cpp/pull/29353), [`62df1815b`](https://github.com/ggml-org/llama.cpp/commit/62df1815b)). Do not rebuild it.
- **Target c, done for decode:** chunked CUDA graphs are in b12004 ([#30804](https://github.com/ggml-org/llama.cpp/pull/30804), [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d)).
- **Target d, done in part:** the MMQ gate, up and GLU fusion is in b12004 ([#29948](https://github.com/ggml-org/llama.cpp/pull/29948), [`c49edf096`](https://github.com/ggml-org/llama.cpp/commit/c49edf096)). It does **not** cover Q4_0, the FFN type of the production model.
- **Target d, in progress:** [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) fuses GLU into the MMQ activation quantization for all Q8_1 MMQ types. It is a cherry-pick candidate.
- **Target a, in progress for drafts only:** [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) and [#29143](https://github.com/ggml-org/llama.cpp/pull/29143) add a reduced output head for draft contexts.
- **Target e, done:** per-architecture MMQ tile tables and stream-k are in b12004 ([#24127](https://github.com/ggml-org/llama.cpp/pull/24127), [#22298](https://github.com/ggml-org/llama.cpp/pull/22298), [#30553](https://github.com/ggml-org/llama.cpp/pull/30553)).
- **Target i, in progress:** [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) fixes an extra decode pass in the shared-prefix slot scheme of `/v1/systemone`.
- **Target f, caution:** a cp.async MMQ change ([#28572](https://github.com/ggml-org/llama.cpp/pull/28572)) was closed after a reviewer said it clashes with plans of their own for the kernel.

## 2. Summary table

Verdict key:

- **Already in b12004:** the change exists in the production tag. A rebase gains nothing. It is prior art.
- **In progress upstream:** an open item exists. Watch it or cherry-pick it.
- **Proposed and rejected:** a closed item exists, with a reason when one is known.
- **Nothing found:** the searches returned no matching item.

| Target | Verdict | Key items | Implication for this project |
|---|---|---|---|
| a. Answer-token-only output head | In progress upstream, for draft contexts only. Nothing found for decision labels. | [#31031](https://github.com/ggml-org/llama.cpp/pull/31031), [#29143](https://github.com/ggml-org/llama.cpp/pull/29143), [#25187](https://github.com/ggml-org/llama.cpp/issues/25187), [#30445](https://github.com/ggml-org/llama.cpp/pull/30445) | The production path computes all 248,320 head rows. A label-only head needs project code. [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) shows a pattern to copy. |
| b. Ragged batching for hybrid memory | Nothing found. | [#14217](https://github.com/ggml-org/llama.cpp/pull/14217), [#25278](https://github.com/ggml-org/llama.cpp/pull/25278), [#15636](https://github.com/ggml-org/llama.cpp/pull/15636) | `split_equal` still needs equal sequence lengths. Ragged batching is open design space. No upstream work exists to rebase onto. |
| c. CUDA graphs for prompt batches | Decode graphs: already in b12004. Prompt batches: nothing found. Shape keying: in progress. | [#30804](https://github.com/ggml-org/llama.cpp/pull/30804), [#31004](https://github.com/ggml-org/llama.cpp/pull/31004), [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) | b12004 captures a graph only after two identical calls. Varying prompt shapes rarely qualify [INFERENCE]. Watch [#31004](https://github.com/ggml-org/llama.cpp/pull/31004). |
| d. Fusion of memory-bound ops | Mostly already in b12004. Q4_0 FFN fusion is missing. GLU-in-quantize is in progress. Conv-state concat fusion: nothing found. | [#29948](https://github.com/ggml-org/llama.cpp/pull/29948), [#30489](https://github.com/ggml-org/llama.cpp/pull/30489), [#22710](https://github.com/ggml-org/llama.cpp/pull/22710), [#25930](https://github.com/ggml-org/llama.cpp/pull/25930) | [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) is a cherry-pick candidate. A Q4_0 extension of [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) is new work. Two norm-plus-quantize fusions were closed. |
| e. MMQ tile tuning for sm_86 | Already in b12004 (tile tables, stream-k). A Q4_0 small-batch tile change was closed without review. | [#24127](https://github.com/ggml-org/llama.cpp/pull/24127), [#30553](https://github.com/ggml-org/llama.cpp/pull/30553), [#28204](https://github.com/ggml-org/llama.cpp/pull/28204), [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) | Tuning means editing `mmq-config-ampere.cuh`. No upstream tuning for sm_86 competes with it. |
| f. MMQ redesign or other int8 GEMM | Nothing found for Ampere. Blackwell and Hopper experiments exist. One was closed. | [#28572](https://github.com/ggml-org/llama.cpp/pull/28572), [#26704](https://github.com/ggml-org/llama.cpp/pull/26704), [#28514](https://github.com/ggml-org/llama.cpp/discussions/28514) | A cp.async MMQ pipeline for sm_86 would be new. A reviewer has plans for this kernel, so conflicts are possible. |
| g. Megakernel or persistent kernel | Nothing found. | [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331), [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | No prior art exists. Chunked graphs cover part of the launch overhead. |
| h. Gated DeltaNet CUDA kernels | Already in b12004 (chunked prefill kernel). Nothing found for short or ragged prompts. | [#29353](https://github.com/ggml-org/llama.cpp/pull/29353), [#30087](https://github.com/ggml-org/llama.cpp/pull/30087), [#20537](https://github.com/ggml-org/llama.cpp/pull/20537), [#22587](https://github.com/ggml-org/llama.cpp/pull/22587) | The chunked kernel needs 64 or more tokens per sequence. Shorter rows use the autoregressive kernel. |
| i. Server decision API | Core API: already in b12004. Speed fixes: in progress upstream. | [#29818](https://github.com/ggml-org/llama.cpp/pull/29818), [#30361](https://github.com/ggml-org/llama.cpp/pull/30361), [#30567](https://github.com/ggml-org/llama.cpp/pull/30567), [#30779](https://github.com/ggml-org/llama.cpp/pull/30779), [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) removes an extra decode pass after [#30567](https://github.com/ggml-org/llama.cpp/pull/30567). It is a cherry-pick candidate. |
| j. Hybrid state copy cost | In progress upstream, in parts. Nothing found on the cost of `seq_cp` itself. | [#28841](https://github.com/ggml-org/llama.cpp/issues/28841), [#29856](https://github.com/ggml-org/llama.cpp/pull/29856), [#30137](https://github.com/ggml-org/llama.cpp/pull/30137), [#30683](https://github.com/ggml-org/llama.cpp/pull/30683), [#31391](https://github.com/ggml-org/llama.cpp/pull/31391) | In b12004, recurrent `seq_cp` copies no tensor data. The state gather runs inside the graph on each pass. |

## 3. Findings by target

Each item table lists the issue or PR number, the title as GitHub shows it, the state, the dates, and a one-line relevance note.

Titles are quoted. They may be shortened, or edited to remove semicolons and contractions. Merged items show the commit hash. A date range shows the day the item opened and the day it merged or closed. Merge dates come from the commit on master. All merged items listed here are in b12004 unless the table says otherwise.

### 3.a Answer-token-only output head

**Question.** Can the model compute only the answer-label logits and skip the other vocabulary rows?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) | Trimmed draft vocab support (1/2) - reduced draft LM head from a user supplie… | Open | 2026-10-23 to open | Adds a reduced draft LM head. It keeps listed vocabulary rows and scatters the result into a full-size logit tensor. This is the closest upstream mechanism. It covers draft contexts only. |
| [#29143](https://github.com/ggml-org/llama.cpp/pull/29143) | feat(qwen35): support d2t draft-vocabulary trim for MTP speculative decoding | Open | 2026-09-19 to open | Trims the draft vocabulary for Qwen3.5 MTP. The head shape becomes embedding size by draft vocabulary size. It covers draft contexts only. |
| [#25187](https://github.com/ggml-org/llama.cpp/issues/25187) | Research: FR-Spec-style draft-vocab trimming for native MTP speculative decod… | Open | 2026-07-01 to open | Requests a trimmed draft vocabulary. The stale bot closed it and it is open again. A maintainer wrote that upstream will support it later. A commenter measured the 248,320-row Q8_0 head at 675 MB. |
| [#26226](https://github.com/ggml-org/llama.cpp/pull/26226) | core : support output vocab size distinct from embedding vocab size | Open | 2026-07-28 to open | Change for an output vocabulary size that differs from the embedding vocabulary. Stale and incomplete. |
| [#26224](https://github.com/ggml-org/llama.cpp/issues/26224) | core: support output vocab size distinct from embedding vocab size | Closed | 2026-07-28 to 2026-09-21 | Issue behind #26226. Closed with no merged change. |
| [#30445](https://github.com/ggml-org/llama.cpp/pull/30445) | graph: skip the LM head for pooled embeddings | Merged [`cb55ffe6e`](https://github.com/ggml-org/llama.cpp/commit/cb55ffe6e) | 2026-10-13 to 2026-10-13 | Skips the LM head for pooled embeddings. It shows that upstream accepts a skipped head. It covers pooled contexts only. |
| [#30265](https://github.com/ggml-org/llama.cpp/pull/30265) | server : add logit_gate to /v1/completions and /v1/chat/completions | Open | 2026-10-10 to open | Adds a logit gate for completions. It reads full logits on the host and applies a softmax over the candidate tokens. It does not shrink the head. |
| [#29022](https://github.com/ggml-org/llama.cpp/issues/29022) | Feature Request: Fast Tool Gating & Single-Pass Selection via Prefill Logit S… | Open | 2026-09-17 to open | Requests selected-token logits after prefill for tool gating. It describes the same need. It proposes no head change. |
| [#30323](https://github.com/ggml-org/llama.cpp/issues/30323) | perf: pooled embeddings context copies full-vocab logits to host on every dec… | Open | 2026-10-11 to open | Reports that pooled-embedding contexts copy full-vocabulary logits to the host. Shows the cost of full logit rows. |
| [#29487](https://github.com/ggml-org/llama.cpp/issues/29487) | Misc. bug: Backend Top-K output buffer is still allocated at full vocabulary… | Open | 2026-09-26 to open | Reports that the backend top-k output buffer still has full vocabulary width. |
| [#27050](https://github.com/ggml-org/llama.cpp/issues/27050) | Server: backend sampling (-bs) gives +48% throughput at 32 slots (706 -> 1046… | Closed | 2026-08-14 to 2026-09-28 | Measures the cost of copying full logits to the host in multi-slot serving. Backend sampling gave +48 percent throughput at 32 slots. |

**Master since b12004.** No commit changes the output head or the logit path. The only graph change is [`fb5997274`](https://github.com/ggml-org/llama.cpp/commit/fb5997274) (KV rotation constants).

**Code in b12004.**

- The qwen35 graph builds the head with `build_lora_mm(model.output, ...)` for all rows. See [`src/models/qwen35.cpp:233`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/qwen35.cpp#L233).
- The decision server reads logits with `llama_get_logits_ith` and indexes them by label token. See [`tools/server/server-context.cpp:2527-2535`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L2527-L2535).
- The H2O-Lightning labels are single tokens. See [`tools/server/server-decision.cpp:178-193`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-decision.cpp#L178-L193).
- The Q8_0 `output.weight` has 248,320 rows of 2,560 values and weighs about 675 MB (figure from a comment on [#25187](https://github.com/ggml-org/llama.cpp/issues/25187)).
- [INFERENCE] The RTX 3060 Ti has 448 GB/s peak bandwidth. Reading 675 MB takes at least 1.5 ms for each pass that has output rows.

**Verdict.** In progress upstream for draft contexts only. Nothing found for decision-label heads.

**Implication.** The mechanism in [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) copies chosen rows into a small head and scatters the result into a full-size tensor filled with minus infinity.
A decision version can skip the scatter and return only the label values [INFERENCE: design idea, not tested].
[#30445](https://github.com/ggml-org/llama.cpp/pull/30445) shows that upstream accepts a skipped head for pooled contexts. Watch [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) because it touches `llama-context.cpp`, `llama-graph.cpp` and `llama.h`.

### 3.b Ragged batching for hybrid and recurrent memory

**Question.** Can one micro-batch hold several sequences of different lengths for a hybrid model?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#14217](https://github.com/ggml-org/llama.cpp/pull/14217) | ubatch : new splitting logic | Merged [`4c9fdfbe1`](https://github.com/ggml-org/llama.cpp/commit/4c9fdfbe1) | 2025-06-16 to 2025-06-20 | Introduced the micro-batch split logic. The history of `split_equal` in `llama-batch.cpp` starts here. |
| [#14511](https://github.com/ggml-org/llama.cpp/pull/14511) | batch : add optional for sequential equal split | Merged [`67d1ef23c`](https://github.com/ggml-org/llama.cpp/commit/67d1ef23c) | 2025-07-03 to 2025-07-04 | Added the optional sequential equal split. |
| [#16442](https://github.com/ggml-org/llama.cpp/pull/16442) | memory : use sequential equal splits for recurrent modules | Merged [`0123ff38f`](https://github.com/ggml-org/llama.cpp/commit/0123ff38f) | 2025-10-06 to 2025-10-07 | Used sequential equal splits for recurrent modules in the memory class. |
| [#25278](https://github.com/ggml-org/llama.cpp/pull/25278) | llama-batch: enable parallel sequences for partial rollback | Merged [`230ea9d21`](https://github.com/ggml-org/llama.cpp/commit/230ea9d21) | 2026-07-03 to 2026-07-08 | Adds `n_keep_tail` to `split_equal` for partial rollback. It is the latest change to the splitter. The equal-length rule stays. |
| [#25004](https://github.com/ggml-org/llama.cpp/pull/25004) | recurrent : support equal splits for recurrent-state rollback | Open | 2026-06-25 to open | Equal splits for recurrent-state rollback. Partly replaced by #25278. |
| [#15636](https://github.com/ggml-org/llama.cpp/pull/15636) | batch : add `pad_equal` [RFC] | Open, draft | 2025-08-28 to open | RFC that pads sequences to equal length. Stale. It is the only padding proposal found. |
| [#30402](https://github.com/ggml-org/llama.cpp/issues/30402) | Eval bug: vllm-d3 answers change when a shorter request shares the batch (spl… | Open | 2026-10-12 to open | Reports that answers of a non-causal decision model change when a shorter request shares the batch. `split_equal` splits the prompts. |
| [#30408](https://github.com/ggml-org/llama.cpp/pull/30408) | model: fix vllm d3 multi seq case | Merged [`cbe28bb3c`](https://github.com/ggml-org/llama.cpp/commit/cbe28bb3c) | 2026-10-12 to 2026-10-13 | Related fix for the non-causal, hybrid, multi-sequence case behind #30402. It touches batch and memory code. |
| [#25840](https://github.com/ggml-org/llama.cpp/issues/25840) | Eval bug: server -np 3 decode drops ~4.4x (166 -> 38 t/s) whenever the two bu… | Closed | 2026-07-17 to 2026-09-03 | With `-np 3` and the default split cache, a gap in busy slot ids forces two ubatches per step. The workaround is `--kv-unified`, which production already uses. |
| [#28495](https://github.com/ggml-org/llama.cpp/issues/28495) | Eval bug: with -np 2 and --kv-unified, prompt processing drops 42-54% from th… | Open | 2026-09-06 to open | Reports 42 to 54 percent lower prompt speed on HIP with `-np 2 --kv-unified`. A V100 user saw the same on CUDA. The flash-attention kernels skip only mask tails. This matters for the production flags. |
| [#20006](https://github.com/ggml-org/llama.cpp/issues/20006) | Eval bug: Qwen3.5-27B batched performance much lower than expected | Closed | 2026-03-01 to 2026-04-27 | Reports slow batched Qwen3.5 on Vulkan. A commenter found state gather and copy operations still costly after a fix. |

**Master since b12004.** No commit changes `src/llama-batch.cpp` or `src/llama-memory-hybrid.cpp`.

**Code in b12004.**

- The hybrid memory calls `balloc.split_equal(...)`. See [`src/llama-memory-hybrid.cpp:90`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-memory-hybrid.cpp#L90).
- `split_equal` builds micro-batches whose sequences all have the same token count. The shortest sequence sets the count. See [`src/llama-batch.cpp:619-712`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-batch.cpp#L619-L712).
- The GDN MMA kernel runs only when a sequence has 64 or more tokens. Otherwise the autoregressive kernel runs. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391).
- The kernel grid is heads times sequences, so every sequence has the same length. No `cu_seqlens`-style layout exists.

**Verdict.** Nothing found. No item proposes ragged or variable-length batching for recurrent memory. The only padding proposal is [#15636](https://github.com/ggml-org/llama.cpp/pull/15636), which is stale.

**Implication.** Any ragged-batch work is new work. The padding RFC shows a cheaper alternative design. The `-np 3` gap-slot item [#25840](https://github.com/ggml-org/llama.cpp/issues/25840) is already avoided by the production flag `--kv-unified`.

### 3.c CUDA graphs for prompt batches and launch overhead

**Question.** Does upstream reduce launch overhead for prompt batches, for example with CUDA graphs or `GGML_CUDA_GRAPH_OPT`?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | CUDA: chunk graphs to amortize graph launch | Merged [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d) | 2026-10-19 to 2026-10-24 | Chunks CUDA graphs to amortize launch cost. It gives 1 to 2 percent decode gain. It already sits in b12004. |
| [#28549](https://github.com/ggml-org/llama.cpp/pull/28549) | Enable CUDA graph for MTP draft | Merged [`2f3fd0252`](https://github.com/ggml-org/llama.cpp/commit/2f3fd0252) | 2026-09-07 to 2026-09-16 | Enables a CUDA graph for the MTP draft. It uses a separate arena and stable cache keys. |
| [#26802](https://github.com/ggml-org/llama.cpp/pull/26802) | CUDA: only disable CUDA graphs when mul_mat_id actually needs a stream sync | Merged [`ebb546b7e`](https://github.com/ggml-org/llama.cpp/commit/ebb546b7e) | 2026-08-09 to 2026-08-11 | Disables graphs only when `mul_mat_id` needs a stream synchronization. |
| [#26771](https://github.com/ggml-org/llama.cpp/pull/26771) | CUDA: Recapture graph after memory pool flush | Merged [`13786fbb3`](https://github.com/ggml-org/llama.cpp/commit/13786fbb3) | 2026-08-08 to 2026-10-20 | Recaptures the graph after a memory pool flush. |
| [#25749](https://github.com/ggml-org/llama.cpp/pull/25749) | Enable CUDA graphs on Volta+Turing | Merged [`3f08ef2c5`](https://github.com/ggml-org/llama.cpp/commit/3f08ef2c5) | 2026-07-16 to 2026-07-16 | Enables CUDA graphs on Volta and Turing. |
| [#16991](https://github.com/ggml-org/llama.cpp/pull/16991) | CUDA: add stream-based concurrency | Merged [`c7af376c2`](https://github.com/ggml-org/llama.cpp/commit/c7af376c2) | 2025-11-04 to 2025-11-30 | Adds stream-based concurrency. It is the origin of `GGML_CUDA_GRAPH_OPT`. |
| [#22522](https://github.com/ggml-org/llama.cpp/pull/22522) | Programmatic Dependent Launch (PDL) for more performance on newer NVIDIA GPUs… | Merged [`e94722822`](https://github.com/ggml-org/llama.cpp/commit/e94722822) | 2026-04-29 to 2026-05-20 | Programmatic dependent launch. It helps Hopper and newer GPUs only. No gain on Ampere. |
| [#31004](https://github.com/ggml-org/llama.cpp/pull/31004) | Implement graph shape keying to prevent graph thrashing | Open, draft | 2026-10-22 to open | Keys CUDA graphs by shape to stop cache thrashing. A reviewer asks for the fix at the llama.cpp level. The author agreed to investigate. |
| [#28666](https://github.com/ggml-org/llama.cpp/pull/28666) | ggml-cuda: add first-node extents to the CUDA graph cache key | Closed, not merged | 2026-09-09 to 2026-09-09 | Earlier try to add first-node extents to the graph key. The author closed it. #31004 continues the work. |
| [#28652](https://github.com/ggml-org/llama.cpp/issues/28652) | Misc. bug: CUDA: alternating variants sharing one graph cache key cause warmu… | Closed | 2026-09-09 to 2026-10-25 | Reports that alternating graph variants thrash the CUDA graph cache. |
| [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) | Feature Request: CUDA graphs for multi-slot decode via shape-stable padded u | Closed | 2026-08-13 to 2026-10-06 | Asks for CUDA graphs in multi-slot decode with padded ubatches. A test showed 2 to 3 percent gain on a 4090. |
| [#21897](https://github.com/ggml-org/llama.cpp/pull/21897) | ggml-cuda: enable concurrent streams for linear attention | Open | 2026-04-14 to open | Runs linear-attention kernels on parallel streams with `GGML_CUDA_GRAPH_OPT=1`. A later comment reports a gain in token generation only. |
| [#29796](https://github.com/ggml-org/llama.cpp/pull/29796) | cuda: skip redundant stream synchronization | Open, draft | 2026-10-01 to open | Draft PR that skips a redundant stream synchronization. |
| [#16548](https://github.com/ggml-org/llama.cpp/pull/16548) | Implement and use cuda graph plans | Open | 2025-10-13 to open | Old proposal for CUDA graph plans. Stale. |
| [#11867](https://github.com/ggml-org/llama.cpp/pull/11867) | Overlap CUDA graph building and processing to minimize GPU idle time and impr… | Open | 2025-02-14 to open | Old proposal to overlap graph build and compute. Stale. |

**Master since b12004.** No commit touches `ggml/src/ggml-cuda`.

**Code in b12004.**

- The code has no batch-size gate for CUDA graphs. The architecture gate is Volta or newer. See [`ggml/src/ggml-cuda/ggml-cuda.cu:4983-4996`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L4983-L4996).
- A graph needs two consecutive calls with unchanged node properties before capture. Any change resets the warmup. See [`ggml/src/ggml-cuda/ggml-cuda.cu:5016-5020`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L5016-L5020).
- The update check is in [`ggml/src/ggml-cuda/ggml-cuda.cu:2898-2940`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L2898-L2940). The graph key is the first node of the graph.
- llama.cpp reuses a previous graph only when the micro-batch parameters match. See [`src/llama-context.cpp:1539`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-context.cpp#L1539).
- `GGML_CUDA_GRAPH_OPT=1` enables multi-stream execution for a three-way `attn_norm` fan-out. It counts only nodes with one row or less, so it affects decode only.

**Verdict.** Decode graphs: already in b12004. Shape keying: in progress ([#31004](https://github.com/ggml-org/llama.cpp/pull/31004)). Prompt-batch graphs: nothing found.

**Implication.** [INFERENCE] Prompt batches with changing shapes rarely give two identical consecutive calls, so graphs rarely engage for them.
A padded or bucketed prompt shape would be new work. [#27009](https://github.com/ggml-org/llama.cpp/issues/27009) asked for padding in decode and reported only 2 to 3 percent gain on a 4090.
Watch [#31004](https://github.com/ggml-org/llama.cpp/pull/31004) because a reviewer asked for separate arenas per shape at the llama.cpp level.

### 3.d Fusion of memory-bound ops

**Question.** Does upstream fuse rms_norm with multiply, SwiGLU, activation quantization, add with rms_norm, or the conv-state concat with ssm_conv?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) | CUDA: fuse FFN gate/up matmuls and GLU in MMQ with interleaved gate/up rows | Merged [`c49edf096`](https://github.com/ggml-org/llama.cpp/commit/c49edf096) | 2026-10-04 to 2026-10-25 | Fuses FFN gate, up and GLU inside MMQ. Q4_0 is not in the supported type list. See `mmq.cuh:437-440`. |
| [#31188](https://github.com/ggml-org/llama.cpp/pull/31188) | CUDA: fuse MoE gate/up matmuls and GLU in MMQ | Merged [`f5222f25c`](https://github.com/ggml-org/llama.cpp/commit/f5222f25c) | 2026-10-26 to 2026-10-28 | The same MMQ fusion for MoE gate and up. |
| [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) | ggml-cuda: fuse GLU into the MMQ Q8_1 activation quantization | Open | 2026-10-14 to open | Fuses GLU into the MMQ activation quantization. It gives +1.0 to +1.8 percent in prompt processing on a 4090 with Q4_K models. It is one commit, rebased on master on 2026-10-30. |
| [#22478](https://github.com/ggml-org/llama.cpp/pull/22478) | CUDA: fuse SSM_CONV + ADD(bias) + SILU | Merged [`098705a29`](https://github.com/ggml-org/llama.cpp/commit/098705a29) | 2026-04-28 to 2026-04-30 | Fuses SSM_CONV, bias add and SiLU in CUDA. |
| [#20128](https://github.com/ggml-org/llama.cpp/pull/20128) | CUDA: use shared mem for ssm_conv | Merged [`1e38a7a6f`](https://github.com/ggml-org/llama.cpp/commit/1e38a7a6f) | 2026-03-05 to 2026-03-06 | Uses shared memory in the CUDA `ssm_conv` kernel. |
| [#15631](https://github.com/ggml-org/llama.cpp/pull/15631) | CUDA: fuse adds, fuse add with rms norm | Merged [`009b709d6`](https://github.com/ggml-org/llama.cpp/commit/009b709d6) | 2025-08-28 to 2025-08-29 | Fuses add and RMS norm. |
| [#26767](https://github.com/ggml-org/llama.cpp/pull/26767) | CUDA: fuse rms_norm + mul + rope (+ view + set_rows) | Merged [`687e77892`](https://github.com/ggml-org/llama.cpp/commit/687e77892) | 2026-08-08 to 2026-08-09 | Fuses RMS norm, multiply and RoPE. |
| [#29393](https://github.com/ggml-org/llama.cpp/pull/29393) | CUDA: fuse RMS_NORM + SCALE into one kernel | Merged [`1ab7e5ad2`](https://github.com/ggml-org/llama.cpp/commit/1ab7e5ad2) | 2026-09-24 to 2026-09-25 | Fuses RMS norm and scale. |
| [#25441](https://github.com/ggml-org/llama.cpp/pull/25441) | CUDA: dedup MoE gate/up activation quantization | Merged [`5839ba352`](https://github.com/ggml-org/llama.cpp/commit/5839ba352) | 2026-07-08 to 2026-07-16 | Removes repeated MoE gate and up activation quantization. |
| [#23940](https://github.com/ggml-org/llama.cpp/pull/23940) | Remove redundant CUDA copies after gated_delta_net. | Merged [`5a460dea9`](https://github.com/ggml-org/llama.cpp/commit/5a460dea9) | 2026-05-31 to 2026-07-03 | Removes copies after the GDN op. |
| [#29807](https://github.com/ggml-org/llama.cpp/pull/29807) | CUDA: Remove redundant CUDA copies after SSM_SCAN | Merged [`50e3e3e48`](https://github.com/ggml-org/llama.cpp/commit/50e3e3e48) | 2026-10-01 to 2026-10-09 | Removes copies after SSM_SCAN. |
| [#22710](https://github.com/ggml-org/llama.cpp/pull/22710) | Fuse rms_norm, mul, quantize_q8_1 | Closed, not merged | 2026-05-05 to 2026-08-10 | Fuse RMS norm, multiply and Q8_1 quantization. The author closed it. Maintainers called it complex. |
| [#25930](https://github.com/ggml-org/llama.cpp/pull/25930) | CUDA: fuse q8_1 quantization into the producer rms_norm | Closed, not merged | 2026-07-20 to 2026-07-21 | Fuse Q8_1 quantization into the RMS norm producer. A maintainer closed it as too sloppy and asked for a more principled design. |
| [#29187](https://github.com/ggml-org/llama.cpp/pull/29187) | CUDA: Fuse GDN alpha/beta projections | Open, draft | 2026-09-20 to open | Fuses the GDN alpha and beta projections. Gain fell to 2 to 4 percent after #29633. On hold. |
| [#20520](https://github.com/ggml-org/llama.cpp/pull/20520) | CUDA: add new op template rms_norm_f32_vec4 for float4 vectorized load/store | Open | 2026-03-13 to open | Vectorizes RMS norm loads. No end-to-end change. |
| [#26468](https://github.com/ggml-org/llama.cpp/pull/26468) | ggml : fuse soft_max sweeps into fewer passes | Open | 2026-08-02 to open | Fuses soft-max sweeps. |
| [#26486](https://github.com/ggml-org/llama.cpp/pull/26486) | ggml : vectorize rms_norm reduce and fuse the scale write | Open, draft | 2026-08-03 to open | Vectorizes the RMS norm reduction. |
| [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | rfc: Memory eliding fusions | Open, draft | 2026-09-21 to open | RFC for fusions that elide memory traffic. |
| [#15602](https://github.com/ggml-org/llama.cpp/issues/15602) | Feature Request: Repeated Unecessary Activation Quantization Ops | Open | 2025-08-26 to open | Asks to avoid repeated activation quantization. |
| [#31179](https://github.com/ggml-org/llama.cpp/pull/31179) | ggml-cpu : rewrite concat as one row kernel for all types | Merged [`4d71b42fd`](https://github.com/ggml-org/llama.cpp/commit/4d71b42fd) | 2026-10-26 to 2026-10-26 | Rewrites CPU concat as a row kernel. CPU only. |
| [#30149](https://github.com/ggml-org/llama.cpp/pull/30149) | vulkan: add tiled concat for transposed inputs | Closed, not merged | 2026-10-08 to 2026-10-19 | Vulkan tiled concat for transposed inputs. Not CUDA. |
| [#28303](https://github.com/ggml-org/llama.cpp/pull/28303) | HIP : add shared-memory tiled transpose fast path for F32 concat on RDNA3.5 | Open | 2026-09-03 to open | HIP tiled transpose for F32 concat on RDNA3.5. Not CUDA. |

**Master since b12004.** No commit touches `norm.cu`, `quantize.cu` or `ggml-cuda.cu`.

**Code in b12004.**

- The MMQ gate, up and GLU fusion supports only Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS. See [`ggml/src/ggml-cuda/mmq.cuh:437-440`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L437-L440). Q4_0 is absent.
- The dense FFN of the production model uses Q4_0, so it takes the unfused path. [INFERENCE: the type comes from the model recipe. This survey did not dump the GGUF.]
- [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) fuses GLU into the Q8_1 quantization of the down projection. Its text says it covers all Q8_1 MMQ types. It gives +1.0 to +1.8 percent in prompt processing on a 4090 with Q4_K models. Q4_0 on sm_86 is not measured.
- The conv-state concat is `ggml_concat(conv_states, qkv_mixed, 0)`. See [`src/models/delta-net-base.cpp:469`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/delta-net-base.cpp#L469). The `ssm_conv` call follows in [`src/models/qwen35.cpp:403`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/models/qwen35.cpp#L403).
- CUDA has a `concat_non_cont` path. See [`ggml/src/ggml-cuda/concat.cu:85`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/concat.cu#L85). CUDA fuses SSM_CONV, add and SiLU ([#22478](https://github.com/ggml-org/llama.cpp/pull/22478)) but not the concat.

**Verdict.** Mostly already in b12004. GLU-in-quantize ([#30489](https://github.com/ggml-org/llama.cpp/pull/30489)) is in progress. Norm-plus-quantize fusion was proposed and closed ([#22710](https://github.com/ggml-org/llama.cpp/pull/22710), [#25930](https://github.com/ggml-org/llama.cpp/pull/25930)). Conv-state concat fusion for CUDA: nothing found.

**Implication.** Cherry-pick [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) and measure it on Q4_0. A Q4_0 loader for the [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) fusion is new work.
A reviewer said the norm-plus-quantize idea is known and needs a more principled design. A new attempt must show how the backend changes the graph.
Items for the conv-state concat exist only for CPU, HIP, Vulkan and Hexagon (for example [#31179](https://github.com/ggml-org/llama.cpp/pull/31179)).

### 3.e MMQ tile tuning for sm_86

**Question.** Does upstream tune MMQ tiles, `mmq_x`, `mmq_y` or stream-k for sm_86 with Q4_0 or Q8_0?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#24127](https://github.com/ggml-org/llama.cpp/pull/24127) | CUDA: refactor MMQ kernel configuration | Merged [`6eddde06a`](https://github.com/ggml-org/llama.cpp/commit/6eddde06a) | 2026-06-04 to 2026-07-13 | Moves MMQ tile sizes into per-architecture config tables. Q4_0 and Q8_0 on Ampere sit in `mmq-config-ampere.cuh`. |
| [#22298](https://github.com/ggml-org/llama.cpp/pull/22298) | CUDA: reduce MMQ stream-k overhead | Merged [`9725a313b`](https://github.com/ggml-org/llama.cpp/commit/9725a313b) | 2026-04-23 to 2026-04-25 | Reduces the stream-k overhead in MMQ. |
| [#30553](https://github.com/ggml-org/llama.cpp/pull/30553) | cuda: cap MMQ Stream-K blocks at available work | Merged [`4c1004c49`](https://github.com/ggml-org/llama.cpp/commit/4c1004c49) | 2026-10-14 to 2026-10-19 | Caps stream-k blocks at the available work. |
| [#26079](https://github.com/ggml-org/llama.cpp/pull/26079) | CUDA: adding switch points per HW and quant type to tune the mvq->MMQ decode… | Merged [`2b5621094`](https://github.com/ggml-org/llama.cpp/commit/2b5621094) | 2026-07-24 to 2026-08-20 | Sets MMVQ to MMQ switch points per GPU and quantization type. |
| [#28285](https://github.com/ggml-org/llama.cpp/pull/28285) | tune MMVQ to MMQ crossover for SM87 | Merged [`8c1a25166`](https://github.com/ggml-org/llama.cpp/commit/8c1a25166) | 2026-09-03 to 2026-09-03 | Tunes the MMVQ to MMQ switch for SM87. |
| [#28912](https://github.com/ggml-org/llama.cpp/pull/28912) | tune MMVQ to MMQ crossover for SM70 (Volta) | Merged [`68d9053af`](https://github.com/ggml-org/llama.cpp/commit/68d9053af) | 2026-09-14 to 2026-09-21 | Tunes the MMVQ to MMQ switch for SM70. |
| [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) | Feature Request: Add an sm_86 entry to the MMVQ cutoff table (Q4_0 crosses at… | Open | 2026-08-31 to open | Asks for an sm_86 entry in the switch table. Evidence says the best switch depends on SM count and not on compute capability. |
| [#28204](https://github.com/ggml-org/llama.cpp/pull/28204) | cuda : MMQ q4_0 small-batch tile config for Ampere+ | Closed, not merged | 2026-09-01 to 2026-09-10 | Small-batch Q4_0 tile for Ampere and newer. The author closed it without review. |
| [#29330](https://github.com/ggml-org/llama.cpp/pull/29330) | cuda: optimize MMQ configuration for prefill phase with larger tile sizes | Closed, not merged | 2026-09-23 to 2026-09-23 | Larger prefill tiles. Closed for a template violation. |
| [#21849](https://github.com/ggml-org/llama.cpp/pull/21849) | ggml-cuda: make mul_mat_q tile selection tunable per-type and per-arch | Open | 2026-04-13 to open | Per-type and per-architecture tile choice. Replaced by #24127. |
| [#30493](https://github.com/ggml-org/llama.cpp/pull/30493) | ggml-cuda: vectorize the MMQ activation tile copy to int2 | Open, draft | 2026-10-14 to open | Vectorizes the MMQ activation tile copy. It loses up to 8 percent on a 4090. |
| [#30235](https://github.com/ggml-org/llama.cpp/issues/30235) | CUDA: Pascal DP4A MMQ config uses I=64 for Q8_0, I=128 gives +12% pp512 on sm… | Open | 2026-10-09 to open | Reports that I=128 for Q8_0 gives +12 percent on Pascal. |
| [#30797](https://github.com/ggml-org/llama.cpp/pull/30797) | ggml-cuda: use MMA for Q5_K, Q4_K, Q4_1, Q5_1, IQ1_S and Q2_K J128 minimum co… | Open | 2026-10-19 to open | Adds an MMA path for more K-quants. Not Q4_0. |

**Master since b12004.** No commit touches `mmq*`.

**Code in b12004.**

- Tile tables sit in `mmq-config-ampere.cuh` (383 lines). The refactor [#24127](https://github.com/ggml-org/llama.cpp/pull/24127) renamed `mmq_x` and `mmq_y` to J and I.
- The table macro is `CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)`. See [`ggml/src/ggml-cuda/mmq.cuh:208`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L208).
- Q4_0 and Q8_0 on Ampere use 256 threads, occupancy 1 and I=128. J runs from 8 to 128 in steps of 8. Stream-k is on.
- MMQ loads A tiles with `ldmatrix` but has no `cp.async` use. Only the flash-attention code includes `cp-async.cuh`.
- MMVQ handles batches of up to 8 columns (`MMVQ_MAX_BATCH_SIZE`) on sm_86. The switch table has no sm_86 entry. See [#28090](https://github.com/ggml-org/llama.cpp/issues/28090).

**Verdict.** Already in b12004 for tile tables and stream-k. A Q4_0 small-batch tile change was proposed and closed without review ([#28204](https://github.com/ggml-org/llama.cpp/pull/28204)).

**Implication.** Tuning for sm_86 means editing the Ampere table. No open upstream item competes with it.
The evidence in [#28090](https://github.com/ggml-org/llama.cpp/issues/28090) says the best switch point moves with the GPU and its clock. A power-capped RTX 3060 Ti may behave like the pinned-clock A10, so a local sweep is useful.

### 3.f MMQ redesign or another int8 GEMM

**Question.** Does upstream plan a new MMQ design, or a CUTLASS, cuBLASLt, `ldmatrix` or `cp.async` int8 GEMM?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#28572](https://github.com/ggml-org/llama.cpp/pull/28572) | ggml-cuda: pipeline NVFP4 MMQ tile loads with cp.async and TMA on Blackwell (… | Closed, not merged | 2026-09-07 to 2026-09-17 | Pipelines NVFP4 tile loads with cp.async and TMA. It gives +14 percent on Blackwell. A maintainer closed it and cited own plans for the kernel. |
| [#26704](https://github.com/ggml-org/llama.cpp/pull/26704) | CUDA: Add experimental SM120 CUTLASS MoE prefill for MXFP4 and NVFP4 | Open | 2026-08-07 to open | Experimental CUTLASS prefill for MXFP4 and NVFP4 on SM120. Blackwell only. |
| [#28514](https://github.com/ggml-org/llama.cpp/discussions/28514) | CUDA: NVFP4 prefill on Blackwell leaves ~1.45x performance on the table. Comp… | Open | 2026-09-06 to open | Discussion with a proof of concept. NVFP4 prefill on Blackwell leaves about 1.45 times on the table. |
| [#24777](https://github.com/ggml-org/llama.cpp/pull/24777) | cuda: opt-in Hopper (sm_90a) wgmma path for Q1_0 prefill | Closed, not merged | 2026-06-18 to 2026-06-19 | Hopper wgmma path for Q1_0 prefill. Not Ampere. |
| [#29699](https://github.com/ggml-org/llama.cpp/pull/29699) | musa: opt-in muDNN INT8 `MUL_MAT`, a work-gated decode path, and vectorized `… | Closed, not merged | 2026-09-30 to 2026-09-30 | MUSA muDNN INT8 path. The target is not an NVIDIA GPU. |
| [#7860](https://github.com/ggml-org/llama.cpp/pull/7860) | CUDA: int8 tensor cores for MMQ (q4_K, q5_K, q6_K) | Merged [`bdcb8f422`](https://github.com/ggml-org/llama.cpp/commit/bdcb8f422) | 2024-06-10 to 2024-06-11 | Introduced int8 tensor cores for MMQ. Ancestor of the present kernel. |

**Master since b12004.** No commit touches `mmq*` or the cuBLAS call sites.

**Code in b12004.**

- The CUDA backend links the cuBLASLt library but has no int8 cuBLASLt call. The cuBLAS calls are GemmEx and batched variants. See [`ggml/src/ggml-cuda/ggml-cuda.cu:1697-1750`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L1697-L1750).
- No source file uses CUTLASS headers.
- `cp.async` appears in `fattn-mma-f16.cuh` and `cp-async.cuh` only.

**Verdict.** Nothing found for Ampere or for Q4_0. The Blackwell cp.async change [#28572](https://github.com/ggml-org/llama.cpp/pull/28572) was proposed and rejected. A reviewer wrote that it adds too much technical debt and clashes with the reviewer's own plans for the kernel.
The CUTLASS item [#26704](https://github.com/ggml-org/llama.cpp/pull/26704) is open and limited to SM120.

**Implication.** A cp.async pipeline for MMQ on sm_86 would be new. A kernel owner has plans, so a local patch may conflict with future upstream work. Keep the patch small and contained in the MMQ files.
A reviewer asked the CUTLASS author to drop MXFP4 support and async loading of repacked tensors. Expect resistance to large structural changes.

### 3.g Megakernel or persistent kernel

**Question.** Does upstream have a megakernel or a persistent kernel for CUDA decode or prefill?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331) | Experiment: tile-level task DAG executor for batch=1 CPU decode (+13.7% tg on… | Open | 2026-08-18 to open | Task-DAG executor for CPU decode. Gives 13.7 percent on one Apple chip. Not CUDA. |
| [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) | CUDA: chunk graphs to amortize graph launch | Merged [`280141e6d`](https://github.com/ggml-org/llama.cpp/commit/280141e6d) | 2026-10-19 to 2026-10-24 | Chunked CUDA graphs. The closest CUDA work. It reduces launch gaps and is not a megakernel. |
| [#22522](https://github.com/ggml-org/llama.cpp/pull/22522) | Programmatic Dependent Launch (PDL) for more performance on newer NVIDIA GPUs… | Merged [`e94722822`](https://github.com/ggml-org/llama.cpp/commit/e94722822) | 2026-04-29 to 2026-05-20 | Programmatic dependent launch. Hopper and newer only. |
| [#29247](https://github.com/ggml-org/llama.cpp/pull/29247) | rfc: Memory eliding fusions | Open, draft | 2026-09-21 to open | RFC for fusions that elide memory traffic. Not a persistent kernel. |
| [#29187](https://github.com/ggml-org/llama.cpp/pull/29187) | CUDA: Fuse GDN alpha/beta projections | Open, draft | 2026-09-20 to open | A reviewer suggested NVRTC auto-fusion. A short discussion of the scope followed. |

**Master since b12004.** No commit touches CUDA code.

**Verdict.** Nothing found. Thirteen queries covered megakernels, persistent kernels and whole-layer kernels. They are g01 to g08 in `queries.tsv` and G01 to G05 in `queries2.tsv`. None returned a CUDA item for this idea. The hits for "persistent" are about MoE expert caches and prompt caches.

The listed items are loosely related. [#27331](https://github.com/ggml-org/llama.cpp/discussions/27331) is a CPU experiment. [#30804](https://github.com/ggml-org/llama.cpp/pull/30804) reduces launch gaps with chunked graphs and is not a megakernel.

**Implication.** No prior art exists to reuse or to collide with. Chunked graphs already remove part of the launch overhead in decode.

### 3.h Gated DeltaNet CUDA kernels

**Question.** What CUDA work exists for Gated DeltaNet, especially for prefill and for short or ragged prompts?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#19504](https://github.com/ggml-org/llama.cpp/pull/19504) | ggml: add GATED_DELTA_NET op | Merged [`c5a778891`](https://github.com/ggml-org/llama.cpp/commit/c5a778891) | 2026-02-11 to 2026-03-07 | Added the GATED_DELTA_NET op. |
| [#20340](https://github.com/ggml-org/llama.cpp/pull/20340) | llama : enable chunked fused GDN path | Merged [`d28961d81`](https://github.com/ggml-org/llama.cpp/commit/d28961d81) | 2026-03-10 to 2026-03-11 | Enabled the chunked fused GDN path in the graph. |
| [#20537](https://github.com/ggml-org/llama.cpp/pull/20537) | CUDA: GDN hide memory latency | Merged [`34818ea6c`](https://github.com/ggml-org/llama.cpp/commit/34818ea6c) | 2026-03-14 to 2026-03-16 | Hides memory latency in the CUDA GDN kernel. |
| [#30087](https://github.com/ggml-org/llama.cpp/pull/30087) | ggml-cuda: assign four GDN state columns per warp | Merged [`24e41838e`](https://github.com/ggml-org/llama.cpp/commit/24e41838e) | 2026-10-07 to 2026-10-08 | Assigns four GDN state columns per warp. |
| [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) | cuda, hip: GDN chunked kernel | Merged [`62df1815b`](https://github.com/ggml-org/llama.cpp/commit/62df1815b) | 2026-09-24 to 2026-10-14 | Adds the chunked GDN CUDA kernel in `gated-delta-net-mma.cu`. It gives 3 to 6 percent prompt gain on a dense model and 11 to 21 percent on a MoE model. |
| [#23940](https://github.com/ggml-org/llama.cpp/pull/23940) | Remove redundant CUDA copies after gated_delta_net. | Merged [`5a460dea9`](https://github.com/ggml-org/llama.cpp/commit/5a460dea9) | 2026-05-31 to 2026-07-03 | Removes copies after the GDN op. |
| [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | CUDA: Support of GDN chunked kernel for prefill | Closed, not merged | 2026-07-22 to 2026-10-06 | Earlier chunked kernel. Replaced by #29353. |
| [#24561](https://github.com/ggml-org/llama.cpp/pull/24561) | CUDA/HIP: chunked MFMA prefill kernel for GATED_DELTA_NET (CDNA) | Closed, not merged | 2026-06-13 to 2026-07-10 | CDNA MFMA version. The author closed it unless someone volunteers to maintain it. |
| [#22587](https://github.com/ggml-org/llama.cpp/pull/22587) | CUDA: row-per-warp kernel for GATED_DELTA_NET | Open | 2026-05-01 to open | Row-per-warp decode kernel. A reviewer wants the decode kernel to stay simple. |
| [#21897](https://github.com/ggml-org/llama.cpp/pull/21897) | ggml-cuda: enable concurrent streams for linear attention | Open | 2026-04-14 to open | Concurrent streams for linear attention. |
| [#18102](https://github.com/ggml-org/llama.cpp/pull/18102) | ggml-cuda: Delta-Net linear attention for Qwen3-Next | Open | 2025-12-16 to open | Old delta-net kernel. Stale. |
| [#25162](https://github.com/ggml-org/llama.cpp/issues/25162) | Performance regression on Turing GPUs after 9e58d4d69 — Qwen35 SSM kernels af… | Closed | 2026-06-30 to 2026-08-03 | Turing regression after a GDN change. |

**Master since b12004.** No commit touches `gated*`.

**Code in b12004.**

- [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) added `gated-delta-net-mma.cu`. It runs only for a scalar gate and head size 128. It also needs `H_k == 16`, `H` of 16, 32, 48 or 64, and 64 or more tokens in each sequence. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391) and [`ggml/src/ggml-cuda/gated_delta_net.cu:276`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated_delta_net.cu#L276).
- The same condition needs op parameter 0 to be smaller than the token count. That parameter sets how many state snapshots the op keeps.
- The reported gain in prompt processing is 3 to 6 percent on a dense 27B model. It is 11 to 21 percent on a 35B MoE model.
- `gated_delta_net.cu` keeps the autoregressive kernel and a TODO for a faster prefill kernel. See [`ggml/src/ggml-cuda/gated_delta_net.cu:192`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated_delta_net.cu#L192).
- The decode-side tuning commits are [`34818ea6c`](https://github.com/ggml-org/llama.cpp/commit/34818ea6c) ([#20537](https://github.com/ggml-org/llama.cpp/pull/20537)) and [`24e41838e`](https://github.com/ggml-org/llama.cpp/commit/24e41838e) ([#30087](https://github.com/ggml-org/llama.cpp/pull/30087)).

**Verdict.** Already in b12004 for the chunked prefill kernel. Nothing found for variable-length or short-prompt cases.

**Implication.** Do not rebuild the chunked kernel. Check how often decision prompts have fewer than 64 tokens per sequence, because those rows use the slower autoregressive kernel [INFERENCE].
A reviewer wants the autoregressive kernel to stay simple ([#22587](https://github.com/ggml-org/llama.cpp/pull/22587)). Decode-kernel rewrites may meet resistance.

### 3.i Server decision API

**Question.** Did the `/v1/systemone` API or its server code change in ways that affect speed?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#29818](https://github.com/ggml-org/llama.cpp/pull/29818) | llama, server: add /v1/systemone API (models: laya, julia-1, lev, openjev, ke… | Merged [`a4cb4c61f`](https://github.com/ggml-org/llama.cpp/commit/a4cb4c61f) | 2026-10-01 to 2026-10-02 | Adds `/v1/systemone` and five decision models. |
| [#30361](https://github.com/ggml-org/llama.cpp/pull/30361) | model : add H2O-Lightning-4B decision model | Merged [`7e3eeee04`](https://github.com/ggml-org/llama.cpp/commit/7e3eeee04) | 2026-10-12 to 2026-10-20 | Adds the H2O-Lightning-4B decision type. |
| [#30567](https://github.com/ggml-org/llama.cpp/pull/30567) | server: prefer empty slots and keep states overwritten by child tasks | Merged [`b0d239164`](https://github.com/ggml-org/llama.cpp/commit/b0d239164) | 2026-10-15 to 2026-10-15 | Takes empty slots first for child tasks and saves overwritten states to the prompt cache. |
| [#30498](https://github.com/ggml-org/llama.cpp/pull/30498) | server: save only restorable slot states to the prompt cache | Merged [`2639a3b4e`](https://github.com/ggml-org/llama.cpp/commit/2639a3b4e) | 2026-10-14 to 2026-10-15 | Saves only restorable slot states to the prompt cache. |
| [#29856](https://github.com/ggml-org/llama.cpp/pull/29856) | graph: fix CI realloc abort by gathering the recurrent states once | Merged [`436f6f89e`](https://github.com/ggml-org/llama.cpp/commit/436f6f89e) | 2026-10-02 to 2026-10-03 | Gathers recurrent states once in `build_rs`. Decode speed unchanged. |
| [#30408](https://github.com/ggml-org/llama.cpp/pull/30408) | model: fix vllm d3 multi seq case | Merged [`cbe28bb3c`](https://github.com/ggml-org/llama.cpp/commit/cbe28bb3c) | 2026-10-12 to 2026-10-13 | Fixes the non-causal hybrid multi-sequence case. |
| [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) | server: visit parent slots before child slots | Open | 2026-10-18 to open | Visits parent slots before child slots. Without it a child with a lower slot id than its parent runs its suffix in a third decode pass. |
| [#30710](https://github.com/ggml-org/llama.cpp/pull/30710) | server: cap shared decision prefixes at the first output | Open | 2026-10-17 to open | Caps the shared prefix at the first output position. Fixes a 500 error when questions share option tokens. |
| [#30482](https://github.com/ggml-org/llama.cpp/issues/30482) | Misc. bug: /v1/systemone returns 500 "the question and its options must fit i… | Open | 2026-10-13 to open | Issue behind #30710. `/v1/systemone` returns 500 when two questions share a prefix. |
| [#30682](https://github.com/ggml-org/llama.cpp/pull/30682) | feat: Do not disable prompt caching for text + decision models | Open, draft | 2026-10-16 to open | Keeps prompt caching for text plus decision models. |
| [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | server : share prompt prefix cells between slots (--slot-attach) | Open | 2026-10-22 to open | Shares prompt prefix cells between slots with `--slot-attach`. |
| [#30683](https://github.com/ggml-org/llama.cpp/pull/30683) | llama-server: Avoids extra copying of existing checkpoints | Open | 2026-10-16 to open | Reuses existing checkpoints instead of copying again. |
| [#29918](https://github.com/ggml-org/llama.cpp/pull/29918) | server : add --cache-reuse-hybrid for recurrent/hybrid models | Open, draft | 2026-10-03 to open | Adds `--cache-reuse-hybrid` for recurrent and hybrid models. |
| [#26204](https://github.com/ggml-org/llama.cpp/pull/26204) | server : add /slots endpoint action=clone_to (KV clone between slots) | Open | 2026-07-27 to open | Adds a slot clone endpoint. |
| [#30780](https://github.com/ggml-org/llama.cpp/discussions/30780) | server : reuse image features within a SystemOne request | Open | 2026-10-18 to open | Discussion about sharing image features inside one decision request. |
| [#30073](https://github.com/ggml-org/llama.cpp/issues/30073) | Eval bug: decision models fail on large input size | Open | 2026-10-07 to open | Reports that decision models fail on large input. |
| [#30064](https://github.com/ggml-org/llama.cpp/issues/30064) | Eval bug: Clef-GGUF Q8_0 /v1/systemone probabilities collapse toward uniform,… | Closed | 2026-10-06 to 2026-10-07 | Q8_0 decision probabilities collapsed toward uniform while BF16 stayed correct. |
| [#29902](https://github.com/ggml-org/llama.cpp/issues/29902) | Misc. bug: llama-server with laya model aborts when the questions of a batch… | Closed | 2026-10-03 to 2026-10-03 | A batch whose questions exceed the micro-batch size aborted. |

**Master since b12004.** No commit touches `tools/server/server-decision.cpp` or `tools/server/server-context.cpp`.

**Code in b12004.**

- `server_decision_group_tasks` puts questions in groups of up to one task per slot. The first task is the parent. It evaluates the shared prefix. See [`tools/server/server-decision.cpp:1753-1783`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-decision.cpp#L1753-L1783).
- The children wait for the parent. The parent then copies its state to each child. See [`tools/server/server-context.cpp:4360-4372`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L4360-L4372) and [`tools/server/server-context.cpp:4785-4789`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L4785-L4789).
- The copy helpers call `mem.seq_cp` and clone the prompt. See [`tools/server/server-context.cpp:845-864`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/tools/server/server-context.cpp#L845-L864).
- [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) says that since [#30567](https://github.com/ggml-org/llama.cpp/pull/30567) a child task can get a slot id below the slot id of its parent. The server visits slots in id order. The child then misses one batch and runs its suffix in a third decode pass.
- The fix [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) is open. Its test shows 5 waits across two requests on master and 4 with the fix.
- The API first shipped in release b11361, as a comment on [#29818](https://github.com/ggml-org/llama.cpp/pull/29818) states. The H2O-Lightning decision type came in [`7e3eeee04`](https://github.com/ggml-org/llama.cpp/commit/7e3eeee04).

**Verdict.** Core API: already in b12004. Speed fixes: in progress upstream. No item targets the label-logit readout.

**Implication.** Cherry-pick [#30779](https://github.com/ggml-org/llama.cpp/pull/30779) if the production scheme has parent and child slots with ids in the wrong order. Check with a trace first.
Prompt-cache items ([#30682](https://github.com/ggml-org/llama.cpp/pull/30682), [#30970](https://github.com/ggml-org/llama.cpp/pull/30970), [#30683](https://github.com/ggml-org/llama.cpp/pull/30683), [#29918](https://github.com/ggml-org/llama.cpp/pull/29918)) matter for repeated prefixes. They are watch items.

### 3.j Hybrid state copy cost

**Question.** What does upstream do about the cost of copying and gathering recurrent state in hybrid models?

| Item | Title | State | Dates | Relevance |
|---|---|---|---|---|
| [#29856](https://github.com/ggml-org/llama.cpp/pull/29856) | graph: fix CI realloc abort by gathering the recurrent states once | Merged [`436f6f89e`](https://github.com/ggml-org/llama.cpp/commit/436f6f89e) | 2026-10-02 to 2026-10-03 | Gathers recurrent states once. Fixes a reserve problem. |
| [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) | Feature Request: skip redundant recurrent-state gather when n_rs == 1 && n_se… | Open | 2026-09-13 to open | Asks to skip the 96 redundant state gathers when one sequence uses one recurrent slot. Measured 2.2 percent decode cost on Vulkan. |
| [#29121](https://github.com/ggml-org/llama.cpp/pull/29121) | llama : update the GDN recurrent state in place on the CPU (#28841) | Open | 2026-09-19 to open | In-place GDN state update on CPU. |
| [#27991](https://github.com/ggml-org/llama.cpp/pull/27991) | kv cache : optimize restoring non-contiguous cells | Merged [`2d8d612e4`](https://github.com/ggml-org/llama.cpp/commit/2d8d612e4) | 2026-08-29 to 2026-08-31 | Speeds up restore of non-contiguous KV cells. |
| [#26004](https://github.com/ggml-org/llama.cpp/pull/26004) | server : preserve context checkpoints across slot save/restore | Merged [`033df86b6`](https://github.com/ggml-org/llama.cpp/commit/033df86b6) | 2026-07-22 to 2026-10-08 | Keeps checkpoints across slot save and restore. |
| [#20087](https://github.com/ggml-org/llama.cpp/pull/20087) | Hybrid model cache: add `--checkpoint-every-nb` | Merged [`f5ddcd169`](https://github.com/ggml-org/llama.cpp/commit/f5ddcd169) | 2026-03-03 to 2026-03-06 | Adds `--checkpoint-every-nb` for hybrid models. |
| [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) | llama-server: Use reference-counted checkpoints instead of deep-copies | Open | 2026-10-08 to open | Uses reference-counted checkpoints instead of deep copies. |
| [#30683](https://github.com/ggml-org/llama.cpp/pull/30683) | llama-server: Avoids extra copying of existing checkpoints | Open | 2026-10-16 to open | Reuses existing checkpoints instead of copying again. |
| [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | server : share prompt prefix cells between slots (--slot-attach) | Open | 2026-10-22 to open | Shares prefix cells between slots. |
| [#29918](https://github.com/ggml-org/llama.cpp/pull/29918) | server : add --cache-reuse-hybrid for recurrent/hybrid models | Open, draft | 2026-10-03 to open | Adds `--cache-reuse-hybrid`. |
| [#26204](https://github.com/ggml-org/llama.cpp/pull/26204) | server : add /slots endpoint action=clone_to (KV clone between slots) | Open | 2026-07-27 to open | Adds a slot clone endpoint. |
| [#24785](https://github.com/ggml-org/llama.cpp/pull/24785) | server: add recurrent state shrink/expand for prompt cache (#22746) | Open | 2026-06-18 to open | Recurrent state shrink and expand for the prompt cache. |
| [#30252](https://github.com/ggml-org/llama.cpp/issues/30252) | server : saving idle slots to the prompt cache stalls all requests for second… | Open | 2026-10-09 to open | Saving idle slots stalls requests when KV cells interleave. |
| [#31391](https://github.com/ggml-org/llama.cpp/pull/31391) | cuda : gather batched tensor reads into mapped host memory (PR to master) | Open | 2026-10-29 to open | Gathers batched tensor reads into mapped host memory. Idle slot save drops from 140 to 7 ms in one test. |
| [#29324](https://github.com/ggml-org/llama.cpp/issues/29324) | Misc. bug: Misc. bug: --cache-ram -1 is not "no limit": the token cap stays a… | Open | 2026-09-23 to open | `--cache-ram -1` lets host memory grow with hybrid models. |
| [#28118](https://github.com/ggml-org/llama.cpp/pull/28118) | server : keep speculative recurrent-state checkpoints on-device | Open, draft | 2026-08-31 to open | Keeps speculative checkpoints on the device. |
| [#31368](https://github.com/ggml-org/llama.cpp/pull/31368) | memory: refuse rollbacks past the snapshots of the last ubatch | Merged [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc), after b12004 | 2026-10-29 to 2026-10-30 | Post-b12004 commit. Limits rollback to the snapshots of the last micro-batch. |

**Master since b12004.** The only commit is [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc). It adds a per-cell `rs_max` field and changes `seq_rm` in `src/llama-memory-recurrent.cpp`. It adds no tensor copy.

**Code in b12004.**

- Recurrent `seq_cp` links the destination sequence to the source cell. It copies no tensor data. See [`src/llama-memory-recurrent.cpp:262-297`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-memory-recurrent.cpp#L262-L297).
- The data move happens in the graph. Each `build_rs` call runs one `ggml_get_rows` over the states and copies the extra states back. See [`src/llama-graph.cpp:3937-3949`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/src/llama-graph.cpp#L3937-L3949).
- Qwen3.5 calls `build_rs` twice per linear layer, once for the conv state and once for the SSM state. [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) counts 96 gathers per graph for a model with 48 linear layers.
- [INFERENCE] A child slot pays the state copy on its first decode, because its cell is shared with the parent.
- [#28841](https://github.com/ggml-org/llama.cpp/issues/28841) measured 2.2 percent of decode time for the gathers on Vulkan with one sequence, at a commit before [#29856](https://github.com/ggml-org/llama.cpp/pull/29856). The cost for CUDA with several sequences is not measured here.
- A commenter on [#29324](https://github.com/ggml-org/llama.cpp/issues/29324) computed about 3 MiB of SSM state per layer for a 27B model with 48 value heads. The state size of the production model is not measured here.

**Verdict.** In progress upstream, in parts. Nothing found that reduces the cost of `seq_cp` or of the per-pass gather for multi-sequence batches.

**Implication.** Measure the gather and copy kernels for `-np 3` before any work. Upstream items focus on prompt-cache reuse and host memory, so a GPU-side gather fix would be new.


## 4. Commits since b12004

Command: `git log --format='%h %cd %s' --date=short b12004..origin/master`. It lists 13 commits. The tag b12017 points at the newest one.

Command for the requested paths:
`git log --format='%h %cd %s' --date=short b12004..origin/master -- 'ggml/src/ggml-cuda/mmq*' 'ggml/src/ggml-cuda/gated*' 'ggml/src/ggml-cuda/fattn*' 'ggml/src/ggml-cuda/ggml-cuda.cu' 'ggml/src/ggml-cuda/norm.cu' 'ggml/src/ggml-cuda/quantize.cu' 'src/llama-batch.cpp' 'src/llama-memory-*.cpp' 'tools/server/server-decision.cpp'`.
Result: one commit.

| Commit | Date | Subject | Touches a requested path | Effect on targets a to j |
|---|---|---|---|---|
| [`5047748f1`](https://github.com/ggml-org/llama.cpp/commit/5047748f1) | 2026-10-30 | rpc : delete fs_path_to_utf8() (#31424) | No | None. Outside the target paths. |
| [`792fcf4e7`](https://github.com/ggml-org/llama.cpp/commit/792fcf4e7) | 2026-10-30 | opencl: add initial iq1 support (#31341) | No | None. OpenCL backend, outside the target paths. |
| [`a34de8618`](https://github.com/ggml-org/llama.cpp/commit/a34de8618) | 2026-10-30 | common : delete fs_path_to_utf8() (#31423) | No | None. Outside the target paths. |
| [`aa588a346`](https://github.com/ggml-org/llama.cpp/commit/aa588a346) | 2026-10-30 | common : drop the Windows argv reconstruction (#31417) | No | None. Outside the target paths. |
| [`889f2ec96`](https://github.com/ggml-org/llama.cpp/commit/889f2ec96) | 2026-10-30 | quant : allow quantization for LFM3-Audio codec decoder (#31386) | No | None. Outside the target paths. |
| [`1ab23c33b`](https://github.com/ggml-org/llama.cpp/commit/1ab23c33b) | 2026-10-30 | vendor : update cpp-httplib to 0.64.0 (#31400) | No | None. Outside the target paths. |
| [`27b8464e8`](https://github.com/ggml-org/llama.cpp/commit/27b8464e8) | 2026-10-30 | ci : add missing ccache-clear to windows behavior check (#31413) | No | None. Outside the target paths. |
| [`d63fbfe79`](https://github.com/ggml-org/llama.cpp/commit/d63fbfe79) | 2026-10-30 | cmake : embed an app manifest in all Windows exes (#31329) | No | None. Outside the target paths. |
| [`e216f4ec8`](https://github.com/ggml-org/llama.cpp/commit/e216f4ec8) | 2026-10-30 | ci : extract most apple jobs into daily workflow (#31408) | No | None. Outside the target paths. |
| [`344846b4f`](https://github.com/ggml-org/llama.cpp/commit/344846b4f) | 2026-10-30 | gemma3n, gemma4 : create token_embd before the tied output (#31394) | No | None. Gemma model loading, outside the target paths. |
| [`fb5997274`](https://github.com/ggml-org/llama.cpp/commit/fb5997274) | 2026-10-30 | kv-cache : keep rotation constants on the GPU (#31073) | No | None. KV cache rotation constants move to the GPU. It touches `llama-graph.cpp`, not a target path. |
| [`ab3768cdc`](https://github.com/ggml-org/llama.cpp/commit/ab3768cdc) | 2026-10-30 | memory: refuse rollbacks past the snapshots of the last ubatch (#31368) | Yes: `src/llama-memory-recurrent.cpp` | None expected. It limits recurrent rollback to the last micro-batch snapshots. The server does not use rollback [INFERENCE]. |
| [`e0f142d5d`](https://github.com/ggml-org/llama.cpp/commit/e0f142d5d) | 2026-10-29 | opencl: add initial q2_0 support (#31340) | No | None. OpenCL backend, outside the target paths. |

The full list with file names is in `commits-since-b12004.tsv`.
Directory checks: `git log b12004..origin/master -- ggml/src/ggml-cuda` returns nothing. `tools/server` shows two commits, which touch `server-models.cpp` and `server-tools.cpp`. `src/llama-context.cpp` shows none.

## 5. Other findings that affect the production flags

1. **FA mask skipping with unified KV (open, [#28495](https://github.com/ggml-org/llama.cpp/issues/28495)).**
   - The CUDA flash-attention code scans the mask for the last needed KV cell in two cases. It needs `Q->ne[1] >= 1024` or `Q->ne[3] > 1`. See [`ggml/src/ggml-cuda/fattn-common.cuh:1117`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/fattn-common.cuh#L1117).
   - A unified cache gives `Q->ne[3] == 1`, so short batches skip the scan. Each sequence then walks cells that belong to other sequences.
   - Reporters measured 42 to 54 percent lower prompt speed on HIP. A V100 user measured about 46 to 49 percent with a second conversation resident.
   - The production flags use `-np 3 --kv-unified`. Only 8 of the 32 layers use attention, so the effect on this model needs a measurement [INFERENCE].
2. **Slot order in the decision grouping (open, [#30779](https://github.com/ggml-org/llama.cpp/pull/30779)).** See section 3.i. The issue appears after [#30567](https://github.com/ggml-org/llama.cpp/pull/30567), which is already in b12004.
3. **Gap slots with the split cache (closed, [#25840](https://github.com/ggml-org/llama.cpp/issues/25840)).** With the default split cache, busy slots with a gap cannot share one micro-batch. The production flag `--kv-unified` avoids this.
4. **MMQ fusion type list.** Q4_0 is not in the list at [`ggml/src/ggml-cuda/mmq.cuh:437-440`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/mmq.cuh#L437-L440). The FFN of the production model is Q4_0 [INFERENCE: from the model recipe].
5. **GDN token threshold.** The chunked kernel needs 64 or more tokens per sequence. See [`ggml/src/ggml-cuda/gated-delta-net-mma.cu:388-391`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/gated-delta-net-mma.cu#L388-L391).
6. **CUDA graph warmup.** A graph needs two identical consecutive calls. See [`ggml/src/ggml-cuda/ggml-cuda.cu:5016-5020`](https://github.com/ggml-org/llama.cpp/blob/3c809f4/ggml/src/ggml-cuda/ggml-cuda.cu#L5016-L5020).
7. **No MMQ cp.async.** Only the flash-attention files use `cp-async.cuh`.
8. **A decision-model quality item.** [#30064](https://github.com/ggml-org/llama.cpp/issues/30064) reported that Q8_0 decision probabilities collapse toward uniform while BF16 stays correct. The issue is closed. Check that the mix8 GGUF does not show the same effect before any speed claim.

## 6. Method and limits

Search method:

1. Write queries in `queries.tsv` (151 queries) and `queries2.tsv` (49 queries). Each has an ID, a target letter and a text.
2. Run `upstream_search.py`. It calls the GitHub GraphQL API through the `gh` command. Each query runs against issues, pull requests and discussions of the repository.
3. Take the top 30 hits for each type and query. The scripts write `search-raw.json`, `search-results.tsv`, `search2-raw.json` and `search2-results.tsv`.
4. Read the issue, the PR files and the comments of 169 items with the REST API. The raw JSON is in `items/`.
5. Read the b12004 source files that the survey cites. Read the git history of a blobless clone of master.

Counts:

- 200 queries returned 7,871 rows. They contain 2,322 unique items: 1,113 pull requests, 891 issues and 318 discussions.
- The first 151 queries have 6,956 rows and 2,211 unique items. The 49 gap queries added 111 new items. None of the new items changes a verdict.
- Dates of merged PRs come from the commit date on master. Dates of other items come from GitHub.

Limits:

- GitHub search is lexical. An item with unusual words can be missed. The top 30 limit can hide older items for broad queries.
- The survey did not read every diff. It read titles, descriptions, files and comments of the cited items.
- "Nothing found" means the queries returned no matching item. It does not prove that no work exists on private branches or forks.
- States are from 2026-10-30. Open items can change.
- Claims marked [INFERENCE] come from code reading and not from a run. No benchmark ran for this survey.
- The raw files hold author names for the search hits. This document names no authors.

## 7. Files

All files are in the `phase0` folder next to this document.

- `upstream-survey.md`: this document.
- `queries.tsv`, `queries2.tsv`: the 200 search queries.
- `upstream_search.py`: the search script.
- `search-raw.json`, `search-results.tsv`: output of the first 151 queries.
- `search2-raw.json`, `search2-results.tsv`: output of the 49 gap queries.
- `items/`: one JSON file for each item that was read in detail.
- `commits-since-b12004.tsv`: the 13 commits after b12004 with file names.
