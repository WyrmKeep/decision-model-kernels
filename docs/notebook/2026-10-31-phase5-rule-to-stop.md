# Phase 5: the rule to stop after the tests of the changes

Written 2026-10-31 at about 21:30 UTC.

## Task of this entry

This entry applies the [rule to stop](2026-10-31-phase4-ranked-targets.md#rule-to-stop-the-project) of phase 4 again, after the tests of the changes of phase 5. The rule stops the project when the expected gain is less than 3 % for each request type.

## Results of the tests of the targets

| Target | Change | Condition | Result |
|---|---|---|---|
| `E` | [0001](../changes/0001-glu-into-quantize.md) | In the stack | `lat_ms` -1.94 % |
| `P` | [0007](../changes/0007-input-copies.md) | In the stack | `lat_ms` -0.93 % |
| `D` | [0002](../changes/0002-gdn-value-split.md) | Not used | The value-split path is 1.15 to 1.31 times as slow |
| `B` | [0004](../changes/0004-gpu-mask.md) | In the stack | `lat_ms` -1.06 % |
| `C` | [0005](../changes/0005-concat-conv.md) | In the stack | `lat_ms` -3.02 % |
| `H` | No change document | Not used | No rule that the code can use gives a gain |
| `J` | [0008](../changes/0008-ragged-suffix.md) | Not used | `lat_ms` -2.46 %, but one limit for the error is not true |
| `G` | No change document | Not used | 1.042 times at 1,025 tokens or more. The limit was 1.08. |
| `A` | [0003](../changes/0003-answer-head.md) | In the stack | `lat_ms` -1.48 % on the stack |

Source: the change documents and the result entries in the [notebook](README.md).

The [second test of the stack](2026-10-31-stack-2-prediction.md) compares the stack with the image in production. The stack is approximately 9 % faster for each request type.

## Targets that stay

The table gives the largest gain of the phase 4 prediction for one request type, and the confidence. The column "Expected gain at this time" uses two corrections:

- The stack removed approximately 9 % of the time of each request. If a target removes the same time in ms, its part of a request is 1.1 times larger. We multiply each gain by 1.1.
- The confidence of `N` goes from 0.20 to 0.10. `N` comes after `G`, and the test of `G` gave much less than its prediction. The cause of the small gain of `G` is not clear. Thus a new MMQ kernel has a lower probability of a large gain.

| Target | Largest gain for one request type (phase 4) | Confidence (phase 4) | Expected gain at this time |
|---|---:|---:|---:|
| `N`. New MMQ kernels | 11.0 % (`short2q`) | 0.20 | 1.2 % (confidence 0.10) |
| `F`. Fusions of RMS norm kernels, element-wise kernels and quantization kernels | 3.8 % (`long`) | 0.50 | 2.1 % |
| `L`. Smaller KV window for flash attention | 1.3 % (`medium`, `long`) | 0.50 | 0.7 % |
| `K`. CUDA graphs for passes with more than one token | 2.2 % (`short2q`) | 0.30 | 0.7 % |
| `M`. One kernel for each layer | 6.1 % (`short2q`) | 0.10 | 0.7 % |
| `O`. Cache of the prompt tokens | 1.0 % (`short2q`) | 0.50 | 0.6 % |

Source: `tables/ranked.tsv` in [2026-10-31-phase4-model](../../data/2026-10-31-phase4-model/README.md), and the corrections above.

Two more items are not in the ranked list:

- The uploads that stay before the first kernel of a pass. The [result of change 0007](2026-10-31-0007-input-copies-result.md) shows that the largest upload is the rows of the token embeddings. The [result of change 0004](2026-10-31-0004-gpu-mask-result.md) gives 15.5 ms of upload time that stays for the `long` request. If a change removes all of this time, the gain is 2.2 % of `long_ms` (711.6 ms in the second test of the stack). This is less than 3 % before the confidence.
- The server starts the third question of a request after the first question. The [second result of change 0008](2026-10-31-0008-ragged-suffix-retest-result.md) gives 1 % to 2 % of `lat_ms` or less for a change of this. That change gives a gain only together with ragged batching, and we do not use ragged batching.

## Decision

The largest expected gain of one target for one request type is 2.1 % (`F`, `long`). Without the correction of the confidence of `N`, `N` gives 2.4 % (`short2q`). Each value is less than 3 %. Thus the rule to stop stops the work on the kernels.

The next steps are the 24-hour test of the stack with a load and the change of production. These steps do not change the code.
