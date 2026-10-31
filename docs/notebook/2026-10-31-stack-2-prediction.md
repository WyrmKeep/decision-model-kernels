# Test of the stack 2: `dmk/stack` at `419b8f612` against the image in production (prediction)

Written 2026-10-31 at about 20:45 UTC, before the first run of the test.

## Task of this entry

This entry gives the prediction for the second test of the stack. At this time, the stack also has change 0003. We compare the build of the stack with the image that the server in production uses. This test is the last test of speed before the stack goes into production. We do not change this entry after the first run of the test.

## The changes in the stack

| Change | Target | Equivalence class | Result for `lat_ms` against the build before the change |
|---|---|---|---:|
| [0001](../changes/0001-glu-into-quantize.md) | `E` | `exact` | -1.94 % |
| [0007](../changes/0007-input-copies.md) | `P` | `exact` | -0.93 % |
| [0002](../changes/0002-gdn-value-split.md) | `D` (test cases only) | `exact` | No change to the program |
| [0004](../changes/0004-gpu-mask.md) | `B` | `exact` | -1.06 % |
| [0005](../changes/0005-concat-conv.md) | `C` | `exact` | -3.02 % |
| [0003](../changes/0003-answer-head.md) | `A` | `exact` | -1.48 % against the stack at `fbfd9cb8c` |

Source: the change documents in the first column.

## Equivalence class

`exact`. Each change in the stack is `exact`. Thus the answers of the stack must be the same as the answers of the image in production, bit for bit. The [phase 1 result](2026-10-30-phase1-equivalence-result.md) shows that build `D` and the image give the same answers bit for bit. The [first test of the stack](2026-10-31-stack-1-result.md) shows the same for the stack and build `D`.

## Arms

| Arm | Build | Commit |
|---|---|---|
| `P` | `stock`: the image `server-cuda-b12004` of llama.cpp, as in production | `3c809f4` (tag `b12004`) |
| `S2` | `build:stack-419b8f6` | `419b8f612` (`dmk/stack`) |

## Schedule

The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`. Each block runs each arm one time, in a random sequence.

```
python3 harness/make_schedule.py --prefix stack2-bench --blocks 6 --seed 62001 \
    --arm P=stock --arm S2=build:stack-419b8f6 --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix stack2-llb --blocks 4 --seed 62002 \
    --arm P=stock --arm S2=build:stack-419b8f6 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

For each metric, the result is the mean of the ratios of the blocks, `S2` divided by `P`, with a 95 % bootstrap interval. The script `harness/analysis/ab_stats.py` calculates them.

## Estimate

We multiply three ratios for each metric. The first ratio is build `D` against the image (phase 1). The second ratio is the stack at `fbfd9cb8c` against build `D` (the first test of the stack). The third ratio is change 0003 against the stack at `fbfd9cb8c`. For `llama bench`, we use 1 for change 0003, because the change does not touch the path of `llama bench`. Its test gave -0.82 %, -0.60 % and -0.16 %, in the range of ±1 %.

## Predicted gain

| Metric | Ratios multiplied | Predicted change (%) | Range (%) |
|---|---:|---:|---|
| `lat_ms` | 0.902 | -9.8 | -7.8 to -11.8 |
| `short_ms` | 0.902 | -9.8 | -7.8 to -11.8 |
| `medium_ms` | 0.902 | -9.8 | -7.8 to -11.8 |
| `long_ms` | 0.906 | -9.4 | -7.4 to -11.4 |
| `rps8` | 1.107 | +10.7 | +8.7 to +12.7 |
| `llama bench`, 64 tokens | 1.060 | +6.0 | +4.0 to +8.0 |
| `llama bench`, 256 tokens | 1.077 | +7.7 | +5.7 to +9.7 |
| `llama bench`, 3,000 tokens | 1.095 | +9.5 | +7.5 to +11.5 |

Source: the [phase 1 result](2026-10-30-phase1-equivalence-result.md), the [first test of the stack](2026-10-31-stack-1-result.md) and the [result of change 0003](2026-10-31-0003-answer-head-result.md).

The range is the predicted change ±2.0 points of percent. The first test of the stack had a range of ±1.5 points, and one result was 0.6 points out of its range. The intervals of the three tests together are approximately ±1.2 points for `short_ms`. The load of the host is also different in each test.

## Tolerances and risk

- The answers of the 12 runs of the private benchmark must be the same as the answers of the image, bit for bit. Each arm must also give the same answers in all of its runs.
- The guards of the benchmark must be correct in all runs.
- If a result is not in its range, the result entry gives the cause. This also applies to a result that is better than the range.
- Risk: the load of the host can change the result. The harness records the load of each run.

## Data

The data set will be `data/2026-10-31-stack-2/`. The result entry will be `2026-10-31-stack-2-result.md`.
