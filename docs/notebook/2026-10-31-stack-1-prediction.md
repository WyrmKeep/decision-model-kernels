# Test of the stack 1: `dmk/stack` at `fbfd9cb8c` against build `D` (prediction)

Written 2026-10-31 at about 08:35 UTC, before the first run of the test.

## Task of this entry

This entry gives the prediction for the first test of the stack. The stack is the branch `dmk/stack` of the fork. It has each change that is complete and correct at this time. We compare the build of the stack with build `D`, the baseline of the project. We do not change this entry after the first run of the test.

## The changes in the stack

The stack is tag `b12004` with these changes, in this sequence:

| Change | Target | Equivalence class | Result for `lat_ms` against the build before the change |
|---|---|---|---:|
| [0001](../changes/0001-glu-into-quantize.md) | `E` | `exact` | -1.94 % |
| [0007](../changes/0007-input-copies.md) | `P` | `exact` | -0.93 % |
| [0002](../changes/0002-gdn-value-split.md) | `D` (test cases only) | `exact` | No change to the program |
| [0004](../changes/0004-gpu-mask.md) | `B` | `exact` | -1.06 % |
| [0005](../changes/0005-concat-conv.md) | `C` | `exact` | -3.02 % |

Source: the change documents in the first column. For change 0001, the build before the change was build `D`. For the changes 0004, 0005 and 0007, it was the build of `b12004` with change 0001.

## Equivalence class

`exact`. Each change in the stack is `exact`. Thus the answers of the stack must be the same as the answers of build `D`, bit for bit. If they are not the same, two changes have an effect on each other. Then we stop and find the cause before we use the stack.

## Arms

| Arm | Build | Commit |
|---|---|---|
| `D` | `build:b12004-sm86` | `3c809f4` (tag `b12004`) |
| `S` | `build:stack-fbfd9cb` | `fbfd9cb8c` (`dmk/stack`) |

## Schedule

The test has 6 blocks of the private benchmark and 4 blocks of `llama bench`. Each block runs each arm one time, in a random sequence. We use 6 blocks and not 4, because this result can put the stack into production.

```
python3 harness/make_schedule.py --prefix stack1-bench --blocks 6 --seed 61001 \
    --arm D=build:b12004-sm86 --arm S=build:stack-fbfd9cb --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix stack1-llb --blocks 4 --seed 61002 \
    --arm D=build:b12004-sm86 --arm S=build:stack-fbfd9cb --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

For each metric, the result is the mean of the ratios of the blocks, `S` divided by `D`, with a 95 % bootstrap interval. The script `harness/analysis/ab_stats.py` calculates them.

## Estimate

We multiply the ratios of the changes for each metric. This method thinks that each change removes different time. The changes 0004 and 0007 each change the uploads of a pass. Thus a small part of their gains can be the same time.

## Predicted gain

| Metric | Ratios of the changes, multiplied | Predicted change (%) | Range (%) |
|---|---:|---:|---|
| `lat_ms` | 0.932 | -6.8 | -5.3 to -8.3 |
| `short_ms` | 0.961 | -3.9 | -2.4 to -5.4 |
| `medium_ms` | 0.929 | -7.2 | -5.7 to -8.7 |
| `long_ms` | 0.921 | -8.0 | -6.5 to -9.5 |
| `rps8` | 1.083 | +8.3 | +6.8 to +9.8 |
| `llama bench`, 64 tokens | 1.052 | +5.2 | +3.7 to +6.7 |
| `llama bench`, 256 tokens | 1.075 | +7.5 | +6.0 to +9.0 |
| `llama bench`, 3,000 tokens | 1.097 | +9.7 | +8.2 to +11.2 |

Source: the Results sections of the change documents in the table "The changes in the stack". For change 0007, `short_ms` was -0.21 % and `rps8` was +0.80 %. For change 0004, `short_ms` was +0.08 %.

The range is the predicted change ±1.5 points of percent. Each test of a change had 4 blocks, and the load of the host was different in each test.

## Tolerances and risk

- The answers of the 12 runs of the private benchmark must be the same as the answers of build `D`, bit for bit. Each arm must also give the same answers in all of its runs.
- The guards of the benchmark must be correct in all runs.
- If a result is not in its range, the result entry gives the cause. This also applies to a result that is better than the range.
- Risk: the load of the host can change the result. The harness records the load of each run.

## Data

The data set will be `data/2026-10-31-stack-1/`. The result entry will be `2026-10-31-stack-1-result.md`.
