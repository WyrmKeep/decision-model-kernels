# Change 0005: concat folded into `ssm_conv` (prediction)

Written 2026-10-31 at about 05:15 UTC, before the first run of the test.

## Task of this entry

This entry gives the prediction for change 0005 before we measure it. The change removes the `concat` kernel before `ssm_conv` in the Gated DeltaNet layers. The entry gives the change, the equivalence class, the arms, the schedule, the estimate, the tolerances and the predicted gain for each request type. We do not change this entry after the first run of the test.

## Cause

Each Gated DeltaNet layer makes the input of `ssm_conv` with a `concat`. The `concat` puts the conv state (3 columns) before the new input (1 column for each token). The new input is the transposed output of the `QKV` projection. Thus the tokens of the new input are not adjacent to each other in memory. The `concat` then uses the slow kernel `concat_non_cont<unsigned int,0>`.

The kernel uses 12.9 ms in a pass of 2,048 tokens. It uses 0.5 ms to 1.6 ms in a pass of 8 to 256 tokens. A second kernel then copies the last 3 columns of the same tensor to the cache as the new conv state. The [time model of phase 4](2026-10-31-phase4-ranked-targets.md) gives these numbers for the target `C`.

## Change

One kernel replaces four nodes of the graph: `concat`, the copy of the new conv state, `ssm_conv` and `SILU`. The kernel reads the conv state and the new input from the two buffers where they are. It writes the output of `SILU` and the new conv state in the cache. Thus the program does not write the tensor of the `concat` and does not read it again.

The CUDA backend finds the four nodes in the graph and runs the new kernel. A change in the graph code of the models puts the four nodes adjacent to each other. A new rule keeps the two inputs of the `concat` in memory until the new kernel ends. Other backends and other models do not change.

## Equivalence class

The equivalence class is `exact`. The new kernel does the same multiply-add operations as the kernel `ssm_conv` of arm `L`. It does them in the same sequence for each output value. The new conv state is a copy of 3 values for each row. Only the movement of data changes.

The tests of the equivalence class are:

- The answers of the quality pass are the same bit for bit as the answers of arm `L`. `answers_compare.py` must give `cross_identical_all` true, and `self_repeat_a` and `self_repeat_b` true.
- `test-backend-ops` passes for `SSM_CONV`, `SSM_CONV_BIAS_SILU`, `CONCAT` and the new test case. The NMSE limit is 1e-7, the same as the other cases.
- `compute-sanitizer` finds 0 errors in the new kernel with the tools `memcheck`, `racecheck`, `initcheck` and `synccheck`. `memcheck` also runs on one request of the model.

If one answer is different, the equivalence class is `re-associated`. We then tell the owner of the project before we continue.

## Arms

| Short name | Arm | Meaning |
|---|---|---|
| `L` | `build:lead-30489` | The branch `dmk/stack`, with change 0001 and without change 0005 |
| `C` | `build:0005-concat-conv` | The same branch and change 0005, in the same image |

## Schedule

The schedule has 4 blocks of the private benchmark and 4 blocks of `llama bench`. Each block has one run of each arm in a random sequence. The test runs in one measurement window of less than 60 minutes. The commands are:

```
python3 harness/make_schedule.py --prefix c0005-bench --blocks 4 --seed 50051 \
    --arm L=build:lead-30489 --arm C=build:0005-concat-conv --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix c0005-llb --blocks 4 --seed 50052 \
    --arm L=build:lead-30489 --arm C=build:0005-concat-conv --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

## Estimate

For each block and each metric, the ratio is the value of `C` divided by the value of `L`. The estimate is the mean of the 4 ratios. The interval is the 95 % bootstrap interval of the blocks, with 10,000 samples and seed 12345. `ab_stats.py` calculates all of it.

The metrics are `lat_ms`, `short_ms`, `medium_ms` and `long_ms` for the private benchmark. For `llama bench` they are the speed of the prompt at 64, 256 and 3,000 tokens. The data set has the load of the host, as in the test of change 0001.

## Predicted gain

The prediction starts from the numbers of the phase 4 model. The phase 4 model removes 90 % of the time of the `concat` kernel. We keep this rule and the numbers.

| Request type | Time of a request (ms) | Time of the `concat` kernel (ms) | Predicted gain (ms) | Predicted gain (%) |
|---|---:|---:|---:|---:|
| `short1q` | 47.1 | 0.99 | 0.9 | 1.9 |
| `short2q` | 87.1 | 2.65 | 2.4 | 2.7 |
| `medium` | 360.0 | 10.86 | 9.8 | 2.7 |
| `long` | 797.0 | 23.78 | 21.4 | 2.7 |
| `lat_ms` (mean of 48 cases) | 240.9 | 7.14 | 6.4 | 2.7 |
| `short_ms` (32 cases) | 72.1 | 2.04 | 1.8 | 2.5 |

The source is the files `pass_kernels.tsv` and `requests.tsv` in the [data set of the phase 4 model](../../data/2026-10-31-phase4-model/README.md). The rows `lat_ms` and `short_ms` are the mean of the types, with the number of cases as weight. We calculated them for this entry.

The time of arm `L` is 1.9 % less than the time of build D. The gain in ms does not change. The gain in percent of arm `L` is more than the numbers above.

### Speed of the prompt in `llama bench`

The phase 4 model has no rows for `llama bench`. We calculate them for this entry with the same rule. A pass of 2,048 tokens uses 416 ms, and its `concat` kernel uses 12.9 ms. The pass that follows it has about 950 tokens, and its `concat` kernel uses about 5.8 ms.

| Prompt (tokens) | Time of the prompt (ms) | Time of the `concat` kernel (ms) | Predicted gain of the speed (%) |
|---:|---:|---:|---:|
| 64 | 22.7 | 0.5 to 0.7 | +2.1 to +2.8 |
| 256 | 60.7 | 1.65 | +2.5 |
| 3,000 | 652.6 | 18.7 | +2.7 |

The time of the prompt is the number of tokens divided by the speed of build D in the [test of change 0001](2026-10-31-lead-30489.md). The `concat` time of a prompt of 64 tokens is a range. The range is from the pass type with 9 to 32 tokens to the pass type with 65 to 128 tokens.

### Items that the prediction does not include

The threads of the new kernel read values that are adjacent in memory. At 2,048 tokens, the kernel `ssm_conv` of arm `L` runs at 67 % of the DRAM bandwidth. For small passes it uses 30 µs for each launch. The new kernel can be faster than that kernel. Then the gain is larger than the numbers above.

The new kernel also writes the new conv state. This work can make it slower. The prediction includes it in the 10 % of the `concat` time that stays.

## Tolerances and risk

We declare these limits before the measurement.

- The prediction is true for `lat_ms` if two conditions are true. The gain is from 0.5 times to 1.6 times the predicted gain. The 95 % interval does not include 0.
- A gain of less than 0.5 times the predicted gain means that the time model is incorrect. We then find the cause with a profile before we write the result.
- The confidence is 0.70, the same as in the ranked list.
- The risks are in three items. The answers can be different. The four nodes can be not adjacent to each other in the graph. The output of the new kernel and its input can use the same memory.

## Data

The data set of this change will have the path `data/2026-10-31-0005-concat-conv/`. The results entry will have the name `2026-10-31-0005-concat-conv-result.md`.
