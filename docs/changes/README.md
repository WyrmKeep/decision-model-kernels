# Change documents

This folder has one document for each change to llama.cpp.

| Number | Change | Equivalence class | Condition |
|---|---|---|---|
| [`0001`](0001-glu-into-quantize.md) | GLU fused into the quantization of the activation in MMQ (upstream pull request #30489) | `exact` | In the branch `dmk/stack`. Not in production. |
| [`0002`](0002-gdn-value-split.md) | Test cases for the shape of `gdn_single` and a comment at the selection of its path. We do not use the value-split path. | `exact` | In the branch `dmk/stack`. The program does not change. |
| [`0003`](0003-answer-head.md) | The output head calculates the answer tokens only. The server gives a list of tokens to `llama_context` for a batch of decisions. | `exact` | In the branch `dmk/stack`. `lat_ms` is 1.48 % lower on the stack and 0.87 % lower on arm `L`. Not in production. |
| [`0004`](0004-gpu-mask.md) | The attention mask on the GPU | `exact` | In the branch `dmk/stack`. `lat_ms` is 1.06 % lower and `long_ms` is 1.52 % lower. Not in production. |
| [`0005`](0005-concat-conv.md) | One CUDA kernel replaces `concat`, the copy of the conv state, `ssm_conv` and `SILU` | `exact` | In the branch `dmk/stack`. `lat_ms` is 3.02 % lower. Not in production. |
| [`0007`](0007-input-copies.md) | The scheduler of ggml skips the uploads that no `COMPUTE` node reads | `exact` | In the branch `dmk/stack`. `medium_ms` is 1.0 % lower and `long_ms` is 1.3 % lower. Not in production. |
| [`0008`](0008-ragged-suffix.md) | Ragged batching of the question suffixes: one pass for the suffixes of different lengths | `re-associated` | We do not use it. One limit of the decision is not true in the second test. The branch `dmk/0008-ragged-suffix` keeps the change. |

Source: the change documents in this folder.

## Rules

- Name each file `NNNN-<name>.md`. `NNNN` is the number of the change. The first change is `0001`. `<name>` is a short name of the change. The name has only `a-z`, `0-9`, and `-`.
- The file has the same text as the pull request of the change.
- A person who reads GitHub cannot see the pull requests of the project. The file gives the same text to this person.
- Write the file before you open the pull request. When the pull request changes, change the file at the same time.
- Use the eight sections below, in the same sequence, with the same headings.
- Write in ASD-STE100 Issue 9. Run the [checker](../../tools/ste/README.md) on the file before you open the pull request.
- Use relative links only. Link to the files in [data/](../../data/README.md) and to the [notebook](../notebook/README.md).

## The eight sections

| Section | Contents of the section |
|---|---|
| Summary | The change, and the cause of the change. Use 2 to 4 sentences. |
| Equivalence class | The equivalence class of the change: `exact`, `re-associated`, or `approximation`. We select the equivalence class before we measure. The [method](../method.md) gives the tests that each equivalence class must have. |
| Results | A table. For each request type it gives the time before (ms), the time after (ms), the difference (%), the 95 % confidence interval, the number of runs, the power limit, and the clocks. The first row is the baseline. The source is below the table. |
| Prediction | The prediction that we wrote before the change. The result. The cause of each difference between them. |
| Correctness | Each test, the threshold that we declared before the test, and the result (`pass` or `fail`). |
| Reproduce | The full commands, the commit hashes, and the build flags. |
| Limits | The risks. The parts of the change with no test. The procedure to go back to the baseline. |
| Data | Relative links to the files in `data/`. |

## Template

Put a copy of this text in a new file. Change each text between `<` and `>`.

```markdown
# <Name of the change>

## Summary

<2 to 4 sentences: the change, and the cause of the change.>

## Equivalence class

<exact, re-associated, or approximation. The date on which we selected the class.>

## Results

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---:|
| <type> | <baseline> | <new build> | <difference> | <low to high> | <number> | <value> | <value> |

Source: <data set, with a relative link>.

## Prediction

<The prediction, the date, the result, and the cause of each difference.>

## Correctness

| Test | Threshold | Result (pass or fail) |
|---|---|---|
| <test> | <value declared before the test> | <result> |

## Reproduce

1. <Command that gets the commit.>
2. <Command that builds the binaries.>
3. <Command that runs the test.>

## Limits

<The risks, what the tests do not test, and the procedure to go back to the baseline.>

## Data

- [<name of the data set>](../../data/<folder>/README.md)
```
