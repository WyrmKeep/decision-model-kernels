# Data set 2026-10-31-phase4-model

This data set holds the time model of the [phase 4 entry](../../docs/notebook/2026-10-31-phase4-ranked-targets.md). The time model calculates the gain of each target for each request type. It uses no GPU. Its input numbers come from other data sets.

## Question

Which is the gain of each target for each request type, and which targets come first? The [prediction entry](../../docs/notebook/2026-10-31-phase4-ranked-targets.md) gives the answer. The results entries of the targets come after it.

## Dates

2026-10-31. The input measurements are from measurement window 2, 2026-10-31, 00:02 to 01:14 UTC.

## Project commit

The commit of this repository that adds this folder.

## Build commits

None. The time model runs no build. The input measurements used build D (tag `b12004` for `sm_86`) and, for target `E`, the branch `dmk/lead-30489`.

## Build flags

None.

## Image

None.

## Model

`H2O Lightning 4B`, quantization mix8, for the input measurements.

## Driver

None for the time model. The input measurements give their driver.

## GPU

None for the time model. The input measurements used an `RTX 3060 Ti`.

## Power limit

None for the time model. The input measurements used 200 W.

## Clocks

`default` in the input measurements.

## Server flags

`default` in the input measurements.

## Commands

```
python3 make_tables.py
```

The script reads `raw/` and writes `tables/`. It uses only the standard library of Python 3.

## Arms

None.

## Raw files

| File | Contents | Source of the numbers |
|---|---|---|
| `raw/pass_kernels.tsv` | Kernel time of each operation class for each pass type, the mean of the passes, in ms | The table of the operation classes of `nsys` (`scope` is `ALL`) in [2026-10-31-profiles](../2026-10-31-profiles/README.md). `concat_ms` and `head_ms` come from the `label` level of the same table. |
| `raw/requests.tsv` | For each request type: the number of cases, the time of one request, the passes, the passes with an output, the copy time to the GPU, the host time of the attention mask, the idle time in the passes, the time to make the prompt, the weight floor and the roofline floor | The request table of `nsys`, the data of the copies, and the table that divides the request time into parts, in [2026-10-31-profiles](../2026-10-31-profiles/README.md) |
| `raw/targets.tsv` | For each target: the equivalence class, the effort, the confidence, the function and its parameters | The assumptions of the project. The prediction entry gives the cause of each one. Target `E` uses the gain that we measured in [`2026-10-31-lead-30489`](../2026-10-31-lead-30489/README.md). |

The numbers in `raw/` are copies of numbers from the other data sets. They have the same decimal places as in the source. Thus they are inputs, not raw files of the harness.

### Columns of `raw/pass_kernels.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `pass_type` | None | The pass type and the token range |
| `tokens` | None | The token range of the pass |
| `all_ms` | ms | All kernels of one pass |
| `mmq_ms` | ms | MMQ kernels, with the stream-k kernels that add the sums of the thread blocks |
| `gdn_ms` | ms | GDN kernels |
| `glu_ms` | ms | GLU kernels |
| `fa_ms` | ms | Kernels for flash attention |
| `quantize_ms` | ms | Quantization kernels for MMQ and MMVQ |
| `copy_ms` | ms | Copy kernels, with the concat |
| `concat_ms` | ms | The concat kernel only (part of `copy_ms`) |
| `rms_norm_ms` | ms | RMS norm kernels |
| `ssm_conv_ms` | ms | `ssm_conv` kernels |
| `elementwise_ms` | ms | Element-wise kernels |
| `mmvq_ms` | ms | MMVQ kernels, with the output head |
| `head_ms` | ms | The output head only (part of `mmvq_ms`) |
| `gdn_kernel` | None | `gdn_single` (chunked kernel) or `ar` (autoregressive kernel) |

### Columns of `raw/requests.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | `short1q`, `short2q`, `medium` or `long` |
| `cases` | None | Number of cases of this type in the benchmark |
| `wall_ms` | ms | Time of one request at the client |
| `passes` | None | The pass types of one request, in sequence |
| `output_passes` | None | Number of passes that calculate the output head |
| `upload_ms` | ms | Time of the copies to the GPU for one request |
| `mask_fill_ms` | ms | Host time to write the attention masks of one request |
| `idle_in_passes_ms` | ms | Idle time of the GPU in the passes of one request |
| `build_ms` | ms | Host time to make the prompt and its tokens |
| `weight_floor_ms` | ms | The weight floor of one request |
| `roofline_ms` | ms | The roofline floor of one request |

### Columns of `raw/targets.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `id` | None | The letter of the target |
| `target` | None | The name of the target. The phase 4 entry gives some names with other words. |
| `class` | None | The equivalence class: `exact` or `re-associated` |
| `effort` | None | `S` (1 day or less), `M` (2 to 5 days), `L` (1 to 2 weeks), `XL` (more than 2 weeks) |
| `confidence` | None | The confidence of the target |
| `model` | None | The name of the function in `make_tables.py` that calculates the gain |
| `parameters` | None | The parameters of that function, as `name=value` items |

## Tables

### `tables/floors.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | The request type |
| `cases` | None | Number of cases |
| `wall_ms` | ms | Time of one request |
| `kernel_ms` | ms | The sum of `all_ms` of the passes of the request |
| `weight_floor_ms` | ms | As in `raw/requests.tsv` |
| `roofline_ms` | ms | As in `raw/requests.tsv` |
| `wall_over_roofline` | None | `wall_ms` divided by `roofline_ms` |

### `tables/savings.tsv` and `tables/ranked.tsv`

The two tables have the same columns. The rows of `ranked.tsv` are in the sequence of `expected_lat_pct`, the largest first.

| Column | Unit | Meaning |
|---|---|---|
| `id`, `target`, `class`, `effort`, `confidence` | None | As in `raw/targets.tsv` |
| `<type>_ms` | ms | The gain of the prediction for one request of this type |
| `<type>_pct` | % | The same, in percent of `wall_ms` of this type |
| `lat_ms` | ms | The gain of the prediction for `lat_ms`, the mean of the 48 cases |
| `lat_pct` | % | The same, in percent of `lat_ms` of the time model |
| `max_expected_pct` | % | The largest value of `<type>_pct` × `confidence` of the 4 types |
| `expected_lat_pct` | % | `lat_pct` × `confidence` |

### `tables/sum.tsv`

The total gain of the targets `A`, `B`, `C`, `D`, `E`, `F`, `G` and `J`. The function `combined` in `make_tables.py` applies `J` first. The other targets then use the passes that stay. The comments in the script give the sequence. The table has one row for each request type and one row for `lat_ms`.

| Column | Unit | Meaning |
|---|---|---|
| `scope` | None | The request type, or `lat_ms` for the mean of the 48 cases |
| `targets` | None | The targets of the total gain |
| `time_now_ms` | ms | The time of the time model with no change |
| `simple_sum_ms` | ms | The sum of the gains of the targets in `savings.tsv`. It counts some kernel time two times. Use it only to compare with `all_pass_saved_ms`. |
| `all_pass_saved_ms` | ms | The total gain if each target gives its full gain |
| `all_pass_time_ms` | ms | `time_now_ms` minus `all_pass_saved_ms` |
| `all_pass_pct` | % | The change |
| `expected_saved_ms` | ms | The mean of the total gain for all 256 possible results. In each result, each target gives its full gain or no gain. The probability of the full gain is the confidence of the target. |
| `expected_time_ms` | ms | `time_now_ms` minus `expected_saved_ms` |
| `expected_pct` | % | The change |

## Files not added

None.
