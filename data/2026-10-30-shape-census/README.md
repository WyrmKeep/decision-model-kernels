# Shape and launch census of the decision model

This data set has the census of the decision model for the `RTX 3060 Ti`. The census is a model of one pass. It gives the shapes, the kernel launches, the bytes, and the floors of each pass of each request type. The census does not measure the GPU.

The data set has the files that the CPU steps read and the tables that they make. [`reports/census-report.txt`](reports/census-report.txt) shows the numbers in text. You can make all tables and the report again from `raw/`. This does not use a GPU, a model file, or a tokenizer.

## Question

The census gives answers for the profile of phase 3:

- The number of passes of each request type of the benchmark.
- The work of each pass: the shape and the kernel of each matmul, the other operations, and the bytes of the weights.
- The floor of each pass and of each request. The floors use the nominal rates of the GPU.
- A check of the model against three captures of the production server. They show the launches of 20 requests.

At this time, the [notebook](../../docs/notebook/README.md) has no prediction entry for this census. The results entry is [phase 3: where the time goes](../../docs/notebook/2026-10-31-phase3-where-the-time-goes.md).

## Dates

- First run: 2026-10-30. The scripts of that run made the tables from the model file, the tokenizer, and the captures.
- Last run: 2026-10-31. The scripts of this folder made all tables and the report again. The section [Changes to the files of the first run](#changes-to-the-files-of-the-first-run) shows the differences.
- The three captures started on 2026-10-30 at 19:40:00 UTC (`long`), 19:40:12 UTC (`medium`), and 19:40:17 UTC (`short`).

## Project commit

The commit that adds this data set also adds the code in `harness/profile/census/`. The commit before it is `e91d24b53413fb9922120760e5c8317a95a6d9c2`.

The first run did not use the code of this repository. The code was in a different folder. The section [Changes to the files of the first run](#changes-to-the-files-of-the-first-run) shows the differences.

## Build commits

The step `02_tokens.py` uses `llama-tokenize` of build `D` of tag `b12004`. The commit is `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. The model of the census follows the sources of this commit.

The three captures come from a copy of the production server. [INFERENCE] It used the stock image of `b12004`, and thus the same commit. No other step uses llama.cpp.

## Build flags

The steps of the census do not build a program. Build `D` has these `cmake` flags:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

The data set [`2026-10-30-window1`](../2026-10-30-window1/README.md) gives more information about build `D`.

## Image

The production server used the stock image `ghcr.io/ggml-org/llama.cpp@sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352`. Its tag is `server-cuda-b12004`. The captures do not record the image. [INFERENCE] The server of the captures used this image.

No step of the census uses an image. The steps run on the CPU of a host.

## Model

The model is H2O Lightning 4B in the "mix8" quantization. The weights of the feed-forward layers are Q4_0. The other weights are Q8_0. The GGUF file has the architecture `qwen35`, 426 tensors, and 3349945376 bytes.

The model has 32 layers. 24 layers are recurrent layers (Gated DeltaNet) and 8 layers are attention layers. The census reads only the start of the file, the template, the 255 decision labels, and the vocabulary. The data set has no model file and no path of one.

## Driver

The driver of the captures is 595.71.05. The driver API is CUDA 13.2. The steps of the census use no driver.

## GPU

The GPU is an `NVIDIA GeForce RTX 3060 Ti` (GA104, `sm_86`). It has 38 SMs. The census models this GPU. The three captures ran on it.

The floors use two nominal rates. The DRAM rate is 448 GB/s. The int8 rate is 144.8 TOPS, at an SM clock of 1861 MHz. The data set [`2026-10-30-card-limits`](../2026-10-30-card-limits/README.md) has the rates that a test measured.

## Power limit

The power limit of the captures is 200 W. This is the default value. [INFERENCE] The captures do not record the limit. The production server used the default value. The steps of the census do not use it.

## Clocks

The value is `default`. [INFERENCE] The production server did not set the clocks and did not lock them. The steps of the census do not use clocks.

## Server flags

The production server used these flags:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified
```

Each capture is a run of a copy of the production server with the profiler `nsys`. A client sent 4 long requests, 4 medium requests, and 12 short requests, in the three captures.

## Commands

Make all tables in `tables/` and the report in `reports/` again. The script uses `python3` only:

```
bash make_tables.sh
```

Make the tables in a different folder and compare them with the tables of this folder:

```
out=$(mktemp -d)
bash make_tables.sh --out "$out"
diff -r "$out/tables" tables
diff -r "$out/reports" reports
```

Make the raw files again. This uses the model file, the tokenizer, and the three captures. The folder with the captures has the files `long`, `medium`, and `short` with the extension `.nsys-rep`:

```
export DMK_GGUF=<model>.gguf
export GGUF_PY=<llama.cpp>/gguf-py
export DMK_TOKENIZE=<build D>/llama-tokenize
export DMK_LEGACY_TRACES=<folder with the captures>
export DMK_NSYS=<nsys command line tool>
bash ../../harness/profile/census/run_all.sh
```

`run_all.sh` lists each step with its input. It runs the steps `1`, `2`, `T`, `3`, `4`, and `5` of the table in the section [Arms](#arms). The flag `--tables-only` runs the steps `3`, `4`, and `5`.

The commands of the steps `1`, `2`, and `T` are:

```
uv run --with numpy --with pyyaml --with tqdm --with requests python 01_gguf_header.py \
    --gguf <model>.gguf --gguf-py <llama.cpp>/gguf-py --out raw/model-header
uv run --with jinja2 --with numpy --with pyyaml --with tqdm --with requests python 02_tokens.py \
    --tokenize <build D>/llama-tokenize --gguf <model>.gguf --gguf-py <llama.cpp>/gguf-py --out raw/tokens
bash trace_export.sh --in <folder with the captures> --out <folder for the SQLite files>
python3 trace_census.py --in <folder for the SQLite files> --out raw/traces
```

## Arms

This data set has no arms. Each step is one program of `harness/profile/census/`. The steps `1`, `2`, and `T` use private data. The steps `3`, `4`, and `5` use only this folder.

| Step | Program | Input | Output |
|---|---|---|---|
| `1` | `01_gguf_header.py` | The start of the GGUF file. | `raw/model-header/` |
| `2` | `02_tokens.py` | `bench/cases.json`, `llama-tokenize`, the GGUF file, and a run of `2026-10-30-window1`. | `raw/tokens/` |
| `T` | `trace_export.sh` and `trace_census.py` | The three captures. | `raw/traces/` |
| `3` | `03_census.py` | `raw/model-header/` and `raw/tokens/`. | `tables/census*.tsv` and `tables/request-floors.tsv` |
| `4` | `04_crosscheck.py` | `raw/traces/` and `raw/tokens/`. | `tables/crosscheck-*.tsv` |
| `5` | `05_report.py` | `raw/` and `tables/`. | `reports/census-report.txt` |

The benchmark has short requests, medium requests, and long requests. A short request has 1 or 2 questions. A medium request and a long request have 3 questions. The tables use these four names: `short-1q`, `short-2q`, `medium-3q`, and `long-3q`.

## Raw files

The files in `raw/` are the output of the steps `1`, `2`, and `T`. They are TSV files. The scripts of the steps `3`, `4`, and `5` read them.

| File | Contents |
|---|---|
| `raw/model-header/tensors.tsv` | One row for each of the 426 tensors of the GGUF file. |
| `raw/model-header/hparams.tsv` | One row for each parameter of the model. |
| `raw/model-header/layers.tsv` | One row for each of the 32 layers. |
| `raw/tokens/token-counts.tsv` | One row for each question of the benchmark and each value of `variant`. |
| `raw/tokens/cases-plan.tsv` | One row for each case and each value of `variant`, with the tokens of the shared prefix and of each suffix. |
| `raw/tokens/ref-line-tokens.tsv` | The tokens of 100 random lines `Ref: <10 hex digits>`. |
| `raw/tokens/token-check.tsv` | The mean prompt tokens of each type of request, and the difference from the mean of a benchmark run. |
| `raw/traces/trace-kernels.tsv` | One row for each launch shape of a kernel in a capture. |
| `raw/traces/trace-passes.tsv` | One row for each pass, each request, and each capture. |
| `raw/traces/trace-mmq-launches.tsv` | One row for each launch of an MMQ kernel or an MMVQ kernel in 37 passes. |

The column `variant` has two values. The value `raw` is the state of the case. The value `ref` has the line `Ref: <10 hex digits>` before the state. The benchmark sends the value `ref`.

## Tables

`make_tables.sh` makes the tables in `tables/` and the report in `reports/`. `columns.tsv` has the table, the column, the unit, and the meaning of each column of each table in `raw/` and `tables/`.

| File | Contents |
|---|---|
| `tables/census.tsv` | One row for each group of operations of each pass of the 6 cases. |
| `tables/census-sequence.tsv` | One row for each matmul of each pass of the 6 cases. |
| `tables/census-passes.tsv` | One row for each pass of the 6 cases, with work, bytes, and floors. |
| `tables/census-cases.tsv` | One row for each case, each value of `variant`, and each slot assignment. |
| `tables/census-summary.tsv` | The mean, the lowest value, and the highest value for each group of `census-cases.tsv`. |
| `tables/census-head.tsv` | The rows of the output head that the passes calculate and that the API reads, for each of the 6 cases. |
| `tables/request-floors.tsv` | The passes, the launches, the weights, and the floors of the 4 requests that the profile runs use. |
| `tables/crosscheck-requests.tsv` | One row for each request of a capture. It shows if a slot assignment of the model gives the passes of the request. |
| `tables/crosscheck-passes.tsv` | One row for each pass of a capture. It has the launches and the time of the capture and of the model. |
| `tables/crosscheck-keys.tsv` | The matmul launches that only one side has. The table has no rows. |
| `tables/crosscheck-kernels.tsv` | The launches of the flash attention kernels and of the Gated DeltaNet kernels, for each capture. |
| `tables/crosscheck-summary.tsv` | One row for each check, with the number of items that agree. |
| `reports/census-report.txt` | The report in text. It has all tables in text with the sources. |

A slot assignment shows which task slot has each question of a request. A request with one question has 1 slot assignment. A request with two or three questions has 6 slot assignments. The column `slots_Q0_Q1_Q2` has the slots of the questions. The passes change with the slot assignment, because the server runs the questions in the sequence of the slots.

The census uses one case for each of the 6 tasks. This case has the median prompt tokens of the task. The tables `census.tsv`, `census-sequence.tsv`, `census-passes.tsv`, and `census-head.tsv` have only these cases. A case can have more than one slot structure. The tables have the first slot assignment of each slot structure.

The tables write the passes of a request as one text. `P<n>` is a prefix pass of one sequence with `n` tokens. `S<s>x<t>` is a pass of `s` sequences with `t` tokens each. `F<n>` is a pass with the full prompt of `n` tokens.

A `+` is between two passes. Example: `P86+S2x51+S1x69`. The column `slot_structure` has only the number of sequences of each pass. Example: `P1 S2 S1`.

### Table of the columns: `raw/model-header/hparams.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `parameter` | None | Name of the parameter. It is a GGUF key or a value that the script calculates. |
| `value` | see the column unit | Value of the parameter. It is a number or a text. |
| `source` | None | The GGUF key, or the rule that gives the value. |
| `unit` | None | Unit of the value. The field is empty if the value has no unit. |

### Table of the columns: `raw/model-header/layers.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `layer` | None | Number of the layer, 0 to 31. |
| `kind` | None | `recurrent` (Gated DeltaNet) or `attention`. |
| `weight_bytes` | B | Bytes of all tensors of the layer: q4_0_bytes + q8_0_bytes + f32_bytes. |
| `q4_0_bytes` | B | Bytes of the Q4_0 tensors of the layer. |
| `q8_0_bytes` | B | Bytes of the Q8_0 tensors of the layer. |
| `f32_bytes` | B | Bytes of the F32 tensors of the layer. They are the weights of the RMS norms and the small tensors of the recurrent layers. |

### Table of the columns: `raw/model-header/tensors.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `name` | None | Name of the tensor in the GGUF file. |
| `layer` | None | Number of the layer of a `blk.N` tensor. The value is -1 for a tensor that is not in a layer. |
| `role` | None | The tensor name without the blk.N prefix. |
| `ne0_k_in` | None | Shape of the tensor in `ggml`, dimension 0. For a weight matrix it is K, the input dimension. |
| `ne1_m_out` | None | Shape of the tensor in `ggml`, dimension 1. For a weight matrix it is M, the number of output rows. |
| `ne2` | None | Shape of the tensor in `ggml`, dimension 2. It is 1 if the tensor has no such dimension. |
| `ne3` | None | Shape of the tensor in `ggml`, dimension 3. It is 1 if the tensor has no such dimension. |
| `ggml_type` | None | Type of the tensor in `ggml`: `Q4_0`, `Q8_0` or `F32`. |
| `elements` | None | Number of values of the tensor. |
| `bytes` | B | Bytes of the tensor data in the file. A block of 32 Q4_0 values has 18 B. A block of 32 Q8_0 values has 34 B. |
| `data_offset_b` | B | Position of the tensor data, from the start of the file. |

### Table of the columns: `raw/tokens/token-counts.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `variant` | None | `raw` is the state of the case. `ref` has the line `Ref: <10 hex digits>` before the state. The benchmark sends `ref`. |
| `case` | None | Name of the case in `bench/cases.json`. |
| `task` | None | Task of the case: `check`, `commit`, `triage`, `style`, `syslog-medium` or `syslog-long`. |
| `size` | None | Type of the request: `short`, `medium` or `long`. |
| `n_questions` | None | Number of questions of the request. |
| `question` | None | Name of the question. |
| `qtype` | None | Type of the question: `choice`, `noul` or `score`. |
| `n_options` | None | Number of answers that the question has. |
| `prompt_tokens` | tokens | Tokens of the prompt that the server makes for this question. |
| `shared_prefix_tokens` | tokens | Tokens of the shared prefix of the request. The value is 0 for a request with one question. |
| `suffix_tokens` | tokens | prompt_tokens minus shared_prefix_tokens. |

### Table of the columns: `raw/tokens/cases-plan.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `variant` | None | `raw` or `ref`. See `token-counts.tsv`. |
| `case` | None | Name of the case in `bench/cases.json`. |
| `task` | None | Task of the case. |
| `size` | None | Type of the request: `short`, `medium` or `long`. |
| `n_questions` | None | Number of questions of the request. |
| `sum_prompt_tokens` | tokens | Sum of the prompt tokens of all questions. This is the value of usage.input_tokens of the server. |
| `shared_prefix_tokens` | tokens | Tokens of the shared prefix of the request. |
| `suffix_tokens_per_question` | tokens | The suffix tokens of each question. One space is between two values. |
| `n_distinct_suffix_lengths` | None | Number of different values in suffix_tokens_per_question. |
| `min_prompt_tokens` | tokens | Prompt tokens of the shortest question. |

### Table of the columns: `raw/tokens/ref-line-tokens.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `stat` | None | `min`, `median`, `mean`, `max` or `samples` of the 100 random lines `Ref: <10 hex digits>`. |
| `tokens` | tokens | Tokens of one line. For the row samples the value is the number of lines. |

### Table of the columns: `raw/tokens/token-check.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `size` | None | Type of the request: `short`, `medium` or `long`. |
| `n_cases` | None | Number of cases of this type. |
| `mean_sum_prompt_tokens_raw_here` | tokens | Mean of `sum_prompt_tokens` for `raw`, from `cases-plan.tsv`. |
| `mean_input_tokens_production_run` | tokens | quality.input_tokens_mean of a benchmark run, as the server counts it. |
| `diff_tokens` | tokens | The first value minus the second value. The value 0 shows that the two means are equal. |

### Table of the columns: `tables/request-floors.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | short-1q, short-2q, medium-3q or long-3q. |
| `rep_case` | None | The case that the profile runs use for the request type. |
| `slot_structure` | None | Number of sequences of each pass of the request. |
| `pass_structure` | None | The passes of the request, with a `+` between two passes. `P<n>` is a prefix pass of `n` tokens. `F<n>` is a pass with the full prompt of `n` tokens. `S<s>x<t>` is a pass of `s` sequences with `t` tokens each. It is the column `passes` of `census-cases.tsv`. |
| `n_passes` | None | Number of passes of the request. |
| `tokens_evaluated` | tokens | Sum of `N_tokens` for the passes. |
| `weight_gb_read` | GB | Weight bytes that the passes read, with the output head. 1 GB is 10^9 B. |
| `floor_weights_ms` | ms | Sum for the passes of the weight bytes divided by 448 GB/s. |
| `floor_pass_max_ms` | ms | Sum for the passes of the larger value of the weight floor and the int8 floor. |
| `floor_roofline_ms` | ms | Sum for the passes of the roofline floor. |
| `n_mmq` | None | Number of MMQ launches of all passes. |
| `n_mmvq` | None | Number of MMVQ launches of all passes. |

### Table of the columns: `tables/census-head.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | short-1q, short-2q, medium-3q or long-3q. |
| `task` | None | Task of the case that the census uses for the request type. |
| `rep_case` | None | The case that the census uses for the request type. |
| `n_questions` | None | Number of questions of the request. |
| `passes_with_head` | None | Number of passes that have an output row and thus read the head weights. |
| `head_rows_computed_per_request` | rows | Output rows times the 248320 rows of the vocabulary. |
| `head_rows_needed_per_request` | rows | Answers of all questions. |
| `head_weight_mb_read_per_request` | MB | Head weight bytes that the request reads. |
| `head_weight_mb_needed_per_request` | MB | Head weight bytes of the rows that the API reads. |
| `head_flops_computed_gflop` | GFLOP | Head work that the passes calculate. |
| `head_flops_needed_gflop` | GFLOP | Head work of the rows that the API reads. |
| `head_time_at_448GBps_ms` | ms | Head weight bytes read divided by 448 GB/s. |

### Table of the columns: `tables/crosscheck-keys.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `side` | None | `predicted only` or `observed only`. |
| `path` | None | MMQ or MMVQ. |
| `wtype` | None | Type of the weights. |
| `tile_J_or_ncols` | None | Tile width `J` for MMQ, or number of columns for MMVQ. |
| `need_check_or_fused` | None | need_check for MMQ, or 1 if fused for MMVQ. |
| `grid_x` | None | grid.x of the launch. |
| `fixup_or_small_k` | None | Flag of the fix-up kernel for MMQ, or flag `small_k` for MMVQ. |
| `n_launches_summed_over_passes` | None | Number of launches of this key that only one side has, in all passes together. The table has no rows if the two sides agree. |

### Table of the columns: `tables/crosscheck-kernels.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `trace` | None | The capture: `long`, `medium` or `short`. |
| `kernel` | None | Flash attention kernel template, or Gated DeltaNet kernel with its grid. |
| `observed_launches` | None | Launches in the capture. |
| `predicted_launches` | None | Launches that the model gives. |

### Table of the columns: `tables/crosscheck-summary.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `check` | None | Name of the check. |
| `passed` | None | Number of items in which the capture and the model agree. |
| `total` | None | Number of items of the check. |
| `all_passed` | None | `yes` if `passed` and `total` are equal. |

### Tables with more than 12 columns

The tables below have more than 12 columns. The next tables show the groups of columns. `columns.tsv` has the unit and the meaning of each column.

`raw/traces/trace-kernels.tsv` has 27 columns:

| Group | Columns | Contents |
|---|---|---|
| Name | `trace`, `kernel_short`, `kernel_full`, `category` | The capture, the names of the kernel, and its group. |
| Launch | `grid_x` to `block_z`, `smem_static_b`, `smem_dynamic_b`, `regs` | The grid, the block, the shared memory, and the registers. |
| Time | `n_launches`, `total_us`, `mean_us`, `min_us`, `max_us`, `share_pct_of_trace_kernel_time` | The number of launches and their times. |
| Passes | `n_passes_containing` to `n_passes_suffix` | The launches in each pass that has the kernel, for each pass type. |

`raw/traces/trace-passes.tsv` has 106 columns. The column `row_kind` shows if the row is a `pass`, a `request`, or a `trace_total`.

| Group | Columns | Contents |
|---|---|---|
| Row | `row_kind` to `pass_kind_evidence` | The type of the row, the capture, the numbers of the request and of the pass, the case, and the pass type with its facts. |
| Shape | `n_tokens`, `n_kv_padded`, `n_outputs`, `n_passes` | The tokens, the KV window, the output rows, and the number of passes. |
| Time | `start_ns` to `gap_total_excl_d2d_ms` | The start, the time from the first to the last kernel, the GPU busy time, and the gaps. |
| Gaps | `gap_share_of_wall_pct` to `gpu_busy_pct_of_span_incl_idle_before` | The length of the gaps between the kernels, and the idle time before the pass or the request. |
| Copies at the start | `h2d_lead_ms` to `h2d_tail_ms` | The copies to the GPU before the first kernel. |
| Gap groups | `gap_n_lt1us` to `gap_t_gt100us_ms` | The number and the sum of the gaps in 5 groups of length. |
| Kernels | `n_kernels` to `n_other` | The number of kernels of each type, and the names of two kernels. |
| Kernel groups | `n_cat_*` and `t_cat_*_ms` | The number and the time of the kernels of 11 groups. |
| CUDA runtime | `n_cuda_graph_launches` to `streams_used_memcpy` | The CUDA graph launches, the copies, the time in the CUDA runtime, the launch queue, and the CUDA streams. |

`raw/traces/trace-mmq-launches.tsv` has 45 columns. One row is one launch of an MMQ kernel or an MMVQ kernel in a pass.

| Group | Columns | Contents |
|---|---|---|
| Pass | `trace` to `n_kv_padded` | The capture, the numbers, the type of the pass, its tokens, and its KV window. |
| Kernel | `launch_kind` to `halve_iters` | The type of the launch, its position in the pass, the name of the kernel, and its template parameters. |
| Launch | `grid_x` to `start_in_pass_us` | The grid, the block, the shared memory, the registers, and the time. |
| Quantize kernel | `prev_quant_kernel` to `prev_quant_reuse_index` | The quantize kernel before the launch. |
| Fix-up kernel | `has_fixup` to `fixup_grid_z` | The fix-up kernel after the launch. |

`tables/census.tsv` has 29 columns:

| Group | Columns | Contents |
|---|---|---|
| Request | `request_type` to `slots_Q0_Q1_Q2` | The request type, the case, and the slot assignment. |
| Pass | `pass_idx` to `n_outputs` | The number of the pass, its type, and its tokens. |
| Operation | `op` to `K` | The group of operations, the number of layers, and the shape. |
| Kernel | `wtype` to `fixup_predicted` | The weight type, the CUDA path, the kernel of the model, and its grid. |
| Work and bytes | `tile_eff_pct` to `other_bytes_total_mb` | The tile percent, the work, and the bytes of the weights and of other data. |

`tables/census-sequence.tsv` has 18 columns. One row is one matmul. The columns `rep_case` to `n_outputs` give the pass. The columns `order` to `K` give the matmul. The columns `path` to `fixup_predicted` give the kernel of the model and its grid.

`tables/census-passes.tsv` has 35 columns:

| Group | Columns | Contents |
|---|---|---|
| Pass | `request_type` to `n_outputs` | The request type, the case, the slot assignment, and the tokens of the pass. |
| KV window | `start_pos` to `fa_ncols1` | The position of the pass, the KV window, and the flash attention kernel. |
| Launches | `n_matmul` to `gdn_kernel` | The number of launches of each kernel class, and the Gated DeltaNet kernel. |
| Work and bytes | `matmul_gflop` to `mmout_mb` | The work of the matmul launches, of flash attention, and of the Gated DeltaNet kernel, and the bytes of the weights, the head, the layer state, the KV cache, and other data. |
| Floors | `floor_weights_ms` to `floor_roofline_ms` | The weight-read floor, the int8 floors, and the roofline floor of the pass. |

`tables/census-cases.tsv` has 30 columns. One row is one case, one value of `variant`, and one slot assignment. The first 12 columns give the case, the slot assignment, and the passes. The next 16 columns give the launches, the work, the bytes, and the floors of all passes. The last two columns give the rows of the output head.

`tables/census-summary.tsv` has 58 columns. The first 7 columns give the group of rows. Each of the 17 quantities has three columns with the ends `_mean`, `_min`, and `_max`.

`tables/crosscheck-requests.tsv` has 13 columns. The first 3 columns give the request. The next 4 columns give the passes of the capture. The last 6 columns show if a slot assignment of the model gives them.

`tables/crosscheck-passes.tsv` has 44 columns:

| Group | Columns | Contents |
|---|---|---|
| Pass | `trace` to `n_seqs_model` | The capture, the numbers, and the tokens of the pass. |
| Launch counts | `n_kernels_observed` to `n_gdn_pred` | The launches of the capture (`_obs`) and of the model (`_pred`) for each kernel class. |
| Launch keys | `launch_keys_pred` to `launch_keys_matching` | The matmul launches that have the same key in the capture and in the model. |
| KV window | `n_kv_padded_obs` to `kv_window_equal` | The KV window of the capture and of the model. |
| Floors | `model_floor_weights_ms` to `model_floor_roofline_ms` | The floors of the model. |
| Capture time | `observed_gpu_busy_ms` to `observed_gap_total_ms` | The GPU busy time, the time from the first to the last kernel, and the gaps of the capture. |
| Ratios | `busy_over_pass_max_floor`, `busy_over_roofline_floor` | The GPU busy time divided by two floors. |
| Kernel time | `model_mmq_gflop` to `observed_gdn_kernel_ms` | The work and the bytes of the model, and the time of each kernel class in the capture. |

## Files not added

The folder with the captures has three files. They are the output of `nsys`. They are not in this data set. The scan of one copy of each file with `bash tools/publish/scan-dir.sh` shows 7 findings of the rule `dmk-uuid` in each file. The files have the UUID of the GPU. The tables in `raw/traces/` have no such data.

| File | Bytes | SHA-256 | Start (UTC) |
|---|---|---|---|
| `long.nsys-rep` | 2808361 | `eb8adad8bb03cd16ba76f91475bc9dbd12ea1194ffaa201f1f5cb649abc8b51a` | 2026-10-30 19:40:00 |
| `medium.nsys-rep` | 2155090 | `dd003abfb08cf31489a69526f74bea30a7899695fe791b9b7197f4e7bf7e5fa3` | 2026-10-30 19:40:12 |
| `short.nsys-rep` | 3267612 | `dc48d1d56315b976bf144203dcb136285e832bda5ec8eae1e56382d71557a0e5` | 2026-10-30 19:40:17 |

The start is the value `TARGET_INFO_SESSION_START_TIME` in the `.sqlite` file that `nsys` makes from each capture. The captures ran with Nsight Systems 2026.3.2.

| File | Cause |
|---|---|
| The model file | It has 3.3 GB. The data set has only the tables that the script makes from the start of the file. |
| The prompts | Step `2` writes them to a temporary folder. The tables have their token counts. |
| The `.sqlite` files of the captures | Step `T` writes them to a temporary folder. The tables have the data that the census uses. |
| `trace-notes.md` of the first run | `trace_census.py` writes the check of the passes to its output. The report refers to this script. |

The scan of a folder with Nsight Compute reports shows the IP address of a container in each report. The IP address is private data. This is not applicable to this data set, because it has no such report.

## Findings

The numbers come from the tables of this folder. The floors use the nominal rates of the section [GPU](#gpu).

| Finding | Value | File |
|---|---|---|
| Passes of a request | 1 for `short-1q`, 3 for `short-2q`, 4 for `medium-3q`, and 5 for `long-3q`. | `tables/request-floors.tsv` |
| Weights that a pass reads | 3.335 GB with the output head and 2.620 GB without it. At 448 GB/s the times are 7.444 ms and 5.848 ms. | `tables/census-passes.tsv` |
| Weight-read floor of a request | 7.444 ms, 20.737 ms, 28.181 ms, and 34.029 ms for `short-1q`, `short-2q`, `medium-3q`, and `long-3q`. | `tables/request-floors.tsv` |
| Floor of the larger value of weights and int8 for each pass | 7.644 ms, 20.737 ms, 84.569 ms, and 177.970 ms for the four request types. | `tables/request-floors.tsv` |
| Launches of a pass | 245 MMQ and 3 MMVQ if the pass has an output. 248 MMVQ if the pass has 8 tokens or less. | `tables/census-passes.tsv` |
| Output head | The pass calculates 248320 rows for each output. The API reads 2 to 8 rows of them. | `tables/census-head.tsv` |
| Launch counts of the model | The model gives the launch count of each kernel class in 72 of 72 passes of the captures. | `tables/crosscheck-summary.tsv` |
| Matmul launches of the model | The model gives the key of each launch in 37 of 37 passes with a launch list. These passes have 9119 launches. | `tables/crosscheck-summary.tsv` |
| Requests | A slot assignment of the model gives the passes of 20 of 20 requests of the captures. | `tables/crosscheck-requests.tsv` |
| KV window of flash attention | The KV window of the capture is the same as the KV window of the model in 33 of 72 passes. It is larger in 39 passes, at most 5.0 times. | `tables/crosscheck-passes.tsv` |
| GPU busy time of the captures | The GPU busy time of a pass is 1.6 to 6.1 times `floor_pass_max_ms` and 1.4 to 2.4 times `floor_roofline_ms`. The captures have the overhead of `nsys`. | `tables/crosscheck-passes.tsv` |

The report shows two more results in its sections 8.2 and 8.3. The MMQ kernels get 31 to 53 int8 TOPS, which is 22 to 37 % of 144.8 TOPS. The pass of 8 tokens or less reads weights at 365 GB/s, which is 81 % of 448 GB/s.

## Changes to the files of the first run

The tables `census*.tsv` and `crosscheck-*.tsv` are the same as the tables of the first run. We made these changes:

- `raw/model-header/hparams.tsv` has two different texts in the column `source`. The first run read the model file from a server. The rows `gguf_file_bytes` and `head_bytes_fetched` name the local file.
- `01_gguf_header.py` reads a local GGUF file with the flag `--gguf`. It does not connect to a server.
- `02_tokens.py` reads the template, the 255 decision labels, and the spellings of the labels from the GGUF file. The first run used three copies. They are private data. This run uses the public benchmark `bench/cases.json`. 28 of the 48 case texts are different from the texts of the first run. The four files in `raw/tokens/` are the same as the files of the first run.
- `trace_census.py` has the optional flag `--old-export`. It compares the new tables with the tables of the first run.
- `03_census.py` writes the new table `tables/request-floors.tsv`.
- `05_report.py` writes `reports/census-report.txt`. The first run made a `README.md`. We removed the text with paths and host names.
- `trace-kernels.tsv`, `trace-passes.tsv`, and `trace-mmq-launches.tsv` are the same as the files of the first run. `cmp` shows no difference.

## Limits of the data

- The census does not measure. Its work, bytes, and floors come from the shapes of the model and from the nominal rates of the GPU.
- The bytes that the other operations of a layer read and write are an estimate. [INFERENCE] The highest FP16 and FP32 rates are also estimates.
- The census models the KV window as the cells of the request. The captures show a larger KV window in 39 of 72 passes. [INFERENCE] The cause is that the cells of slots with no work stay in the unified KV cache. The model does not correct this.
- The three captures have the overhead of `nsys`. A first estimate is 3 % for the long capture and 25 % for the short capture. We did not make sure of it.
- The captures have 20 requests. The data set does not show how frequently each slot assignment occurs in production.
- The short requests of the captures have a random `Ref:` value. Thus the prefix of a short request changes by some tokens. The check against the captures uses the prefix of the capture and the suffixes of the case.
