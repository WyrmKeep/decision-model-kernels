# Profiles of the decision model: where the time of a request goes

This data set has the measurements of phase 3 for the `RTX 3060 Ti`. It has four parts. The first part divides the request time with the build `ablate-timing`, and it uses no profiler. The second part has the Nsight Systems captures of the server. The third part has two lead tests. The fourth part has the Nsight Compute profiles of kernels.

The data set has the raw files of the runs and the tables that the scripts make. The folder `reports/` has the reports in text. The [notebook entry](../../docs/notebook/2026-10-31-phase3-where-the-time-goes.md) shows the numbers with their meaning.

You can make the tables of the first part and of the lead #30779 again from `raw/`. The tables of Nsight Systems and Nsight Compute use the capture files. The capture files are not in this data set. The section [Files not added](#files-not-added) gives the cause.

## Question

The profile gives answers for phase 3:

- The parts of the wall time of a request: HTTP, the prompt, the host work for the passes, the submit step, and the GPU wait.
- The time that `DMK_TIMING` adds to a request, and the change of the time from the first to the second of two equal requests.
- The time of the kernels of a pass. The gaps between the kernels and between the passes. The time of the CUDA calls of the host.
- The time that Nsight Systems adds to a request.
- The distance of each kernel from its limit. The stall reasons and the instructions that use the time of an MMQ kernel.
- The lead #28495: the KV window of flash attention can be wider than the cells of the sequence. The cause can be cells of other task slots in the cache of `--kv-unified`. The test measures the time that this adds.
- The lead #30779: a child task on a task slot with a lower number than the parent slot can get more passes. The test checks the passes and the time.

At this time, the [notebook](../../docs/notebook/README.md) has no prediction entry for this profile. The results entry is [phase 3: where the time goes](../../docs/notebook/2026-10-31-phase3-where-the-time-goes.md).

## Dates

- First run: `p3-dec-timing`, on 2026-10-31 at 00:10 UTC.
- Last run: `p3-ncu-extra-long-full`, on 2026-10-31 at 01:13 UTC.
- The tables and the reports: made on 2026-10-31. The section [Arms](#arms) gives the start of each run.

## Project commit

The commit that adds this data set also adds the code in `harness/profile/`. The commit before it is `e91d24b53413fb9922120760e5c8317a95a6d9c2`.

The runs and the first run of the scripts used the same code in a work folder that is not in the repository. The code in `harness/profile/` has no path of a host. The tables of `make_tables.sh` have the same bytes as the tables of the first run of the scripts. Two tables are new: the tables of `ncu_source.py` in `tables/ncu/`. The summary tables in `tables/summary/` are also new.

## Build commits

| Arm | Source | Commit |
|---|---|---|
| `build:b12004-sm86` | Build `D` of tag `b12004`. It is the baseline build. | `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. The source had 0 changes. |
| `build:ablate-timing` | The branch `dmk/ablate-timing` of the fork of this project. It has 3 commits on `b12004`. | `aaf86b0528b37f1125a397efecf169a06b51447e`. The source had 0 changes. |

The 3 commits of the branch are `c3747185f` (the code of `DMK_ABLATE`), `5d35fda2c` (`DMK_TIMING` in the library), and `aaf86b052` (the request line of the server). The fork is not in this repository. The server log shows `DMK_ABLATE active: mask=0x0 items="" sync=0 sync_dump=0 timing=1`. Thus the runs of this arm used the lines of `DMK_TIMING` and no part of `DMK_ABLATE`.

We made the two builds with `harness/build/build.sh`, on 2026-10-30 at 21:50 UTC (`D`) and at 22:35 UTC (`ablate-timing`). The data set [`2026-10-30-window1`](../2026-10-30-window1/README.md) gives more information about build `D`.

## Build flags

The two builds used the same tools and the same `cmake` flags. The tools are the image `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`, `nvcc` 12.8.93, `gcc` 14.2.0, and `cmake` 3.28.3. The flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

## Image

The runs used the stock image `ghcr.io/ggml-org/llama.cpp@sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352`. Its tag is `server-cuda-b12004`. The harness puts the programs of the build in the container at `/opt/dmk`. The container starts the server of the build.

The runs with a profiler put the folder of the profiler in the container too. The tools are Nsight Systems 2026.3.2 and Nsight Compute 2026.3.1.

## Model

The model is H2O Lightning 4B in the "mix8" quantization. The weights of the feed-forward layers are Q4_0. The other weights are Q8_0. The embedding and the output head are Q8_0. The model has 32 layers. 24 layers are recurrent layers (Gated DeltaNet) and 8 layers are attention layers.

The data set has no model file and no path of one. The client of each run sent the 48 cases of the private benchmark. The file `bench/cases.json` of this repository has the same 48 case names and the same token counts. [INFERENCE] The texts of the cases are different.

## Driver

The driver is 595.71.05. The `.meta` file of each run shows it.

## GPU

The GPU is an `NVIDIA GeForce RTX 3060 Ti` (GA104, `sm_86`). It has 38 SMs. We stopped the production server for the runs. The schedule ran one run after the other.

## Power limit

The power limit is 200 W. The `.meta` file of each run shows it. The harness does not change it.

## Clocks

The value is `default`. The harness does not set the clocks and does not lock them. The `.smi` file of each run has the SM clock. The runs with Nsight Compute use `--clock-control none`.

## Server flags

The server of each run used these flags:

```
-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots
```

Some runs have a difference:

| Runs | Difference |
|---|---|
| `p3-lead-kvwin-nokvu-timing`, `p3-lead-kvwin-nokvu-nsys` | `--no-kv-unified` and not `--kv-unified`. |
| All runs of `4-ncu-*` | These runs have the flag `--no-warmup` too. Each run sends one request to a new server. |
| `p3-dec-timing`, `p3-lead-kvwin-timing`, `p3-lead-kvwin-nokvu-timing` | These runs have `DMK_TIMING=1` in `DMK_DOCKER_EXTRA="-e DMK_TIMING=1"`. The build has an env file with the same value. [INFERENCE] The effect is the same. |

The runs of `2-nsys` and the Nsight Systems runs of `3-leads` start the server with Nsight Systems. The flags of the profiler are `profile --trace=cuda,osrt,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none`. The runs of `4-ncu-*` start the server with Nsight Compute. The schedule files in `raw/4-ncu-*/` have the flags, the names of the kernels, and the number of launches of each run.

## Commands

The schedule files are in `harness/profile/sched/`. The lines with the profiler of `2-nsys` and `3-leads` have the word `@REMOTE_DIR@` for the work folder of the GPU host. `make_sched.sh` writes copies with your work folder:

```
bash harness/profile/make_sched.sh --remote-dir <work folder of the GPU host> --out <schedule folder>
```

Run each schedule with the harness. The results go to one folder for each schedule:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/1-decomp.txt <results>/1-decomp
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/2-nsys.txt <results>/2-nsys
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/3-leads.txt <results>/3-leads
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/4-ncu-smoke.txt <results>/4-ncu-smoke
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/4-ncu-core.txt <results>/4-ncu-core
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/4-ncu-extra.txt <results>/4-ncu-extra
```

The folder `harness/profile/extra/` is the folder `DMK_EXTRA_DIR` of the env file. The files `NOTES.txt` of `harness/profile/nsys/` and `harness/profile/ncu/` give the steps to start the profilers on the GPU host.

Make the tables and the reports of the first part and of the lead #30779 again from `raw/`. The script writes to a different folder. It uses `python3` only:

```
out=$(mktemp -d)
bash make_tables.sh --out "$out"
diff -r "$out/tables/decomp" tables/decomp
diff -r "$out/tables/leads-childpass" tables/leads-childpass
diff -r "$out/reports/decomp" reports/decomp
diff -r "$out/reports/leads-childpass" reports/leads-childpass
```

Make all tables again. This uses the capture files, Nsight Systems, and Nsight Compute. The folder with the captures has one folder for each schedule, as `raw/` has. The files are `*.nsys-rep` in `2-nsys` and in `3-leads`, and `*.ncu-repz` in the three folders `4-ncu-*`:

```
DMK_NSYS=<nsys command line tool> DMK_NCU=<ncu command line tool> \
    bash make_tables.sh --captures <folder with the captures>
```

The script runs `harness/profile/analyze_all.sh` with the flag `--no-mmq-launches`. Then it runs `harness/profile/ncu/ncu_source.py` and `make_summary.py`. Without `--captures` the script has to have `--out`, because it removes the tables of its output folder.

## Arms

The data set has two builds. Each schedule is one part of the profile. The column Start is the value `date_utc` of the `.meta` file of the first run.

| Schedule | Runs | Build | Start (UTC) | Contents |
|---|---|---|---|---|
| `1-decomp` | `p3-dec-timing` | `build:ablate-timing` | 2026-10-31 00:10 | 348 requests with `DMK_TIMING=1`. The server log has the lines of `DMK_TIMING`. |
| `1-decomp` | `p3-dec-ref` | `build:b12004-sm86` | 2026-10-31 00:12 | The same 348 requests with no `DMK_TIMING`. |
| `2-nsys` | `p3-nsys-short1q`, `p3-nsys-short2q`, `p3-nsys-medium3q`, `p3-nsys-long3q` | `build:b12004-sm86` | 2026-10-31 00:14 | One run for each request type with Nsight Systems. 2 warm-up requests and 10 requests that count. |
| `2-nsys` | `p3-nsys-all48` | `build:b12004-sm86` | 2026-10-31 00:15 | The 48 cases, one time each, with Nsight Systems. |
| `2-nsys` | `p3-nsys-ctl` | `build:b12004-sm86` | 2026-10-31 00:14 | The same sequences with no Nsight Systems. |
| `3-leads` | `p3-lead-kvwin-timing`, `p3-lead-kvwin-nokvu-timing` | `build:ablate-timing` | 2026-10-31 01:04 | The lead tests with `DMK_TIMING=1`, with `--kv-unified` and with `--no-kv-unified`. |
| `3-leads` | `p3-lead-kvwin-nsys`, `p3-lead-kvwin-nokvu-nsys` | `build:b12004-sm86` | 2026-10-31 01:04 | The same lead tests with Nsight Systems. |
| `4-ncu-smoke` | `p3-ncu-smoke-a`, `p3-ncu-smoke-b` | `build:b12004-sm86` | 2026-10-31 01:05 | 2 runs with Nsight Compute. They check which launches the profiler selects. |
| `4-ncu-core` | 6 runs `p3-ncu-core-*` | `build:b12004-sm86` | 2026-10-31 01:05 | The MMQ kernels, the MMVQ kernels, `gdn_single`, and `concat_non_cont` of short requests and long requests. |
| `4-ncu-extra` | 5 runs `p3-ncu-extra-*` | `build:b12004-sm86` | 2026-10-31 01:10 | MMQ at 1,288 tokens, the small MMQ kernels, flash attention, the small kernels, and the full set of metrics for one MMQ kernel. |

The request types are `short1q`, `short2q`, `medium3q`, and `long3q`. A `short` request has 1 or 2 questions. A `medium` request and a `long` request have 3 questions. A request of the types `medium3q` and `long3q` has a prefix pass and suffix passes, as the census shows.

The lead test sends 2 warm-up requests and 10 long requests. The first long request (T1) goes to a cache with a small number of cells. Then 6 other cases follow. The requests T2, T3, and T4 send the case of T1 again.

## Raw files

The folder `raw/` has one folder for each schedule. The files of a run have the name of the run and one extension. The harness makes them.

| Extension | Contents |
|---|---|
| `.client.json` | The JSON file of the client. It has the case, the phase, the wall time, and the times at which the client sends and receives each request. |
| `.client.out` | The standard output of the client. |
| `.server.log` | The log of the server. The runs with `DMK_TIMING=1` have the lines of `DMK_TIMING`. |
| `.launch.log` | The output of `start_server.sh`. |
| `.smi` | The samples of the sampler of the GPU, one line each 250 ms. The [harness document](../../harness/README.md) gives the fields. |
| `.meta` | The facts of the run: the arm, the start, the driver, and the power limit. |
| `.load0`, `.load1` | The line of `/proc/loadavg` before the server starts and after the run. |
| `.txt` | The copy of the schedule file in the three folders `4-ncu-*`. |

Some runs of `4-ncu-*` have no `.client.json` and no `.client.out`. Nsight Compute ends the server after the last profiled launch, and the client then sees an error. The two runs `p3-ncu-smoke-a` and `p3-ncu-smoke-b` have no client file.

The folders `1-decomp`, `2-nsys`, and `3-leads` have the files of 2, 6, and 4 runs. They have 16, 48, and 32 files. The folders `4-ncu-smoke`, `4-ncu-core`, and `4-ncu-extra` have 13, 45, and 37 files.

## Tables

`make_tables.sh` makes the tables in `tables/` and the reports in `reports/`. `columns.tsv` has the table, the column, the unit, and the meaning of each column of each table in `tables/`. The tables of `tables/leads-nsys/` are the tables of `tables/nsys/` for the two Nsight Systems runs of `3-leads`. They have the same columns.

The first line of a table of Nsight Compute has the name of each column and a short meaning in the characters `[` and `]`. The column `column` of `columns.tsv` is the name before the `[`.

| File | Contents |
|---|---|
| `tables/decomp/requests.tsv` | One row for each request of the two runs of `1-decomp`, with all times of the lines of `DMK_TIMING`. |
| `tables/decomp/passes.tsv` | One row for each pass of the run `p3-dec-timing`. |
| `tables/decomp/stack.tsv` | The mean time of each step of a request, for each request type and each client phase. |
| `tables/decomp/components.tsv` | The mean, the median, and the 95th percentile of each component, with its part of the wall time. |
| `tables/decomp/summary-wall.tsv` | The wall time for each request type and each client phase. |
| `tables/decomp/fractions.tsv` | The part of each step in the wall time. The sum of each row is 100 %. |
| `tables/decomp/hostbound.tsv` | The submit step and the GPU wait against the GPU span and the floors. |
| `tables/decomp/waterfall-short1q*.tsv` | The three tables of the steps of a `short1q` request and its references. |
| `tables/decomp/instrumentation.tsv` | The time of `p3-dec-timing` against the time of `p3-dec-ref`. |
| `tables/decomp/warm-path.tsv` | The change from the first to the second of two equal requests, and the change of its parts. |
| `tables/decomp/warm-path-work.tsv` | The work of the server in the first and in the second request of a pair. |
| `tables/decomp/clock-check.tsv` | The check that the clocks of the client and the server agree. |
| `tables/nsys/requests.tsv` | One row for each request of the Nsight Systems runs of `2-nsys`, with the GPU busy time and the gaps. |
| `tables/nsys/passes.tsv` | One row for each pass and each request, with the gaps before the pass and the CUDA calls. |
| `tables/nsys/classes.tsv` | The launches and the kernel time of a pass, for each pass type and kernel class. |
| `tables/nsys/kernels.tsv` | One row for each shape of a kernel launch in each capture. |
| `tables/nsys/osrt-request-start.tsv` | The functions of the system that the host threads use at the start of the first requests. |
| `tables/nsys/overhead*.tsv` | The wall time with and with no Nsight Systems, for each pair of requests. |
| `tables/leads-nsys/*.tsv` | The same tables for the two Nsight Systems runs of `3-leads`. |
| `tables/leads-kvwin/*.tsv` | The KV window, the time of flash attention, and the GPU busy time of each pass of the lead #28495. |
| `tables/leads-childpass/childpass-requests.tsv` | One row for each request of the three server logs with the lines of `DMK_TIMING`, for the lead #30779. |
| `tables/ncu/alignment.tsv` | For each run of Nsight Compute: the number of launches in its list, the number of launches that agree with the list, and the type of the run. |
| `tables/ncu/kernels.tsv` | One row for each profiled launch, with the metrics. |
| `tables/ncu/classes.tsv` | The mean of the metrics for each kernel class of each run. |
| `tables/ncu/sol.tsv` | The speed of light for each kernel class: the time, the limit, the stall reasons, and the DRAM rate. |
| `tables/ncu/ffn_up_q4_0_N2048-instructions.tsv` | One row for each SASS line of one MMQ kernel, with the stall samples and the instructions that the kernel ran. |
| `tables/ncu/ffn_up_q4_0_N2048-opcodes.tsv` | The same data for each opcode. |
| `tables/summary/*.tsv` | The 20 tables of the notebook entry. `make_summary.py` makes them from the other tables. |
| `reports/*/*.txt` | The reports in text. The folder `reports/leads-childpass/` has the text that `analyze_childpass.py` writes. |

A report has the work folder of the script and the run time of the script. These values change with each run. The script replaces the work folder with `raw/` and `reports/`, and it removes the line `wall time of this analysis`.

The next sections give the columns of the tables of `tables/summary/`. `columns.tsv` has these columns too, and the columns of all other tables. This is a difference from the model README. The data set has 53 tables with 921 columns, and a README with all of them is too long.

### Table of the columns: `tables/summary/request-time-decomposition.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type: `short1q`, `short2q`, `medium3q`, or `long3q`. |
| `requests` | count | Number of requests of the phase `unique` in `tables/decomp/stack.tsv`. |
| `wall_ms` | ms | Mean wall time of the requests. |
| `before_submit_ms` | ms | Mean time of the steps before the submit step. They are HTTP, the prompt, the start of the task, and the host work to prepare the passes. |
| `submit_ms` | ms | Mean host time to submit the passes to the GPU, for all passes of the request. It is not the time of the GPU. |
| `gpu_wait_ms` | ms | Mean GPU wait, for all passes of the request. |
| `other_steps_ms` | ms | Mean time of the other steps: the host work between the passes and after the last pass, the end in the server, the answer, and HTTP. |
| `gpu_span_ms` | ms | Mean GPU span: from the start of the first pass to the end of the last GPU wait. |
| `floor_pass_max_ms` | ms | Mean sum of the larger value of the weight-read floor and the int8 floor for the passes of the request. |
| `floor_roofline_ms` | ms | Mean sum of the roofline floor for the passes of the request. |
| `span_over_floor_pass_max` | ratio | The GPU span divided by `floor_pass_max_ms`. |

### Table of the columns: `tables/summary/short1q-waterfall.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `step` | None | One step of a `short1q` request. The last row is the wall time. |
| `mean_ms` | ms | Mean time of the step. |
| `cumulative_ms` | ms | Sum of this step and the steps above it. |
| `share_of_wall_pct` | % | The mean time of the step, as a part of the wall time. |

### Table of the columns: `tables/summary/short1q-reference.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `quantity` | None | The name of the value. |
| `unit` | None | `ms` or `ratio`. |
| `value` | See unit | The value for a `short1q` request. |

### Table of the columns: `tables/summary/instrumentation-cost.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `requests_each_arm` | count | Number of requests in each of the two runs. |
| `timing_build_mean_ms` | ms | Mean wall time with the build `ablate-timing` and `DMK_TIMING=1`. |
| `reference_build_mean_ms` | ms | Mean wall time with the build `b12004-sm86`. |
| `difference_pct` | % | The time with `DMK_TIMING` minus the reference time, as a part of the reference time. A value below 0 means that the build with `DMK_TIMING` is faster. |

### Table of the columns: `tables/summary/warm-path.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `case` | None | Name of the case. |
| `pairs` | count | Number of pairs. A pair is two requests with the same text, one after the other. |
| `first_wall_ms` | ms | Mean wall time of the first request of the pairs. |
| `second_wall_ms` | ms | Mean wall time of the second request of the pairs. |
| `wall_change_ms` | ms | The second time minus the first time, mean of the pairs. |
| `submit_change_ms` | ms | The same change for the submit step. |
| `gpu_wait_change_ms` | ms | The same change for the GPU wait. |

### Table of the columns: `tables/summary/nsys-overhead.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `pairs` | count | Number of pairs of the same case, one with Nsight Systems and one without it. |
| `without_nsys_mean_ms` | ms | Mean wall time without Nsight Systems. |
| `with_nsys_mean_ms` | ms | Mean wall time with Nsight Systems. |
| `difference_ms` | ms | The time with Nsight Systems minus the time without it. |
| `difference_pct` | % | The difference as a part of the time without Nsight Systems. |
| `ci95_halfwidth_ms` | ms | The difference is in the interval from minus this value to this value, with a probability of 95 %. |

### Table of the columns: `tables/summary/nsys-request-time.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `requests` | count | Number of requests of the run `p3-nsys-all48`, without the requests of the warm-up. |
| `passes_per_request` | count | Mean number of passes in a request. |
| `client_wall_ms` | ms | Mean wall time. |
| `send_to_first_kernel_ms` | ms | Mean time from the start of the request at the client to the first kernel. |
| `gpu_busy_ms` | ms | Mean of the GPU busy time of the request. |
| `idle_inside_passes_ms` | ms | Mean idle time of the GPU in the passes. |
| `idle_between_passes_ms` | ms | Mean idle time of the GPU between two passes. |
| `last_kernel_to_recv_ms` | ms | Mean time from the end of the last kernel to the end of the request at the client. |
| `gpu_busy_share_pct` | % | The GPU busy time as a part of the wall time. |

### Table of the columns: `tables/summary/nsys-gap-before-pass.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `boundary` | None | The position of the gap: the passes before and after it. `request-start` is the gap before the first pass. |
| `passes` | count | Number of passes with this position. |
| `gap_mean_ms` | ms | Mean gap: the sum of the four parts that follow. |
| `gpu_done_to_sync_return_ms` | ms | Mean time from the end of the last kernel of the previous pass to the end of the sync call. |
| `no_cuda_call_ms` | ms | Mean time in which the decode thread has no CUDA call. |
| `no_cuda_call_in_os_wait_ms` | ms | The part of `no_cuda_call_ms` in a system wait. |
| `no_cuda_call_busy_ms` | ms | The part of `no_cuda_call_ms` in which the CPU works. |
| `cuda_calls_ms` | ms | Mean time in CUDA calls. |
| `first_launch_to_first_kernel_ms` | ms | Mean time from the first launch of the pass to its first kernel. |

### Table of the columns: `tables/summary/nsys-uploads.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `requests` | count | Number of requests of the run `p3-nsys-all48`, without the warm-up. |
| `upload_copies` | count | Mean number of uploads in a request. |
| `uploaded_mb` | MB | Mean data of the uploads in a request. |
| `upload_gpu_ms` | ms | Mean GPU time of the uploads in a request. |
| `upload_rate_gb_s` | GB/s | The data of the uploads divided by their GPU time. |

### Table of the columns: `tables/summary/nsys-host-cuda-calls.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `requests` | count | Number of requests. |
| `launch_calls` | count | Mean number of launches in a request. |
| `launch_calls_ms` | ms | Mean time of the decode thread in the launches of a request. |
| `sync_calls` | count | Mean number of sync calls in a request. |
| `sync_calls_ms` | ms | Mean time of the decode thread in the sync calls of a request. |
| `all_cuda_calls_ms` | ms | Mean time of the decode thread in all CUDA calls of a request. |
| `command_buffer_full_events` | count | Mean number of times in a request that the command buffer is full. |
| `command_buffer_full_ms` | ms | Mean time in a request in which the command buffer is full. |
| `launch_queue_depth_max` | launches | Largest launch queue in the requests. |

### Table of the columns: `tables/summary/nsys-kernel-ms-per-pass.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `pass_type` | None | Type of the pass: `prefix-large`, `prefix-small`, `suffix`, or `single`. |
| `tokens_in_pass` | None | Group of the number of tokens of the pass. |
| `passes` | count | Number of passes in the group. |
| `launches_per_pass` | count | Mean number of launches of a pass. |
| `all_kernels_ms` | ms | Mean kernel time of a pass. |
| `MMQ_ms` | ms | Mean kernel time of the kernel class `MMQ`. The columns that follow give the same value for the other kernel classes. |
| `GDN_ms` | ms | Kernel class `GDN`: the Gated DeltaNet kernels. |
| `glu_ms` | ms | Kernel class `glu`. |
| `FA_ms` | ms | Kernel class `FA`: the flash-attention kernels. |
| `quantize_ms` | ms | Kernel class `quantize`. |
| `copy_ms` | ms | Kernel class `copy`. |
| `rms_norm_ms` | ms | Kernel class `rms_norm`. |
| `ssm_conv_ms` | ms | Kernel class `ssm_conv`. |
| `elementwise_ms` | ms | Kernel class `elementwise`. |
| `MMVQ_ms` | ms | Kernel class `MMVQ`. |
| `other_ms` | ms | All other kernels. |

### Table of the columns: `tables/summary/ncu-run-alignment.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | Name of the run, without the prefix `p3-ncu-`. |
| `planned_launches` | count | Number of launches in the list of the run. |
| `matching_launches` | count | Number of profiled launches that have the kernel and the grid of the list. |
| `shift_launches` | launches | The difference in launches between the start of the run and the start of the list. A value below 0 means that the run starts before the first launch of the list. |
| `status` | None | `ALIGNED`, `SHIFTED`, or `MISALIGNED`. |

### Table of the columns: `tables/summary/ncu-mmq-large-n.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `n_tokens` | tokens | Number of tokens of the pass. |
| `op` | None | Operation of the model that the kernel does. |
| `weight_type` | None | Type of the weights: `Q4_0` or `Q8_0`. |
| `launches_per_pass` | count | Number of launches of this kernel in a pass. |
| `duration_us` | µs | Time of one launch, as Nsight Compute measures it. |
| `share_of_pass_pct` | % | The time of all launches of this kernel as a part of the kernel time of the pass. |
| `int8_of_peak_pct` | % | The int8 operations of the model, divided by the time and by the int8 peak of 149.3 TOP/s. |
| `imma_pipe_active_pct` | % | Part of the time in which the IMMA pipe is active. |
| `dram_of_422_gb_s_pct` | % | The DRAM data rate as a part of 422 GB/s. |
| `top_stall_reasons` | None | The three largest stall reasons, with their part of the stalls. |

### Table of the columns: `tables/summary/ncu-mmq-resources.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `kernel` | None | Weight type and operation of the kernel. |
| `grid` | None | Grid of the launch: the number of thread blocks in three dimensions. |
| `block` | None | Thread block: the number of threads in three dimensions. |
| `registers_per_thread` | count | Registers of each thread. |
| `dynamic_smem_b` | B | Bytes of shared memory of a thread block that the launch sets. |
| `blocks_per_sm_limit_registers` | count | Limit of the thread blocks on an SM by the registers. |
| `blocks_per_sm_limit_smem` | count | Limit of the thread blocks on an SM by the shared memory. |
| `blocks_per_sm_limit_warps` | count | Limit of the thread blocks on an SM by the warps. |
| `blocks_per_sm_limit_blocks` | count | Limit of the thread blocks on an SM by the limit of the SM for thread blocks. |
| `warps_active_pct` | % | Part of the warp slots of an SM that have an active warp. |
| `issue_slots_active_pct` | % | Part of the issue slots that start an instruction. |
| `imma_pipe_active_pct` | % | Part of the time in which the IMMA pipe is active. |
| `sm_clock_mhz` | MHz | SM clock during the launch. |

### Table of the columns: `tables/summary/ncu-mmq-small-n.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `n_tokens` | tokens | Number of tokens of the pass. |
| `tile_columns_j` | count | Number of columns of the tile of the kernel (the template value `J`). |
| `op` | None | Operation of the model that the kernel does. |
| `weight_type` | None | Type of the weights. |
| `duration_us` | µs | Time of one launch. |
| `limit` | None | The limit that sets the floor of the kernel: `DRAM`, `INT8`, `FP16`, `FP32`, or `launch`. |
| `of_limit_pct` | % | The floor of the kernel divided by the time of the launch. |
| `imma_pipe_active_pct` | % | Part of the time in which the IMMA pipe is active. |
| `int8_of_peak_pct` | % | The int8 operations of the model, divided by the time and by the int8 peak. |

### Table of the columns: `tables/summary/ncu-other-kernels.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | Name of the run, without the prefix `p3-ncu-`. |
| `run_alignment` | None | `ALIGNED`, `SHIFTED`, or `MISALIGNED`. |
| `kernel` | None | Name of the kernel and its grid. |
| `duration_us` | µs | Time of one launch. |
| `limit` | None | The limit that sets the floor of the kernel: `DRAM`, `INT8`, `FP16`, `FP32`, or `launch`. |
| `of_limit_pct` | % | The floor of the kernel divided by the time of the launch. |
| `dram_gb_s` | GB/s | DRAM data rate. |
| `dram_of_422_gb_s_pct` | % | The DRAM data rate as a part of 422 GB/s. |
| `top_stall_reasons` | None | The three largest stall reasons, with their part of the stalls. |

### Table of the columns: `tables/summary/ncu-instruction-mix.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `opcode` | None | Opcode of the SASS instruction. The last row sums all opcodes. |
| `sass_lines` | count | Number of SASS lines with this opcode. |
| `stall_samples_pct` | % | The stall samples at these lines, as a part of all stall samples. |
| `instructions_executed_millions` | 10^6 | Number of instructions that the kernel ran at these lines, divided by 10^6. |
| `instructions_per_imma` | ratio | The number of instructions that the kernel ran at these lines, divided by the number of IMMA instructions that it ran. |

### Table of the columns: `tables/summary/lead-28495-nsys-phases.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `cache_mode` | None | `kv-unified` or `no-kv-unified`. |
| `phase` | None | Phase of the test: `T2`, `T3`, or `T4`. Phase `T1` is the reference. |
| `passes_compared` | None | The number of passes of the phase that have a pass in `T1` with the same position and the same number of tokens, and the number of passes of the phase. |
| `fa_ms` | ms | Sum of the flash-attention time of the passes that the table compares. |
| `fa_same_passes_of_t1_ms` | ms | The same sum for the same passes of `T1`. |
| `fa_ratio` | ratio | `fa_ms` divided by `fa_same_passes_of_t1_ms`. |
| `gpu_busy_ms` | ms | Sum of the GPU busy time of the passes that the table compares. |
| `gpu_busy_same_passes_of_t1_ms` | ms | The same sum for the same passes of `T1`. |
| `gpu_busy_ratio` | ratio | `gpu_busy_ms` divided by `gpu_busy_same_passes_of_t1_ms`. |

### Table of the columns: `tables/summary/lead-30779-passes.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type. |
| `requests` | count | Number of requests in the three server logs with `DMK_TIMING`. |
| `passes_equal_census` | count | Number of requests for which the census model for `--kv-unified` gives the passes. |
| `requests_with_more_passes_than_census` | count | Number of requests that have more passes than the census model gives. |
| `requests_child_below_parent` | count | Number of requests with a child slot that has a lower number than the parent slot. |
| `requests_with_more_decode_calls_than_fixed_order` | count | Number of requests that use `llama_decode` more times than the fixed-order model. |
| `requests_with_other_pass_sizes_than_fixed_order` | count | Number of requests with passes of other lengths than the passes of the fixed-order model. |

### Table of the columns: `tables/summary/lead-30779-cost.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `request_type` | None | Request type: `medium3q` or `long3q`. |
| `child_below_parent` | None | `yes` when a child slot has a lower number than the parent slot. |
| `requests` | count | Number of requests. |
| `request_total_mean_ms` | ms | Mean request time that the server writes in the request line. |
| `decode_calls_mean` | count | Mean number of uses of `llama_decode` in a request. |

## Files not added

The data set does not have all files of the runs. This section lists them with the bytes and the SHA-256 of each file. A scan of the files with `bash tools/publish/scan-dir.sh` shows the cause. The scan has three types of finding.

- The `.nsys-rep` files have the UUID of the GPU (the rule `dmk-uuid`). The publish check also blocks this file type.
- The report in each `.ncu-repz` file has the IP address of a container (the rule `dmk-private-ipv4`). The IP address is private data.
- The schedule files `1-decomp.txt`, `2-nsys.txt`, and `3-leads.txt` have the path of the work folder of the GPU host (the rule `dmk-server-path`). The folder `harness/profile/sched/` has the schedule files without this path. Its lines with the profiler have the word `@REMOTE_DIR@` for it. The files `4-ncu-*.txt` have no such path, and `raw/` has them.

The `.sqlite` files come from `nsys export` of the `.nsys-rep` files. They are not raw files, and the publish check blocks them. `make_tables.sh` makes them in a temporary folder and removes them.

The tables `mmq-launches.tsv` of `tables/nsys/` and `tables/leads-nsys/` are not in the data set. They have 8 MB and 2.65 MB, and the publish check stops files of more than 5 MiB. `make_tables.sh` uses the flag `--no-mmq-launches`.

| File | Bytes | SHA-256 | Cause |
|---|---:|---|---|
| `1-decomp/1-decomp.txt` | 2321 | `c6d3303fc397c81f53c40c37bf387ade4344f3ee843515ac7a5d8f844ac6b6da` | The path of the work folder of the GPU host is in the file (rule `dmk-server-path`). |
| `2-nsys/2-nsys.txt` | 9463 | `d55ea389dca5ade187a964f13b6de85588eba0d34137e5877f3cb96c38ce5010` | The path of the work folder of the GPU host is in the file (rule `dmk-server-path`). |
| `2-nsys/p3-nsys-all48.nsys-rep` | 9101759 | `ade61f0c34c4f680ff00ede1ea1baa8a57df5926a76d0f5f7a525d5927381293` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `2-nsys/p3-nsys-all48.sqlite` | 27377664 | `c48c62915701c77ebf7db0c41b59bb307ecd41ad520a56330a8929dceeb422e7` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `2-nsys/p3-nsys-long3q.nsys-rep` | 3841827 | `63a7363021e3b1e0df8f4b510bf9d50b0a2ad0fe6481ef2560d553cdec056b0d` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `2-nsys/p3-nsys-long3q.sqlite` | 11644928 | `1438f94ebcd07c1ce6536a0428db5be4a20ce23f1f3bdf229521214bef101595` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `2-nsys/p3-nsys-medium3q.nsys-rep` | 3171848 | `ddb3fa1b84569dff16a777b97a4d29a47dcc7f3bd73bdd1a53ff5951cd0f4301` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `2-nsys/p3-nsys-medium3q.sqlite` | 9682944 | `f95f9befef8e427934bebdf5bb56299e4e40a5fd3a70b3a41edc182a69853b10` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `2-nsys/p3-nsys-short1q.nsys-rep` | 1127118 | `d1315984e3ee545d075bc68bc12da542fd8792cfd5e682c145a188032c2306e8` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `2-nsys/p3-nsys-short1q.sqlite` | 3620864 | `273cc2b35697b539d451bcb822627aafe2f5d7ac026e2b7c0c55159bb32dfb80` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `2-nsys/p3-nsys-short2q.nsys-rep` | 2610112 | `e40746001198c174e65590ef50849d2ca8594af7ec5531df2b4da875db53021a` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `2-nsys/p3-nsys-short2q.sqlite` | 8007680 | `4438cb44cb6349d45ea1c0959a89313e164432419c9c07c042a90ae1d17c2c22` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `3-leads/3-leads.txt` | 4790 | `c21939f7f0c6c4f1c9947b065128643b053405eb40c321fdf2b71fcbe3beffd6` | The path of the work folder of the GPU host is in the file (rule `dmk-server-path`). |
| `3-leads/p3-lead-kvwin-nokvu-nsys.nsys-rep` | 3570763 | `160e946d8f05f94c4e16363106ef55f3b1e3f319d3f0e079c4bb6c1ab7a7d71d` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `3-leads/p3-lead-kvwin-nokvu-nsys.sqlite` | 10878976 | `1c4dcc43c520d4c7d62cf129b853b9b50a2a1b2d694e3d98d43fca65db19367d` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `3-leads/p3-lead-kvwin-nsys.nsys-rep` | 3522473 | `156924c369f99146fd05f2d14b2bab0b6813ab59e86a734524fd7c99e7305537` | The UUID of the GPU is in the file (rule `dmk-uuid`). |
| `3-leads/p3-lead-kvwin-nsys.sqlite` | 10743808 | `dfdc174c78c055805c74103bed9f155ef6ab5145076e61da149986dd32bf5275` | The file comes from `nsys export` of a capture. It is not a raw file. |
| `4-ncu-smoke/p3-ncu-smoke-a.ncu-repz` | 1296354 | `8797f69f1d78298a61d53e5033fedba8a3f87297d834b5417991bb23122d24fc` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-smoke/p3-ncu-smoke-b.ncu-repz` | 1138959 | `e7641226f820a92e62a74c248b7182bb8ff82abbfa1e2206934aae7afcae5b89` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-long-p0.ncu-repz` | 353343 | `b7d3b4bc6809c99d3c0befb77f65144b199e33b1f461afe1b8e4d31818aef4d7` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-long-p1.ncu-repz` | 1030972 | `67f4d12c117e7e465841354f240906c7ac0690c0d49ea74b623d2329742c7939` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-long-p2.ncu-repz` | 1202002 | `8c13d48ceb4223ff338ad18c0d1d5d235d9c28bd3f0b70b7d23625b323c2c4ff` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-mv.ncu-repz` | 1211772 | `dba3004cb5c6319c09b126374e5540e139fddab30d1410f1835e494f55caf950` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-short-p0.ncu-repz` | 1063865 | `fdb950800f0abe1641d07c2993235327e6149456d93b2fdddc2bb2150ee8a0b9` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-core/p3-ncu-core-short-p1.ncu-repz` | 1066296 | `feff9634cc7fc76c5db29b14fd3a67d5a23aa66ddcd18b6d86b406ae6152fe73` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-extra/p3-ncu-extra-long-att.ncu-repz` | 605659 | `23a5470c9117eef38a2ad9ddda2010d5c563a9c601b9a318102fea67591c28af` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-extra/p3-ncu-extra-long-ew.ncu-repz` | 257302 | `a13490b006e152b4651f28b43190af762e1dcb1ff0ffe80af604be12b0773232` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-extra/p3-ncu-extra-long-full.ncu-repz` | 6656360 | `f5300179c56f8d18f7a4aec8211f4be03661c9c2c5e17f9ba7b74df5e055c354` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-extra/p3-ncu-extra-long-p4.ncu-repz` | 1040580 | `c05aec79e6111a3125904b6cd809b6ec2f9f76b6f24ca195ea965d567d0ac4b9` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |
| `4-ncu-extra/p3-ncu-extra-med-p0.ncu-repz` | 993046 | `0f3f36a0bcefb68cadc1ff6f353919cc818cbf16a002b212004500af75bf2fbe` | The IP address of a container is in the report (rule `dmk-private-ipv4`). |

## Limits of the data

- The tables of Nsight Systems and Nsight Compute come from the capture files. The repository does not have them. `make_tables.sh` makes these tables again only with `--captures`.
- The table `summary/ncu-run-alignment.tsv` shows five misaligned runs and one shifted run of Nsight Compute. The rows of these runs have the name of the kernel and its grid. They have no model floor. [INFERENCE] The cause is kernels that run at the start of the server, as the run `p3-ncu-smoke-a` shows.
- The 13 runs of Nsight Compute profile 6 to 17 launches each. The data set does not have the time of each of the 1,300 launches of a pass.
- Each run of Nsight Compute sends one request to a new server. A pass of a server that has run for some time can have other task slots and other kernel shapes.
- The census of the shapes gives the KV window of flash attention correctly in 33 of 72 passes of the three captures. The KV window of the captures is wider by 5.0 times at most in the other 39 passes. The lead #28495 uses this fact.
- The code in `harness/profile/ncu/` has the constant `COST` with 25 and 42 passes of the profiler for each launch. The runs show 19 and 45. The constant sets only the estimate of the run time of a schedule.
- The runs of `1-decomp` and `3-leads` use the cases of the private benchmark. The file `bench/cases.json` has the same names and token counts.
