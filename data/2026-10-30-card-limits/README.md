# Limits of the GPU: tensor core, memory and cuBLASLt

This data set has the raw files and the tables of three small programs. The programs ran on one `RTX 3060 Ti` on 2026-10-30. They measure the highest rates of the GPU. [`summary.md`](summary.md) shows the numbers.

The data set also has the sources of the programs, the scripts, and the list of weight shapes. You can build the programs again, run them again, and make all tables again.

## Question

The test answers three questions. The numbers give the limits for the kernel targets of phase 4.

- The highest rate of the tensor cores for int8 work and for FP16 work.
- The highest read rate of the DRAM and of the L2 cache.
- The time of a cuBLASLt int8 GEMM for the weight shapes of the model, with 8 to 2048 tokens.

At this time, the [notebook](../../docs/notebook/README.md) has no prediction entry and no results entry for this test.

## Dates

- First run: 2026-10-30.
- Last run: 2026-10-30.
- Time of the run: 22:52:01 UTC to 22:56:19 UTC. The benchmarks ran for 256 s.

## Project commit

The commit that adds this data set also adds the sources and the scripts in this folder. The commit before it is `0d93d5fa29c00428858baf4ee4d221fdee712444`.

The run did not use a script of this repository. The scripts of the run were in a different folder. The section [Changes to the files of the run](#changes-to-the-files-of-the-run) shows the differences.

## Build commits

None. The programs do not use llama.cpp. The run did not build a fork or a branch.

## Build flags

The command is `./build.sh`. It runs in a build container and it does not use a GPU. It makes the programs `mb-mma`, `mb-membw` and `mb-cublaslt-i8`. The flags of `nvcc` are:

```
nvcc -std=c++17 -O3 -gencode arch=compute_86,code=sm_86 -Xcompiler -Wall,-Wextra \
     -Xcompiler -static-libstdc++,-static-libgcc -Xptxas -v
```

The program `mb-cublaslt-i8` also has `-lcublasLt`. The compiler is `nvcc` 12.8.93 from CUDA 12.8.1. The host compiler is `g++` 13.3.0.

The build image of the run came from `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`. The digest of that image is `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66`. We did not publish the build image. It has `libcublasLt.so.12.8.4.1`. The image of the run has the same version.

`raw/env-public.txt` has the SHA-256 of the three programs of the run. We built the sources of this folder again, in the same build image, on 2026-10-30. The three new programs have the same SHA-256 as the programs of the run. We did not build in the image `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`.

## Image

The image is `ghcr.io/ggml-org/llama.cpp@sha256:f25bfa096802cdf35ceb7d124816387a1e3ed5cbfc869eb242246cd29ab29352`. Its tag is `server-cuda-b12004`. The image ID on the GPU host is `sha256:21e387d87dd526d2722a26ce65479b207b13e96bd72784ff701f3db201223c64`.

No server ran in the image. Each program ran in a new container of this image. The container had the programs as read-only files. The image has the cuBLASLt library of the run. The GPU host gives the NVML library to the container.

## Model

No model ran. The programs use random int8 numbers. They do not read the weights of the model.

The list of shapes `shapes/qwen35-4b-mix8.tsv` comes from the "mix8" GGUF file of H2O Lightning 4B. The script reads only the start of the file. The file has the architecture `qwen35`, 426 tensors and 3349945376 bytes. The weights of the feed-forward layers are Q4_0. The other weights are Q8_0. The data set has no model file and no path of one.

## Driver

The driver is 595.71.05. The CUDA runtime of the programs is 12.8. The driver API is 13.2. The VBIOS is 94.04.3A.40.A7.

## GPU

The GPU is an `NVIDIA GeForce RTX 3060 Ti` (GA104, `sm_86`). It has 38 SMs, 7835 MiB of VRAM, a memory bus of 256 bit and an L2 cache of 3.0 MiB. The highest SM clock is 2100 MHz. The highest memory clock is 7001 MHz.

## Power limit

The power limit is 200 W. This is the default value. The range is 100 W to 220 W. The run did not change it.

## Clocks

The value is `default`. The run did not set the clocks and did not lock them. The line `Applications Clocks Setting` is `Not Active` before and after the run.

With no load, the SM clock was 210 MHz before the run and 240 MHz after it. During the phases, the GPU used the P-state P2 with a memory clock of 6801 MHz. The table `tables/phase_clocks_power.tsv` shows the SM clock of each phase.

## Server flags

The value is `default`. No server ran for this test. The production server of the project used the same GPU. It held 4370 MiB of VRAM and did no work. Before the first program started, 3319 MiB of VRAM were free.

## Commands

Build the programs:

```
./build.sh
```

Run all phases. The command runs for about 6 minutes:

```
./run.sh
```

The run used these commands in the containers. They are the lines of `raw/phases.tsv`:

```
mb-mma --mode all --seconds 2 --series /out/mma_all.series.tsv
mb-mma --mode sweep --sweep-seconds 1 --series /out/mma_sweep.series.tsv
mb-mma --mode s8k32 --data zero --seconds 3 --series /out/mma_zero.series.tsv
mb-mma --mode s8k32 --seconds 20 --series /out/mma_sustained.series.tsv
mb-membw --mode all --seconds 1.5 --l2-seconds 1 --series /out/membw_all.series.tsv
mb-membw --mode sweep --series /out/membw_sweep.series.tsv
mb-cublaslt-i8 --shapes /shapes/qwen35-4b-mix8.tsv --seconds 0.25 --max-algos 16 --retime 3 --peak-tops 151.54 --series /out/cublaslt.series.tsv
```

Make the tables and `summary.md` from the files in `raw/`:

```
python3 summarize.py --raw raw --tables tables --summary summary.md
```

Make the list of shapes again. The script uses `numpy` and the `gguf-py` folder of llama.cpp b12004:

```
uv run --with numpy --with pyyaml --with tqdm python shapes/gen_shapes.py \
    --gguf-py <llama.cpp>/gguf-py --gguf <model>.gguf \
    --origin-label 'the GGUF header of the "mix8" file' -o shapes/qwen35-4b-mix8.tsv
```

## Arms

This test has no arms. Each phase is one program in one mode. The name of the phase is the start of the name of its raw files.

| Phase | Program and mode | Contents |
|---|---|---|
| `mma_all` | `mb-mma --mode all` | The rate of 5 mma modes. Each mode runs for 2 s. |
| `mma_sweep` | `mb-mma --mode sweep` | The rate of `s8k32` for 3 values of `ilp` and for 4 to 32 warps for each SM. Each pair runs for 1 s. |
| `mma_zero` | `mb-mma --data zero` | The rate of `s8k32` with operands that are all zero. It runs for 3 s. |
| `mma_sustained` | `mb-mma --seconds 20` | The rate of `s8k32` with random operands for 20 s. |
| `membw_all` | `mb-membw --mode all` | The DRAM read, write and copy rates, and the L2 read rate for 10 buffer dimensions. |
| `membw_sweep` | `mb-membw --mode sweep` | The DRAM read rate for 1 to 6 thread blocks for each SM and 1 to 8 loads in flight. |
| `cublaslt` | `mb-cublaslt-i8` | The time of a cuBLASLt int8 GEMM for 58 shapes and 3 layouts. |

## Raw files

The programs and `run.sh` wrote the files in `raw/`. A table of the columns of each file follows the list.

| File | Contents |
|---|---|
| `env-public.txt` | The facts of the run: the time, the kernel, the driver, the image, the SHA-256 of the programs, and `nvidia-smi -q` before and after the run. The section [Changes to the files of the run](#changes-to-the-files-of-the-run) shows which lines we changed. |
| `phases.tsv` | The start, the end and the exit code of each phase. It has no first row with names. |
| `smi.csv` | The samples of `nvidia-smi` for the full run, 10 each second. |
| `smi.fields` | The field names that `nvidia-smi` used for `smi.csv`. |
| `mma_all.tsv`, `mma_sweep.tsv`, `mma_zero.tsv`, `mma_sustained.tsv` | The result rows of `mb-mma`. |
| `membw_all.tsv`, `membw_sweep.tsv` | The result rows of `mb-membw`. |
| `cublaslt.tsv` | The result rows of `mb-cublaslt-i8`. It has 174 rows, 3 for each of the 58 shapes. |
| `<phase>.series.tsv` | The NVML samples of the phase, 50 each second. |
| `<phase>.log` | The text that the program wrote to its error output. It has the device query and one line for each step. |

The `.log` files have the PCI address of the GPU in the line that starts with `# GPU 0:`. Rule 7 of the [data README](../README.md) does not name a PCI address, thus we kept the files as the programs wrote them. Each `.log` file also starts with this text:

```
ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11
```

The sources do not have this text. All programs stopped with exit code 0. We do not know the cause of the lines.

### Table of the columns: `phases.tsv`

| Column | Unit | Meaning |
|---|---|---|
| 1 | None | The name of the phase. |
| 2 | s | The start of the phase, as a Unix time. |
| 3 | s | The end of the phase, as a Unix time. |
| 4 | None | The exit code of the program. The value 0 means that the program had no error. |
| 5 | None | The command in the container. |

### Table of the columns: `smi.csv`

The first row has the names of the fields. The names come from `nvidia-smi`. The time is the local time of the host. The difference from UTC is 0.

| Column | Unit | Meaning |
|---|---|---|
| `timestamp` | None | The time of the sample, in the shape `YYYY/MM/DD HH:MM:SS.mmm`. |
| `power.draw` | W | The power of the GPU. The driver gives a short mean. |
| `power.draw.instant` | W | The power of the GPU at the time of the sample. |
| `clocks.current.sm` | MHz | The SM clock. |
| `clocks.current.memory` | MHz | The memory clock. |
| `pstate` | None | The P-state, for example `P2`. |
| `temperature.gpu` | °C | The temperature of the GPU. |
| `utilization.gpu` | % | The percent of the last period in which a kernel ran. |
| `memory.used` | MiB | The VRAM that programs use. |
| `clocks_event_reasons.active` | None | A bit mask of the causes that lower the clock. Bit `0x4` is the power cap of the driver. |

### Table of the columns: `mma_*.tsv`

The four files have the same columns. Each row is one test of one mode.

| Column | Unit | Meaning |
|---|---|---|
| `bench` | None | The name of the program, `mma`. |
| `mode` | None | The mma instruction. `s8k32` is `m16n8k32` with int8 and int32. `s8k16` is `m16n8k16` with int8 and int32. `f16f32` is `m16n8k16` with FP16 and an FP32 accumulator. `f16f16` is `m16n8k16` with FP16 and an FP16 accumulator. `bf16f32` is `m16n8k16` with bf16 and an FP32 accumulator. |
| `data` | None | The operands: `rand` for random numbers, `zero` for zero. |
| `ilp` | None | The number of mma chains in each warp. A chain does not use the results of the other chains. |
| `blocks_per_sm` | None | The number of thread blocks for each SM. Each thread block has 4 warps. |
| `warps_per_sm` | None | The number of warps for each SM. |
| `launches` | None | The number of kernel launches in the test. |
| `seconds` | s | The sum of the times of the launches. |
| `mma_total` | None | The number of mma instructions of all warps in all launches. |
| `tops` | TOPS | The rate. A multiply-add counts as 2 operations. The unit is TFLOPS for the FP16 and bf16 modes. |
| `tops_last_half` | TOPS | The rate of the last 50 % of the launches. |
| `tops_launch_min` | TOPS | The lowest rate of one launch. |
| `tops_launch_max` | TOPS | The highest rate of one launch. |
| `ops_per_clk_sm` | operations for each clock for each SM | `tops` divided by the number of SMs and by `sm_mhz_mean`. |
| `nominal_ops_per_clk_sm` | operations for each clock for each SM | The nominal rate. The section [Limits of the data](#limits-of-the-data) gives its source. |
| `eff_vs_nominal_pct` | % | `ops_per_clk_sm` divided by `nominal_ops_per_clk_sm`. |

The columns about the GPU follow. All three programs write them.

| Column | Unit | Meaning |
|---|---|---|
| `sm_mhz_mean`, `sm_mhz_min`, `sm_mhz_max` | MHz | The mean, the lowest and the highest SM clock of the NVML samples in the test. |
| `mem_mhz_mean` | MHz | The mean memory clock. |
| `power_w_mean`, `power_w_max` | W | The mean and the highest NVML power. NVML gives a short mean. |
| `power_inst_w_mean`, `power_inst_w_max` | W | The mean and the highest power at the time of the sample. |
| `temp_c_max` | °C | The highest temperature. |
| `pstate` | None | The P-state at the end of the test. |
| `throttle_or` | None | The OR of the bit masks of the causes that lower the clock, for all samples. |
| `swcap_pct` | % | The percent of the samples in which the power cap of the driver is active. |
| `vram_used_mib_max` | MiB | The highest VRAM that programs use. |
| `hw_samples` | None | The number of NVML samples in the test. |

### Table of the columns: `membw_*.tsv`

The two files have the same columns, and then the columns about the GPU.

| Column | Unit | Meaning |
|---|---|---|
| `bench` | None | The name of the program, `membw`. |
| `test` | None | `dram_read`, `dram_write` and `dram_copy` use a buffer of 768 MiB or more. `l2_read` reads a small buffer. It does this many times. |
| `variant` | None | The instruction that reads or writes, or the copy method. For reads: `cs`, `cg`, `nc` and `plain`. For writes: `cs` and `wb`. For copies: `ldcs_stcs`, `ldplain_stwb` and `memcpy_d2d`. `l2_read` uses `cg`. |
| `buf_mib` | MiB | The dimension of the buffer. A copy has two buffers of this dimension. |
| `blocks_per_sm` | None | The number of thread blocks for each SM. |
| `unroll` | None | The number of loads in flight for each thread. |
| `passes_per_launch` | None | The number of passes that one launch makes through the buffer. |
| `launches` | None | The number of launches. |
| `seconds` | s | The sum of the times of the launches. |
| `gbps` | GB/s | The rate. The unit is 10^9 bytes each second. A copy counts the bytes that it reads and the bytes that it writes. |
| `gbps_median_launch` | GB/s | The median rate of one launch. |
| `gbps_max_launch` | GB/s | The highest rate of one launch. |
| `nominal_gbps` | GB/s | The nominal rate: the mean memory clock times 2 times the width of the memory bus in bytes. `NA` for `l2_read`. |
| `eff_vs_nominal_pct` | % | `gbps` divided by `nominal_gbps`. |
| `bytes_per_pass` | bytes | The bytes of one pass. A copy counts the bytes that it reads and the bytes that it writes. |

### Table of the columns: `cublaslt.tsv`

Each row is one shape in one layout. The columns about the GPU follow the last column.

| Column | Unit | Meaning |
|---|---|---|
| `bench` | None | The name of the program, `cublaslt_i8`. |
| `variant` | None | The layout: `col`, `col32` or `col32t`. The section [Known fault of the layout `col32`](#known-fault-of-the-layout-col32) tells more. |
| `label` | None | The weights of the shape. `ref_gemm` is a reference with equal `M`, `N` and `K`. It is not part of the model. |
| `count` | None | The number of weights with this shape that one pass of the model multiplies. The value is 0 for `ref_gemm`. |
| `wtype` | None | The quantization type of the weights in the model. |
| `M`, `N`, `K` | None | `M` is the number of rows of the weights. `N` is the number of tokens. `K` is the number of columns of the weights. One GEMM does `2*M*N*K` operations. |
| `status` | None | `OK`, `SKIP_MEM` (the shape uses more than 1800 MiB of memory), `SKIP_DIM` (the layout does not let the shape run), or `NO_ALGO`. |
| `algo` | None | The algorithm that cuBLASLt uses, as `id/tile/stages/splitk/red/swz/opt`. |
| `workspace_bytes` | bytes | The memory that the algorithm uses for temporary data. |
| `waves` | None | The `wavesCount` of cuBLASLt. The value 1 means that the kernel fills the GPU one time. |
| `a_copies` | None | The number of copies of the weights. The test uses the copies in turn, thus the weights come from the DRAM. |
| `graph` | None | The value 1 means that the test ran the GEMM in a CUDA graph. |
| `reps_per_graph` | None | The number of GEMM runs in one CUDA graph. |
| `candidates` | None | The number of algorithms that the test measured for a short time. |
| `quick_best_us` | µs | The best time of the short measurement. |
| `time_us` | µs | The time of one GEMM. It is the best of 3 long measurements. |
| `tops` | TOPS | `2*M*N*K` divided by `time_us`. |
| `a_read_gbps` | GB/s | The weights that the GEMM reads each second, with 1 byte for each weight. |
| `eff_vs_peak_pct` | % | `tops` divided by the `--peak-tops` value, 151.54 TOPS. |
| `eff_vs_nominal_pct` | % | `tops` divided by the nominal rate at the mean SM clock of the test. |
| `check` | None | The result of the check of the layout: `OK`, or `MISMATCH_<wrong>_of_<all>`. All rows of one layout have the same value. |
| `mem_mib` | MiB | The device memory that the test uses for the shape. |

### Table of the columns: `*.series.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `bench` | None | The name of the program. |
| `phase` | None | The work of the program at the sample: `warmup`, `measure`, `idle`, or the text of the shape and the step. |
| `t_s` | s | The time from the first sample. |
| `epoch_s` | s | The time of the sample, as a Unix time. |
| `sm_mhz`, `mem_mhz` | MHz | The SM clock and the memory clock. |
| `power_w`, `power_inst_w` | W | The NVML power, and the power at the time of the sample. |
| `temp_c` | °C | The temperature. |
| `pstate` | None | The P-state as a number. The value 2 means P2. |
| `throttle` | None | The bit mask of the causes that lower the clock. |
| `util_pct` | % | The percent of the time in which a kernel runs on the GPU. |
| `vram_used_mib` | MiB | The VRAM that programs use. |

## Tables

`summarize.py` makes the tables in `tables/`. The numbers in `summary.md` come from the same data.

A layout has a correct result if its `check` value is `OK` in all rows of `raw/cublaslt.tsv`. The layouts `col` and `col32t` have a correct result. The layout `col32` does not. The best time of a shape and all sums use only the layouts with a correct result.

| File | Contents |
|---|---|
| `tensor_rates.tsv` | The rate of each mma mode, and of the runs with zero operands and of 20 s. |
| `s8k32_sweep.tsv` | The rate of `s8k32` for each `ilp` and each number of warps. |
| `sustained_2s.tsv` | The clocks, the power and the temperature of the run of 20 s, in intervals of 2 s. |
| `dram_l2_bandwidth.tsv` | The DRAM read, write and copy rates, and the L2 read rates. |
| `dram_read_sweep.tsv` | The DRAM read rate for each number of thread blocks and loads. |
| `cublaslt_best_verified.tsv` | The best layout with a correct result, and its time, for each shape. |
| `cublaslt_model_sums.tsv` | The sum of the best times of all weights of the model, for each number of tokens. |
| `phase_clocks_power.tsv` | The clocks, the power and the temperature of each phase, from `smi.csv`. |

### Table of the columns: `tensor_rates.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `run` | None | The phase: `all`, `zero` or `sustained`. |
| `mode`, `data`, `ilp`, `warps_per_sm` | None | The values of the raw file of the same name. |
| `seconds` | s | The run time. |
| `tops` | TOPS | The rate. The unit is TFLOPS for the FP16 and bf16 modes. |
| `tops_last_half` | TOPS | The rate of the last 50 % of the launches. |
| `ops_per_clk_sm`, `nominal_ops_per_clk_sm` | operations for each clock for each SM | The rate that the test measures, and the nominal rate. |
| `pct_of_nominal` | % | The rate that the test measures, divided by the nominal rate. |
| `sm_mhz_mean` | MHz | The mean SM clock. |
| `power_w_mean`, `power_w_max` | W | The mean and the highest power. |
| `swcap_pct` | % | The percent of the samples in which the power cap is active. |

### Table of the columns: `s8k32_sweep.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `ilp` | None | The number of mma chains in each warp. |
| `warps_per_sm` | None | The number of warps for each SM. |
| `tops` | TOPS | The rate. |
| `ops_per_clk_sm` | operations for each clock for each SM | The rate for each clock for each SM. |
| `sm_mhz_mean` | MHz | The mean SM clock. |
| `power_w_mean` | W | The mean power. |

### Table of the columns: `sustained_2s.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `t_start_s` | s | The start of the interval, from the start of the run of 20 s. |
| `sm_mhz_mean`, `mem_mhz_mean` | MHz | The mean SM clock and memory clock in the interval. |
| `power_w_mean`, `power_inst_w_mean` | W | The mean NVML power, and the mean power at the time of the sample, in the interval. |
| `temp_c_max` | °C | The highest temperature in the interval. |

### Table of the columns: `dram_l2_bandwidth.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `test` | None | `dram_read`, `dram_write`, `dram_copy` or `l2_read`. |
| `variant` | None | The instruction that reads or writes, or the copy method. |
| `buf_mib` | MiB | The dimension of the buffer. |
| `gbps`, `gbps_median_launch` | GB/s | The rate, and the median rate of one launch. |
| `nominal_gbps` | GB/s | The nominal rate of the DRAM. `NA` for `l2_read`. |
| `pct_of_nominal` | % | `gbps` divided by `nominal_gbps`. |
| `mem_mhz_mean`, `sm_mhz_mean` | MHz | The mean memory clock and SM clock. |
| `power_w_mean` | W | The mean power. |

### Table of the columns: `dram_read_sweep.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `blocks_per_sm` | None | The number of thread blocks for each SM. |
| `unroll` | None | The number of loads in flight for each thread. |
| `gbps` | GB/s | The DRAM read rate. |

### Table of the columns: `cublaslt_best_verified.tsv`

| Column | Unit | Meaning |
|---|---|---|
| `label`, `M`, `N`, `K`, `count`, `wtype` | None | The values of `raw/cublaslt.tsv`. |
| `best_verified_variant` | None | The layout with a correct result and the lowest time. |
| `time_us` | µs | The time of that layout. |
| `tops` | TOPS | The rate of that layout. |
| `pct_of_mma_peak` | % | `tops` divided by the `s8k32` rate of `mma_all`, 151.54 TOPS. |
| `a_read_gbps` | GB/s | The weights that the GEMM reads each second, with 1 byte for each weight. |
| `sm_mhz_mean` | MHz | The mean SM clock in the test of that layout. |
| `pct_of_nominal_at_run_clock` | % | `tops` divided by the nominal rate at `sm_mhz_mean`. |
| `algo` | None | The algorithm of that layout. |
| `us_col`, `us_col32_not_verified`, `us_col32t` | µs | The time of each layout. `NA` means no time. This data set does not use the times of `col32`. |
| `best_incl_unverified_variant`, `best_incl_unverified_us` | None, µs | The best layout and its time if the table also lets `col32` be the best layout. Use these two columns only to see the effect of `col32`. |

### Table of the columns: `cublaslt_model_sums.tsv`

For each `n_tokens`, the table adds the best time of each group of weights of the model. A group is the set of weights with the same `label`, `M` and `K`. The time of a group is the best time of one GEMM, multiplied by `count`. The sum has no attention, no recurrent layer and no other operation. `NA` means that the table has no sum, because a shape has no row.

| Column | Unit | Meaning |
|---|---|---|
| `n_tokens` | None | The value of `N`. |
| `ms_with_head` | ms | The sum of all groups, with the output head. It is `NA` for 2048 tokens. |
| `ms_without_head` | ms | The sum of all groups, without the output head. |
| `ms_without_head_col_only` | ms | The same sum with the layout `col` only. |
| `ms_without_head_incl_not_verified` | ms | The same sum if the table also lets `col32` be the best layout. This data set does not use it. |
| `gguf_dram_floor_without_head_ms` | ms | The weight-read floor without the output head: the bytes of the weights divided by the DRAM read rate of 422.5 GB/s. Q4_0 has 18 bytes for 32 weights. Q8_0 has 34 bytes for 32 weights. |
| `gguf_dram_floor_with_head_ms` | ms | The weight-read floor with the output head. |
| `ratio_to_floor_without_head` | None | `ms_without_head` divided by `gguf_dram_floor_without_head_ms`. |
| `tops_without_head` | TOPS | The operations of the groups without the output head, divided by `ms_without_head`. |

### Table of the columns: `phase_clocks_power.tsv`

The table uses the samples of `smi.csv` with `utilization.gpu` of 50 % or more, from the start to the end of the phase.

| Column | Unit | Meaning |
|---|---|---|
| `phase` | None | The phase. |
| `seconds` | s | The time from the start to the end of the phase. |
| `exit_code` | None | The exit code of the program. |
| `busy_samples` | None | The number of samples that the table uses. |
| `sm_mhz_median`, `sm_mhz_p05`, `sm_mhz_max` | MHz | The median, the 5th percentile and the highest SM clock. |
| `mem_mhz_median` | MHz | The median memory clock. |
| `power_w_mean`, `power_w_max` | W | The mean and the highest `power.draw`. |
| `power_inst_w_mean`, `power_inst_w_max` | W | The mean and the highest `power.draw.instant`. |
| `temp_c_max` | °C | The highest temperature. |
| `pstates` | None | The P-states of the samples, with the number of samples of each. |
| `swcap_pct` | % | The percent of the samples in which the power cap is active. |
| `vram_used_mib_max` | MiB | The highest VRAM that programs use. |

## Files not added

| File | Cause |
|---|---|
| `raw/env.txt` | It has the host name, the UUID of the GPU, the PCI address of the GPU, the number of the server program, the path of the shapes file on the server, and the names of the other containers of the server. `raw/env-public.txt` is a copy without these items. |
| `raw/smi.err` | It is empty. |
| `summary.md` of the run | It used the layout `col32` for the best time. `summarize.py` makes a new `summary.md`. |
| `deploy.sh` | It sends the files to a server. The name of the server is private data. |

## Known fault of the layout `col32`

The program `mb-cublaslt-i8` checks each layout on a problem of 64 by 48 by 128. It compares the result with the result of the CPU. The column `check` has the results: `col` is `OK`, `col32t` is `OK`, and `col32` is `MISMATCH_1023_of_3072`. The problem has 3072 results. 1023 of them are incorrect.

The fault is in the program of this data set. It is not a fault of cuBLASLt. The function `geometry` in `src/cublaslt_i8.cu` sets `ldB = 32 * roundup(N, 8)` for the two layouts. Matrix B has `N` rows. The cuBLAS document gives a rule for the leading dimension `ld` of each layout:

- `CUBLASLT_ORDER_COL4_4R2_8C` (the layout `col32t`) has tiles of 32 columns and 8 rows. The minimum of `ld` is `32 * roundup(rows, 8)`.
- `CUBLASLT_ORDER_COL32_2R_4R4` (the layout `col32`) has tiles of 32 columns and 32 rows. The minimum of `ld` is `32 * roundup(rows, 32)`.

The check problem has `N` = 48. For `col32`, `ld` must be 2048 or more, but the program sets 1536. Thus the tiles of different groups of columns overlap. The check of `col32t` is correct, because 8 divides 48.

The rule `N % 8 == 0` that the program uses for `col32` is also too weak. The rule must be `N % 32 == 0`. We read the rules in the document of cuBLAS 13.4. We did not read the document of version 12.8. It is an [INFERENCE] that the rules are the same.

The result for the times of `col32`:

- For `N` = 8 and 16, `ld` of `col32` is too small. The times are not correct.
- For `N` = 32, 64, 128, 512 and 2048, `ld` is the same as the correct value. The times can be correct. The check did not use such a shape, thus no check shows that the times are correct. [INFERENCE]

Thus `summarize.py` does not use `col32`. The best layout and all sums use `col` and `col32t`. The columns `us_col32_not_verified`, `best_incl_unverified_*` and `ms_without_head_incl_not_verified` show the effect if the times of `col32` are correct. For 128 tokens, the sum without the output head is 13.35 ms with `col` and `col32t`, and 12.77 ms with `col32`.

The checks of `col` and `col32t` used one small problem only. The rows with a time have no check of their shapes.

The repair has two parts. We did not run it, because the run of the programs uses the GPU:

1. Set `ldB = 32 * roundup(N, 32)` for `col32`, and let only shapes with `N % 32 == 0` use `col32`.
2. Run the check on a problem with `N` = 64. The numbers 8 and 32 divide 64.

A new run of only `col32` uses the GPU for about 1 minute. [INFERENCE: the `cublaslt` phase ran for 88 s for 3 layouts.]

## L2 cache

The L2 cache of this GPU is 3.0 MiB. The device query in each `raw/*.log` file shows this value. The rates of the test agree with it:

| Buffer (MiB) | 3 | 3.5 | 6 | 12 |
|---|---|---|---|---|
| GB/s | 1374 | 1148 | 464 | 429 |

The rate is above 1100 GB/s until 3.5 MiB. For 12 MiB or more, it is the DRAM rate of about 422 GB/s. A data sheet can give a different value. We did not check one.

## Changes to the files of the run

We made these changes to the files of the run before we added them.

- `raw/env-public.txt` is a copy of `env.txt` with changes. The lines with a change end with `[cleaned]`. We replaced the host name and the kernel string with `Linux 6.12.54`. We removed the UUID, the PCI address of the GPU, and the number of the server program. We replaced the list of containers with one fact: the production server ran and did no work. We replaced the path of the shapes file with its name, and we added the digest of the image.
- `shapes/qwen35-4b-mix8.tsv` has a different comment line 2. The line of the run had the path of the model on the server. The data rows are the same. The SHA-256 of the data rows (the lines that do not start with `#`) is `613aceb3629ea05323c170b986808cb94dcacf2dd5e7a11e448f094929bad2ea`. The SHA-256 of the file that the run used is `532d8c8c545152ed60c9ba7d738331281be9feece1c308583a8250a070d47d36`. The SHA-256 of the file in this folder is `90518e0eaf4defb1ea099cff87fdb9da1c75f1ebbdde36d459d4c485f2283e9a`.
- `run.sh` has four differences from the script of the run. A container name starts with `dmk-mb-`. The default GPU is `0`, not the UUID of the GPU. The environment variable `MB_BLOCKING_NAMES` is optional, and it replaces the check for containers of other work. The file `env.txt` has no host name, no PCI address, no number of a program and no list of containers. The commands of the programs did not change.
- `build.sh` has the environment variables `MB_BUILDER_IMAGE` and `MB_CONTAINER_CLI`. It installs `g++` in the container if the image has none. We did not run that step.
- `shapes/gen_shapes.py` finds `gguf-py` with `--gguf-py` or with `GGUF_PY`. It has the new flag `--origin-label`. It writes no path of a server.
- `summarize.py` is new. The script of the run also used `col32` for the best time. The tables and `summary.md` in this folder come from the new script.
- `src/` has the files that made the programs of the run. We did not change them.

## Limits of the data

- We made one run. The data does not show the difference between runs.
- We calculate the nominal rates. We do not measure them. For each clock for each SM, the rates are 2048 int8 operations and 1024 FP16 operations with an FP16 accumulator. The rate is 512 operations for FP16 or bf16 with an FP32 accumulator. [INFERENCE] They are the dense rates of the GA10x GeForce architecture. We did not compare them with a data sheet.
- The nominal rate of the DRAM is the mean memory clock times 2 times 32 bytes. The memory clock was 6801 MHz, thus the nominal rate is 435.3 GB/s.
- The power cap limits the clock. The rate of `s8k32` was 151.54 TOPS at 1952 MHz for 2 s, and 149.26 TOPS at 1919 MHz for 20 s. Random operands used 158 W, and zero operands used 96 W. The rate changes with the data and with the time.
- NVML gives a short mean of the power. It does not give the power at one time. The column `power_inst_w_mean` is the power at the time of the sample.
- The cuBLASLt test is a GEMM with int8 only. It does not change 4-bit weights to int8 values and it has no scales. Q4_0 has 0.5625 bytes for each weight and Q8_0 has 1.0625 bytes, but the test reads 1 byte. The times are a reference. They are not the times of MMQ.
- Some tests of cuBLASLt ran at a lower SM clock. 46 of the 171 rows with a time have a mean SM clock below 1800 MHz. 16 rows have a mean clock below 1700 MHz. The column `eff_vs_nominal_pct` removes the effect of the clock.
- The output head with 2048 tokens uses more than 1800 MiB of memory. The test did not run it, thus the sum with the output head has no value for 2048 tokens.
- The sums add the GEMM times only. The model also has attention, recurrent layers and other operations.
