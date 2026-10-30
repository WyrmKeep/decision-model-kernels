# Summary of the limits of the GPU

This file shows the numbers of the run on 2026-10-30. The script `summarize.py` made it from the files in `raw/`. The folder `tables/` has the same numbers as TSV files. The README of the data set gives the columns and the method.

## Tensor core rate

Each mode is one `mma.sync` instruction in a loop. The loop has no access to memory. Each mode runs for 2 s. The nominal rate is the highest rate that you can calculate from the architecture of the GPU. We did not measure it.

| Mode | TOPS | Rate (operations for each clock for each SM) | Nominal rate (operations for each clock for each SM) | Percent of the nominal rate | SM clock (MHz) | Power (W) |
|---|---|---|---|---|---|---|
| `s8k32` | 151.54 | 2043 | 2048 | 99.8 | 1952 | 158 |
| `s8k16` | 147.70 | 2040 | 2048 | 99.6 | 1905 | 199 |
| `f16f32` | 38.02 | 509 | 512 | 99.4 | 1965 | 134 |
| `f16f16` | 75.62 | 1018 | 1024 | 99.4 | 1956 | 194 |
| `bf16f32` | 37.87 | 511 | 512 | 99.8 | 1950 | 126 |

The unit TOPS is 10^12 operations each second. A multiply-add counts as 2 operations. The modes `f16f32`, `f16f16` and `bf16f32` use floating point values. For these modes the column has the unit TFLOPS.

### Run of 20 s

The mode `s8k32` runs for 20 s. The rate is 149.26 TOPS for the full time and 149.04 TOPS in the last 10 s. The mean SM clock is 1919 MHz. The power of the GPU is 198 W (mean) and 200 W (highest). The power cap is active in 100 percent of the samples. The highest temperature is 72 °C.

### Operands that are all zero

With operands that are all zero, the rate is 152.88 TOPS at 96 W. With random operands, the rate is 151.54 TOPS at 158 W. The power changes with the data.

### Number of warps

The table shows the rate of `s8k32` in TOPS. A row is the value of `ilp`. This is the number of mma instructions that a warp keeps in flight. These instructions do not use the results of each other. A column is the number of warps of each SM. The sign - shows a pair that the run did not use.

| `ilp` \ warps | 4 | 8 | 16 | 20 | 32 |
|---|---|---|---|---|---|
| 2 | 146.6 | 150.3 | 150.3 | - | 149.7 |
| 4 | 149.4 | 149.4 | 149.4 | - | 149.4 |
| 8 | 149.4 | 149.4 | 149.4 | 149.4 | - |

## Memory bandwidth

The unit GB/s is 10^9 bytes each second. A copy counts the bytes that it reads and the bytes that it writes. The nominal rate of the DRAM is the memory clock times 2 times the width of the memory bus.

| Test | Method | Buffer (MiB) | GB/s | Nominal rate (GB/s) | Percent of the nominal rate | Memory clock (MHz) |
|---|---|---|---|---|---|---|
| `dram_read` | `cs` | 1024 | 422.5 | 435.3 | 97.1 | 6801 |
| `dram_read` | `cg` | 1024 | 422.1 | 435.3 | 97.0 | 6801 |
| `dram_read` | `nc` | 1024 | 422.1 | 435.3 | 97.0 | 6801 |
| `dram_read` | `plain` | 1024 | 422.1 | 435.3 | 97.0 | 6801 |
| `dram_write` | `cs` | 1024 | 416.3 | 435.3 | 95.6 | 6801 |
| `dram_write` | `wb` | 1024 | 416.1 | 435.3 | 95.6 | 6801 |
| `dram_copy` | `ldcs_stcs` | 768 | 407.9 | 435.3 | 93.7 | 6801 |
| `dram_copy` | `ldplain_stwb` | 768 | 407.6 | 435.3 | 93.7 | 6801 |
| `dram_copy` | `memcpy_d2d` | 768 | 411.2 | 435.3 | 94.5 | 6801 |

### L2 cache

The program reads a buffer many times with `ld.global.cg`. The L2 cache of this GPU holds 3 MiB. A buffer of 12 MiB or more has the rate of the DRAM.

| Buffer (MiB) | GB/s |
|---|---|
| 0.25 | 1215 |
| 0.50 | 1356 |
| 1.00 | 1229 |
| 2.00 | 1380 |
| 3.00 | 1374 |
| 3.50 | 1148 |
| 6.00 | 464 |
| 12.00 | 429 |
| 32.00 | 422 |
| 128.00 | 422 |

### Number of loads in flight

The table shows the DRAM read rate in GB/s. A row is the number of thread blocks of each SM. A column is the number of loads in flight for each thread.

| Blocks \ loads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 1 | 392 | 418 | 423 | 423 |
| 2 | 420 | 423 | 423 | 423 |
| 3 | 422 | 423 | 423 | 423 |
| 4 | 422 | 423 | 423 | 422 |
| 6 | 423 | 423 | 422 | 422 |

## cuBLASLt int8 GEMM

The test multiplies a matrix of int8 weights with a matrix of int8 inputs. The weights have the shapes of the model. `N` is the number of tokens.

Each weight has 1 byte, and the GEMM has no scales. Q4_0 has 0.5625 bytes for each weight, and Q8_0 has 1.0625 bytes. Thus the rates are a reference. They are not the rates of MMQ.

### Layouts and check

The program checks each layout on a problem of 64 by 48 by 128 against the result of the CPU. Only a layout that has the correct result can give a best time. The layout `col32` does not have the correct result: the check gives `MISMATCH_1023_of_3072`. Thus this file does not use its times. The README gives the cause.

| Layout | Check | Rows with a time | Rows that the memory limit removes | Used for the best time |
|---|---|---|---|---|
| `col` | `OK` | 57 | 1 | yes |
| `col32` | `MISMATCH_1023_of_3072` | 57 | 1 | no |
| `col32t` | `OK` | 57 | 1 | yes |

### Best time of each shape

Only the layouts that have the correct result are in this table. The column Percent of the highest rate is the rate of the shape divided by the `s8k32` rate of the first table (151.54 TOPS). The column Percent of the nominal rate uses the mean SM clock of the run of the shape. The column Read rate is 1 byte for each weight divided by the time.

| Label | `M` | `N` | `K` | Layout | Time (µs) | TOPS | Percent of the highest rate | SM clock (MHz) | Percent of the nominal rate | Read rate (GB/s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `attn_gate` | 4096 | 8 | 2560 | `col` | 28.5 | 5.9 | 3.9 | 1828 | 4.1 | 368 |
| `attn_gate` | 4096 | 16 | 2560 | `col` | 28.9 | 11.6 | 7.7 | 1934 | 7.7 | 363 |
| `attn_gate` | 4096 | 32 | 2560 | `col` | 29.8 | 22.6 | 14.9 | 1935 | 15.0 | 352 |
| `attn_gate` | 4096 | 64 | 2560 | `col` | 31.8 | 42.2 | 27.9 | 1935 | 28.0 | 330 |
| `attn_gate` | 4096 | 128 | 2560 | `col32t` | 39.6 | 67.8 | 44.8 | 1845 | 47.2 | 265 |
| `attn_gate` | 4096 | 512 | 2560 | `col` | 113.9 | 94.3 | 62.2 | 1720 | 70.4 | 92 |
| `attn_gate` | 4096 | 2048 | 2560 | `col` | 419.6 | 102.4 | 67.5 | 1785 | 73.7 | 25 |
| `attn_k+attn_v` | 1024 | 8 | 2560 | `col` | 13.7 | 3.1 | 2.0 | 1883 | 2.1 | 192 |
| `attn_k+attn_v` | 1024 | 16 | 2560 | `col` | 13.7 | 6.2 | 4.1 | 1935 | 4.1 | 192 |
| `attn_k+attn_v` | 1024 | 32 | 2560 | `col` | 13.7 | 12.3 | 8.1 | 1924 | 8.2 | 192 |
| `attn_k+attn_v` | 1024 | 64 | 2560 | `col` | 13.8 | 24.4 | 16.1 | 1935 | 16.2 | 190 |
| `attn_k+attn_v` | 1024 | 128 | 2560 | `col` | 14.1 | 47.6 | 31.4 | 1935 | 31.6 | 186 |
| `attn_k+attn_v` | 1024 | 512 | 2560 | `col32t` | 33.7 | 79.6 | 52.5 | 1815 | 56.4 | 78 |
| `attn_k+attn_v` | 1024 | 2048 | 2560 | `col` | 112.6 | 95.4 | 62.9 | 1815 | 67.5 | 23 |
| `attn_qkv+attn_q` | 8192 | 8 | 2560 | `col` | 55.1 | 6.1 | 4.0 | 1878 | 4.2 | 380 |
| `attn_qkv+attn_q` | 8192 | 16 | 2560 | `col` | 56.0 | 12.0 | 7.9 | 1920 | 8.0 | 375 |
| `attn_qkv+attn_q` | 8192 | 32 | 2560 | `col32t` | 57.2 | 23.5 | 15.5 | 1890 | 15.9 | 367 |
| `attn_qkv+attn_q` | 8192 | 64 | 2560 | `col` | 61.1 | 44.0 | 29.0 | 1839 | 30.7 | 344 |
| `attn_qkv+attn_q` | 8192 | 128 | 2560 | `col32t` | 68.2 | 78.7 | 51.9 | 1575 | 64.2 | 308 |
| `attn_qkv+attn_q` | 8192 | 512 | 2560 | `col` | 230.3 | 93.2 | 61.5 | 1650 | 72.6 | 91 |
| `attn_qkv+attn_q` | 8192 | 2048 | 2560 | `col` | 831.4 | 103.3 | 68.2 | 1680 | 79.0 | 25 |
| `ffn_down` | 2560 | 8 | 9216 | `col` | 61.9 | 6.1 | 4.0 | 1665 | 4.7 | 381 |
| `ffn_down` | 2560 | 16 | 9216 | `col` | 61.4 | 12.3 | 8.1 | 1965 | 8.0 | 384 |
| `ffn_down` | 2560 | 32 | 9216 | `col` | 62.2 | 24.3 | 16.0 | 1935 | 16.1 | 380 |
| `ffn_down` | 2560 | 64 | 9216 | `col` | 63.8 | 47.3 | 31.2 | 1895 | 32.1 | 370 |
| `ffn_down` | 2560 | 128 | 9216 | `col32t` | 100.1 | 60.3 | 39.8 | 1888 | 41.1 | 236 |
| `ffn_down` | 2560 | 512 | 9216 | `col32t` | 273.9 | 88.2 | 58.2 | 1730 | 65.5 | 86 |
| `ffn_down` | 2560 | 2048 | 9216 | `col` | 898.0 | 107.6 | 71.0 | 1785 | 77.5 | 26 |
| `ffn_gate+ffn_up` | 9216 | 8 | 2560 | `col` | 61.6 | 6.1 | 4.0 | 1790 | 4.4 | 383 |
| `ffn_gate+ffn_up` | 9216 | 16 | 2560 | `col` | 62.4 | 12.1 | 8.0 | 1935 | 8.0 | 378 |
| `ffn_gate+ffn_up` | 9216 | 32 | 2560 | `col32t` | 63.8 | 23.7 | 15.6 | 1740 | 17.5 | 370 |
| `ffn_gate+ffn_up` | 9216 | 64 | 2560 | `col32t` | 67.5 | 44.7 | 29.5 | 1815 | 31.7 | 350 |
| `ffn_gate+ffn_up` | 9216 | 128 | 2560 | `col32t` | 73.5 | 82.2 | 54.2 | 1695 | 62.3 | 321 |
| `ffn_gate+ffn_up` | 9216 | 512 | 2560 | `col` | 240.7 | 100.4 | 66.2 | 1644 | 78.5 | 98 |
| `ffn_gate+ffn_up` | 9216 | 2048 | 2560 | `col` | 943.2 | 102.5 | 67.6 | 1650 | 79.8 | 25 |
| `output_head` | 248320 | 8 | 2560 | `col` | 1553.3 | 6.5 | 4.3 | 1770 | 4.8 | 409 |
| `output_head` | 248320 | 16 | 2560 | `col` | 1579.8 | 12.9 | 8.5 | 1830 | 9.0 | 402 |
| `output_head` | 248320 | 32 | 2560 | `col32t` | 1597.0 | 25.5 | 16.8 | 1575 | 20.8 | 398 |
| `output_head` | 248320 | 64 | 2560 | `col32t` | 1683.9 | 48.3 | 31.9 | 1815 | 34.2 | 378 |
| `output_head` | 248320 | 128 | 2560 | `col` | 1929.3 | 84.3 | 55.7 | 1815 | 59.7 | 330 |
| `output_head` | 248320 | 512 | 2560 | `col` | 6217.5 | 104.7 | 69.1 | 1672 | 80.5 | 102 |
| `ref_gemm` | 4096 | 4096 | 4096 | `col` | 1226.2 | 112.1 | 74.0 | 1890 | 76.2 | 14 |
| `ref_gemm` | 8192 | 8192 | 8192 | `col` | 9133.2 | 120.4 | 79.4 | 1560 | 99.2 | 7 |
| `ssm_alpha+ssm_beta` | 32 | 8 | 2560 | `col` | 9.6 | 0.1 | 0.1 | 1900 | 0.1 | 9 |
| `ssm_alpha+ssm_beta` | 32 | 16 | 2560 | `col` | 9.8 | 0.3 | 0.2 | 1950 | 0.2 | 8 |
| `ssm_alpha+ssm_beta` | 32 | 32 | 2560 | `col` | 9.8 | 0.5 | 0.3 | 1950 | 0.4 | 8 |
| `ssm_alpha+ssm_beta` | 32 | 64 | 2560 | `col` | 9.9 | 1.1 | 0.7 | 1950 | 0.7 | 8 |
| `ssm_alpha+ssm_beta` | 32 | 128 | 2560 | `col` | 9.9 | 2.1 | 1.4 | 1950 | 1.4 | 8 |
| `ssm_alpha+ssm_beta` | 32 | 512 | 2560 | `col` | 11.9 | 7.1 | 4.7 | 1950 | 4.7 | 7 |
| `ssm_alpha+ssm_beta` | 32 | 2048 | 2560 | `col` | 17.3 | 19.4 | 12.8 | 1949 | 12.8 | 5 |
| `ssm_out+attn_output` | 2560 | 8 | 4096 | `col` | 28.9 | 5.8 | 3.8 | 1845 | 4.0 | 362 |
| `ssm_out+attn_output` | 2560 | 16 | 4096 | `col` | 29.2 | 11.5 | 7.6 | 1935 | 7.6 | 359 |
| `ssm_out+attn_output` | 2560 | 32 | 4096 | `col` | 29.9 | 22.5 | 14.8 | 1851 | 15.6 | 351 |
| `ssm_out+attn_output` | 2560 | 64 | 4096 | `col` | 31.2 | 43.1 | 28.4 | 1935 | 28.6 | 336 |
| `ssm_out+attn_output` | 2560 | 128 | 4096 | `col32t` | 50.3 | 53.4 | 35.2 | 1860 | 36.9 | 209 |
| `ssm_out+attn_output` | 2560 | 512 | 4096 | `col32t` | 129.1 | 83.2 | 54.9 | 1830 | 58.4 | 81 |
| `ssm_out+attn_output` | 2560 | 2048 | 4096 | `col32t` | 423.0 | 101.5 | 67.0 | 1680 | 77.7 | 25 |

The output head with `N` = 2048 has no row. This shape uses more than 1800 MiB of memory, thus the test does not run it.

### Sum for the model

For each `N`, the table gives the sum of one time for each group of weights. All weights of a group have the same shape. The time of a group is the best time of one GEMM, multiplied by the number of weights in the group. The sum has no attention, no recurrent layer and no other operation.

The weight-read floor is the number of GGUF weight bytes divided by the best DRAM read rate (422.5 GB/s). The column Ratio is the sum without the output head divided by the weight-read floor without the output head.

| `N` | Sum with the output head (ms) | Sum without the output head (ms) | Same, layout `col` only (ms) | Same, with `col32` (not correct) (ms) | Weight-read floor without the output head (ms) | Ratio | TOPS |
|---|---|---|---|---|---|---|---|
| 8 | 11.52 | 9.97 | 9.97 | 9.97 | 6.30 | 1.58 | 5.7 |
| 16 | 11.65 | 10.07 | 10.07 | 10.07 | 6.30 | 1.60 | 11.3 |
| 32 | 11.86 | 10.26 | 10.32 | 10.26 | 6.30 | 1.63 | 22.3 |
| 64 | 12.45 | 10.77 | 10.81 | 10.62 | 6.30 | 1.71 | 42.4 |
| 128 | 15.28 | 13.35 | 15.14 | 12.77 | 6.30 | 2.12 | 68.4 |
| 512 | 45.73 | 39.51 | 40.07 | 39.04 | 6.30 | 6.28 | 92.5 |
| 2048 | - | 141.94 | 142.13 | 140.84 | 6.30 | 22.55 | 103.0 |

The weight-read floor with the output head is 7.89 ms.

## Clocks and power of each phase

`nvidia-smi` made 2569 samples, 10 each second. The table uses only the samples with a GPU use of 50 percent or more. The column Power cap is the percent of these samples in which the power cap is active.

| Phase | Seconds | SM clock median (MHz) | SM clock highest (MHz) | Memory clock (MHz) | Power mean (W) | Power highest (W) | Temperature highest (°C) | P-states | Power cap (percent) |
|---|---|---|---|---|---|---|---|---|---|
| `mma_all` | 18 | 1950 | 1965 | 6801 | 160 | 200 | 64 | `P2:165` | 64 |
| `mma_sweep` | 30 | 1920 | 1965 | 6801 | 193 | 200 | 70 | `P2:275` | 100 |
| `mma_zero` | 6 | 1965 | 1965 | 6801 | 86 | 99 | 59 | `P2:45` | 0 |
| `mma_sustained` | 23 | 1920 | 1935 | 6801 | 193 | 200 | 72 | `P2:215` | 100 |
| `membw_all` | 38 | 1950 | 1965 | 6801 | 129 | 152 | 64 | `P2:359` | 0 |
| `membw_sweep` | 31 | 1965 | 1965 | 6801 | 126 | 148 | 63 | `P2:294` | 0 |
| `cublaslt` | 89 | 1860 | 1965 | 6801 | 176 | 200 | 73 | `P2:858` | 84 |

