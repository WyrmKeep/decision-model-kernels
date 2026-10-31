# 0001: GLU fused into the quantization of the activation in MMQ

## Summary

This change is pull request 30489 of upstream llama.cpp. The project did not write it. The pull request goes on tag `b12004` with no change.

In `b12004`, the feed-forward layer of "mix8" runs one kernel for SiLU of `gate` times `up`. That kernel writes an F32 tensor. The quantization kernel of MMQ then reads the tensor and writes Q8_1 data. The change makes the quantization kernel read `gate` and `up`, calculate SiLU of `gate` times `up` in registers, and write the Q8_1 data. The program does not write the F32 tensor and does not read it again.

This is the first change of the project. Each subsequent change starts from it.

## Equivalence class

`exact`. We selected this equivalence class on 2026-10-30, before the first measurement. [INFERENCE: the date is the time of the file that has the selection. The file is not in this repository.]

The fused kernel does the same operations as the two kernels that it replaces. It does them in the same sequence and with the same precision. Before the change, the program wrote an F32 value and read it again. This does not change the value.

The [method](../method.md) lists a fusion of kernels as an example of the equivalence class `re-associated`. This fusion does not change the sequence of the sums. If the answers are different, the tests of `re-associated` apply.

## Results

Build D is the baseline: tag `b12004`, built from source. Build L is build D with this change. The test has 4 blocks. Each block has one run of each build. The difference is the mean of the 4 block ratios.

The interval is the 95 % bootstrap interval of the blocks. The power limit was 200 W. We did not lock the clocks.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| All 48 cases (`lat_ms`) | 241.146 | 236.476 | -1.94 | -2.38 to -1.50 | 4 | 200 | 1,845 to 1,860 |
| Short requests | 73.361 | 72.459 | -1.23 | -2.14 to -0.34 | 4 | 200 | 1,845 to 1,860 |
| Medium requests | 362.909 | 355.704 | -1.99 | -2.24 to -1.73 | 4 | 200 | 1,845 to 1,860 |
| Long requests | 791.394 | 773.625 | -2.25 | -2.45 to -2.04 | 4 | 200 | 1,845 to 1,860 |

Source: table `speed-bench.tsv` in the [data set of the speed test](../../data/2026-10-31-lead-30489/README.md). The clock is the median SM clock of each run. The table `runs.tsv` has the clock of each run.

The next table has the numbers of the benchmark for 8 requests at the same time, and the prompt speed of `llama bench`. A larger number is better.

| Measurement | Before | After | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| Requests/s, 8 at the same time | 4.195 | 4.279 | +2.01 | +1.66 to +2.35 | 4 | 200 | 1,845 to 1,860 |
| Tokens/s, prompt of 64 tokens | 2,814.826 | 2,842.722 | +1.00 | +0.13 to +1.86 | 4 | 200 | 1,845 to 1,935 |
| Tokens/s, prompt of 256 tokens | 4,220.310 | 4,310.613 | +2.14 | +1.68 to +2.60 | 4 | 200 | 1,845 to 1,935 |
| Tokens/s, prompt of 3,000 tokens | 4,596.702 | 4,702.955 | +2.31 | +1.98 to +2.72 | 4 | 200 | 1,845 to 1,935 |

Source: tables `speed-bench.tsv` and `speed-llama-bench.tsv` in the [data set of the speed test](../../data/2026-10-31-lead-30489/README.md).

The [notebook entry](../notebook/2026-10-31-lead-30489.md) gives the table of each run, the ratio of each block, and the host load.

## Prediction

We wrote the prediction on 2026-10-30, before the measurement, in a file that is not in this repository. The notebook has no prediction entry for this change. The [notebook entry](../notebook/2026-10-31-lead-30489.md) gives the text of the prediction. It also gives the ceiling and its source.

| Item | Prediction | Measurement | Same as the prediction |
|---|---|---|---|
| Equivalence class | `exact` | The answers are the same bit for bit on 100 decisions. The SASS of the matrix multiply is the same. | Yes, for the inputs of the test |
| Speed increase, short request | 2.0 % or less | 1.23 % (interval 0.34 % to 2.14 %) | Yes. It is 61 % of the ceiling. |
| Speed increase, medium request | 2.7 % or less | 1.99 % (interval 1.73 % to 2.24 %) | Yes. It is 74 % of the ceiling. |
| Speed increase, long request | 2.9 % or less | 2.25 % (interval 2.04 % to 2.45 %) | Yes. It is 77 % of the ceiling. |
| Largest change | Long request, medium request, and 3,000 tokens. The short request and the short prompt change less. | Long request 2.25 %, medium request 1.99 %, short request 1.23 %. Prompt speed: +2.31 % at 3,000 tokens, +2.14 % at 256 tokens, +1.00 % at 64 tokens. | Not fully. The sequence of the requests is correct. The speed at 256 tokens changes almost as much as at 3,000 tokens. |
| Range of the author | +1.0 % to +1.8 % for the prompt speed, on a different GPU and type | +1.00 % to +2.31 % for the three prompt lengths | Not fully. The values at 256 and 3,000 tokens are more than the range. |

The ceiling is the time of the SwiGLU kernel that the change removes, less the time to read the second tensor in the quantization kernel. The prediction thinks that the quantization kernel waits for memory. It thinks that the kernel uses about 50 % of its time to read the second tensor. The unit of the ceiling is the kernel time.

The kernel time is about 90 % to 94 % of the time of a request. The ceiling in the time of a request is then about 1.8 %, 2.5 %, and 2.7 %. [INFERENCE: we calculated this from the `nsys` data of the notebook entry. The data are not in this repository.]

The speed increases are less than the ceilings. The difference is 0.7 % to 0.8 %. The measurement did not look at the kernels. We do not know the cause of the difference.

## Correctness

For the equivalence class `exact`, the answers must be the same bit for bit as the answers of build D. We did the tests of the answers and the SASS compare with the speed test. We did the other tests after it. The thresholds of the other tests come from the source of `test-backend-ops` and from build D. The [correctness baseline of phase 2](../notebook/2026-10-30-phase2-correctness-baseline.md) gives the 6 tests of build D with `FAIL`.

| Test | Threshold | Result |
|---|---|---|
| Answers of build D, 4 runs against each other | Bit for bit the same | pass |
| Answers of build L, 4 runs against each other | Bit for bit the same | pass |
| Answers of build D against build L, 100 decisions | Bit for bit the same, and no change of a top answer | pass: the largest probability difference is 0.0 |
| Guards of the benchmark, and quality numbers | All guards true, 0 errors, and the same numbers as build D | pass: 94 correct, 97 agree, `mean_dp` 0.013322 |
| SASS of `sm_86` | No change in a kernel that the model uses, and no change in `REG`, `STACK`, `SHARED`, `LOCAL` or `CONSTANT` | pass for 6,744 of 6,748 names. The 4 other names are NVFP4 kernels that the model does not use. We did not compare 5 pairs of kernels with a new name. |
| `test-backend-ops`, 310 new cases of the pull request, CUDA against CPU | NMSE of 5e-4 or less, set by the test source | pass: 310 of 310 |
| `test-backend-ops`, 187 operations of the model graph | The same result for each test as build D | pass: 0 tests with a different result. The two builds have the same 6 tests with `FAIL`, as in the phase 2 baseline. |
| `compute-sanitizer`, `memcheck`, `synccheck` and `racecheck` on the 310 new cases | 0 errors | pass: 0 errors, 0 problems of shared memory |
| `compute-sanitizer`, `initcheck` on the 310 new cases | 0 errors in the new kernels, and no new errors in the kernels that did not change | pass for the new kernels (0 errors). The 942,080 errors are all in `mul_mat_q` kernels. Build D has the same type of errors, see the text below. |
| `compute-sanitizer`, four tools on the model, one chunk of 256 tokens | 0 errors, as build D | pass: 0 errors with each tool, the same perplexity 17.5145 as build D |

Source: the [data set of the speed test](../../data/2026-10-31-lead-30489/README.md) for the answers, the guards, and the SASS. The [data set of the correctness tests](../../data/2026-10-31-0001-correctness/README.md) for the other tests.

The first `initcheck` run found 942,080 errors, and the log shows the first 100. All 100 are in `mul_mat_q_stream_k_fixup`. Two more runs show where the errors are:

- With only the kernels `quantize_mmq_q8_1` checked, `initcheck` finds 0 errors. This is the new kernel.
- With the kernels `mul_mat_q` not checked, `initcheck` finds 0 errors in all other kernels of the new cases.

The kernels `mul_mat_q` and `mul_mat_q_stream_k_fixup` have the same SASS in build D and build L. The two builds give 2,944 `initcheck` errors on the 121 Q4_0 tests of `MUL_MAT`. Build D has no test with the cases of the pull request, thus we have no baseline for the number 942,080. On the model, `initcheck` finds 0 errors in build L, as in build D.

## Reproduce

1. Get the code of build L from the fork. The commit is a `git cherry-pick -x` of the one commit `730f0425773c0ece8fbdd5e73b80b749e6bf81b4` of pull request 30489. The commit before it is the commit of tag `b12004`.

```
git clone https://github.com/WyrmKeep/llama.cpp.git
cd llama.cpp
git checkout 6b8b65e6672f07805000d3c9683b65c72cc62303
```

2. Make build D from the commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b` with the same procedure. Build D is the baseline.
3. Build the two commits with `harness/build/build.sh`. It uses the builder image of the project. The `cmake` flags are:

```
-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON
-DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
-DLLAMA_USE_PREBUILT_UI=OFF -DGGML_CCACHE=OFF
```

4. Make the two schedules. The seeds are 30489 and 30490.

```
python3 harness/make_schedule.py --prefix lead30489-bench --blocks 4 --seed 30489 \
    --arm D=build:b12004-sm86 --arm L=build:lead-30489 --kind bench \
    --args "--write-reference {name}"
python3 harness/make_schedule.py --prefix lead30489-llb --blocks 4 --seed 30490 \
    --arm D=build:b12004-sm86 --arm L=build:lead-30489 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

5. Run the schedules and make the tables. The [data set of the speed test](../../data/2026-10-31-lead-30489/README.md) gives the commands.

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
python3 data/2026-10-31-lead-30489/make_tables.py
```

6. Run the tests of correctness. The [data set of the correctness tests](../../data/2026-10-31-0001-correctness/README.md) gives the commands of `test-backend-ops` and `compute-sanitizer`.

The server flags of the benchmark are `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`.

## Limits

- **Where it applies.** The function `ggml_cuda_should_fuse_glu_mmq` has the conditions:
  - A GLU node has an MMQ matrix multiply as a user in 7 nodes or less.
  - The tensors `gate`, `up` and the result are F32. The test `ggml_is_contiguous` gives true for them.
  - The value `ne[0] % 4` is 0. The tensors `gate` and `up` align to 16 bytes.
  - The pass has more than 8 tokens. A pass of 8 tokens or less uses MMVQ, and the change does not apply.
  - The value `stream_context().concurrent_events` is empty.
- **Host load.** The load of the host was 1.61 to 2.56 in block 1 and 3.00 to 18.70 in blocks 2 to 4. It was 14.6 to 18.7 in all runs of `llama bench`. The ratio of `lat_ms` is -2.18 % in block 1 and -1.86 % as the mean of blocks 2 to 4. In `llama bench`, build L ran first in all 4 blocks. The test cannot show which effect is from the build and which effect is from the sequence.
- **Number of blocks.** The test has 4 blocks. The bootstrap uses only 4 values. [INFERENCE: the interval is too small.]
- **SASS.** The tool for the SASS lists 6 kernels `quantize_mmq_q8_1` of build D as removed. It lists 6 kernels of build L as added, with a different name. A check by a person showed the same SASS for one pair, but the data set does not hold that result. We did not compare the 5 other pairs. [INFERENCE: the pair for the layout `D4` is one of them, and the Q8_0 matrix multiplies use it.]
- **Answers.** The answers are the same on 100 decisions of one benchmark. The test does not use other inputs.
- **Tool `compute-sanitizer`.** The tool ran on the new test cases and on a run of `llama perplexity` with 256 tokens. It did not run on a request to the server. The `initcheck` errors in the `mul_mat_q` kernels have no baseline for the new test cases.
- **Not in the test.** The change has an effect on types and operations that the model does not use, for example MMQ with other types and `MUL_MAT_ID`. The 4 kernels `quantize_mmq_nvfp4` have a different line number and no other difference. We ran the tests of the pull request on this GPU and with this model only.
- **Prediction.** The notebook has no prediction entry written before the measurement. The rule of the method is to commit one before the measurement.
- **Go back to the baseline.** There are two methods:
  - Build the commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b`. This is build D.
  - On a branch with subsequent changes, use `git revert 6b8b65e6672f07805000d3c9683b65c72cc62303`. The subsequent changes start from this commit. The command can make a change of them necessary.

  No code in the program stops only this change. The value `GGML_CUDA_DISABLE_FUSION` stops all fusions of the CUDA backend. Its result is not build D.

## Data

- [Speed test, answers and SASS compare](../../data/2026-10-31-lead-30489/README.md)
- [Correctness tests](../../data/2026-10-31-0001-correctness/README.md)
- [Notebook entry of the result](../notebook/2026-10-31-lead-30489.md)
