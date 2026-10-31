# Attention mask on the GPU

## Summary

The change makes the attention mask on the GPU. Before the change, the CPU wrote the attention mask of each pass and sent it to the GPU. A large pass has an attention mask of 28 MB, and the copy uses time. The change sends one small input with the position and the sequences of each KV cell and of each token (16 bytes for each). A new operation, `KQ_MASK`, makes the attention mask from it. The `long` request is 1.52 % faster and the `medium` request is 1.12 % faster.

## Equivalence class

`exact`. We selected it on 2026-10-31 at 05:24 UTC, before the first run of the speed test. The kernel compares integers and writes the same two values as the CPU: 0 and `-inf`. The answers must be the same as the answers of the build before the change, bit for bit.

## Results

The test has 4 blocks, with one run of each arm in each block. Arm `D` is build `lead-30489`. Arm `L` is build `0004-gpu-mask`. The first row is the baseline, for all requests.

| Request type | Before (ms) | After (ms) | Difference (%) | 95 % interval (%) | Runs | Power limit (W) | Clock (MHz) |
|---|---:|---:|---:|---|---:|---:|---|
| All requests (`lat_ms`) | 237.49 | 234.98 | -1.06 | -1.56 to -0.54 | 4 + 4 | 200 | 1,845 to 1,860 |
| `short` (`short_ms`) | 73.17 | 73.22 | +0.08 | -0.72 to +0.88 | 4 + 4 | 200 | 1,845 to 1,860 |
| `medium` (`medium_ms`) | 357.14 | 353.13 | -1.12 | -1.45 to -0.79 | 4 + 4 | 200 | 1,845 to 1,860 |
| `long` (`long_ms`) | 775.90 | 764.14 | -1.52 | -1.79 to -1.25 | 4 + 4 | 200 | 1,845 to 1,860 |

The speed of 8 requests at the same time (`rps8`) is 1.09 % higher (+0.84 % to +1.33 %). The prompt speed of `llama bench` has no change for 64 tokens (-0.00 %) and 256 tokens (+0.27 %). For 3,000 tokens it is 1.73 % higher (+1.48 % to +1.93 %).

Source: the tables [speed-bench.tsv](../../data/2026-10-31-0004-gpu-mask/tables/speed-bench.tsv) and [speed-llama-bench.tsv](../../data/2026-10-31-0004-gpu-mask/tables/speed-llama-bench.tsv) in the [data set](../../data/2026-10-31-0004-gpu-mask/README.md). The host had a high load during the test.

## Prediction

We wrote the prediction on 2026-10-31 at 05:24 UTC: `lat_ms` -2.8 % (range -1.8 % to -3.8 %), `medium_ms` -2.8 %, `long_ms` -3.4 %, `short_ms` -1.3 %. The [prediction entry](../notebook/2026-10-31-0004-gpu-mask-prediction.md) has the full text.

The gain in the test is 37 % to 44 % of the predicted gain. The [results entry](../notebook/2026-10-31-0004-gpu-mask-result.md) gives the cause. The model of phase 4 gave the upload time of all inputs as the copy time of the attention mask. The attention mask has 28 % of that time for the `medium` request and the `long` request. The copy of the attention mask plus the CPU time of the attention mask gives the gain in the test in 0.5 ms or less. The part of the upload time that is not the attention mask stays in the new build.

## Correctness

| Test | Threshold declared before the test | Result |
|---|---|---|
| Answers of 8 runs, `answers_compare.py` | `self_repeat` true for the two arms, `cross_identical_all` true, `cross_max_abs_dp` 0.0 | Pass. All true, and 0.0. |
| `test-backend-ops -o KQ_MASK` | All cases the same on the GPU and on the CPU, error limit 0 | Pass. 24 of 24 cases. |
| Compare of the attention mask from the GPU with the attention mask from the CPU, on the requests of the benchmark | 0 different values in each pass | Pass. 1,285 passes, 0 different values. |
| `compute-sanitizer --tool memcheck`, one request of each type | 0 errors | Pass. The server log has no line of an error. The two runs of `llama perplexity` have 0 errors. The server log has no summary line. |

## Reproduce

1. Get the commit from the fork of llama.cpp of this project: branch `dmk/0004-gpu-mask`, commit `cf3913bedc87cb1e089e6cac562ca693620cb90e`. The commit before it is `6b8b65e6672f07805000d3c9683b65c72cc62303` (branch `dmk/lead-30489`, arm `D`).
2. Build the two commits: `harness/build/build.sh <source folder> <output folder>`. The flags are in the data set README: `-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=86-real`, with `nvcc` 12.8.93 and `gcc` 14.2.0.
3. Make the schedules: `python3 harness/make_schedule.py --prefix gpumask-bench --blocks 4 --seed 4001 --arm D=build:lead-30489 --arm L=build:0004-gpu-mask --kind bench --args "--write-reference {name}"`, and the same command with `--prefix gpumask-llb --seed 4002 --kind tool --args "bench -m /models/{model} -ngl 999 -fa auto -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"`.
4. Run the schedules with `harness/run_schedule.sh`. The server flags are `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots`.
5. Make the tables: `python3 data/2026-10-31-0004-gpu-mask/make_tables.py`.
6. Run the tests: `test-backend-ops test -o KQ_MASK`. Start the server with `LLAMA_KQ_MASK_CHECK=1` and `-lv 4`, and run the private benchmark with `--passes 1`. Run the server in `compute-sanitizer --tool memcheck`.

## Limits

- The test has 4 blocks, and the host had a high load.
- The device path is on if three conditions are true. The cells of the pass are in one group of sequences. The model has no `SWA` and no `ALiBi`. The pass has 32 sequences or less.
- The server in production has `--kv-unified`. The other cases use the CPU path as before. No test uses these cases with a model.
- The change has no test with more than one GPU and no test with the remote backend of `ggml`. It adds one operation number to `ggml` and a patch number to the remote interface.
- The server log of `memcheck` has no summary line of `compute-sanitizer`.
- Before the rule for the GPU, small tasks of this change ran on the GPU while measurement windows of other changes were open. The [results entry](../notebook/2026-10-31-0004-gpu-mask-result.md) gives the times.
- To go back to the baseline, set `LLAMA_KQ_MASK_DEVICE=0` for the server, or use build `D`.

## Data

- The data set [`2026-10-31-0004-gpu-mask`](../../data/2026-10-31-0004-gpu-mask/README.md)
