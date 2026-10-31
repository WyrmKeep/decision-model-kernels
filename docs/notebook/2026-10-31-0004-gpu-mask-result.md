# Change 0004: attention mask on the GPU (result)

Written 2026-10-31, after the measurement window of the speed test (07:11 to 07:29 UTC).

## Task of this entry

This entry gives the result of change 0004. The change makes the attention mask on the GPU. The [prediction entry](2026-10-31-0004-gpu-mask-prediction.md) has the prediction. We wrote it on 2026-10-31 at 05:24 UTC, in commit `7dd4b953b8a041f9941833775f5e61fd14b54b16`, before the first run of the speed test. We did not change it after that time. The [README of the data set](../../data/2026-10-31-0004-gpu-mask/README.md) gives the raw files, the tables, and the conditions.

## Result in short

- The answers of the build with the change (arm `L`) are the same as the answers of build `D`, bit for bit. The equivalence class `exact` is true.
- The three other tests of the equivalence class pass. The program compared 1,285 attention masks of the GPU with the attention masks of the CPU. It found 0 different values.
- Arm `L` is faster than arm `D` in the private benchmark. `lat_ms` is 1.06 % lower (95 % interval: -1.56 % to -0.54 %). `long_ms` is 1.52 % lower, and `medium_ms` is 1.12 % lower. `short_ms` has no change.
- The gain in the test is 37 % to 44 % of the predicted gain. It is less than 50 % of the prediction. The point estimates are not in the predicted ranges.
- The cause is in the model of phase 4. The model used the upload time of the profile as the copy time of the attention mask. The attention mask has 28 % of that time. The copy of the attention mask plus the CPU time of the attention mask gives the gain in the test, in 0.5 ms or less.
- In `llama bench`, the prompt speed has no change for the prompts of 64 and 256 tokens. For the prompt of 3,000 tokens, it is 1.73 % higher. The prediction was 0.6 %.

## Correctness tests

The tests are the four tests that the prediction entry declared. The first test gives the equivalence class.

| Test | Result that is necessary | Result in the test | As necessary |
|---|---|---|---|
| Answers | `self_repeat_a`, `self_repeat_b` and `cross_identical_all` are true. `cross_max_abs_dp` is 0.0. No decision has a different top answer. | All three are true. `cross_max_abs_dp` is 0.0. The list of the top answers that changed is empty. The 8 runs of the speed test have 0 errors, and all guards are true. | Yes |
| `test-backend-ops` | All cases of `KQ_MASK` are the same on the GPU and on the CPU. The error limit is 0. | All 24 cases are the same (F16 and F32, 12 cases each). One case has 270,000 tokens. | Yes |
| Compare of the attention masks | 0 different values in each pass | 733 passes with `--kv-unified` and 552 passes without it. All have 0 different values. The largest pass has 6,912 KV cells and 2,048 tokens. | Yes |
| `compute-sanitizer` | 0 errors for one request of each type | The server in `memcheck` gave answers to 4 requests: `short` with 1 question, `short` with 2 questions, `medium` and `long`. Its log has no line of an error. `llama perplexity` in `memcheck` has 0 errors for a pass of 256 tokens and for a pass of 2,048 tokens. | Yes |

The data set has the tables [answers-compare-bench.json](../../data/2026-10-31-0004-gpu-mask/tables/answers-compare-bench.json) and [correctness.tsv](../../data/2026-10-31-0004-gpu-mask/tables/correctness.tsv).

- The server log of the run in `memcheck` has no line `ERROR SUMMARY`. The tool writes that line when the program ends, but the server did not write it. We cannot show that the tool counted 0 errors. We can show that the tool wrote no text about an error during the 4 requests. The two runs of `llama perplexity` have the line with 0 errors.
- The run `0004-check-unified` gave the same answers as the runs of build `lead-30489` in the data set [`2026-10-31-lead-30489`](../../data/2026-10-31-lead-30489/README.md). We compared them with `answers_compare.py` before the speed test. This compare is not in the data set.
- The server logs of the two correctness runs have the line `KQ mask on the device enabled`. The server logs of the speed test do not have it, because their log level is 3. They use the same flags. [INFERENCE: the change is active in the speed test. The time of the requests agrees with this.]

The program `test-backend-ops` also gives the time of the kernel. The kernel makes the attention mask for 6,912 KV cells and 2,048 tokens (28 MB) in 72 µs, at 367 GB/s. The prediction was 85 µs. For 1,024 KV cells and 2,048 tokens it uses 14 µs. For 1 token it uses 2 µs. The file `raw/0004-tbo-kq-mask-perf.out` has all times.

## Speed test with the private benchmark

The test has 4 blocks. Each block has one run of arm `D` and one run of arm `L`. The table gives the difference of arm `L` from arm `D`.

| Metric | Arm `D` | Arm `L` | Difference (%) | 95 % interval (%) | Predicted (%) | Range of the prediction (%) | In range |
|---|---:|---:|---:|---|---:|---|---|
| `lat_ms` | 237.49 | 234.98 | -1.06 | -1.56 to -0.54 | -2.8 | -1.8 to -3.8 | No |
| `short_ms` | 73.17 | 73.22 | +0.08 | -0.72 to +0.88 | -1.3 | -0.3 to -2.3 | No |
| `medium_ms` | 357.14 | 353.13 | -1.12 | -1.45 to -0.79 | -2.8 | -1.8 to -3.8 | No |
| `long_ms` | 775.90 | 764.14 | -1.52 | -1.79 to -1.25 | -3.4 | -2.4 to -4.4 | No |
| `rps8` | 4.265 | 4.312 | +1.09 | +0.84 to +1.33 | None | None | No prediction |

Source: [speed-bench.tsv](../../data/2026-10-31-0004-gpu-mask/tables/speed-bench.tsv).

The gain in ms is the value of arm `D` less the value of arm `L`.

| Metric | Predicted gain (ms) | Gain in the test (ms) | Part of the predicted gain |
|---|---:|---:|---:|
| `short_ms` | 0.95 | -0.06 | None |
| `medium_ms` | 10.0 | 4.02 | 40 % |
| `long_ms` | 26.5 | 11.76 | 44 % |
| `lat_ms` | 6.7 | 2.51 | 37 % |

The sign of the change is as predicted for `lat_ms`, `medium_ms`, `long_ms` and `rps8`. The 95 % interval of each of them does not include 0. The 4 blocks have the same sign for these metrics. For `short_ms`, the interval includes 0. The [table of the blocks](../../data/2026-10-31-0004-gpu-mask/tables/speed-bench-blocks.tsv) shows that `short_ms` is lower in 2 blocks and higher in 2 blocks.

## Speed test with `llama bench`

| Prompt (tokens) | Arm `D` (tokens/s) | Arm `L` (tokens/s) | Difference (%) | 95 % interval (%) | Predicted (%) | Range of the prediction (%) | In range |
|---:|---:|---:|---:|---|---:|---|---|
| 64 | 2,843.9 | 2,843.8 | -0.00 | -0.13 to +0.14 | 0.0 | -0.5 to +0.5 | Yes |
| 256 | 4,302.9 | 4,314.4 | +0.27 | +0.12 to +0.49 | 0.0 | -0.5 to +0.5 | Yes |
| 3,000 | 4,700.4 | 4,781.5 | +1.73 | +1.48 to +1.93 | +0.6 | +0.2 to +1.0 | No |

Source: [speed-llama-bench.tsv](../../data/2026-10-31-0004-gpu-mask/tables/speed-llama-bench.tsv).

For the prompt of 3,000 tokens, the gain is about 11 ms for a prompt time of 638 ms. The attention mask of this prompt has about 15 MB. [INFERENCE: the 2 passes have 2,048 tokens with about 2,300 KV cells, and 952 tokens with about 3,100 KV cells. A value has 2 bytes.] The copy of 15 MB at 6.7 GB/s uses 2.3 ms.

We do not know the CPU time of the attention mask in this test. The sum of the copy and the CPU time can be less than the gain in the test. We did not find the cause of the remaining difference. The test does not show it.

## Prediction against result

The prediction used the gain of the phase-4 model. The model gives two times for each request: the upload time and the CPU time of the attention mask. The prediction used the sum as the gain. The gain in the test is 37 % to 44 % of that sum for the requests with a large attention mask.

The correctness run `0004-check-unified` shows the dimensions of each attention mask. The server log has one line for each pass, with the number of KV cells and of tokens. We added the bytes for the 48 requests of the quality phase. The requests ran one at a time. The next table compares the model with the gain in the test.

| Request type | Passes | Bytes of the attention mask (MB) | Copy at 6.7 GB/s (ms) | CPU time in phase 4 (ms) | Upload time in phase 4 (ms) | Gain by the model (ms) | Gain in the test (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `short1q` | 1 | 0.07 | 0.01 | 0.05 | 0.71 | 0.03 | -0.06 (`short_ms`) |
| `short2q` | 3 | 0.12 | 0.02 | 0.10 | 1.08 | 0.03 | -0.06 (`short_ms`) |
| `medium` | 4 | 7.17 | 1.07 | 2.90 | 7.54 | 3.67 | 4.02 |
| `long` | 5 | 39.76 | 5.93 | 5.80 | 21.44 | 11.33 | 11.76 |

The gain by the model is the copy time plus the CPU time, less the new work. The new work is 0.03 ms for each small request, 0.3 ms for the `medium` request, and 0.4 ms for the `long` request. The prediction entry gives these values. The kernel time from `test-backend-ops` agrees with them.

The gain by the model is in 0.5 ms of the gain in the test for the `medium` request and for the `long` request. The upload time of phase 4 is 7 times the copy time of the attention mask for the `medium` request. It is 3.6 times for the `long` request. The upload time has other parts. Phase 4 gave the sum of all parts as the gain. The change does not remove the other parts.

[INFERENCE: the other parts are the other inputs and the time that the CPU waits for the GPU at each input copy. The server makes one copy and one wait for each input in each pass. The new input `info` has the same number of copies as the attention mask had before the change. We did not measure the parts of the upload time.]

The cause of the difference between the prediction and the result is thus in the model, not in the code. The prediction entry had this risk in the section "Confidence and risks": "The copy of the attention mask can be less than the model gives." The risk was true, and it was larger than we wrote.

| Item | Prediction | Result |
|---|---|---|
| Time of the kernel for the largest pass | 85 µs | 72 µs |
| New work for the `long` request | 0.4 ms | About 0.4 ms. It is in the gain in the test. |
| Gain for the `long` request | 26.5 ms | 11.76 ms |
| Gain for the `medium` request | 10.0 ms | 4.02 ms |
| `short_ms` | -1.3 % | +0.08 %. The interval includes 0. |

The rule of the method applies. A gain of less than 50 % of the prediction has a cause that we must find before the next change. The cause is in the table above.

## Host load

The host had a high load in all runs: 4.02 to 22.15 for the one-minute load average. The two arms of one block ran one after the other, in a sequence that a seed set. The 4 blocks have the same sign of the difference for `lat_ms`, `medium_ms`, `long_ms` and `rps8`. The numbers of arm `D` are near the numbers of arm `L` in the data set `2026-10-31-lead-30489`, which is the same build. The values of `lat_ms` are 237.5 ms here and 239.7 ms there.

## Limits and next work

- The test has 4 blocks. The interval is small, because 4 blocks give a small range of the ratios.
- The device path is on only if three conditions are true. The cells of the pass are in one group of sequences. The model has no `SWA` and no `ALiBi`. The pass has 32 sequences or less.
- The server in production has `--kv-unified`. The correctness runs show the path in use.
- A server without `--kv-unified` uses the CPU path for a pass with more than one sequence.
- No test uses a model with `SWA`, with `ALiBi`, or with more than 32 sequences. These models use the CPU path as before.
- The change has no test with more than one GPU. It has no test with the remote backend of `ggml`. The change adds one operation number to `ggml`, and it changes the patch number of the remote interface.
- To go back to the baseline, set `LLAMA_KQ_MASK_DEVICE=0` for the server, or use build `D`.
- Before the rule for the GPU, we ran small tasks on the GPU when we had no measurement window. They ran at these times on 2026-10-31 (UTC). `test-backend-ops` ran from 05:26:10 to 05:26:17. The server run `0004-check1` ran from 05:26:36 to 05:27:52. About 25 starts of the server did not start, from 05:29 to 05:40:18.
- The measurement windows of other changes were open at these times: 05:11:03 to 05:28:14, 05:28:14 to 05:40:04, and from 05:40:04. These tasks can change the data of those measurement windows. The project has these times. The data set of this change has no run from that time.
- The upload time that stays is large: 15.5 ms of the 21.4 ms of the `long` request in phase 4. The next target is to find which part of the upload time is the time that the CPU waits for the GPU. Change 0007 is for the same part of the code.
