# Phase 1: equivalence of the source build and the stock image (plan)

Written 2026-10-30, 21:35 UTC, before any measurement.

## Purpose

The project compares each change with a baseline build. That baseline must be a build that we make from source. This entry declares how we show that our build is equivalent to the stock image. We declare the margins and tolerances here, before we measure.

## Arms

| Arm | Binaries | Container image |
|---|---|---|
| S (stock) | The `/app` binaries of the image | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004` |
| D (source build) | Our build of tag `b12004` for `sm_86` only, mounted into the same image | The same image |

Both arms use the same model file, the same server flags and the same GPU. The GPU runs at its default 200 W power limit with default clocks. No other process uses the GPU during the test.

## Speed test

- **Harness:** the private benchmark (48 requests, 100 decisions) with its standard client settings: warm-up, one quality pass, 5 timed passes, one throughput pass at 8 concurrent requests.
- **Order:** 6 blocks. Each block has one S run and one D run in a random order from a fixed seed. Each run starts a new server.
- **`llama bench`:** prompt sizes 64, 256 and 3,000 tokens, 5 repetitions each. 4 blocks, with S and D in a random order in each block.
- **Statistic:** for each block, the ratio D/S. The estimate is the mean of the block ratios. The 95 % confidence interval comes from a bootstrap over blocks (10,000 samples).
- **Gate:** the interval must be inside ±2 % for the benchmark's `lat_ms` and for each `llama bench` prompt size. We also report the short, medium and long means, but they are not part of the gate.

## Answer test

1. **Self-repeat.** Each benchmark run saves the answers of its quality pass. Each run starts a new server and sends the same requests in the same order. We compare the answer probabilities of all runs of one arm, bit for bit.
2. **Cross-build.** We compare the S answers with the D answers, bit for bit.
3. **If the answers are not bit-identical,** these tolerances apply:

   | Check | Tolerance |
   |---|---|
   | Largest difference of an answer probability | ≤ 0.001 |
   | Top answer of each decision | Same for all 100 decisions |
   | Labelled decisions correct | Same count |
   | Benchmark guards | All pass |
   | KL divergence, mean per token | ≤ 1e-5 |
   | KL divergence, largest for one token | ≤ 1e-3 |
   | Same top token | ≥ 99.9 % |

4. **KL divergence.** `llama perplexity` with `--kl-divergence-base` on arm S, then `--kl-divergence` on arm D. Text sets: the first 20 chunks of the WikiText-2 test file at a context of 512 tokens, and one file with the 16 medium and long benchmark states.

## Prediction

- **Speed:** no difference. Both arms run the same source with the same compiler and flags. Only the list of GPU architectures is different. The point estimate will be within ±0.5 %.
- **Answers:** bit-identical in the self-repeat test and in the cross-build test, if the `sm_86` machine code of the two builds is identical. If the machine code is different, small differences, inside the tolerances above.
