# decision-model-kernels

This project makes one small decision model faster on one GPU. It changes the code of llama.cpp: the CUDA kernels, the graph, and the batches of the server. It does not change the flags of the server. We tuned the flags before the project started.

## The project

- One model: H2O Lightning 4B, with the "mix8" quantization.
- One GPU: the `RTX 3060 Ti`.
- One program that we change: llama.cpp. The code is in a fork: [github.com/WyrmKeep/llama.cpp](https://github.com/WyrmKeep/llama.cpp). Each change has one branch.
- Three types of output: a [notebook](docs/notebook/README.md) with each test, public repositories with the code and the data, and blog posts.
- The owner wants to know how a request goes from the client to the kernels. Each step has a measurement.
- No pull requests to llama.cpp. The changes are for one GPU and one model.

The project has no model files. You can download the model from Hugging Face. The project is not a benchmark for a different GPU or a different model.

## Where the project is at this time

The project is at the checkpoint of phase 4. We measured where the time goes. We ranked the possible changes and wrote a prediction for each one. The code of llama.cpp has no change at this time.

- A request uses 2.13 to 2.45 times its roofline floor.
- The GPU runs a kernel for 90 % to 94 % of the time of a request. The host is not the limit.
- The ranked list has 15 targets. The first change to use is target `E`: [pull request #30489](https://github.com/ggml-org/llama.cpp/pull/30489) of llama.cpp. We measured it as a lead. `lat_ms` is 1.94 % lower, and the answers are the same bit for bit.
- The owner selects the next step from the [ranked list](docs/notebook/2026-10-31-phase4-ranked-targets.md).

## How we got here

The project started on 2026-10-30 at 21:05 UTC. The checkpoint of phase 4 is at about 02:30 UTC on 2026-10-31. Each row of the table gives the primary result of one phase and a link to its notebook entry. The [long version](docs/story.md) has the full text for each phase.

| Phase | Question | Primary result | Notebook entry |
|---|---|---|---|
| 0 | Which work is in upstream llama.cpp? | 200 queries gave 2,322 different items. Upstream has no ragged batching for hybrid models and no output head for answer tokens. Four upstream items are important for the project: #30489, #30779, #28495 and #31031. | [Upstream inspection](docs/notebook/2026-10-30-phase0-upstream-survey.md) |
| 1 | Does the build that we make give the same result as the stock image? | Yes. `lat_ms` is -0.40 % (interval -0.76 % to -0.09 %), in ±2 %. The answers are the same bit for bit. | [Result of phase 1](docs/notebook/2026-10-30-phase1-equivalence-result.md) |
| 2 | Which numbers must a change keep? | The KL budget is 0.027830. The 52-decision set gives 48 correct. `test-backend-ops` gives 181 of 187. | [Correctness baseline](docs/notebook/2026-10-30-phase2-correctness-baseline.md) |
| 3 | Where does the time of a request go? | The GPU runs a kernel for 90 % to 94 % of the time. MMQ is about 63 % of the kernel time at 32.5 % to 39.2 % of the int8 peak. | [Where the time goes](docs/notebook/2026-10-31-phase3-where-the-time-goes.md) |
| 4 | Which changes come first? | 158.4 ms for `lat_ms` if the 8 targets all pass (prediction). The first change is `E`. | [Ranked list](docs/notebook/2026-10-31-phase4-ranked-targets.md) |

Source: the notebook entries in the last column.

## The primary numbers

### Request times and the roofline floor

The roofline floor is the sum of the times of the kernels of a request at the limits of the GPU. A request uses 2.13 to 2.45 times that floor.

| Request type | Time of one request (ms) | Kernel time (ms) | Weight floor (ms) | Roofline floor (ms) | Ratio of time to roofline floor |
|---|---:|---:|---:|---:|---:|
| `short1q` | 47.1 | 42.4 | 7.7 | 19.2 | 2.45 |
| `short2q` | 87.1 | 85.1 | 20.7 | 39.8 | 2.19 |
| `medium` | 360.0 | 344.9 | 84.6 | 169.1 | 2.13 |
| `long` | 797.0 | 761.3 | 177.9 | 371.5 | 2.15 |

Source: the table "Where the time goes" in the [phase 4 entry](docs/notebook/2026-10-31-phase4-ranked-targets.md#where-the-time-goes) and `tables/floors.tsv` of the [time model](data/2026-10-31-phase4-model/README.md).

### Where the kernel time goes

| Quantity | Value | Unit | Where we measured it |
|---|---:|---|---|
| MMQ part of the kernel time | 63 | % | [Phase 4 entry](docs/notebook/2026-10-31-phase4-ranked-targets.md#where-the-time-goes), fact 2 |
| MMQ speed, as a part of the int8 peak (1,173 tokens or more) | 32.5 to 39.2 | % | [Phase 3 entry](docs/notebook/2026-10-31-phase3-where-the-time-goes.md#mmq-with-many-tokens) |
| GPU busy time, as a part of the wall time | 90.0 to 93.7 | % | [Phase 3 entry](docs/notebook/2026-10-31-phase3-where-the-time-goes.md#where-the-wall-time-goes) |
| MMVQ speed, as a part of the DRAM limit | 90 to 95 | % | [Phase 3 entry](docs/notebook/2026-10-31-phase3-where-the-time-goes.md#mmvq-gdn_single-concat_non_cont-and-small-kernels) |

### The top of the ranked list

These numbers are predictions. We wrote them before the code of a target. The only measured gain is the gain of target `E` as a lead (1.94 % of `lat_ms`).

| Rank | Target | Equivalence class | Confidence (0 to 1) | Predicted gain of `lat_ms` (ms) | Predicted gain of `lat_ms` (%) | Expected gain of `lat_ms` (%) |
|---:|---|---|---:|---:|---:|---:|
| 1 | `G`. MMQ as a pipeline with `cp.async` tile loads | `exact` | 0.40 | 35.7 | 14.8 | 5.9 |
| 2 | `J`. Ragged batching for the suffixes of the questions | `re-associated` | 0.45 | 16.4 | 6.8 | 3.1 |
| 3 | `N`. New MMQ kernels, after `G` | `re-associated` | 0.20 | 24.9 | 10.4 | 2.1 |
| 4 | `B`. Attention mask on the GPU | `exact` | 0.70 | 6.9 | 2.9 | 2.0 |
| 5 | `E`. Pull request #30489: GLU fused into the quantization | `exact` | 0.95 | 4.8 | 2.0 | 1.9 |
| 6 | `C`. Concat folded into `ssm_conv` | `exact` | 0.70 | 6.4 | 2.7 | 1.9 |
| 7 | `F`. Fusions of RMS norm kernels, element-wise kernels and quantization kernels | `exact` | 0.50 | 8.6 | 3.6 | 1.8 |
| 8 | `H`. MMQ tile and occupancy adjustments for this GPU | `exact` or `re-associated` | 0.30 | 10.2 | 4.2 | 1.3 |

Source: the [ranked list of the phase 4 entry](docs/notebook/2026-10-31-phase4-ranked-targets.md#ranked-list) and `tables/ranked.tsv` of the [time model](data/2026-10-31-phase4-model/README.md). The expected gain is the predicted gain multiplied by the confidence.

### The total gain

The gains of the targets do not add, because some targets change the same time. The time model applies the targets `A`, `B`, `C`, `D`, `E`, `F`, `G` and `J` one after the other. These numbers are predictions.

| Request type | Time at this time (ms) | Time if all 8 targets pass (ms) | Change (%) | Time with the expected gain (ms) | Change (%) |
|---|---:|---:|---:|---:|---:|
| `short1q` | 47.1 | 34.4 | -27.0 | 40.3 | -14.4 |
| `short2q` | 87.1 | 53.8 | -38.3 | 69.5 | -20.2 |
| `medium` | 360.0 | 230.2 | -36.1 | 292.5 | -18.7 |
| `long` | 797.0 | 534.4 | -33.0 | 659.9 | -17.2 |
| `lat_ms` | 240.9 | 158.4 | -34.2 | 197.8 | -17.9 |

Source: the section "Prediction of the total gain" in the [phase 4 entry](docs/notebook/2026-10-31-phase4-ranked-targets.md#prediction-of-the-total-gain) and `tables/sum.tsv` of the [time model](data/2026-10-31-phase4-model/README.md). The first version of this prediction added the gains of the targets. It gave 153.7 ms for `lat_ms` [calculated: 240.9 ms minus 87.2 ms] and counted some kernel time two times. The [long version](docs/story.md#phase-4-the-ranked-list-and-a-correction) shows the correction.

### Results at this time

The baseline of all work is build `D`. Build `D` is the build of llama.cpp tag `b12004` that we make for `sm_86`. No change to the code is complete at this time.

| Benchmark | Build | Runs | `lat_ms` (ms) | `short_ms` (ms) | `medium_ms` (ms) | `long_ms` (ms) | `correct` (of 100) |
|---|---|---:|---:|---:|---:|---:|---:|
| Private benchmark | `D` (baseline) | 6 | 238.3 | 72.1 | 359.3 | 784.5 | 94 |
| Private benchmark | Stock image `b12004` | 6 | 239.2 | 72.3 | 360.3 | 786.1 | 94 |
| Public benchmark | `D`, run 1 | 1 | 236.2 | 71.1 | 356.2 | 778.7 | 94 |
| Public benchmark | `D`, run 2 | 1 | 236.9 | 71.3 | 357.4 | 781.1 | 94 |
| Public benchmark | `D`, run 3 | 1 | 237.3 | 71.3 | 357.8 | 781.7 | 94 |

Source: the rows of the private benchmark are the means of 6 blocks. They come from the [result of phase 1](docs/notebook/2026-10-30-phase1-equivalence-result.md) and the table `speed-bench.tsv` of the [first data set](data/2026-10-30-window1/README.md). The rows of the public benchmark are the three baseline runs in the [data set of the baseline of the public benchmark](data/2026-10-31-public-baseline/README.md). The private benchmark is not in this repository. The [public benchmark](bench/README.md) has different words and a different baseline. Do not compare the quality numbers of the two benchmarks.

## The method

- **Prediction before test.** We write the prediction, the limits and the tolerances in the notebook. We commit them before we measure. The [notebook](docs/notebook/README.md) shows the dates.
- **Baseline from source.** We compare each change with build `D`, which we make from source. Phase 1 showed that build `D` is equivalent to the stock image.
- **Equivalence classes.** Each change has an equivalence class before its test: `exact`, `re-associated` or `approximation`. The equivalence class sets the tests of correctness.
- **Tests of correctness.** An `exact` change must give the same answers bit for bit. A `re-associated` change must stay in the limits of `test-backend-ops` and have a KL divergence at noise level. An `approximation` has a limit for the KL divergence.
- **Blocks of runs.** A speed test uses blocks. Each block has one run of each build in a random sequence. The interval is a 95 % bootstrap interval of the blocks.
- **One test at a time.** The GPU does the work of one run only. The server in production is off during a measurement window.
- **Rule to stop.** We stop work on a target when the result is different from the prediction and we cannot find the cause.

The [method](docs/method.md) gives all rules.

## The test device

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce `RTX 3060 Ti` (GA104, `sm_86`, 38 SMs, 8 GiB) |
| GPU power limit | 200 W (the default) |
| Driver | 595.71.05 |
| CUDA | 12.8.1 |
| llama.cpp | tag `b12004` (commit `3c809f4`) |
| Model | H2O Lightning 4B (24 Gated DeltaNet layers and 8 attention layers, as in Qwen3.5-4B) |
| Quantization | Q4_0 for the weights of the feed-forward layers, Q8_0 for all other weights ("mix8") |
| Request of the server | `POST /v1/systemone` of `llama-server` |

## Reproduce

1. Get the model file. [`bench/`](bench/README.md#make-the-model-files) shows how to make the "mix8" file.
2. Build llama.cpp. [`harness/`](harness/README.md#build-the-programs-of-an-arm) shows the build commands.
3. Start the server and run the benchmark. [`bench/`](bench/README.md#use-the-benchmark) shows the commands.
4. Compare two builds with a schedule of blocks. [`harness/`](harness/README.md#make-a-schedule) shows the commands.
5. Make a profile of phase 3 with the code in [`harness/profile/`](harness/profile/README.md).
6. Make an ablation with the code in [`harness/ablate/`](harness/ablate/README.md).

## Folders of the repository

| Path | Contents |
|---|---|
| [`bench/`](bench/README.md) | The public benchmark: cases, labels, client |
| [`harness/`](harness/README.md) | The scripts that run tests of two builds |
| [`harness/profile/`](harness/profile/README.md) | The code of phase 3: the census, the profile runs, and the tables |
| [`harness/ablate/`](harness/ablate/README.md) | The tools of the ablation: the arms, the client, and the tables |
| [`data/`](data/README.md) | The data of each measurement and the tables that we make from it |
| [`docs/`](docs/README.md) | The long version, the method, the model, the kernels, the notebook, the glossary |
| [`docs/story.md`](docs/story.md) | How we got here, phase by phase |
| [`docs/notebook/`](docs/notebook/README.md) | One entry for each test, with its date |
| [`docs/changes/`](docs/changes/README.md) | One document for each change, with the same text as its pull request |
| [`tools/ste/`](tools/ste/README.md) | The ASD-STE100 checker for all documents |
| [`tools/publish/`](tools/publish/README.md) | The checks before the files go to GitHub |

The folder `data/` has these data sets.

| Data set | Contents |
|---|---|
| [`2026-10-30-window1`](data/2026-10-30-window1/README.md) | Build `D` against the stock image, KL divergence, the 52-decision set, `compute-sanitizer` |
| [`2026-10-30-test-backend-ops`](data/2026-10-30-test-backend-ops/README.md) | `test-backend-ops` for the 187 operations of the graph |
| [`2026-10-30-card-limits`](data/2026-10-30-card-limits/README.md) | The highest rates of the GPU for tensor cores, DRAM and L2 cache |
| [`2026-10-30-shape-census`](data/2026-10-30-shape-census/README.md) | The shapes, launches, bytes and floors of each pass |
| [`2026-10-31-public-baseline`](data/2026-10-31-public-baseline/README.md) | The baseline of the public benchmark |
| [`2026-10-31-profiles`](data/2026-10-31-profiles/README.md) | The decomposition of the request time, the profiles, and two lead tests |
| [`2026-10-31-ablation`](data/2026-10-31-ablation/README.md) | The caps of the parts of the GPU work |
| [`2026-10-31-lead-30489`](data/2026-10-31-lead-30489/README.md) | The test of pull request #30489 against build `D` |
| [`2026-10-31-phase4-model`](data/2026-10-31-phase4-model/README.md) | The time model of phase 4 |

## Condition of each change

No change has a change document at this time. A change gets its number when it gets its document. The table shows the changes in the sequence of work of the [phase 4 entry](docs/notebook/2026-10-31-phase4-ranked-targets.md#recommended-sequence). The owner selects the budget and the next step.

| Target | Change | Equivalence class | Condition |
|---|---|---|---|
| `E` | Pull request #30489: GLU fused into the quantization of the activation | `exact` | First in the sequence. Measured as a lead: `lat_ms` is 1.94 % lower. Not in production. |
| `D` | Value-split path of `gdn_single` | `exact` | Second in the sequence. Not started. |
| `H` | MMQ tile and occupancy adjustments for this GPU | `exact` or `re-associated` | Second in the sequence. Not started. |
| `A` | Output head for the answer tokens only | `exact` | Third in the sequence. Not started. |
| `B` | Attention mask on the GPU | `exact` | Fourth in the sequence. Not started. |
| `C` | Concat folded into `ssm_conv` | `exact` | Fourth in the sequence. Not started. |
| `G` | MMQ as a pipeline with `cp.async` tile loads | `exact` | Fifth in the sequence. Not started. |
| `J` | Ragged batching for the suffixes of the questions | `re-associated` | Sixth in the sequence. Not started. |

The targets `F`, `L`, `P`, `K` and `O` come after these changes, if the rule to stop does not stop the project. The phase 4 entry does not recommend `N` and `M`.

## Standard for the text

All documents and pull requests follow ASD-STE100 Issue 9. The [checker](tools/ste/README.md) checks each document before each pull request. The [glossary](docs/glossary.md) gives the terms of the project.

## License

`MIT`. Refer to [LICENSE](LICENSE). The fork of llama.cpp keeps the `MIT` license of llama.cpp. The model (H2O Lightning 4B) has the `Apache 2.0` license. This project does not include it.

## Changes from other persons

This project is the work of one person. You can read the files and copy them. The license gives the rules. The project does not accept pull requests on GitHub, because GitHub has a copy of the project only.
