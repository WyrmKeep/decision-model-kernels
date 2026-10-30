# decision-model-kernels

This project makes one small decision model faster on one GPU. It changes the code of llama.cpp (CUDA kernels, the compute graph and batching). It does not change settings.

## What the project is

- A lab notebook. Each change has a prediction, which we write before the measurement.
- A benchmark harness and the data of each measurement.
- Documents for each change to llama.cpp. The code is in a fork of llama.cpp: [github.com/WyrmKeep/llama.cpp](https://github.com/WyrmKeep/llama.cpp), one branch for each change.

## What the project is not

- It is not a contribution to llama.cpp. The changes are for one GPU and one model.
- It is not a general benchmark of GPUs or models.
- It does not include model files. The model is on Hugging Face.

## Current results

No change is complete yet. The table shows the baseline only.

| Build | Short request (ms) | Medium request (ms) | Long request (ms) | Mean of all requests (ms) |
|---|---:|---:|---:|---:|
| Stock llama.cpp `b12004` (baseline) | 71 | 357 | 779 | 236 |

Source: the private benchmark, 48 requests, measured before this project started. The public benchmark in [bench/](bench/README.md) has its own baseline.

## Hardware and software

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti (GA104, `sm_86`, 38 SMs, 8 GiB) |
| GPU power limit | 200 W (the default) |
| Driver | 595.71.05 |
| CUDA | 12.8.1 |
| llama.cpp | tag `b12004` (commit `3c809f4`) |
| Model | H2O Lightning 4B (Qwen3.5-4B architecture: 24 Gated DeltaNet layers and 8 attention layers) |
| Quantization | Q4_0 for the feed-forward weights, Q8_0 for all other weights ("mix8") |
| API | `POST /v1/systemone` of `llama-server` |

## Quick start

1. Get the model file. The [bench/](bench/README.md) document shows how to make the "mix8" file.
2. Build llama.cpp. The [harness/](harness/README.md) document shows the build commands.
3. Start the server and run the benchmark. The [bench/](bench/README.md) document shows the commands.

## Repository layout

| Path | Contents |
|---|---|
| [bench/](bench/README.md) | The public benchmark: cases, labels, client |
| [harness/](harness/README.md) | Scripts that build llama.cpp and run A/B tests |
| [data/](data/README.md) | Raw data and derived tables of each measurement |
| [docs/](docs/README.md) | Method, model notes, kernel notes, lab notebook, glossary |
| [docs/changes/](docs/changes/README.md) | One document for each change, with the same text as its pull request |
| [tools/ste/](tools/ste/README.md) | The ASD-STE100 checker for all documents |
| [tools/publish/](tools/publish/README.md) | The publication gate |

## Status of each change

| Number | Change | Equivalence class | Status |
|---|---|---|---|
| | No change yet | | |

## Writing standard

All documents and pull requests use ASD-STE100 Issue 9. The [checker](tools/ste/README.md) does a check of each document before each pull request. The [glossary](docs/glossary.md) gives the technical words of the project.

## License

MIT. Refer to [LICENSE](LICENSE). The fork of llama.cpp keeps the MIT license of llama.cpp. The model (H2O Lightning 4B) has the Apache 2.0 license. This project does not include it.

## How to contribute

This is a personal project. You can read it and copy from it under the license. The project does not accept pull requests on GitHub, because GitHub has a copy of the project only.
