# decision-model-kernels

This project makes one small decision model faster on one GPU. It changes the code of llama.cpp: the CUDA kernels, the graph, and the batches of the server. It does not change the flags of the server.

## The project

- A [notebook](docs/notebook/README.md). Each change has a prediction. We write the prediction before the test.
- A benchmark, a harness, and the data of each measurement.
- A document for each change to llama.cpp. The code is in a fork of llama.cpp: [github.com/WyrmKeep/llama.cpp](https://github.com/WyrmKeep/llama.cpp). Each change has one branch.

## Limits of the project

- It does not send pull requests to llama.cpp. The changes are for one GPU and one model.
- It is not a benchmark for a different GPU or a different model.
- It has no model files. You can download the model from Hugging Face.

## Results at this time

No change is complete at this time. The table shows the baseline only.

| Build | Short request (ms) | Medium request (ms) | Long request (ms) | Mean of all requests (ms) |
|---|---:|---:|---:|---:|
| llama.cpp `b12004` without changes (baseline) | 71 | 357 | 779 | 236 |

Source: the private benchmark, with 48 requests. We measured it before this project started. The [public benchmark](bench/README.md) has a different baseline.

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

## Run a test

1. Get the model file. [`bench/`](bench/README.md) shows how to make the "mix8" file.
2. Build llama.cpp. [`harness/`](harness/README.md) shows the build commands.
3. Start the server and run the benchmark. [`bench/`](bench/README.md) shows the commands.

## Folders of the repository

| Path | Contents |
|---|---|
| [`bench/`](bench/README.md) | The public benchmark: cases, labels, client |
| [`harness/`](harness/README.md) | The scripts that run tests of two builds |
| [`data/`](data/README.md) | The data of each measurement and the tables that we make from it |
| [`docs/`](docs/README.md) | The method, the model, the kernels, the notebook, the glossary |
| [`docs/changes/`](docs/changes/README.md) | One document for each change, with the same text as its pull request |
| [`tools/ste/`](tools/ste/README.md) | The ASD-STE100 checker for all documents |
| [`tools/publish/`](tools/publish/README.md) | The checks before the files go to GitHub |

## Condition of each change

| Number | Change | Equivalence class | Condition |
|---|---|---|---|
| | No change at this time | | |

## Standard for the text

All documents and pull requests follow ASD-STE100 Issue 9. The [checker](tools/ste/README.md) checks each document before each pull request. The [glossary](docs/glossary.md) gives the terms of the project.

## License

`MIT`. Refer to [LICENSE](LICENSE). The fork of llama.cpp keeps the `MIT` license of llama.cpp. The model (H2O Lightning 4B) has the `Apache 2.0` license. This project does not include it.

## Changes from other persons

This project is the work of one person. You can read the files and copy them. The license gives the rules. The project does not accept pull requests on GitHub, because GitHub has a copy of the project only.
