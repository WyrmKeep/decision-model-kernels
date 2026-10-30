# Decision-model benchmark

This folder holds the public benchmark for the decision model. It has 48 cases and 100 decisions. Each decision has a label. The benchmark measures the speed and the quality of a `llama-server`. The client sends the requests to `POST /v1/systemone` of that server.

## Results of the benchmark

The benchmark sends each case to the server. A case has one state and one to three questions about the state. The server sends back one answer for each question.

The benchmark records two types of result:

- Speed: the latency of each request.
- Quality: how many answers agree with the labels and with the reference answers.

The quality numbers must stay in the guards after a change to the server. The section [Guards](#guards) lists them.

## Files

| Path | Description |
|---|---|
| `bench.py` | The client. It runs all phases and writes one result file. |
| `summarize.py` | Shows the values of one run and one TSV line. |
| `remote_run.sh` | Makes one full run on the host that has the GPU. |
| `run_experiment.sh` | Runs `remote_run.sh` through SSH and copies the result back. |
| `launch.example.sh` | The example script that starts the server container. |
| `bench.env.example` | Lists all environment variables of the scripts. |
| `cases.json` | The 48 cases with their labels. |
| `reference.json` | The Q8_0 reference answers. |
| `baseline.json` | The guard numbers of the baseline build. |
| `DATA.md` | The fields, units and meaning of each data file. |
| `tools/make_cases.py` | Makes `cases.json`. |
| `tools/make_baseline.py` | Makes `baseline.json` from baseline runs. |
| `tools/count_tokens.py` | Counts the tokens of each case. |
| `dectest/` | The smoke test with 32 cases and 52 decisions. |

The files `reference.json` and `baseline.json` are empty slots until you make them. Each slot holds the text `"status": "not-measured"`. The client does not do the checks that use the data of an empty slot. The section [Make the reference and the baseline](#make-the-reference-and-the-baseline) shows how to put data in the slots.

## The cases

A person or a script made all the cases. They hold no data of a server in use. The model reads each case as one state and a list of questions. The field `size` of a case gives its type: `short`, `medium` or `long`.

| Type | Description | Cases | Decisions | State length, minimum (tokens) | State length, maximum (tokens) | State length, mean (tokens) | Questions for each case, minimum | Questions for each case, maximum |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `short` | Four tasks | 32 | 52 | 11 | 84 | 37 | 1 | 2 |
| `medium` | A part of a syslog | 8 | 24 | 985 | 1,307 | 1,202 | 3 | 3 |
| `long` | A part of a syslog | 8 | 24 | 2,899 | 3,414 | 3,126 | 3 | 3 |

Source: token counts of the state text with the tokenizer of the model. Each question adds 11 to 94 tokens.

The `short` cases come from four tasks.

| Task | Cases | Questions | State |
|---|---:|---|---|
| `triage` | 12 | `category` (six possible answers), `needs_action` (yes or no) | One report of a home server |
| `check` | 8 | `done` (yes or no) | A task and a report about the task |
| `style` | 8 | `relative_date`, `secret` (yes or no) | One sentence of a text |
| `commit` | 4 | `matches` (yes or no) | A commit message and its `diff` |

Source: `cases.json`.

A syslog case holds usual log lines of a home server. A `medium` case has 20 usual lines. A `long` case has 58 usual lines.

75 % of the syslog cases (12 of 16) have a group of three or four fault lines at a random position. The fault types are `storage`, `memory` and `network`, with 4 cases each. The other 4 cases have no fault. The three questions are `disk_error`, `oom_kill` and `worst`.

## How the labels were made

The person who made the benchmark wrote each short case and its label together. No model made a label.

A person can give a different label to three short cases:

- `triage00`: a report that tells how much of a disk is full.
- `triage07`: the CPU does much work while a `transcode` runs that started at a set time.
- `triage08`: a report of the `mover`.

Most of the incorrect answers in the previous tests were for these three cases.

The script `tools/make_cases.py` makes each syslog label together with its case. It selects the fault type first. It then puts the fault lines between the usual lines. The label comes from the type:

- `disk_error` is true for `storage` only.
- `oom_kill` is true for `memory` only.
- `worst` is the fault type, or `none`.

Do not change a label, a question, the seed or a guard to make a result better. If you find a fault in the benchmark, record the fault. Then make a new benchmark with a new baseline.

## The public benchmark

The first benchmark used the names of internal hosts, networks and paths. The public benchmark does not use these names. It uses:

- One host name for all syslog lines.
- IP addresses for examples, from the ranges of `RFC 5737` and from `2001:db8::/32`.
- The domain name `example.net`.
- Paths with general names.

Each new name has the same token count as the name that it replaces. The check used the tokenizer of the model. It found no difference in the 48 cases, in a state or in a question. The questions and the labels did not change.

The words are different. Thus the answers of a model can have small differences. The public benchmark has a different reference and a different baseline. Do not compare quality numbers of the two benchmarks. You can compare speed numbers because the token counts are the same.

## Metrics

The client shows one line at the end of a run. It also writes a result file. `summarize.py` shows these values.

| Metric | Unit | Meaning |
|---|---|---|
| `lat_ms` | ms | The primary metric. Each timed pass gives a mean latency. `lat_ms` is the median of these means. |
| `spread_pct` | % | The range of the mean latencies of the timed passes, divided by their median. |
| Mean and `p95` for each `size` | ms | The mean and the 95th percentile of the latency of the short requests, the medium requests and the long requests. |
| `p95` | ms | The 95th percentile of the latency of all requests in the timed passes. |
| `rps` | requests/s | The number of requests that finish, divided by the time of the `tput` phase. The client sends `--conc` requests at the same time. |
| `decisions_per_s` | decisions/s | The number of decisions that the server finishes, divided by the time of the `tput` phase. |
| `input_tokens_mean` | tokens | The mean of `usage.input_tokens` of the server, for each `size`. |
| `correct` | decisions, of 100 | The number of decisions with a top answer that is the same as the label. |
| `agree` | number, 0 to 1 | The part of the 100 decisions with a top answer that is the same as the reference answer. |
| `mean_dp` | probability, 0 to 1 | The mean of the largest probability difference to the reference answer. |
| `flips` | list | The decisions where the top answer is different from the reference. |
| `cold_ms` | ms | Optional. The latency of one short request after 30 s with no request. |
| VRAM maximum | MiB | The largest value of `memory.used` in the sampler file. |
| Mean power | W | The mean of `power.draw` in the `seq` phase. |
| Mean SM clock | MHz | The mean of `clocks.sm` in the `seq` phase. |
| `pstate` values | None | The `pstate` values that the sampler records in the `seq` phase. |
| `startup_s` | s | The time from the start of the launch script until the server gives a good response to `/health`. |

Source: the result file of `bench.py` and the output of `summarize.py`.

The top answer of a yes or no question is true when the probability is 0.5 or more. The top answer of a `choice` question is the possible answer with the highest probability. The probability difference of a `choice` question is the largest difference of the probabilities of all possible answers.

## Guards

The result of a run is `PASS` when the run stays in all guards. The result is `n/a` when the baseline has no measurement: the quality guards did not run. The result is `FAIL` when the run does not stay in a guard.

| Guard | Rule | Data that the guard uses |
|---|---|---|
| `errors` | No request gives an error | None |
| `correct` | `correct` is not less than the baseline value minus 1 | Baseline and reference with measurements |
| `agree` | `agree` is not less than the baseline value minus 0.02 | Baseline and reference with measurements |
| `mean_dp` | `mean_dp` is not more than the baseline value plus 0.01 | Baseline and reference with measurements |
| `vram` | The VRAM maximum is not more than 6,144 MiB | The sampler file of `remote_run.sh` |

Source: `bench.py` and `summarize.py`.

The VRAM limit is the default of `summarize.py`. You can change it with the flag `--vram-cap-mib` or with `BENCH_VRAM_CAP_MIB`. If the run has no sampler file, the result is `FAIL`. The sampler counts all GPU memory that programs use. Thus you must stop other GPU programs first.

## Sequence of the phases

The client does the phases in this sequence:

1. **`warmup`.** The client sends three requests. It does not count them.
2. **`quality`.** The client sends each case one time, with the state without change. It then calculates the quality numbers from the answers.
3. **`seq`.** The client does 5 timed passes. This is the default. Each timed pass sends all 48 cases, one request at a time, in a random sequence.
4. **`tput`.** The client sends each case three times. It sends `--conc` requests at the same time. The default is 8.
5. **`cold`.** Optional. The flag `--cold N` sends `N` short requests. Each request comes after 30 s with no request.

All requests, but not the requests of the `quality` phase, start with a line `Ref: <hash>`. The hash is different for each request. The line makes sure that the server does not get a response from the cache of a previous request.

## Necessary items

- Python 3.8 or higher. The client uses only code that comes with Python.
- A `llama-server` build that can run the decision model, for example llama.cpp `b12004`.
- For a full run: a container tool, for example `docker` or `podman`, and the programs `nvidia-smi`, `curl` and `bash`.
- For a run through SSH: `ssh`, `rsync` and `python3` on your device.

## Use the benchmark

### Run the client against a server

1. Start the server. It must receive the requests of `POST /v1/systemone`.
2. Make a folder for the results: `mkdir runs`.
3. Run the client:

   ```sh
   python3 bench.py --host 127.0.0.1 --port 18080 --out runs/my-run.json
   ```

4. Show the values of the run:

   ```sh
   python3 summarize.py runs/my-run
   ```

The environment variables `BENCH_HOST` and `BENCH_PORT` are an alternative to the flags `--host` and `--port`. The client has the flags `--host`, `--port` and `--out`. It also has these flags:

| Flag | Default | Meaning |
|---|---|---|
| `--passes N` | 5 | The number of timed passes |
| `--conc N` | 8 | The number of requests at the same time in the `tput` phase |
| `--cold N` | 0 | The number of requests in the `cold` phase |
| `--cases`, `--reference`, `--baseline` | The files in this folder | The paths of other input files |
| `--write-reference LABEL` | None | Writes the answers of the `quality` phase to `LABEL.reference.json` |
| `--startup-s S` | None | The time that the server used to start. The client writes it in the result file. |

Source: `bench.py`.

### Make a full run on the GPU host

A full run also records the GPU memory, the power and the clocks. Do these steps on the host that has the GPU.

1. Make `bench.env` from `bench.env.example`. Change the values in it.
2. Make `launch.sh` from `launch.example.sh`. Change it if you must use other server flags. The file `.gitignore` keeps the two files out of git.
3. Run the script `remote_run.sh`:

   ```sh
   BENCH_ENV_FILE=bench.env bash remote_run.sh my-run
   ```

4. Show the values of the run: `python3 summarize.py runs/my-run`.

### The work of the script

The script `remote_run.sh` does these tasks:

- It starts the GPU sampler. The sampler writes `runs/my-run.smi`.
- It runs `launch.sh`. Then it waits until `/health` gives a response.
- It runs `bench.py` in a Python container with the network of the host.
- It writes the server log to a file and removes the containers.
- It removes the clock locks of the GPU only when you set `RESET_GPU_CLOCKS=1`.

### Run through SSH

1. Set `GPU_HOST` and `GPU_WORK_DIR` in `bench.env` or as environment variables.
2. Run the script `run_experiment.sh` on your device:

   ```sh
   ./run_experiment.sh my-run
   ```

The script `run_experiment.sh` puts a copy of this folder and of `launch.sh` on the GPU host. It runs `remote_run.sh` there. It then puts a copy of the files `runs/my-run.*` on your device and shows the values of the run.

## Make the model files

The two model files of the benchmark come from one download.

1. Download the Q8_0 file from the `h2oai/h2o-lightning-4b-gguf` repository on Hugging Face. The file name is `h2o-lightning-4b-Q8_0.gguf`. It has 4.48 GB.
2. Set `IMAGE` to the server image, for example `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004`. Set `MODEL_DIR` to the folder with the file.
3. Make the `mix8` file. It keeps Q8_0 for the weights of the attention layers and of the recurrent layers. It also keeps Q8_0 for the embedding and for the output weights. It uses Q4_0 for the weights of the feed-forward layers.

   ```sh
   docker run --rm --cpus 6 -v "$MODEL_DIR:/models" --entrypoint /app/llama "$IMAGE" \
     quantize --allow-requantize --token-embedding-type q8_0 \
     --tensor-type attn_qkv=q8_0 --tensor-type attn_gate=q8_0 --tensor-type ssm_out=q8_0 \
     --tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 \
     --tensor-type attn_output=q8_0 --tensor-type ssm_alpha=q8_0 --tensor-type ssm_beta=q8_0 \
     /models/h2o-lightning-4b-Q8_0.gguf /models/h2o-lightning-4b-rq-Q4_0-mix8.gguf Q4_0 6
   ```

A new build of `llama quantize` can make a file with different bytes. Run the benchmark before you use such a file.

## Make the reference and the baseline

The reference answers and the baseline numbers are not available at this time. Measure the reference and the baseline one time. Use a GPU that does no other work, and the server image of the stock build. Do not start a different GPU program at the same time.

The reference is the set of the answers of the Q8_0 model. The baseline is the quality of the build that you compare with. The baseline also holds three guard numbers. The runs of the baseline make the numbers.

1. Set `MODEL` to the Q8_0 file in `bench.env`. Keep the baseline server flags.
2. Make the reference run:

   ```sh
   ./run_experiment.sh ref-q8 --passes 1 --write-reference ref-q8
   ```

3. Make sure that the run has no errors.
4. Put the answers into the slot:

   ```sh
   cp runs/ref-q8.reference.json reference.json
   ```

5. Set `MODEL` to the `mix8` file in `bench.env`.
6. Make three baseline runs:

   ```sh
   for i in 1 2 3; do ./run_experiment.sh base-$i; done
   ```

7. Make `baseline.json` from the three runs:

   ```sh
   python3 tools/make_baseline.py runs/base-1.json runs/base-2.json runs/base-3.json \
     --note "stock server image, mix8 model, date" --out baseline.json
   ```

8. Make one more run: `./run_experiment.sh base-4`. Make sure that the result is `PASS`.
9. Commit `reference.json`, `baseline.json` and the files of the four runs.

`make_baseline.py` stops when the quality numbers of the three runs are different. Find the cause before you continue.

## Token length test

Use this test after you change a case.

1. Start a server on the CPU. Do not use GPU flags:

   ```sh
   docker run --rm -p 127.0.0.1:8080:8080 -v "$MODEL_DIR:/models:ro" "$IMAGE" \
     -m /models/h2o-lightning-4b-Q8_0.gguf --host 0.0.0.0 --port 8080 \
     -ngl 0 -c 512 --no-warmup
   ```

2. Count the tokens of each case:

   ```sh
   python3 tools/count_tokens.py cases.json --url http://127.0.0.1:8080
   ```

3. Compare two case files. The exit code is 1 when a case has a different token count:

   ```sh
   python3 tools/count_tokens.py old-cases.json --compare cases.json --url http://127.0.0.1:8080
   ```

## Make the cases again

The script `tools/make_cases.py` makes `cases.json` from `dectest/cases.json` and a seed that does not change.

```sh
python3 tools/make_cases.py --check
```

The command stops with an error if `cases.json` is different from the output of the script. Do not change the seed, the text of the log lines or the sequence of the random numbers. Such a change gives different cases. Then you cannot compare the new results with the previous results.

## Smoke test

The folder [`dectest/`](dectest/README.md) holds the 32 short cases as a test with 52 decisions. You can use the test without the other cases. Use it for a fast check of the answer quality after a change.
