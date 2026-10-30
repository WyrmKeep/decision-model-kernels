# Decision-model benchmark

This folder holds the public benchmark for the decision model. It has 48 cases with 100 labeled decisions. It measures the speed and the answer quality of a `llama-server` that serves `POST /v1/systemone`.

## What the benchmark measures

The benchmark sends each case to the server. A case has a text (the state) and one to three questions about the text. The server returns one answer for each question.

The benchmark records two kinds of result:

- Speed: the time of each request.
- Quality: the agreement of the answers with the labels and with the reference answers.

A change to the server passes only when the quality numbers stay inside the guards. The section [Guards](#guards) lists them.

## Files

| Path | Role |
|---|---|
| `bench.py` | The client. It runs all phases and writes one result file. |
| `summarize.py` | Prints the summary of one run and one TSV line. |
| `remote_run.sh` | Makes one complete run on the machine that has the GPU. |
| `run_experiment.sh` | Runs `remote_run.sh` through SSH and copies the result back. |
| `launch.example.sh` | Example script that starts the server container. |
| `bench.env.example` | Lists every setting of the scripts. |
| `cases.json` | The 48 cases with their labels. |
| `reference.json` | The Q8_0 reference answers. |
| `baseline.json` | The guard numbers of the baseline build. |
| `DATA.md` | The fields, units and meaning of every data file. |
| `tools/make_cases.py` | Makes `cases.json`. |
| `tools/make_baseline.py` | Makes `baseline.json` from baseline runs. |
| `tools/count_tokens.py` | Counts the tokens of each case. |
| `dectest/` | The smoke test with 32 cases and 52 decisions. |

The files `reference.json` and `baseline.json` are empty slots until you make them. Each slot holds the marker `"status": "not-measured"`. The client skips the checks that need the missing data. The section [Make the reference and the baseline](#make-the-reference-and-the-baseline) shows how to fill the slots.

## The cases

The cases are synthetic. They contain no real server data. The model reads each case as one state and a list of questions.

| Kind | Cases | Decisions | State size (tokens) | Questions per case |
|---|---:|---:|---|---:|
| Short | 32 | 52 | 11 to 84, mean 37 | 1 or 2 |
| Medium syslog excerpt | 8 | 24 | 985 to 1,307, mean 1,202 | 3 |
| Long syslog excerpt | 8 | 24 | 2,899 to 3,414, mean 3,126 | 3 |

Source: token counts of the state text with the tokenizer of the model. Each question adds 11 to 94 tokens.

The short cases come from four tasks.

| Task | Cases | Questions | State |
|---|---:|---|---|
| `triage` | 12 | `category` (six choices), `needs_action` (yes or no) | One notification of a home server |
| `check` | 8 | `done` (yes or no) | A task and a report about the task |
| `style` | 8 | `relative_date`, `secret` (yes or no) | One sentence from a document |
| `commit` | 4 | `matches` (yes or no) | A commit message and its diff |

A syslog case holds routine log lines of a home server. A medium case has 20 routine lines. A long case has 58 routine lines.

Half of the cases have a block of three or four fault lines at a random place. The fault kinds are `storage`, `memory` and `network`. The other cases have no fault. The three questions are `disk_error`, `oom_kill` and `worst`.

## How the labels were made

The author wrote each short case and its label together. No model chose a label.

Three short labels are debatable: `triage00` (a status report that mentions disk usage), `triage07` (high CPU while a scheduled transcode runs) and `triage08` (a mover summary). Most misses of earlier tests were on these three cases.

The script `tools/make_cases.py` makes the syslog labels by construction. It chooses the fault kind first. It then inserts the fault lines into the routine lines. The label follows from the kind:

- `disk_error` is true for `storage` only.
- `oom_kill` is true for `memory` only.
- `worst` is the fault kind, or `none`.

Never change a label, a question, the seed or a guard to improve a result. If you find a fault in the benchmark, record the fault and make a new benchmark version with a new baseline.

## The public copy

The first version of this benchmark used the names of internal hosts, networks and paths. This public copy has generic names instead:

- One host name for all syslog lines.
- Addresses from the documentation ranges of RFC 5737 and from `2001:db8::/32`.
- The domain `example.net`.
- Generic paths.

Each replacement has the same token count as the text it replaced. The check used the tokenizer of the model. It found no difference in the 48 cases, neither in a state nor in a question. The questions and the labels did not change.

The words differ, so the answers of a model can differ a little. The public copy therefore has its own reference and its own baseline. Do not compare quality numbers between the two copies. Speed numbers are comparable, because the token counts match.

## Metrics

The client prints one line at the end of a run. It also writes a result file. `summarize.py` prints these values.

| Metric | Unit | Meaning |
|---|---|---|
| `lat_ms` | ms | Primary metric. The median, over the timed passes, of the mean request time in one pass. |
| `spread_pct` | % | Range of the pass means divided by their median. |
| Mean and p95 for each size | ms | Mean and 95th percentile of the request time of short, medium and long cases. |
| `p95` | ms | 95th percentile of all timed requests. |
| `rps` | requests/s | Completed requests divided by wall time with `--conc` requests in flight. |
| `decisions_per_s` | decisions/s | Answered questions divided by wall time in the same phase. |
| `input_tokens_mean` | tokens | Mean of `usage.input_tokens` of the server, for each size. |
| `correct` | decisions, of 100 | Decisions whose top answer equals the label. |
| `agree` | fraction, 0 to 1 | Part of the 100 decisions whose top answer equals the reference answer. |
| `mean_dp` | probability, 0 to 1 | Mean of the largest probability difference to the reference answer. |
| `flips` | list | Decisions where the top answer differs from the reference. |
| `cold_ms` | ms | Optional. Time of one short request after 30 s of idle time. |
| VRAM peak | MiB | Largest `memory.used` that the sampler saw. |
| Power, SM clock, P-state | W, MHz | Mean power and mean SM clock in the timed pass phase, and the P-states seen. |
| `startup_s` | s | Time from the start of the launch script until the server is healthy. |

The top answer of a yes or no question is true when the probability is 0.5 or more. The top answer of a choice question is the choice with the highest probability. The probability difference of a choice question is the largest difference over all choices.

## Guards

A run passes when every guard passes. The three quality guards show `n/a` when the baseline or the reference has no measured data.

| Guard | Rule | Needs |
|---|---|---|
| Errors | No request fails | Nothing |
| Correct | `correct` is at least the baseline value minus 1 | Measured baseline and reference |
| Agreement | `agree` is at least the baseline value minus 0.02 | Measured baseline and reference |
| Probability difference | `mean_dp` is at most the baseline value plus 0.01 | Measured baseline and reference |
| VRAM | The peak is at most 6,144 MiB | The sampler file of `remote_run.sh` |

The VRAM limit is the default of `summarize.py`. Change it with `--vram-cap-mib` or with `BENCH_VRAM_CAP_MIB`. The VRAM guard fails when the run has no sampler file. The sampler counts all GPU memory in use, so stop other GPU programs first.

## Order of the phases

The client works in this order:

1. **Warmup.** Three requests, not counted.
2. **Quality.** Each case once, with its raw state. The client scores the answers.
3. **Timed passes.** Each pass sends all 48 cases, one request at a time, in a shuffled order. The default is 5 passes.
4. **Throughput.** Each case three times, with `--conc` requests in flight. The default is 8.
5. **Cold.** Optional. `--cold N` sends N short requests, each after 30 s of idle time.

Each request except the quality requests starts with a line `Ref: <hash>`. The hash is different for each request. The line prevents the server from answering from a cache of an earlier request.

## Requirements

- Python 3.8 or later. The client uses only the standard library.
- A `llama-server` build with decision-model support, for example llama.cpp `b12004`.
- For a complete run: a container tool (Docker or Podman), `nvidia-smi`, `curl` and `bash`.
- For a run through SSH: `ssh`, `rsync` and `python3` on your workstation.

## Use the benchmark

### Run the client against a running server

1. Start the server. It must serve `POST /v1/systemone`.
2. Make a folder for the results: `mkdir runs`.
3. Run the client:

   ```sh
   python3 bench.py --host 127.0.0.1 --port 18080 --out runs/my-run.json
   ```

4. Show the summary:

   ```sh
   python3 summarize.py runs/my-run
   ```

Use `BENCH_HOST` and `BENCH_PORT` instead of the options when you prefer variables. The client options are:

| Option | Default | Meaning |
|---|---|---|
| `--passes N` | 5 | Number of timed passes |
| `--conc N` | 8 | Requests in flight in the throughput phase |
| `--cold N` | 0 | Number of cold requests |
| `--cases`, `--reference`, `--baseline` | Files in this folder | Other input files |
| `--write-reference LABEL` | None | Save the quality answers as `LABEL.reference.json` |
| `--startup-s S` | None | Startup time to record in the result file |

### Make a complete run on the GPU machine

A complete run also records the GPU memory, power and clocks. Do these steps on the machine that has the GPU.

1. Make `bench.env` from `bench.env.example`. Edit it.
2. Make `launch.sh` from `launch.example.sh`. Edit it when you need other server flags. The file `.gitignore` keeps both files out of git.
3. Run the runner:

   ```sh
   BENCH_ENV_FILE=bench.env bash remote_run.sh my-run
   ```

4. Show the summary: `python3 summarize.py runs/my-run`.

### What the script does

The script `remote_run.sh` does these things:

- It starts the GPU sampler, which writes `runs/my-run.smi`.
- It runs `launch.sh`, then waits until `/health` answers.
- It runs `bench.py` in a Python container on the host network.
- It saves the server log and removes the containers.
- It resets GPU clock locks only when you set `RESET_GPU_CLOCKS=1`.

### Run through SSH

1. Set `GPU_HOST` and `GPU_WORK_DIR` in `bench.env` or in your shell.
2. Run the driver on your workstation:

   ```sh
   ./run_experiment.sh my-run
   ```

The driver copies this folder and `launch.sh` to the GPU machine. It runs the runner there and copies the files `runs/my-run.*` back. It then prints the summary.

## Make the model files

The benchmark uses two model files. Both come from one download.

1. Download the Q8_0 file from the `h2oai/h2o-lightning-4b-gguf` repository on Hugging Face. The file name is `h2o-lightning-4b-Q8_0.gguf`. It has 4.48 GB.
2. Set `IMAGE` to the server image, for example `ghcr.io/ggml-org/llama.cpp:server-cuda-b12004`. Set `MODEL_DIR` to the folder with the file.
3. Make the "mix8" file. It keeps Q8_0 for the attention, recurrent, embedding and output weights. It uses Q4_0 for the feed-forward weights.

   ```sh
   docker run --rm --cpus 6 -v "$MODEL_DIR:/models" --entrypoint /app/llama "$IMAGE" \
     quantize --allow-requantize --token-embedding-type q8_0 \
     --tensor-type attn_qkv=q8_0 --tensor-type attn_gate=q8_0 --tensor-type ssm_out=q8_0 \
     --tensor-type attn_q=q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0 \
     --tensor-type attn_output=q8_0 --tensor-type ssm_alpha=q8_0 --tensor-type ssm_beta=q8_0 \
     /models/h2o-lightning-4b-Q8_0.gguf /models/h2o-lightning-4b-rq-Q4_0-mix8.gguf Q4_0 6
   ```

A new build of `llama quantize` can make a file that is not byte-identical. Run the benchmark before you use such a file.

## Make the reference and the baseline

The reference answers and the baseline numbers do not exist yet. Measure both once, on a quiet GPU, with the stock server image. Do not start another GPU program at the same time.

The reference is the answer set of the Q8_0 model. The baseline is the quality of the build that you compare against. The baseline also holds three guard numbers. The runs of the baseline make the numbers.

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

5. Set `MODEL` to the mix8 file in `bench.env`.
6. Make three baseline runs:

   ```sh
   for i in 1 2 3; do ./run_experiment.sh base-$i; done
   ```

7. Make `baseline.json` from the three runs:

   ```sh
   python3 tools/make_baseline.py runs/base-1.json runs/base-2.json runs/base-3.json \
     --note "stock server image, mix8 model, date" --out baseline.json
   ```

8. Make one more run: `./run_experiment.sh base-4`. Make sure that the guard says PASS.
9. Commit `reference.json`, `baseline.json` and the files of the four runs.

`make_baseline.py` stops when the three runs have different quality numbers. Find the cause before you continue.

## Token length test

Use this test after any change to a case.

1. Start a server on the CPU. It needs no GPU flags:

   ```sh
   docker run --rm -p 127.0.0.1:8080:8080 -v "$MODEL_DIR:/models:ro" "$IMAGE" \
     -m /models/h2o-lightning-4b-Q8_0.gguf --host 0.0.0.0 --port 8080 \
     -ngl 0 -c 512 --no-warmup
   ```

2. Count the tokens of each case:

   ```sh
   python3 tools/count_tokens.py cases.json --url http://127.0.0.1:8080
   ```

3. Compare two case files. The exit code is 1 when any case differs:

   ```sh
   python3 tools/count_tokens.py old-cases.json --compare cases.json --url http://127.0.0.1:8080
   ```

## Make the cases again

The script `tools/make_cases.py` makes `cases.json` from `dectest/cases.json` and a fixed seed.

```sh
python3 tools/make_cases.py --check
```

The command fails when `cases.json` differs from the script output. Do not change the seed, the line templates or the order of the random draws. Any such change moves the cases and makes older results incomparable.

## Smoke test

The folder [dectest/](dectest/README.md) holds the 32 short cases as a stand-alone test with 52 decisions. Use it for a fast check of answer quality after a change.
