# Data files of the benchmark

This document lists the fields, units and meaning of each data file in this folder. The README of the benchmark explains how to use the files.

| File | Made by | Holds |
|---|---|---|
| `cases.json` | `tools/make_cases.py` | The 48 cases and their labels |
| `dectest/cases.json` | By hand | The 32 short cases of the smoke test |
| `reference.json` | `bench.py --write-reference` | The answers of the reference model |
| `baseline.json` | `tools/make_baseline.py` | The guard numbers of the baseline build |
| `<run>.json` | `bench.py` | The result of one run |
| `<run>.smi` | `remote_run.sh` | The GPU samples of one run |

## cases.json

The file is a JSON list. Each item is one case.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `id` | string | None | The unique name of the case. A short case has the form `triage00`, `check00`, `style00` or `commit00`. A syslog case has the form `medium00-storage` or `long03-none`. |
| `task` | string | None | One of `triage`, `check`, `style`, `commit`, `syslog-medium` and `syslog-long`. |
| `size` | string | None | One of `short`, `medium` and `long`. The client reports the results for each size. |
| `state` | string | None | The text that the questions ask about. A line break is `\n`. |
| `questions` | object | None | The questions. The key is the question name. The value is a question object. |
| `expect` | object | None | The labels. The key is a question name. The value is the correct answer. |

A syslog state has a header line and then one log line for each event. A log line has this form: the time `Oct 30 HH:MM:SS`, the host name `Homelab`, the program with its process number, and the message.

### Question object

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | string | None | `noul` for a yes or no question. `choice` for a question with a fixed list of answers. |
| `instructions` | string | None | The text of the question. |
| `criteria` | object | None | Only for `choice`. The key is a choice name. The value is the description of the choice. |

The server calls a yes or no question `noul`. The file `dectest/cases.json` uses the name `bool` for the same type. When the script builds `cases.json`, it writes `noul` for each `bool`.

### Label values

| Question type | Label type | Meaning |
|---|---|---|
| `noul` | boolean | `true` when the correct answer is yes |
| `choice` | string | The name of the correct choice. It is a key of `criteria`. |

### Label counts

| Question | Cases | Label counts |
|---|---:|---|
| `category` | 12 | `routine` 4, `availability` 3, `backup` 2, `storage` 1, `security` 1, `hardware` 1 |
| `needs_action` | 12 | true 6, false 6 |
| `done` | 8 | true 4, false 4 |
| `relative_date` | 8 | true 2, false 6 |
| `secret` | 8 | true 2, false 6 |
| `matches` | 4 | true 2, false 2 |
| `disk_error` | 16 | true 4, false 12 |
| `oom_kill` | 16 | true 4, false 12 |
| `worst` | 16 | `storage` 4, `memory` 4, `network` 4, `none` 4 |

### Size of the cases

| Size | Cases | State length (characters) | State length (tokens) | Lines in the state |
|---|---:|---|---|---|
| `short` | 32 | 41 to 194, mean 107 | 11 to 84, mean 37 | 1 to 5 |
| `medium` | 8 | 2,054 to 2,513, mean 2,339 | 985 to 1,307, mean 1,202 | 21 to 25 |
| `long` | 8 | 5,732 to 6,497, mean 6,107 | 2,899 to 3,414, mean 3,126 | 59 to 63 |

Source: token counts with the tokenizer of the model, made with `tools/count_tokens.py`.

## reference.json

The file holds the answers of the reference model to the quality requests. The reference model is the Q8_0 file on the stock server build.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `label` | string | None | The name of the reference run |
| `answers` | object | None | The key is a case `id`. The value is an object with one answer for each question name. |

An answer has these fields:

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | string | None | `noul` or `choice` |
| `noul` | number | Probability, 0 to 1 | Only for `noul`. The probability of yes. |
| `choice` | string | None | Only for `choice`. The choice with the highest probability. |
| `probabilities` | object | Probability, 0 to 1 | Only for `choice`. The key is a choice name. |
| `confidence` | number | Probability, 0 to 1 | Only for `choice`. The probability of the chosen answer. |

The slot is empty until you measure it. An empty slot holds `"status": "not-measured"` and a `note`. The client skips the agreement metrics while the marker is present.

## baseline.json

The file holds the numbers of the baseline build. The guard of `bench.py` reads `correct`, `agree` and `mean_dp`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `status` | string | None | `measured`. An empty slot holds `not-measured`. |
| `note` | string | None | A description of the build, the model, the flags and the date |
| `runs` | list of strings | None | The names of the baseline runs |
| `correct` | integer | Decisions, of 100 | The lowest `correct` of the runs |
| `agree` | number | Fraction, 0 to 1 | The lowest `agree` of the runs |
| `mean_dp` | number | Probability, 0 to 1 | The highest `mean_dp` of the runs |
| `lat_ms` | list of numbers | ms | `lat_ms` of each run |
| `short_ms` | list of numbers | ms | The mean latency of the short cases in each run |
| `rps` | number | Requests/s | The median throughput of the runs |
| `conc` | integer | Requests | The requests in flight during the throughput phase |
| `vram_mib` | number | MiB | The largest GPU memory use of the runs, or `null` |

## Result file of a run

`bench.py` writes this file. Times are in milliseconds unless the field says otherwise.

| Field | Meaning |
|---|---|
| `meta` | `started` (Unix time, s), `passes`, `conc`, `startup_s` (s), `cases`, `decisions` |
| `phases` | For each phase name, the start and end as Unix time in seconds. `summarize.py` uses them to select the GPU samples. |
| `quality.model` | The model name that the server reports |
| `quality.correct`, `quality.total` | Decisions that equal the label, and the number of decisions |
| `quality.correct_by_size` | The same count for each size, as text such as `22/52` |
| `quality.input_tokens_mean` | The mean `usage.input_tokens` for each size |
| `quality.agree`, `quality.agree_n` | Agreement with the reference, as a fraction and as a count |
| `quality.mean_dp`, `quality.max_dp` | Mean and largest probability difference to the reference |
| `quality.flips` | The decisions whose top answer differs from the reference |
| `seq.lat_ms` | The primary metric |
| `seq.pass_means` | The mean latency of each pass |
| `seq.spread_pct` | The range of the pass means divided by their median, in % |
| `seq.all`, `seq.by_size.<size>` | `n`, `mean`, `p50`, `p95` and `max` of the request times |
| `seq.reconnects` | The number of times that the client opened the connection again |
| `tput` | `conc`, `requests`, `wall_s` (s), `rps`, `decisions_per_s` and `lat` (request time statistics) |
| `cold_ms` | Optional. The time of each cold request |
| `errors` | The list of failed requests. An empty list means success. |
| `guard` | The result of each guard, as true or false |
| `guard_pass` | True when all guards pass. `null` when the baseline or the reference has no measured data. |

## Sampler file of a run

`remote_run.sh` writes one line every 250 ms. A line has five values with a comma between them.

| Column | Unit | Meaning |
|---|---|---|
| 1 | s | Unix time of the sample |
| 2 | MiB | `memory.used` of the GPU |
| 3 | W | `power.draw` of the GPU |
| 4 | None | P-state, for example `P2` |
| 5 | MHz | `clocks.sm` of the GPU |

## Other files of a run

| File | Made by | Holds |
|---|---|---|
| `<run>.out` | `run_experiment.sh` | The text that the runner printed |
| `<run>.launch.log` | `remote_run.sh` | The output of the launch script |
| `<run>.server.log` | `remote_run.sh` | The log of the server container, with the time of each request |
| `<run>.load0`, `<run>.load1` | `remote_run.sh` | The load average of the GPU machine before and after the run |
| `<run>.launch.sh` | `run_experiment.sh` | A copy of the launch script that the run used |

## Provenance

Each run needs a record of the build that made it. The record lists the commit, the build flags, the driver, the GPU, the power limit, the clocks, the date and the command. Keep the record next to the run files, in the data folder of the project. The launch script copy holds the image and the server flags.
