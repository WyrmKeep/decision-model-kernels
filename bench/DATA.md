# Data files of the benchmark

This document gives the fields, the units, and the meaning of each data file in this folder. The README of the benchmark tells you how to use the files.

| File | Made by | Holds |
|---|---|---|
| `cases.json` | `tools/make_cases.py` | The 48 cases and their labels |
| `dectest/cases.json` | A person | The 32 `short` cases of the smoke test |
| `reference.json` | `bench.py --write-reference` | The reference answers of the Q8_0 model |
| `baseline.json` | `tools/make_baseline.py` | The guard numbers of the baseline build |
| `<run>.json` | `bench.py` | The result of one run |
| `<run>.smi` | `remote_run.sh` | The GPU samples of one run |

## cases.json

The file is a JSON list. Each item is one case.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `id` | string | None | The name of the case. Each case has a different name. A `short` case has a name, for example `triage00`, `check00`, `style00` or `commit00`. A syslog case has a name, for example `medium00-storage` or `long03-none`. |
| `task` | string | None | One of `triage`, `check`, `style`, `commit`, `syslog-medium` and `syslog-long`. |
| `size` | string | None | One of `short`, `medium` and `long`. The client reports the results for each of these values. |
| `state` | string | None | The text that the questions are about. A line break is `\n`. |
| `questions` | object | None | The questions. The key is the question name. The value is a question object. |
| `expect` | object | None | The labels. The key is a question name. The value is the correct answer. |

A syslog state has a line at the start, and then one log line for each entry of the syslog. The line at the start gives information about the log. A log line has four parts in sequence. The parts are the time `Oct 30 HH:MM:SS`, the host name `Homelab`, the program with the number of its run, and the text of the entry. An example of the program part is `ntpd[15401]`.

### Question object

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | string | None | `noul` for a yes or no question. `choice` for a question with a list of possible answers. |
| `instructions` | string | None | The text of the question. |
| `criteria` | object | None | Only for `choice`. The key is the name of a possible answer. The value is the description of that answer. |

The server uses the name `noul` for a yes or no question. The file `dectest/cases.json` uses the name `bool` for the same type. When the script builds `cases.json`, it writes `noul` for each `bool`.

### Label values

| Question type | Label type | Meaning |
|---|---|---|
| `noul` | `true` or `false` | `true` when the correct answer is yes |
| `choice` | string | The name of the correct answer. It is a key of `criteria`. |

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

Source: the `expect` fields of `cases.json`.

### Length of the cases

| `size` | Cases | Characters, smallest | Characters, largest | Characters, mean | Tokens, smallest | Tokens, largest | Tokens, mean | Lines, smallest | Lines, largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short` | 32 | 41 | 194 | 107 | 11 | 84 | 37 | 1 | 5 |
| `medium` | 8 | 2,054 | 2,513 | 2,339 | 985 | 1,307 | 1,202 | 21 | 25 |
| `long` | 8 | 5,732 | 6,497 | 6,107 | 2,899 | 3,414 | 3,126 | 59 | 63 |

Source: the `state` fields of `cases.json`. The tokenizer of the model gives the token numbers, and the script `tools/count_tokens.py` counts them.

## reference.json

The file holds the reference. The reference has the answers of the Q8_0 model file to the requests of the `quality` phase. The server build that made the answers has none of the changes of this project.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `label` | string | None | The name of the reference run |
| `answers` | object | None | The key is a case `id`. The value is an object with one answer for each question name. |

An answer has these fields:

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `type` | string | None | `noul` or `choice` |
| `noul` | number | Probability, 0 to 1 | Only for `noul`. The probability of yes. |
| `choice` | string | None | Only for `choice`. The possible answer with the highest probability. |
| `probabilities` | object | Probability, 0 to 1 | Only for `choice`. The key is the name of a possible answer. |
| `confidence` | number | Probability, 0 to 1 | Only for `choice`. The probability of the answer in `choice`. |

Until you measure the reference, the file has no answers. It holds `"status": "not-measured"` and a `note`. While the file has this mark, the client does not calculate the metrics that compare the answers with the reference.

## baseline.json

The file holds the numbers of the baseline build. The guard of `bench.py` reads `correct`, `agree` and `mean_dp`.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `status` | string | None | `measured`. A file that has no measurement holds `not-measured`. |
| `note` | string | None | A description of the build, the model, the flags and the date |
| `runs` | list of strings | None | The names of the baseline runs |
| `correct` | integer | Decisions, of 100 | The lowest `correct` of the runs |
| `agree` | number | Part of the decisions, 0 to 1 | The lowest `agree` of the runs |
| `mean_dp` | number | Probability, 0 to 1 | The highest `mean_dp` of the runs |
| `lat_ms` | list of numbers | ms | `lat_ms` of each run |
| `short_ms` | list of numbers | ms | The mean latency of the short requests in each run |
| `rps` | number | Requests/s | The median throughput of the runs |
| `conc` | integer | Requests | The requests in flight during the throughput phase |
| `vram_mib` | number | MiB | The largest GPU memory use of the runs, or `null` |

## Result file of a run

`bench.py` writes this file. Times are in milliseconds. If a field has a different unit, the table gives that unit.

| Field | Meaning |
|---|---|
| `meta` | `started` (Unix time, s), `passes`, `conc`, `startup_s` (s), `cases`, `decisions` |
| `phases` | For each phase name, the start and the end as Unix time in seconds. `summarize.py` uses them to select the GPU samples. |
| `quality.model` | The model name that the server reports |
| `quality.correct`, `quality.total` | The number of correct decisions, and the total number of decisions |
| `quality.correct_by_size` | The same number for each `size` value, as text, for example `22/52` |
| `quality.input_tokens_mean` | The mean `usage.input_tokens` for each `size` value |
| `quality.agree`, `quality.agree_n` | The part of the decisions with the same top answer as the reference. `agree` is a number from 0 to 1. `agree_n` is text with the number of these decisions and the total. |
| `quality.mean_dp`, `quality.max_dp` | The mean and the largest difference between the probabilities of the run and of the reference |
| `quality.flips` | The decisions with a top answer that is different from the reference |
| `seq.lat_ms` | The primary metric. It is the median of `seq.pass_means`. |
| `seq.pass_means` | The mean latency of each timed pass |
| `seq.spread_pct` | The range of the means of the timed passes, divided by their median, in % |
| `seq.all`, `seq.by_size.<size>` | `n`, `mean`, `p50`, `p95` and `max` of the latencies of the requests |
| `seq.reconnects` | The number of times that the client opened the connection again |
| `tput` | `conc`, `requests`, `wall_s` (s), `rps`, `decisions_per_s` and `lat` (the same fields as `seq.all`) |
| `cold_ms` | Optional. The time of each cold request |
| `errors` | The list of the requests with an error. An empty list means that the run has no error. |
| `guard` | The result of each guard, as true or false |
| `guard_pass` | True when all guards are true. `null` when the baseline or the reference has no data of a measurement. |

## Sampler file of a run

`remote_run.sh` writes one line at an interval of 250 ms. A line has five values. The values have the character `,` between them.

| Column | Unit | Meaning |
|---|---|---|
| 1 | s | The Unix time of the sample |
| 2 | MiB | `memory.used` of the GPU |
| 3 | W | `power.draw` of the GPU |
| 4 | None | The performance level of the GPU (`pstate`), for example `P2` |
| 5 | MHz | `clocks.sm` of the GPU |

## Other files of a run

| File | Made by | Holds |
|---|---|---|
| `<run>.out` | `run_experiment.sh` | The text that the script shows in the terminal |
| `<run>.launch.log` | `remote_run.sh` | The output of the launch script |
| `<run>.server.log` | `remote_run.sh` | The log of the server container, with the time of each request |
| `<run>.load0`, `<run>.load1` | `remote_run.sh` | The load average of the GPU host before and after the run |
| `<run>.launch.sh` | `run_experiment.sh` | A copy of the launch script that the run used |

## Build information of a run

Each run must have information about the build that made it. The information has the commit, the build flags, the driver, the GPU, the power limit, the clocks, the date, and the command. Keep the information with the run files, in the data folder of the project. The copy of the launch script has the image and the server flags.
