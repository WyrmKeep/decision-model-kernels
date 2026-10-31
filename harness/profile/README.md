# Code for the profile of phase 3

This folder has the code of phase 3. The code makes the census of the shapes, starts the profile runs, and makes the tables of the runs.

Two data sets have the results. The data set [`2026-10-30-shape-census`](../../data/2026-10-30-shape-census/README.md) has the census. The data set [`2026-10-31-profiles`](../../data/2026-10-31-profiles/README.md) has the raw files, the tables, and the reports of the profile. The [notebook entry](../../docs/notebook/2026-10-31-phase3-where-the-time-goes.md) gives the meaning of the numbers.

The code that makes tables uses only the CPU of the host that runs it. The schedule files use the GPU of the GPU host, through `harness/run_schedule.sh`.

## Contents

| Path | Contents |
|---|---|
| `census/` | The census of the shapes, the kernel launches, the bytes, and the floors of each pass. `run_all.sh` runs the steps. |
| `decomp/` | `analyze_decomp.py` divides the wall time of a request in steps. It reads the client JSON and the server log with the lines of `DMK_TIMING`. |
| `nsys/` | `nsys_export.sh` and `nsys_census.py` make tables from the Nsight Systems captures. They find the kernels, the passes, the gaps, and the CUDA calls of each request. |
| `leads/` | `analyze_kvwin.py` is for the lead #28495. `analyze_childpass.py` is for the lead #30779. |
| `ncu/` | `plan_ncu.py` makes the list of launches for the Nsight Compute runs. `analyze_ncu.py` makes the speed of light of each kernel class. `ncu_source.py` makes the table of the SASS lines of one kernel. |
| `extra/prof/` | The clients that run in the client container of the harness. The folder is the folder `DMK_EXTRA_DIR` of the env file. |
| `sched/` | The schedule files of the runs of the profile. |
| `make_sched.sh` | Writes copies of the schedule files with the work folder of the GPU host. |
| `check_sched.py` | Checks schedule files without a run. |
| `dry_run.sh` | Reads schedule files as `harness/run_schedule.sh` does, and makes no run. |
| `analyze_all.sh` | Runs the programs that make tables, on the results of the schedules. |

The folders `decomp/`, `nsys/`, `leads/`, and `ncu/` have the files `NOTES.txt` and `COLUMNS.tsv`. `NOTES.txt` gives the method, the tests, and the limits. `COLUMNS.tsv` gives the table, the column, the unit, and the meaning of each column.

## Schedules

The profile has 6 schedules. The [data set of the profile](../../data/2026-10-31-profiles/README.md) shows the runs of each one.

| File | Contents |
|---|---|
| `sched/1-decomp.txt` | The decomposition of the request time, with and with no `DMK_TIMING`. |
| `sched/2-nsys.txt` | The runs with Nsight Systems and the control run. |
| `sched/3-leads.txt` | The tests of the leads #28495 and #30779. |
| `sched/4-ncu-smoke.txt`, `sched/4-ncu-core.txt`, `sched/4-ncu-extra.txt` | The runs with Nsight Compute. |
| `sched/4-ncu.txt` | The three files `4-ncu-*.txt` in one file. |

Write the copies of the schedule files. The lines with the profiler have the word `@REMOTE_DIR@`, and the script puts your work folder for it:

```
bash harness/profile/make_sched.sh --remote-dir <work folder of the GPU host> --out <schedule folder>
```

Check the copies. These two commands use no GPU and no host:

```
python3 harness/profile/check_sched.py <schedule folder>/*.txt
bash harness/profile/dry_run.sh <schedule folder>/*.txt
```

Run a schedule. The [harness document](../README.md) gives the env file and the limits:

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule folder>/1-decomp.txt <results>/1-decomp
```

Use the same folder name for the results as for the schedule. `analyze_all.sh` reads the folders `1-decomp`, `2-nsys`, `3-leads`, `4-ncu-smoke`, `4-ncu-core`, and `4-ncu-extra`.

## Tables from the results

Make all tables and reports from the results folder:

```
bash harness/profile/analyze_all.sh [--census <census data set>] [--no-mmq-launches] <results> <output folder>
```

The script writes one folder for each part: `decomp`, `nsys`, `leads-nsys`, `leads-kvwin`, `leads-childpass`, and `ncu`. A step that has no input does not run, and the script lists it at the end. A step that has an error does not stop the other steps. The file `summary.txt` of a folder is the entry point.

The script `make_tables.sh` of the data set of the profile runs `analyze_all.sh`, `ncu/ncu_source.py`, and `make_summary.py`. The data set README gives the commands.

## Programs and folders

| Name | Used by | Meaning |
|---|---|---|
| `DMK_NSYS` | `census/`, `nsys/`, `analyze_all.sh` | The path of the command line tool of Nsight Systems. Without it the code uses `nsys` on `PATH`. |
| `DMK_NSYS_DEB` | `census/`, `nsys/`, `analyze_all.sh` | A `.deb` file of Nsight Systems. The code uses the tool in this file. |
| `DMK_NCU` | `ncu/`, `analyze_all.sh` | The path of the command line tool of Nsight Compute. Without it the code uses `ncu` on `PATH`. |
| `DMK_CENSUS_DATA` | `census/`, `decomp/`, `leads/`, `ncu/`, `nsys/`, `analyze_all.sh` | The folder of the census data set. The default is `data/2026-10-30-shape-census` of this repository. |
| `DMK_LEGACY_TRACES` | `census/`, `nsys/`, `ncu/` | The folder with the three captures of the census, as `.nsys-rep` files or `.sqlite` files. The tests of `nsys/` use them. |
| `DMK_BENCH_DIR` | `decomp/`, `extra/`, `nsys/`, `ncu/`, `check_sched.py`, `dry_run.sh` | The folder with the benchmark. The default is the folder `bench` of this repository. |
| `DMK_GGUF`, `DMK_TOKENIZE` | `census/` | The model file and `llama-tokenize`. Only steps 1 and 2 of the census use them. |
| `DMK_TEST_OUT` | `nsys/`, `ncu/` | The folder where the tests keep their files. |

## Tests

The tests use no GPU. A test that has no private data or no tool does not run the steps that use them. It reports this.

| Command | Checks |
|---|---|
| `python3 harness/profile/decomp/analyze_decomp.py --self-test` | The decomposition with test data that the script makes. It has 36 checks. |
| `python3 harness/profile/leads/analyze_childpass.py --self-test` | The pass structure of the lead #30779 with test logs. |
| `python3 harness/profile/leads/analyze_kvwin.py --self-test` | The KV window of the lead #28495 with test data. |
| `bash harness/profile/nsys/test/run_tests.sh` | The Nsight Systems code and the clients. The steps that use `nsys` do not run when the tool is not there. |
| `python3 harness/profile/ncu/analyze_ncu.py --self-test` | The code that reads the reports, the units, the type of the run, and the speed of light. It has 34 checks with `DMK_NCU` and 29 without it. |
