# Harness

The harness compares builds of llama.cpp. It does each test on a GPU host and starts a new server for each run. The local device controls the GPU host through SSH.

The harness runs the [benchmark](../bench/README.md). The [method](../docs/method.md) gives the tests that compare builds. The [data](../data/README.md) folder holds the results.

## Contents

| Path | Contents |
|---|---|
| `harness.env.example` | An example of the env file. |
| `make_schedule.py` | Makes a schedule. It puts the runs in blocks and in a random sequence. |
| `run_schedule.sh` | Does the runs of a schedule on the GPU host. It puts the results on the local device. |
| `remote/start_server.sh` | Starts the server container for one arm. |
| `remote/run_bench.sh` | Does one run of the benchmark. |
| `remote/run_client.sh` | Does one run of a client script. |
| `remote/run_tool.sh` | Does one run of a `llama` command. |
| `analysis/ab_stats.py` | Calculates the ratio of two arms for each block. It gives the mean and an interval. |
| `analysis/answers_compare.py` | Compares the answers of two arms. |

## How the harness works

A schedule is a text file. Each line of a schedule is one run. The harness does the lines in the sequence of the file.

A run uses one arm. An arm is one set of programs that a test compares with a different set. The harness has two types of arm:

| Arm | Programs |
|---|---|
| `stock` | The programs of the image, in its folder `/app`. |
| `build:<name>` | The programs in the folder `builds/<name>` of the work folder on the GPU host. If this folder has a folder `dist`, the harness uses `builds/<name>/dist`. |

The container of a `build:<name>` arm sees its folder at `/opt/dmk` as read-only. A build name has only letters, numbers, `.`, `_` and `-`. All arms of a schedule use the same image, model file, server flags and GPU.

A `bench` run or a `client` run has these parts:

1. The script removes the containers `dmx-server` and `dmx-bench`, and starts the sampler of the GPU.
2. The script starts a new server container for the arm.
3. The script sends requests to `/health` until the server answers. It sends at most 1,200 requests, with 0.25 s between them.
4. The script starts the client in a second container.
5. The script stops the sampler and removes the containers. It keeps the log of the server.

A `tool` run has no server and no client. The script starts the sampler and one container that does the `llama` command of the line.

## Devices

| Device | Parts |
|---|---|
| Local device | `bash`, Python 3, SSH and `rsync`. |
| GPU host | SSH that works with no input from a person. `rsync`, `bash`, `curl`, `awk`, `setsid` and `nvidia-smi`. Docker with the flag `--runtime=nvidia`. |

Python is not necessary on the GPU host. The client runs in a container with a Python image.

## The env file

The env file holds the values of the harness as environment variables. The name of each environment variable starts with `DMK_`.

1. Make a copy of `harness.env.example` in a folder that is not in the repository.
2. Change each value to the value of your GPU host.
3. Set the environment variable `DMK_ENV` to the path of the copy.
4. Keep the copy on your device only. Do not commit it.

The file has the name of your GPU host and the name of your GPU. The scripts read it as `bash` text. Write one line for each environment variable in the shape `NAME=value`. Put the marks `"` around a value that has spaces.

`run_schedule.sh` puts a copy of the file on the GPU host. The copy does not have the lines of `DMK_HOST`, `DMK_BENCH_DIR` and `DMK_EXTRA_DIR`. The scripts on the GPU host read the copy. An environment variable that you set only in your terminal does not go to the GPU host. Put each environment variable in the env file.

### Environment variables of the example file

| Name | Used by | Meaning |
|---|---|---|
| `DMK_HOST` | `run_schedule.sh` | The name of the GPU host for SSH. The example uses `gpu-host`. |
| `DMK_BENCH_DIR` | `run_schedule.sh` | The folder with the benchmark on the local device. Use the folder `bench` of this repository. The script puts a copy of it on the GPU host. |
| `DMK_REMOTE_DIR` | All scripts | The work folder on the GPU host. The scripts keep their files in it. |
| `DMK_GPU` | The scripts on the GPU host | The name of the GPU, in the shape `GPU-<uuid>`. The scripts give it to Docker as `NVIDIA_VISIBLE_DEVICES` and to `nvidia-smi`. |
| `DMK_IMAGE` | `start_server.sh`, `run_tool.sh` | The image of the server and of the tools. It must start `llama-server` by default, because the arm `stock` uses that. It must have `/app/llama` for `tool` lines. The example names the image with `@sha256:`. This name always gives the same image. |
| `DMK_PY_IMAGE` | `run_bench.sh`, `run_client.sh` | The Python image of the client. The example names the image with `@sha256:`. |
| `DMK_MODELS` | `start_server.sh`, `run_tool.sh` | The folder with the model files on the GPU host. The container sees it at `/models` as read-only. |
| `DMK_MODEL` | `start_server.sh` | The name of the model file in `DMK_MODELS`. The server reads this file. |
| `DMK_SHM_DIR` | `run_tool.sh` | A `tmpfs` folder on the GPU host for large temporary files. The tool container sees it at `/shm`. The default is `/dev/shm/dmk`. |

### Optional environment variables

The scripts read these environment variables, but the example file does not have them.

| Name | Used by | Default | Meaning |
|---|---|---|---|
| `DMK_EXTRA_DIR` | `run_schedule.sh` | None | A folder with more files on the local device. The script puts a copy of it on the GPU host. The client container sees it at `/extra` as read-only. |
| `DMK_PORT` | `start_server.sh`, `run_bench.sh`, `run_client.sh` | `18080` | The port of the server on the GPU host. The server accepts connections from the GPU host only. |
| `DMK_SERVER_ARGS` | `start_server.sh` | `-ngl 999 -c 16384 -b 2048 -ub 2048 -np 3 --kv-unified --no-webui --no-slots` | The flags of `llama-server`. |
| `DMK_CONTAINER` | `start_server.sh` | `dmx-server` | The name of the server container. Do not change it. Refer to the section Limits. |
| `DMK_DOCKER_EXTRA` | `start_server.sh` | None | More flags for the `docker run` command of the server. `bash` divides the value at spaces. |
| `DMK_REF_MODELS` | `run_tool.sh` | None | A second folder with model files on the GPU host. The tool container sees it at `/refmodels` as read-only. |
| `DMK_TOOL_CAPS` | `run_tool.sh` | None | The value of the flag `--cap-add` of Docker. It gives the tool container one capability. |
| `DMK_WRAP_<w>` | `run_schedule.sh` | None | The command of the wrapper `<w>`. A `tool@<w>` line uses it. |

A wrapper starts the tool and records or checks its work. The name `<w>` has only letters, numbers and `_`. The first word of the value is the program of the wrapper in the container. The tool container sees the folder `tools` of the work folder on the GPU host at `/tools` as read-only. Put the program of the wrapper in that folder. Example:

```
DMK_WRAP_memcheck="/tools/compute-sanitizer-12.8.93/compute-sanitizer --tool memcheck"
```

`run_schedule.sh` sets `DMK_WRAP` for a `tool@<w>` line, and `run_tool.sh` reads it. Do not set `DMK_WRAP` in the env file. If the env file has no `DMK_WRAP_<w>`, the `tool@<w>` line starts the tool with no wrapper.

## Make a schedule

`make_schedule.py` writes a schedule to the standard output. Each block has one run of each arm, in a random sequence. A slow change of the host then has the same effect on all arms. The seed sets the sequence. You can make the same schedule again.

```
python3 harness/make_schedule.py --prefix PREFIX --blocks N --seed N \
    --arm SHORT=ARM [--arm SHORT=ARM ...] --kind bench|tool \
    [--args "TEXT"] [--model FILE]
```

| Flag | Necessary | Meaning |
|---|---|---|
| `--prefix` | Yes | The first part of each run name. |
| `--blocks` | Yes | The number of blocks. |
| `--seed` | Yes | The seed of the random sequence. The script uses one sequence for all blocks. |
| `--arm SHORT=ARM` | Yes | One arm. `SHORT` is a short name of the arm, for example `S`. `ARM` is `stock` or `build:<name>`. Use the flag one time for each arm. |
| `--kind` | Yes | The type of line: `bench` or `tool`. |
| `--args` | No | The text after the arm on each line. In the text, `{name}` becomes the run name and `{model}` becomes the model file name. Write `{{` for `{` and `}}` for `}`. The default is no text. |
| `--model` | No | The model file name for `{model}`. The default is `h2o-lightning-4b-rq-Q4_0-mix8.gguf`. |

The first line of the output is a comment. It has the values of `--prefix`, `--blocks`, `--seed`, `--arm` and `--kind`, but not the values of `--args` and `--model`. Each other line has this shape:

```
<kind> <prefix>-b<block>-<SHORT> <arm> <text of --args>
```

The block numbers start at 1. The scripts in the folder `analysis` find the runs from this shape of the run name. `SHORT` must have only letters, numbers and `_`.

Example of a `bench` schedule with 3 blocks. Each run keeps the answers of its quality pass:

```
python3 harness/make_schedule.py --prefix p1-bench --blocks 3 --seed 1 \
    --arm S=stock --arm D=build:b12004-sm86 \
    --kind bench --args "--write-reference {name}"
```

```
# make_schedule.py --prefix p1-bench --blocks 3 --seed 1 --arm S=stock --arm D=build:b12004-sm86 --kind bench
bench p1-bench-b1-D build:b12004-sm86 --write-reference p1-bench-b1-D
bench p1-bench-b1-S stock --write-reference p1-bench-b1-S
bench p1-bench-b2-D build:b12004-sm86 --write-reference p1-bench-b2-D
bench p1-bench-b2-S stock --write-reference p1-bench-b2-S
bench p1-bench-b3-S stock --write-reference p1-bench-b3-S
bench p1-bench-b3-D build:b12004-sm86 --write-reference p1-bench-b3-D
```

Example of a `tool` schedule with 1 block. Each run does the `llama` command `bench`:

```
python3 harness/make_schedule.py --prefix p1-llb --blocks 1 --seed 2 \
    --arm S=stock --arm D=build:b12004-sm86 --kind tool \
    --args "bench -m /models/{model} -ngl 999 -fa 1 -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
```

```
# make_schedule.py --prefix p1-llb --blocks 1 --seed 2 --arm S=stock --arm D=build:b12004-sm86 --kind tool
tool p1-llb-b1-D build:b12004-sm86 bench -m /models/h2o-lightning-4b-rq-Q4_0-mix8.gguf -ngl 999 -fa 1 -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json
tool p1-llb-b1-S stock bench -m /models/h2o-lightning-4b-rq-Q4_0-mix8.gguf -ngl 999 -fa 1 -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json
```

The script makes `bench` lines and `tool` lines only. A person writes `client` lines and `tool@<w>` lines.

## Run a schedule

1. Build the programs of each arm. Put them in `builds/<name>` in the work folder on the GPU host.
2. Pull the two images of the env file on the GPU host with `docker pull`.
3. Set `DMK_ENV` to the path of your env file.
4. Make the schedule file with `make_schedule.py`. Add `client` lines and `tool@<w>` lines if you want them.
5. Run `run_schedule.sh` with the schedule file and a new results folder.
6. Read the output. A line that starts with `!!` shows a run with an exit code other than 0.

```
mkdir -p schedules
python3 harness/make_schedule.py --prefix p1-bench --blocks 3 --seed 1 \
    --arm S=stock --arm D=build:b12004-sm86 \
    --kind bench --args "--write-reference {name}" > schedules/p1-bench.txt
export DMK_ENV=/path/to/harness.env
harness/run_schedule.sh schedules/p1-bench.txt results/p1-bench
```

The section Build the programs of an arm gives the commands for step 1. If `DMK_IMAGE` is not on the GPU host, the first run includes the time to pull it in `startup_s`. Step 2 prevents this.

## The schedule script

```
DMK_ENV=<env file> harness/run_schedule.sh <schedule file> <results folder>
```

You give the script two values: the schedule file and the results folder. It has no flags. It does not read `KEY=VALUE` values from the command line. It gets all other values from the env file.

The script does these tasks in this sequence:

1. It makes the results folder. It makes the folders `harness`, `bench` and `runs` in the work folder on the GPU host.
2. It puts a copy of the folder `remote` in `harness`, and a copy of `DMK_BENCH_DIR` in `bench`. If you set `DMK_EXTRA_DIR`, it puts a copy of it in `extra`. Each copy removes the files on the GPU host that the local folder does not have.
3. It writes the copy of the env file to `harness/run.env` on the GPU host.
4. For each line of the schedule, it starts the script on the GPU host through SSH. It writes a line with the time in UTC, the type, the run name and the text after the run name.
5. After each line, it puts a copy of the files `<name>.*` from the folder `runs` of the GPU host in the results folder.
6. At the end, it puts a copy of the schedule file in the results folder and writes `done, <N> failed`.

The script does not use empty lines and lines that start with `#`. The other lines have one of these shapes:

| Line | Script on the GPU host | Meaning |
|---|---|---|
| `bench <name> <arm> [bench.py flags]` | `run_bench.sh` | One run of the benchmark client `bench.py`. |
| `client <name> <arm> <script> [flags]` | `run_client.sh` | One run of a Python script. The script is in `/bench` or in `/extra`. |
| `tool <name> <arm> <command> [flags]` | `run_tool.sh` | One run of the `llama` command, for example `bench` or `perplexity`. |
| `tool@<w> <name> <arm> <command> [flags]` | `run_tool.sh` | The same as `tool`, with the wrapper of `DMK_WRAP_<w>`. |

`bash` on the GPU host reads the text after the run name. Put the marks `"` around a flag that has a space. If a run gives an exit code other than 0, the script writes `!! <name> failed`, counts the run, and goes on with the next line.

The script stops at the first line that has an unknown type or a wrapper name with an incorrect character. It does not put a copy of the schedule file in the results folder in this case. The exit code of the script:

| Code | Meaning |
|---|---|
| 0 | All runs gave exit code 0. |
| 1 | One run or more gave an exit code other than 0. |
| 2 | A line has an unknown type or an incorrect wrapper name. |

## The scripts on the GPU host

`run_schedule.sh` starts these scripts. The environment variables of the copy `run.env` are set when each script starts.

### Server script

Command: `start_server.sh <arm> [server flags]`. The script starts the container `dmx-server` with `docker run -d`. It does not wait for the server.

Docker connects the port `DMK_PORT` of the GPU host to the port 8080 of the container. The container has the GPU `DMK_GPU` and the folder `DMK_MODELS` at `/models`. The flags of the server are `-m /models/$DMK_MODEL --host 0.0.0.0 --port 8080`, then `DMK_SERVER_ARGS`, then the flags that you give.

The arm `stock` uses the entry point of the image. An arm `build:<name>` starts `/opt/dmk/llama-server` and sets `LD_LIBRARY_PATH` to `/opt/dmk`. The script gives exit code 2 for an unknown arm, an incorrect build name, or a folder with no `llama-server`.

### Benchmark script

Command: `run_bench.sh <run-name> <arm> [bench.py flags]`. The script does the five parts of the section How the harness works. It gives the flags to `bench.py`. It also gives `--startup-s` to `bench.py`. This is the time from before the start of the container until the first answer of `/health`.

The client container has the folder `bench` of the work folder at `/bench` as read-only. It has the folder `runs` at `/runs`. `bench.py` writes `/runs/<name>.json`.

The script gives exit code 2 if the file `<name>.json` is in the folder `runs`. It gives exit code 3 if the container does not start or the server does not answer `/health`. In all other cases the exit code is the exit code of `bench.py`.

### Client script

Command: `run_client.sh <run-name> <arm> <script> [script flags]`. The script does the same parts as `run_bench.sh`. It starts `python -u <script>` with the flags in the client container.

`<script>` is a path in the container, `/bench/...` or `/extra/...`. In the flags, `{name}` becomes the run name. The client container has the folder `runs` at `/runs`, and it can write there. A client script must write its files to `/runs/{name}.*`. Then `run_schedule.sh` puts them in the results folder.

The script does not tell the client the port of the server. If you change `DMK_PORT`, give the port in the flags. The script gives exit code 2 if the file `<name>.client.out` is in the folder `runs`. In all other cases the exit code is the exit code of the client script.

### Tool script

Command: `run_tool.sh <run-name> <arm> <command> [flags]`. The script starts the container `dmx-tool` with the `llama` command and the flags. The program is `/app/llama` for `stock` and `/opt/dmk/llama` for `build:<name>`.

The tool container has these paths:

| Path in the container | Folder on the GPU host | Access |
|---|---|---|
| `/models` | `DMK_MODELS` | Read-only |
| `/work` | The folder `work` of the work folder | Read and write |
| `/shm` | `DMK_SHM_DIR` | Read and write |
| `/refmodels` | `DMK_REF_MODELS`, if you set it | Read-only |
| `/tools` | The folder `tools` of the work folder, for a wrapper only | Read-only |

With a wrapper, the program of the wrapper is the entry point. It starts the `llama` program with the command and the flags. The script gives exit code 2 if the file `<name>.out` is in the folder `runs`. In all other cases the exit code is the exit code of the tool.

### Files of one run

The scripts write these files in the folder `runs` of the work folder. `<name>` is the run name.

| File | Written by | Contents |
|---|---|---|
| `<name>.json` | `run_bench.sh` | The result of `bench.py`. The [data file document](../bench/DATA.md) gives its fields. |
| `<name>.reference.json` | `run_bench.sh` | The answers of the quality pass. Only if the line has `--write-reference <name>`. |
| `<name>.client.out` | `run_client.sh` | The standard output and the standard error of the client script. |
| `<name>.out` | `run_tool.sh` | The standard output of the tool. With `-o json` it has JSON. |
| `<name>.err` | `run_tool.sh` | The standard error of the tool. |
| `<name>.meta` | The three scripts | The facts of the run. The table below gives its lines. |
| `<name>.smi` | The three scripts | The samples of the sampler of the GPU. One line each 250 ms. |
| `<name>.server.log` | `run_bench.sh`, `run_client.sh` | The log of the server container. |
| `<name>.launch.log` | `run_bench.sh`, `run_client.sh` | The output of `start_server.sh`. |
| `<name>.load0` | `run_bench.sh`, `run_client.sh` | The line of `/proc/loadavg` before the server starts. |
| `<name>.load1` | `run_bench.sh`, `run_client.sh` | The line of `/proc/loadavg` after the run. |

`bench.py` makes the name of `<name>.reference.json` from the text that you give to `--write-reference`. `run_schedule.sh` puts the file in the results folder only if this text is the run name.

Lines of the file `<name>.meta`:

| Run | Lines |
|---|---|
| `bench`, `client` | `arm=<arm>`. `date_utc=<time>`. One line with the driver version, the power limit, and the causes that limit the clocks. |
| `tool` | `arm=<arm> command=<command> args=<flags>`. `date_utc=<time>`. One line of `/proc/loadavg`. One line with the driver version and the power limit. After the tool stops: `rc=<exit code>` and one line of `/proc/loadavg`. |

A line of `<name>.smi` has five values. A `,` is between two values.

| Position | Quantity | Unit |
|---|---|---|
| 1 | Unix time of the sample, with nine decimal places | s |
| 2 | `memory.used` of the GPU | MiB |
| 3 | `power.draw` of the GPU | W |
| 4 | `pstate` of the GPU | None |
| 5 | `clocks.sm` of the GPU | MHz |

## Compare the results

The two scripts read the results folder of `run_schedule.sh`. They use the Python standard library only.

### `ab_stats.py`

```
python3 harness/analysis/ab_stats.py RESULTS_FOLDER --prefix PREFIX --a SHORT --b SHORT \
    [--margin PERCENT] [--boot N] [--seed N] [--tsv FILE]
```

The script reads the runs `<PREFIX>-b<block>-<SHORT>`. It uses the blocks that have a run of arm A and a run of arm B. For each metric and each block, it calculates the ratio of B to A. The estimate is the mean of the ratios.

The 95 % interval comes from a bootstrap. The script makes `--boot` samples of the ratios. Each sample has as many ratios as there are blocks. The interval is from the 2.5th percentile to the 97.5th percentile of the means of the samples.

| Flag | Default | Meaning |
|---|---|---|
| `--prefix` | None | The prefix of the run names. |
| `--a` | None | The short name of arm A, the reference arm. |
| `--b` | None | The short name of arm B. |
| `--margin` | None | A limit for the difference, in %. The script then adds the column `within_margin`. |
| `--boot` | `10000` | The number of bootstrap samples. |
| `--seed` | `12345` | The seed of the bootstrap. |
| `--tsv` | None | A file for the table. The script writes the same table with four decimal places. |

The script uses these metrics. It uses a metric only if each block that it uses has a value for it.

| Metric | File | Unit | Better if | Meaning |
|---|---|---|---|---|
| `lat_ms` | `<name>.json` | ms | Lower | The primary metric of `bench.py`. It is the median of the mean request times of the timed passes. |
| `short_ms`, `medium_ms`, `long_ms` | `<name>.json` | ms | Lower | The mean time of the short requests, of the medium requests and of the long requests. |
| `rps8` | `<name>.json` | requests/s | Higher | The number of requests that finish each second in the phase `tput`. The name shows the default of 8 requests at one time. |
| `pp<N>_ts` | `<name>.out` | tokens/s | Higher | The speed of prefill for a prompt of `N` tokens. The script reads it from the output of `llama bench`. |

If the file `<name>.json` is in the results folder, the script reads it. If not, the script reads `<name>.out` as the JSON output of `llama bench`.

The script writes the table to the standard output as TSV, with three decimal places. If no block has a run of arm A and a run of arm B, the script stops with the text `no complete blocks`.

| Column | Unit | Meaning |
|---|---|---|
| `metric` | None | The name of the metric. |
| `blocks` | Blocks | The number of blocks that the script used. |
| `a_mean` | Unit of the metric | The mean of arm A for these blocks. |
| `b_mean` | Unit of the metric | The mean of arm B for these blocks. |
| `diff_pct` | % | 100 times (the mean of the ratios minus 1). A value less than 0 for a time metric shows that B is faster. |
| `ci_lo_pct` | % | The lowest value of the 95 % interval of the difference. |
| `ci_hi_pct` | % | The highest value of the 95 % interval of the difference. |
| `within_margin` | `True` or `False` | `True` if the interval is in the range from minus the limit to plus the limit. The script writes this column only with `--margin`. |

The values `a_mean` and `b_mean` are the means of the values of each arm. The ratio of `b_mean` to `a_mean` can be different from the mean of the ratios of the blocks.

### `answers_compare.py`

```
python3 harness/analysis/answers_compare.py --a 'RESULTS_FOLDER/PREFIX-b*-S.reference.json' \
    --b 'RESULTS_FOLDER/PREFIX-b*-D.reference.json'
```

Put the marks `'` around each path. If your terminal expands the `*`, the script gets incorrect flags. The files are the `<name>.reference.json` files that `bench.py --write-reference` writes. Each file has the answers of one quality pass: one answer for each question of each case.

The script writes one JSON object to the standard output. It compares the values of the answers. For a yes or no question, the top answer is yes if the probability is 0.5 or more. For a `choice` question, the top answer is the value of `choice` in the answer. A difference of probabilities is the largest difference of all probabilities of the answer.

| Key | Meaning |
|---|---|
| `files_a`, `files_b` | The file names of each arm, in the sequence of their names. |
| `decisions` | The number of decisions in one file. |
| `self_repeat_a`, `self_repeat_b` | `true` if all files of the arm have the same values. |
| `self_a_max_dp`, `self_b_max_dp` | The largest difference of a probability between the first file of the arm and a file of the arm. |
| `cross_identical_all` | `true` if all files of arm A and all files of arm B contain the same values. |
| `cross_max_abs_dp` | The largest difference of a probability between a file of A and a file of B, for all decisions. |
| `cross_top_flips` | The decisions, as `<case>.<question>`, with a different top answer in one pair of files or more. |

If a path gives no file, the script stops with the text `no files for <path>`.

## Build the programs of an arm

The harness has no build script. You build llama.cpp with CMake. Then you put the programs on the GPU host in the structure that the harness uses.

| Path in the work folder on the GPU host | Contents |
|---|---|
| `builds/<name>/dist/`, or `builds/<name>/` if the folder has no `dist` | The program `llama-server`, the program `llama`, and the shared libraries. |

A `bench` line and a `client` line use `llama-server`. A `tool` line uses `llama`. The harness does not use `llama-bench` or `llama-perplexity`. A `tool` line uses the commands `llama bench` and `llama perplexity` for the same tasks. [INFERENCE: The container gets the CUDA libraries from the image. Use the same CUDA version in the build.]

1. Get the fork of llama.cpp with `git clone`. Check out the commit for your test.
2. Run `cmake` with the flags of the table below.
3. Build the programs with `cmake --build`.
4. Put `llama-server`, `llama` and the shared libraries in a new folder `dist`.
5. Put a copy of `dist` in `builds/<name>/dist` in the work folder on the GPU host.
6. Make sure that the programs start. Use a `tool` line for the command `version`.
7. Write the full CMake command and the commit in the README of each data set.

The baseline build uses the tag `b12004` (commit `3c809f4`).

```
git clone https://github.com/WyrmKeep/llama.cpp
cd llama.cpp
git checkout <commit>
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
mkdir dist
cp -a build/bin/llama-server build/bin/llama build/bin/*.so* dist/
ssh gpu-host mkdir -p /srv/dmk/builds/b12004-sm86/dist
rsync -a dist/ gpu-host:/srv/dmk/builds/b12004-sm86/dist/
```

| Flag | Meaning |
|---|---|
| `-DGGML_CUDA=ON` | Builds the CUDA backend. |
| `-DCMAKE_CUDA_ARCHITECTURES=86` | Makes code for the architecture `sm_86` only. This is the architecture of the GPU of the project. |
| `-DCMAKE_BUILD_TYPE=Release` | Makes the release build. |

This build is for one GPU architecture only. A build can have more flags. Record the flags of each build in each data set.

The line below is the `tool` line for step 6. Put it in a schedule file and run `run_schedule.sh`. The file `check-1.out` in the results folder then shows the version and the commit of the build.

```
tool check-1 build:b12004-sm86 version
```

## Data of the harness

A test gives these files. The [data](../data/README.md) document gives the rules for a data set.

| Output | Folder in a data set |
|---|---|
| The `<name>.*` files of each run, in the results folder | The folder `raw/`, with no change |
| The copy of the schedule file | The folder `raw/` |
| The table of `ab_stats.py --tsv` | The folder `tables/` |
| The output of `answers_compare.py` | The folder `tables/`, as a JSON file |

Use `tools/publish/scan-dir.sh` on the folder of the data set before you add it to the repository.

## Limits

- A run name must be new. If the folder `runs` of the GPU host has the result file of that name, the run gives exit code 2. The harness does not remove files from that folder.
- Each run removes the containers `dmx-server` and `dmx-bench`. A `tool` run removes `dmx-tool`. Do not use these names for other containers.
- `run_bench.sh` and `run_client.sh` use the name `dmx-server` for the log, the check of the server and the removal. If you change `DMK_CONTAINER`, these scripts cannot find the server container.
- The harness does not change the power limit or the clocks of the GPU. It records the power limit in each `.meta` file.
- Use the GPU for no other work during a schedule.
