# Harness

The harness compares builds of llama.cpp. It does each test on a GPU host and starts a new server for each run. The local device controls the GPU host through SSH. The harness also has a script that builds llama.cpp.

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
| `remote/run_soak.sh` | Does a soak test of one arm. |
| `remote/san_wrap.sh` | The wrapper of the server for the sanitizer phase of the soak test. |
| `remote/san_client.py` | The client of the sanitizer phase of the soak test. |
| `build/build.sh` | Builds llama.cpp in a container on the local device. |
| `build/Containerfile.builder` | The file of the builder image. |
| `build/sass_compare.py` | Compares the SASS of two builds. |
| `analysis/ab_stats.py` | Calculates the ratio of two arms for each block. It gives the mean and an interval. |
| `analysis/answers_compare.py` | Compares the answers of two arms. |
| `analysis/soak_check.py` | Checks the results of a soak test. It writes `PASS` or `FAIL` for each condition. |
| `ablate/` | The tools of the ablation: the script that makes the schedules, the client, the script that makes the tables, the SASS check and the files of the arms. Its [README](ablate/README.md) gives the steps. |
| `profile/` | The code of phase 3: the census, the profile runs, and the scripts that make the tables. The [README](profile/README.md) gives the commands. |

## How the harness works

A schedule is a text file. Each line of a schedule is one run. The harness does the lines in the sequence of the file.

A run uses one arm. An arm is one set of programs that a test compares with a different set. The harness has two types of arm:

| Arm | Programs |
|---|---|
| `stock` | The programs of the image, in its folder `/app`. |
| `build:<name>` | The programs in the folder `builds/<name>` of the work folder on the GPU host. If this folder has a folder `dist`, the harness uses `builds/<name>/dist`. |

The container of a `build:<name>` arm sees its folder at `/opt/dmk` as read-only. The program starts in the folder `/opt/dmk`, and `LD_LIBRARY_PATH` is `/opt/dmk`. The libraries of the build load, and the libraries in `/app` do not. A build name has only letters, numbers, `.`, `_` and `-`. All arms of a schedule use the same image, model file, server flags and GPU, if no line has an override.

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
| Local device | `bash`, Python 3, SSH and `rsync`. `podman` for the script `build/build.sh`. |
| GPU host | SSH that works with no input from a person. `rsync`, `bash`, `curl`, `awk`, `setsid` and `nvidia-smi`. Docker with the flag `--runtime=nvidia`. For a soak test, the host must also have `dmesg`, `flock`, `gzip`, `sed` and `timeout`, and the user must read `dmesg`. |

Python is not necessary on the GPU host. The client runs in a container with a Python image.

## The env file

The env file holds the values of the harness as environment variables. The name of each environment variable starts with `DMK_`.

1. Make a copy of `harness.env.example` in a folder that is not in the repository.
2. Change each value to the value of your GPU host.
3. Set the environment variable `DMK_ENV` to the path of the copy.
4. Keep the copy on your device only. Do not commit it.

The file has the name of your GPU host and the name of your GPU. The scripts read it as `bash` text. Write one line for each environment variable in the shape `NAME=value`. Put the marks `"` around a value that has spaces.

`run_schedule.sh` puts a copy of the file on the GPU host. The copy does not have the lines of `DMK_HOST`, `DMK_BENCH_DIR` and `DMK_EXTRA_DIR`. The scripts on the GPU host read the copy. An environment variable that you set only in your terminal does not go to the GPU host. Put each environment variable in the env file, or in an override of a schedule line.

### Environment variables of the example file

| Name | Used by | Meaning |
|---|---|---|
| `DMK_HOST` | `run_schedule.sh` | The name of the GPU host for SSH. The example uses `gpu-host`. |
| `DMK_BENCH_DIR` | `run_schedule.sh` | The folder with the benchmark on the local device. Use the folder `bench` of this repository. The script puts a copy of it on the GPU host. |
| `DMK_REMOTE_DIR` | All scripts | The work folder on the GPU host. The scripts keep their files in it. |
| `DMK_GPU` | The scripts on the GPU host | The name of the GPU, in the shape `GPU-<uuid>`. The scripts give it to Docker as `NVIDIA_VISIBLE_DEVICES` and to `nvidia-smi`. |
| `DMK_IMAGE` | `start_server.sh`, `run_tool.sh` | The image of the server and of the tools. It must have `/app/llama-server` for the arm `stock` and `/app/llama` for `tool` lines. The example names the image with `@sha256:`. This name always gives the same image. |
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
| `DMK_TOOL_CAPS` | `run_tool.sh`, `start_server.sh` | None | The value of the flag `--cap-add` of Docker. It gives the tool container, and the server container, one capability. |
| `DMK_SERVER_WRAP` | `start_server.sh` | None | The command of a wrapper for `llama-server`, for example a profiler. The section Server script gives the rules. |
| `DMK_STOP_TIMEOUT` | `run_client.sh` | `600` | The longest time, in seconds, that `run_client.sh` waits while it stops the server container. It uses this value only with `DMK_SERVER_WRAP`. |
| `DMK_WRAP_<w>` | `run_schedule.sh` | None | The command of the wrapper `<w>`. A `tool@<w>` line uses it. |
| `DMK_SOAK_SANITIZER` | `run_soak.sh` | None | Necessary for the soak test. The path of `compute-sanitizer` in the container, in the folder `/tools`. |
| `DMK_SOAK_INTERVAL` | `run_soak.sh` | `1800` | The time between two checks of the soak test, in seconds. |
| `DMK_SOAK_BENCH_ARGS` | `run_soak.sh` | None | The flags of `bench.py` for each benchmark run of the soak test. |
| `DMK_SOAK_PASS_TIMEOUT` | `run_soak.sh` | `1200` | The longest time of one benchmark run of the soak test, in seconds. |
| `DMK_SOAK_SAN_TIMEOUT` | `run_soak.sh` | `1800` | The longest time of the client in one sanitizer phase, in seconds. |
| `DMK_SOAK_SMI_MS` | `run_soak.sh` | `1000` | The time between two samples of the GPU sampler of the soak test, in ms. |
| `DMK_SOAK_MIN_UNALLOC_MIB` | `run_soak.sh` | `0` | The soak test stops if `btrfs filesystem usage` shows less than this value, in MiB, in the line `Device unallocated`. The value 0 turns this limit off. |

A wrapper starts the tool and records or checks its work. The name `<w>` has only letters, numbers and `_`. The first word of the value is the program of the wrapper in the container. The tool container sees the folder `tools` of the work folder on the GPU host at `/tools` as read-only. Put the program of the wrapper in that folder. Example:

```
DMK_WRAP_memcheck="/tools/compute-sanitizer-12.8.93/compute-sanitizer --tool memcheck"
```

`run_schedule.sh` sets `DMK_WRAP` for a `tool@<w>` line, and `run_tool.sh` reads it. Do not set `DMK_WRAP` in the env file. If the env file has no `DMK_WRAP_<w>`, the `tool@<w>` line starts the tool with no wrapper.

A server wrapper does the same task for `llama-server`. `DMK_SERVER_WRAP` holds its command, and `start_server.sh` reads it. You can set it in the env file or as an override on a schedule line. The section Server script gives the rules.

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

The script makes `bench` lines and `tool` lines only. A person writes `client` lines, `tool@<w>` lines, and the overrides at the start of a line.

## Run a schedule

1. Build the programs of each arm. Put them in `builds/<name>` in the work folder on the GPU host.
2. Pull the two images of the env file on the GPU host with `docker pull`.
3. Set `DMK_ENV` to the path of your env file.
4. Make the schedule file with `make_schedule.py`. Add `client` lines, `tool@<w>` lines, and overrides if you want them.
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

You give the script two values: the schedule file and the results folder. It has no flags. It gets all other values from the env file. A line of the schedule can replace some of these values for that line only. The section Overrides at the start of a line gives the rule.

The script does these tasks in this sequence:

1. It makes the results folder. It makes the folders `harness`, `bench` and `runs` in the work folder on the GPU host.
2. It puts a copy of the folder `remote` in `harness`, and a copy of `DMK_BENCH_DIR` in `bench`. If you set `DMK_EXTRA_DIR`, it puts a copy of it in `extra`. Each copy removes the files on the GPU host that the local folder does not have.
3. It writes the copy of the env file to `harness/run.env` on the GPU host.
4. For each line of the schedule, it starts the script on the GPU host through SSH. It writes a line with the time in UTC, the overrides, the type, the run name and the text after the run name.
5. After each line, it puts a copy of the files `<name>.*` from the folder `runs` of the GPU host in the results folder.
6. At the end, it puts a copy of the schedule file in the results folder and writes `done, <N> failed`.

The script does not use empty lines and lines that start with `#`. The other lines have one of these shapes. Each shape can have overrides in front of the type.

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

### Overrides at the start of a line

A line can start with one or more overrides. An override has the shape `DMK_NAME=value`. The name is in the shape `DMK_[A-Z0-9_]+`. A space and the next word come after each override.

A value has no space and no mark `"` or `'`. Or you put the marks `"` around the value. The value then has no mark `"` or `'` in it. The value `""` is empty text. For `DMK_SERVER_WRAP`, an empty value turns the wrapper off.

The override applies to that line only. The script sets it on the GPU host after it reads the copy `run.env`. It then replaces the value of the env file for that run. The line of the output of the script shows the overrides in the shape `DMK_NAME='value'`, before the type.

An override of `DMK_HOST`, `DMK_BENCH_DIR` or `DMK_EXTRA_DIR` has no effect. The script reads these values before the first line. The scripts on the GPU host do not read them. Do not use an override for `DMK_REMOTE_DIR`. The script uses the value of the env file to start the scripts and to find the files of the run.

A word that is not an override is the type of the line. If the script does not know that type, it stops with exit code 2. This is also true for a word with an incorrect override shape.

The first line below runs the benchmark with the model file `other.gguf` of `DMK_MODELS`:

```
DMK_MODEL=other.gguf bench r1 stock
```

The next line runs a `client` run with a profiler as the wrapper of the server. The profiler writes its report to the folder `runs`:

```
DMK_SERVER_WRAP="/tools/nsys/nsys profile -o /runs/r2" client r2 build:b12004-sm86 /bench/dectest/run_dectest.py --out /runs/{name}.answers.json
```

## The scripts on the GPU host

`run_schedule.sh` starts these scripts. The scripts read the environment variables of the copy `run.env` when they start. The overrides of a line replace these values.

### Server script

Command: `start_server.sh <arm> [server flags]`. The script starts the container `dmx-server` with `docker run -d`. It does not wait for the server.

Docker connects the port `DMK_PORT` of the GPU host to the port 8080 of the container. The container has the GPU `DMK_GPU` and the folder `DMK_MODELS` at `/models`. The flags of the server are `-m /models/$DMK_MODEL --host 0.0.0.0 --port 8080`, then `DMK_SERVER_ARGS`, then the flags that you give.

The arm `stock` sets the entry point `/app/llama-server`. This is the same program as the entry point of the image. An arm `build:<name>` sets the entry point `/opt/dmk/llama-server`. The program starts in the folder `/opt/dmk`, and `LD_LIBRARY_PATH` is `/opt/dmk`. The script gives exit code 2 for an unknown arm, an incorrect build name, or a folder with no `llama-server`.

If you set `DMK_TOOL_CAPS`, the script gives the container that capability with the flag `--cap-add`.

If you set `DMK_SERVER_WRAP`, the script runs `llama-server` with a wrapper, for example a profiler. The script divides the value at spaces. The first word is the entry point of the container. The other words and the server program come after it. The flags of the server come last.

Put the program of the wrapper in the folder `tools` of the work folder on the GPU host. With `DMK_SERVER_WRAP`, the server container has two more paths:

| Path in the container | Folder on the GPU host | Access |
|---|---|---|
| `/tools` | The folder `tools` of the work folder | Read-only |
| `/runs` | The folder `runs` of the work folder | Read and write |

The wrapper can write its reports to `/runs`. `run_schedule.sh` puts the files `<name>.*` of the folder `runs` in the results folder. Give each report the name `<name>.<extension>`, where `<name>` is the run name.

### Benchmark script

Command: `run_bench.sh <run-name> <arm> [bench.py flags]`. The script does the five parts of the section How the harness works. It gives the flags to `bench.py`. It also gives `--startup-s` to `bench.py`. This is the time from before the start of the container until the first answer of `/health`.

The client container has the folder `bench` of the work folder at `/bench` as read-only. It has the folder `runs` at `/runs`. `bench.py` writes `/runs/<name>.json`.

The script gives exit code 2 if the file `<name>.json` is in the folder `runs`. It gives exit code 3 if the container does not start or the server does not answer `/health`. In all other cases the exit code is the exit code of `bench.py`.

The script removes the server container with `docker rm -f`. It does not stop the server first. With `DMK_SERVER_WRAP`, the wrapper then gets no `SIGTERM`. [INFERENCE: `docker rm -f` kills the server. A profiler then cannot write its report.] Use a `client` line for a wrapper that writes a report when the server stops.

### Client script

Command: `run_client.sh <run-name> <arm> <script> [script flags]`. The script does the same parts as `run_bench.sh`. It starts `python -u <script>` with the flags in the client container.

`<script>` is a path in the container, `/bench/...` or `/extra/...`. In the flags, `{name}` becomes the run name. The client container has the folder `runs` at `/runs`, and it can write there. A client script must write its files to `/runs/{name}.*`. Then `run_schedule.sh` puts them in the results folder.

The script does not tell the client the port of the server. If you change `DMK_PORT`, give the port in the flags. The script gives exit code 2 if the file `<name>.client.out` is in the folder `runs`. In all other cases the exit code is the exit code of the client script.

With `DMK_SERVER_WRAP`, the script stops the server container with `docker stop` before it reads the log of the server. `docker stop` sends `SIGTERM` to the container. The script waits at most `DMK_STOP_TIMEOUT` seconds. A profiler can then write its report.

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

## Soak test

A soak test keeps the server of one arm up for many hours. The benchmark runs against this server again and again. At a regular interval, the script examines the server, the GPU, `dmesg` and the log of the server. A soak test finds faults that a short run does not find. Examples are a leak of memory, a slow decrease of the speed, and a GPU error.

The script `remote/run_soak.sh` does the test on the GPU host. The script `analysis/soak_check.py` reads the results and writes `PASS` or `FAIL` for each condition.

### Steps to start a soak test

1. Make sure that no other program uses the GPU. The script does not stop the production server.
2. Put a copy of the folder `remote` in the folder `harness` of the work folder on the GPU host.
3. Put a copy of the folder `bench` in the folder `bench` of the work folder.
4. Put the env file in `harness/run.env`, without the lines of `DMK_HOST`, `DMK_BENCH_DIR` and `DMK_EXTRA_DIR`.
5. Add the line `DMK_SOAK_SANITIZER=<path>` to `harness/run.env`.
6. Start the script with `setsid` and `nohup`. The script then continues if the SSH connection stops.
7. Read `runs/<name>/<name>.soak.log` to see the work of the script.
8. Put a copy of the folder `runs/<name>` on the local device, and run `soak_check.py` on it.

Steps 2 to 4 are the same as the first steps of `run_schedule.sh`. This example starts a soak test of 24 hours:

```
ssh gpu-host "cd /srv/dmk && set -a && . harness/run.env && set +a && \
    setsid nohup bash harness/run_soak.sh soak-1 build:b12004-sm86 24h \
    > runs/soak-1.out 2>&1 < /dev/null &"
rsync -a gpu-host:/srv/dmk/runs/soak-1/ results/soak-1/
python3 harness/analysis/soak_check.py results/soak-1
```

### The script of the soak test

```
harness/remote/run_soak.sh <run name> <arm> <time>
```

The time is a number of seconds, or numbers with the units `s`, `m`, `h` and `d`. The values `90m`, `24h` and `1h30m` are correct. The run name must be new. The script makes the folder `runs/<run name>` in the work folder and writes all files there. It does these tasks in this sequence:

1. It reads the time of the last line of `dmesg`. Only the lines after this time count.
2. It does the first sanitizer phase.
3. It starts the server of the arm with `start_server.sh` and waits for `/health`. The flags of the server are the flags of the other runs of the harness.
4. It does the first check. The time of the soak test starts.
5. It starts `bench.py` with the flag `--write-reference`. It does this again and again until the time of the soak test ends.
6. It does a check after the first run of `bench.py`, a check at each interval, and a check at the end.
7. It stops the server and does the second sanitizer phase.
8. It does the last check.

The script stops the soak test before the time ends in these cases:

- The server container does not run.
- Three runs of `bench.py`, one after the other, give an exit code other than 0.
- `btrfs filesystem usage` shows less than `DMK_SOAK_MIN_UNALLOC_MIB` MiB in the line `Device unallocated`.

If the script gets `SIGTERM`, `SIGINT` or `SIGHUP`, it stops the client, removes the containers and writes `<name>.status`. Do not use `kill -9`. The script then cannot remove the containers.

The server container has no `--restart` flag. If the server stops, Docker does not start it again. Condition `d` is then not true.

A server can write many log lines. The server container keeps at most 3 log files of 50 MB each. At each check, the script writes the new lines to a `gzip` file.

### The sanitizer phase

The phase starts the server of the arm below `compute-sanitizer` with the tool `memcheck`. The flags of the server are the same as in the soak test. The script `san_client.py` sends one request of each type, one at a time. The types are the short cases with each number of questions, the first `medium` case and the first `long` case. Each request must give the code 200 and an answer for each question.

Then the script runs `docker stop`, and the container gets `SIGTERM`. The wrapper `san_wrap.sh` stops only the server. The sanitizer then stops and writes the line `ERROR SUMMARY`. The phase is correct if this line has 0 errors, the exit code of the container is 0, and all requests are correct. The phase uses some minutes, because the sanitizer makes the server slow.

### The checks

A check writes one line of JSON to `<name>.checks.jsonl`. The table gives the keys of the line.

| Key | Meaning |
|---|---|
| `tag` | The type of the check: `start`, `warm` (after the first run of `bench.py`), `periodic`, `end`, or `final` (after the second sanitizer phase, with no server). |
| `state`, `restarts`, `exit_code`, `oom`, `started_at` | The values of `docker inspect` for the server container. |
| `mem_mib`, `power_w`, `pstate`, `clock_sm`, `clock_mem`, `temp_c` | The values of `nvidia-smi`: `memory.used`, `power.draw`, `pstate`, `clocks.sm`, `clocks.mem` and `temperature.gpu`. |
| `dmesg_hits` | The number of lines with `Xid` or `NVRM` in `dmesg` after the start time of the test. |
| `log_error_hits` | The number of log lines of the server that agree with a regular expression of `<name>.error-patterns`. |
| `unalloc_mib` | The value of `Device unallocated` of `btrfs`, in MiB. It is `null` if the folder of Docker is not on `btrfs`. |
| `passes_done`, `passes_failed` | The number of runs of `bench.py` that are completed, and the number of runs with an exit code other than 0. |

### Files of a soak test

The folder `runs/<name>` has these files. `NNNN` is the number of the run of `bench.py`.

| File | Contents |
|---|---|
| `<name>-pNNNN.json` | The result of `bench.py` for the run `NNNN`. |
| `<name>-pNNNN.reference.json` | The answers of the quality phase of the run `NNNN`. |
| `<name>.passes` | One row for each run of `bench.py`: the number, the exit code, the start time, the end time and the file name. The character `\t` is between two values. |
| `<name>.bench.out` | The output of each run of `bench.py`. |
| `<name>.checks.jsonl` | The checks. |
| `<name>.server.log.gz` | The log of the server. Each line starts with the time of Docker. |
| `<name>.error-patterns` | The regular expressions for the error lines of the log, one in each line. |
| `<name>.errors.log` | The lines of the log that agree with a regular expression. |
| `<name>.dmesg.log` | The lines of `dmesg` with `Xid` or `NVRM` after the start time of the test. |
| `<name>.events.log` | The lines of `docker events` for the server container: `die`, `kill`, `oom`, `restart` and `stop`. |
| `<name>.san-start.*`, `<name>.san-end.*` | The files of the two sanitizer phases: `.result`, `.server.log`, `.client.out`, `.answers.json` and `.launch.log`. |
| `<name>.smi` | The samples of the GPU sampler. |
| `<name>.meta`, `<name>.status` | The facts of the test and the state at the end. |
| `<name>.soak.log` | The log of the script, with the time in UTC. |
| `<name>.build.sha256` | The checksums of `llama-server` and of the libraries of the build. The arm `stock` has no such file. |
| `<name>.launch.log`, `<name>.load0`, `<name>.load1` | The same files as for `run_bench.sh`. |

### The check script

```
python3 harness/analysis/soak_check.py RUN_FOLDER [--reference GLOB ...] [--slowdown-pct N] [--mem-mib N] [--window N]
```

The script writes one line for each condition. The last line is `RESULT: PASS` or `RESULT: FAIL`. The exit code is 0 only if all conditions are true. The script uses the Python standard library only.

| Condition | The condition is true if |
|---|---|
| `a` | Each run of `bench.py` has all guards true. |
| `b` | The answers of each run are the same, bit for bit, as the answers in one set of the group of correct sets. The group has the different sets in the files of `--reference`. Without `--reference`, the group has only the set of the first run. |
| `c` | `dmesg` has no line with `Xid` or `NVRM` after the start time of the test. |
| `d` | `RestartCount` of the server container is 0, the container did not stop, and `docker events` shows no `die`, `kill`, `oom`, `restart` or `stop`. |
| `e` | The log of the server and the results of the runs have no HTTP 5xx line, no CUDA error line, and no line with `Aborted`, `GGML_ASSERT` or `SIGSEGV`. |
| `f` | The two sanitizer phases have `ERROR SUMMARY: 0 errors`, and each request gave the code 200. |
| `g` | The mean `lat_ms` of the last runs is at most `--slowdown-pct` above the mean of the first runs. |
| `h` | `memory.used` at the end is at most `--mem-mib` away from `memory.used` after the first run. |
| `i` | The script stopped after the last phase. The soak test ran for the full time. No gap between two checks is more than two times the interval plus 60 s. |

| Flag | Default | Meaning |
|---|---|---|
| `--reference` | None | A path of answer files of a build that is correct. It can have `*`. Put the marks `'` around it. You can use the flag more than one time. |
| `--slowdown-pct` | `2` | The largest increase of `lat_ms`, in %. |
| `--mem-mib` | `64` | The largest change of `memory.used`, in MiB. |
| `--window` | `10` | The number of runs at the start and at the end for condition `g`. If there are less than 20 runs, each group has `runs // 2` runs. |

One server can give more than one set of answers for the same requests. The state of the server after the requests before changes the numbers by a small quantity. Test runs of two builds gave the same 3 sets, and the largest change of a probability was 0.02. No top selection changed. Use the files of all correct sets for `--reference`. Condition `b` is then not true only if the server gives a new set.

The script reads the regular expressions for condition `e` from `<name>.error-patterns`. It also writes lines with `info` for the space of Docker, and for the clocks, the power and the temperature of the GPU.

## Build the programs of an arm

The script `build/build.sh` builds llama.cpp for the harness. It runs on the local device, and `podman` must be on that device. You then put the programs on the GPU host in the structure that the harness uses.

| Path in the work folder on the GPU host | Contents |
|---|---|
| `builds/<name>/dist/`, or `builds/<name>/` if the folder has no `dist` | The program `llama-server`, the program `llama`, and the shared libraries. |

A `bench` line and a `client` line use `llama-server`. A `tool` line uses `llama`. The harness does not use `llama-bench` or `llama-perplexity`. A `tool` line uses the commands `llama bench` and `llama perplexity` for the same tasks.

### Steps to build and deploy an arm

1. Get the source of llama.cpp with `git clone`. Check out the commit for your test.
2. Run `harness/build/build.sh` with the source folder and a new output folder.
3. Read `dist/MANIFEST.txt` in the output folder. Make sure that the line `src commit` shows the commit that you want.
4. Make sure that the line `src dirty files` shows 0 for a baseline build.
5. Put a copy of `dist` in `builds/<name>/dist` in the work folder on the GPU host.
6. Make sure that the programs start. Use a `tool` line for the command `version`.
7. Write the commit and the CMake command from the manifest in the README of each data set.

### The build script

```
harness/build/build.sh <src-dir> <out-dir> [more flags for cmake]
```

`<src-dir>` is a clone of llama.cpp, and `<out-dir>` is the output folder. The script makes the output folder if it is not there. If you give it less than two values, it gives exit code 2. It gives exit code 1 if `<src-dir>` is not a llama.cpp folder. In all other cases the exit code is the exit code of the build.

The script builds in a container of the builder image. The file `build/Containerfile.builder` makes the image, and the script makes the image if it is not there. The file starts from the image `docker.io/nvidia/cuda:12.8.1-devel-ubuntu24.04`. It names that image with its digest `sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66`. This name always gives the same image. The builder image also has `gcc-14`, `cmake`, `ccache`, `python3` and `git`.

The script puts `<src-dir>` in the container at `/src` as read-only, and `<out-dir>` at `/out`. If `<src-dir>` comes from `git worktree add`, the script also puts the primary repository in the container as read-only. The `git` commands for the manifest use it.

The script reads these environment variables. All are optional.

| Name | Default | Meaning |
|---|---|---|
| `DMK_BUILDER_IMAGE` | `localhost/dmk-builder:cuda12.8.1-gcc14` | The tag of the builder image. |
| `DMK_CUDA_ARCH` | `86-real` | The value of `CMAKE_CUDA_ARCHITECTURES`. |
| `DMK_JOBS` | `16` | The value of the flag `-j` of `cmake --build`. |
| `DMK_TARGETS` | `llama-server llama-app llama-bench llama-perplexity llama-tokenize test-backend-ops` | The CMake targets to build. The value `all` builds all targets. |
| `DMK_CLEAN` | None | The value `1` removes `<out-dir>/build` before the build starts. |
| `DMK_CCACHE` | `0` | The value `1` turns `ccache` on. The section Builds of the same source gives the cause of the default. |
| `DMK_CCACHE_DIR` | The folder `ccache` in the folder of `build.sh` | The folder of the cache of `ccache`. The script puts it in the container at `/ccache`. |

The script writes these paths in `<out-dir>`:

| Path | Contents |
|---|---|
| `build/` | The CMake build folder. The script keeps it. A second run in the same output folder then builds only the changes. |
| `dist/` | The files to deploy. It has the structure of `/app` in the stock image. |
| `dist/MANIFEST.txt` | The manifest of the build. |
| `build.log` | The full log of the last run. |

The target `llama-app` writes the program `llama`. The script copies the program of each target and all shared libraries of `build/bin` into `dist/`. All files are in one folder. With the default targets, `dist/` has the same files as `/app` in the stock image. It also has `llama-bench`, `llama-perplexity`, `llama-tokenize`, `test-backend-ops` and `MANIFEST.txt`.

The manifest has these parts:

| Part | Contents |
|---|---|
| `built` | The date of the build in UTC, and the time that the build used. |
| `source` | The commit, the output of `git describe`, the branch, the number of files that `git status` shows, and the SHA-256 checksum of `git diff HEAD`. |
| `toolchain` | The image of the `FROM` line with its digest, the builder image, and the versions of the tools `nvcc`, `gcc-14`, `g++-14`, `cmake` and `ccache`. |
| `cmake configure` | The `cmake` commands with all flags, and some entries of `CMakeCache.txt`. |
| `files` | The SHA-256 checksum, the length and the name of each file. |
| `ccache` | The output of `ccache --show-stats`. The manifest has this part only if `DMK_CCACHE` is `1`. |

### The flags of CMake

The script gives these flags to `cmake`. The last column shows the difference from the file `.devops/cuda.Dockerfile` of llama.cpp at b12004.

| Flag | Meaning | Difference from the upstream file |
|---|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | Makes the release build. | The upstream file sets no value. llama.cpp then uses `Release`. The build type is the same. |
| `-DGGML_NATIVE=OFF` | The code is not only for the CPU that builds it. | None. |
| `-DGGML_CUDA=ON` | Builds the CUDA backend. | None. |
| `-DGGML_BACKEND_DL=ON` | Builds each backend as a shared library. The program loads it when it runs. | None. |
| `-DGGML_CPU_ALL_VARIANTS=ON` | Builds the CPU backend in all variants. | None. |
| `-DLLAMA_BUILD_TESTS=ON` | Builds the tests, with `test-backend-ops`. | The upstream file has `OFF`. |
| `-DCMAKE_CUDA_ARCHITECTURES=86-real` | Makes the SASS for `sm_86` only. | The upstream file sets no value. The default list of llama.cpp then has more architectures. |
| `-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined` | The build accepts a shared library that has symbols that it does not find. | None. |
| `-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON` | A program finds its shared libraries in the folder of the program. | The upstream file does not set it. |
| `-DLLAMA_USE_PREBUILT_UI=OFF` | The build does not download the WebUI. It can run with no network. The harness starts the server with `--no-webui`. | The upstream file does not set it. The default is `ON`, and the stock image has the WebUI. |
| `-DGGML_CCACHE=OFF` | The build does not use `ccache`. | The upstream file does not set it. The upstream build step has no `ccache`. |

With `DMK_CCACHE=1`, the script sets `-DGGML_CCACHE=ON`. The flags that you give to the script come last on the `cmake` command line. They replace the defaults. This example turns one CUDA flag off:

```
harness/build/build.sh /path/to/llama.cpp out/p01-name -DGGML_CUDA_FA=OFF
```

### Builds of the same source

We built b12004 four times from the same source. All programs and libraries in `dist/` are the same, bit for bit. Only `libggml-cuda.so` is different. The checksum of `libggml-cuda.so` changes in each build, but its SASS does not change. [INFERENCE: `nvcc` puts a text that is different in each build into the names of symbols.]

A check of a build must then compare the SASS, and not the checksum. The script `build/sass_compare.py` does this.

`ccache` can change the SASS of a kernel. In one build with `CCACHE_BASEDIR` set, the SASS of one kernel was different from the SASS of the stock image. The script does not set `CCACHE_BASEDIR`. A build without it and with a cold cache had the same SASS as a build without `ccache`. Use a build without `ccache` for a baseline.

### Steps to compare the SASS of two builds

The script `build/sass_compare.py` compares the SASS of `libggml-cuda.so` of two builds for `sm_86`. The script uses `cuobjdump` and Python 3. The builder image has these two programs.

1. Make one empty folder for the cubins of each build.
2. Start a container of the builder image with `podman run`. Put the output folders of the builds and the script in it.
3. Run `cuobjdump -arch sm_86 -xelf all <libggml-cuda.so>` in each empty folder, for the library of that build.
4. Run `python3 sass_compare.py <A> <B> <out> [name-regex ...]` with the two folders of cubins.
5. Read the summary that the script writes.

The files of each folder have the names `libggml-cuda.<N>.sm_86.cubin`. The two folders must have the same values of `<N>`. This example uses the output folders of two builds:

```
podman run --rm --security-opt label=disable -v "$PWD":/w -w /w \
    -v /path/to/repository/harness/build/sass_compare.py:/dmk/sass_compare.py:ro \
    localhost/dmk-builder:cuda12.8.1-gcc14 bash -c '
  mkdir cubins-A cubins-B
  (cd cubins-A && cuobjdump -arch sm_86 -xelf all /w/out/b12004-sm86/dist/libggml-cuda.so)
  (cd cubins-B && cuobjdump -arch sm_86 -xelf all /w/out/p01-name/dist/libggml-cuda.so)
  python3 /dmk/sass_compare.py cubins-A cubins-B result "mul_mat_q|flash_attn"'
```

The script finds the same function in the two builds by its name. `nvcc` puts a hash that is different in each build into the names of some functions. The script removes this hash first. It also removes the instruction positions and the comments from the SASS text before it compares the text.

The script also compares the 128 bits of the code of each instruction. These bits have control bits that the text does not show. A name can be in more than one function of the library. The script then compares the group of all functions with that name. The values `REG`, `STACK`, `SHARED`, `LOCAL` and `CONSTANT` come from `cuobjdump -res-usage`. For a name with more than one function, the last function counts.

The script writes the summary as JSON on the standard output. It tests each `name-regex` against each function name with `re.search`. It writes these files to `<out>`:

| File in `<out>` | Contents |
|---|---|
| `sass-compare.json` | The summary, the names with different SASS, and the first 50 names that only one build has. |
| `key-kernels.tsv` | One row for each name that a `name-regex` finds. It has the same data for A and for B in two columns. |
| `diff-<hash>.txt` | The difference of the SASS text of a name that a `name-regex` finds, if the SASS is different. |

The summary has these keys:

| Key | Meaning |
|---|---|
| `cubins_per_side` | The number of cubins of each build. |
| `function_bodies_A`, `function_bodies_B` | The number of functions of each build. |
| `distinct_normalized_names_A`, `distinct_normalized_names_B` | The number of different names of each build, after the script removes the hash. |
| `names_in_both` | The number of names that are in the two builds. |
| `names_only_A`, `names_only_B` | The number of names that only one build has. |
| `names_sass_identical`, `names_sass_different` | The number of names with the same SASS text, and with different SASS text. |
| `names_encoding_identical_incl_control_bits`, `names_encoding_different` | The number of names with the same code of the instructions, and with different code. |
| `instr_total_A`, `instr_total_B` | The number of SASS instructions of each build. |
| `names_resusage_different` | The number of names with different values of `REG`, `STACK`, `SHARED`, `LOCAL` or `CONSTANT`. |

The two builds have the same SASS when `names_sass_different`, `names_encoding_different`, `names_only_A`, `names_only_B` and `names_resusage_different` are all 0.

We compared the first baseline build with the stock image. The build `b12004-sm86` and the stock image have the same SASS for all 6,754 kernel names. They also have the same code of the instructions and the same values of `REG`, `STACK`, `SHARED`, `LOCAL` and `CONSTANT`. Each side has 146 cubins and 6,154,400 instructions.

### How the build runs in the harness

The container of the arm `build:<name>` has the folder of the build at `/opt/dmk` as read-only. The program starts in the folder `/opt/dmk`, and `LD_LIBRARY_PATH` is `/opt/dmk`. The libraries of the build load, and the libraries in `/app` do not.

The builder image has CUDA 12.8.1. The upstream file `.devops/cuda.Dockerfile` at b12004 sets CUDA 12.8.1 for the image that runs the server. [INFERENCE: The container gets the CUDA libraries from the image. The CUDA version of the image must be the CUDA version of the builder image.]

Put a copy of `dist/` on the GPU host in `builds/<name>/dist`. For example:

```
ssh gpu-host mkdir -p /srv/dmk/builds/b12004-sm86/dist
rsync -a out/b12004-sm86/dist/ gpu-host:/srv/dmk/builds/b12004-sm86/dist/
```

### The baseline build

The baseline build `b12004-sm86` is the tag `b12004` (commit `3c809f4`) of llama.cpp, built with the flags above. The upstream repository has the tag. The build used `nvcc` 12.8.93, `gcc-14` 14.2.0 and `cmake` 3.28.3.

```
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout b12004
harness/build/build.sh llama.cpp out/b12004-sm86
```

The manifest of the baseline build shows the commit `3c809f44383bcebc318b8cddd8c6d11aaa861e9b` and 0 for `src dirty files`.

### Check of the programs

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
- `run_bench.sh` and `run_client.sh` use the name `dmx-server` for the log, the check of the server and the removal. `run_client.sh` also uses it to stop the server. If you change `DMK_CONTAINER`, these scripts cannot find the server container.
- The harness does not change the power limit or the clocks of the GPU. It records the power limit in each `.meta` file.
- Use the GPU for no other work during a schedule or during a soak test.
- `run_bench.sh` does not stop the server before it removes it. A wrapper of the server gets no `SIGTERM` in a `bench` run. Use a `client` line.
- A soak test holds the GPU for many hours. `run_soak.sh` removes the containers `dmx-server` and `dmx-bench` when it starts.
- `run_soak.sh` gives exit code 2 for an incorrect command or for a `dmesg` that the user cannot read. It gives exit code 3 if a server does not start.
