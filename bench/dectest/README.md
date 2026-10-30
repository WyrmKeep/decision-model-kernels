# Smoke test of the decision model

This folder holds a small test of answer quality. It has 32 short cases with 52 labeled decisions. The cases are the short cases of the [benchmark](../README.md). The benchmark builds its short cases from `cases.json` in this folder.

## What the test is

The test is a smoke test, not a benchmark. A difference of one to three answers between two builds is noise. Use the test for a fast check that a change did not break the answers. Use the benchmark for speed and for the quality guards.

| Task | Cases | Questions | Decisions |
|---|---:|---|---:|
| `triage` | 12 | `category` (six choices), `needs_action` (yes or no) | 24 |
| `check` | 8 | `done` (yes or no) | 8 |
| `style` | 8 | `relative_date`, `secret` (yes or no) | 16 |
| `commit` | 4 | `matches` (yes or no) | 4 |
| Total | 32 | | 52 |

The cases are synthetic. They contain no real server data. The fields of `cases.json` are the same as in the benchmark file. See [DATA.md](../DATA.md). One difference exists. The yes or no questions have the type `bool` here and the type `noul` in the benchmark.

## Labels

The author wrote each case and its label together. No model chose a label. Three labels are debatable: `triage00`, `triage07` and `triage08`. Most misses of earlier tests were on these three cases.

This folder holds no model answers. Make your own answers with the commands below. Never change a label to improve a result.

## Run the test

1. Start a server that serves `POST /v1/systemone`.
2. Run the test and save the answers:

   ```sh
   python3 run_dectest.py --url http://127.0.0.1:18080 --out answers-build-a.json
   ```

3. Read the result. The script prints the number of correct decisions, the count for each task and the list of misses.

To compare two builds, make an answer file for each build. Then compare the two files:

```sh
python3 run_dectest.py --compare answers-build-a.json answers-build-b.json
```

The script prints how many decisions have the same top answer in both files. It also lists the decisions that differ. The `BENCH_URL` variable can replace the `--url` option.

## Answer file

The answer file is a JSON object. The key is a case `id`. The value is an object with one answer for each question name. An answer has the same fields as an answer in `reference.json`. See [DATA.md](../DATA.md).
