# Smoke test of the decision model

This folder holds a small test of answer quality. It has 32 `short` cases with 52 decisions. Each decision has a label. The cases are the `short` cases of the [benchmark](../README.md). The benchmark builds its `short` cases from the file `cases.json` in this folder.

## About the test

The test is a smoke test, not a benchmark. A difference of one to three answers between two builds is noise. Use the test for a fast check that a change did not break the answers. Use the benchmark for speed and for the quality guards.

| Task | Cases | Questions | Decisions |
|---|---:|---|---:|
| `triage` | 12 | `category` (six possible answers), `needs_action` (yes or no) | 24 |
| `check` | 8 | `done` (yes or no) | 8 |
| `style` | 8 | `relative_date`, `secret` (yes or no) | 16 |
| `commit` | 4 | `matches` (yes or no) | 4 |
| Total | 32 | | 52 |

Source: the file `cases.json` in this folder.

The cases are new text that a person wrote. They have no data from a server in use.

The fields of the cases are the same as the fields in `../cases.json` of the benchmark. See [DATA.md](../DATA.md). There is one difference. The yes or no questions have the type `bool` here and the type `noul` in the benchmark.

## Labels

A person wrote each case and its label together. A model did not make a label. A person can give a different label for three cases: `triage00`, `triage07` and `triage08`. In the previous tests, most of the incorrect answers were for these three cases.

This folder has no answers of a model. Make new answers with the commands below. Do not change a label to make a result better.

## Run the test

1. Start a server that receives `POST /v1/systemone` requests.
2. Run the test and write the answers to a file:

   ```sh
   python3 run_dectest.py --url http://127.0.0.1:18080 --out answers-build-a.json
   ```

3. Read the result. The script shows the number of correct decisions. It also shows the number for each task and the decisions with an incorrect answer.

To compare two builds, make an answer file for each build. Then compare the two files:

```sh
python3 run_dectest.py --compare answers-build-a.json answers-build-b.json
```

The script shows how many decisions have the same top answer in the two files. It also shows the decisions with a different top answer. The environment variable `BENCH_URL` can replace the flag `--url`.

## Answer file

The answer file is a JSON object. The key is a case `id`. The value is an object with one answer for each question name. An answer has the same fields as an answer in `../reference.json`. See [DATA.md](../DATA.md).
