#!/usr/bin/env python3
"""Make the derived tables of the data set 2026-10-31-ablation from the files in raw/.

Usage: python3 data/2026-10-31-ablation/make_tables.py
The script writes into the folder tables/ next to it. It calls harness/ablate/summarise_ablate.py of this
repository for the tables of the ablation runs, and it writes runs.tsv and sass.tsv itself. It uses only the
Python standard library. It changes no file in raw/. Run it again after each change of a file in raw/.
"""
import glob
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
SUMMARISE = os.path.join(ROOT, "harness", "ablate", "summarise_ablate.py")


def raw(name):
    return os.path.join(RAW, name)


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            assert len(r) == len(header), (name, r)
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")
    print(f"{name}: {len(rows)} rows")


def num(v, places):
    return None if v is None else f"{v:.{places}f}"


def stage_of(run):
    for prefix, name in (("abl-smoke-", "smoke"), ("abl-s1-", "stage 1"), ("abl-ops-", "sync"), ("abl-llb-", "llama bench")):
        if run.startswith(prefix):
            return name
    raise SystemExit(f"make_tables.py: unknown run name {run}")


def read_meta(run):
    """Facts of one run from its .meta file. A client run has the lines of run_client.sh, a tool run those of run_tool.sh."""
    meta = open(raw(run + ".meta")).read().splitlines()
    d = {"run": run, "arm": meta[0].split()[0].split("=", 1)[1]}
    d["start_utc"] = next(m for m in meta if m.startswith("date_utc=")).split("=", 1)[1]
    if os.path.exists(raw(run + ".load0")):  # a client run
        d["line_type"] = "client"
        d["load_before"] = float(open(raw(run + ".load0")).read().split()[0])
        d["load_after"] = float(open(raw(run + ".load1")).read().split()[0])
        info = meta[2].split(", ")
    else:  # a tool run: the load averages are lines 3 and 6 of the meta file
        d["line_type"] = "tool"
        d["load_before"] = float(meta[2].split()[0])
        d["load_after"] = float(meta[5].split()[0])
        info = meta[3].split(", ")
    d["driver"] = info[0]
    d["power_limit_w"] = info[1].replace(" W", "")
    return d


def smi_stats(run):
    """Samples of the GPU sampler: five values on each line, separated by a comma and a space."""
    rows = [l.strip().split(", ") for l in open(raw(run + ".smi")) if l.strip()]
    rows = [r for r in rows if len(r) == 5 and r[4].isdigit()]
    return {
        "mem": max(float(r[1]) for r in rows),
        "power": statistics.median(float(r[2]) for r in rows),
        "clock": statistics.median(int(r[4]) for r in rows),
    }


def client_json(run):
    """Facts of the result file of ablate_bench.py, or None."""
    p = raw(run + ".json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return {"requests": len(j["requests"]), "errors": len(j["errors"]), "seq_s": j["meta"]["seq_s"],
            "reconnects": j["meta"]["reconnects"]}


def llama_bench_commit(run):
    """The commit that llama bench reports in its JSON output, or None."""
    p = raw(run + ".out")
    if not os.path.exists(p):
        return None
    commits = {row["build_commit"] for row in json.load(open(p))}
    assert len(commits) == 1, (run, commits)
    return commits.pop()


def runs_table():
    runs = [read_meta(os.path.basename(p)[:-5]) for p in glob.glob(raw("abl-*.meta"))]
    runs.sort(key=lambda d: (d["start_utc"], d["run"]))
    rows = []
    for d in runs:
        c = client_json(d["run"])
        smi = smi_stats(d["run"])
        rows.append([
            d["run"], stage_of(d["run"]), d["line_type"], d["arm"], d["start_utc"],
            num(d["load_before"], 2), num(d["load_after"], 2), d["driver"], d["power_limit_w"],
            c["requests"] if c else None, c["errors"] if c else None, num(c["seq_s"], 1) if c else None,
            llama_bench_commit(d["run"]),
            num(smi["clock"], 1), num(smi["power"], 2), num(smi["mem"], 0),
        ])
    write_tsv("runs.tsv", [
        "run", "stage", "line_type", "arm", "start_utc", "load1_before", "load1_after", "driver", "power_limit_w",
        "requests", "errors", "seq_s", "llama_bench_commit",
        "sm_clock_mhz_median", "power_draw_w_median", "memory_used_mib_max"], rows)


def sass_table():
    """The summary of sass_compare.py for the build ablate-timing against the baseline build."""
    s = json.load(open(raw("sass-compare-ablate-timing.json")))["summary"]
    write_tsv("sass.tsv", ["quantity", "value"], [[k, v] for k, v in s.items()])


def main():
    if not os.path.exists(SUMMARISE):
        sys.exit(f"make_tables.py: {SUMMARISE} not found")
    os.makedirs(OUT, exist_ok=True)
    subprocess.run([sys.executable, SUMMARISE, "--tsv-dir", OUT, RAW], check=True)
    runs_table()
    sass_table()


main()
