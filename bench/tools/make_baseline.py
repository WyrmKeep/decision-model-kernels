#!/usr/bin/env python3
"""Build baseline.json from the result files of repeated baseline runs.

Usage:
    python3 tools/make_baseline.py RUN1.json RUN2.json RUN3.json \\
        --note "stock image, model X, flags Y, date" [--out baseline.json]

Each RUN file is a result file of bench.py. The runs must come from the same
configuration and must have run with a measured reference.json (so they have
agreement numbers). The guard of bench.py reads three numbers from the output:
correct, agree and mean_dp. The other fields are for the reader.

The quality numbers of repeated runs of one build should be identical. The tool
stops when they differ. With --allow-differ it keeps the worst value of each.
"""
import argparse
import json
import pathlib
import statistics
import sys


def vram_peak_mib(run_json):
    """Peak GPU memory in MiB from the sampler file next to the run file, or None."""
    smi = pathlib.Path(run_json).with_suffix(".smi")
    if not smi.exists():
        return None
    peak = None
    for line in smi.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            used = float(parts[1])
        except ValueError:
            continue
        peak = used if peak is None else max(peak, used)
    return peak


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="result files of baseline runs")
    ap.add_argument("--note", required=True, help="what the baseline is: build, model, flags, date")
    ap.add_argument("--out", default="baseline.json")
    ap.add_argument("--allow-differ", action="store_true", help="keep the worst quality value when runs differ")
    args = ap.parse_args()

    runs = [json.load(open(p)) for p in args.runs]
    for p, r in zip(args.runs, runs):
        q = r["quality"]
        if "agree" not in q:
            sys.exit(f"{p}: no agreement numbers. Run with a measured reference.json.")
        if r["errors"]:
            sys.exit(f"{p}: the run had errors: {r['errors'][:3]}")

    correct = [r["quality"]["correct"] for r in runs]
    agree = [r["quality"]["agree"] for r in runs]
    mean_dp = [r["quality"]["mean_dp"] for r in runs]
    same = len(set(correct)) == 1 and len(set(agree)) == 1 and max(mean_dp) - min(mean_dp) < 1e-9
    if not same and not args.allow_differ:
        sys.exit(f"quality differs between runs: correct {correct}, agree {agree}, mean_dp {mean_dp}. "
                 "Find the cause, or use --allow-differ.")

    vram = [v for v in (vram_peak_mib(p) for p in args.runs) if v is not None]
    out = {
        "status": "measured",
        "note": args.note,
        "runs": [pathlib.Path(p).stem for p in args.runs],
        "correct": min(correct),
        "agree": min(agree),
        "mean_dp": max(mean_dp),
        "lat_ms": [round(r["seq"]["lat_ms"], 1) for r in runs],
        "short_ms": [round(r["seq"]["by_size"]["short"]["mean"], 1) for r in runs],
        "rps": round(statistics.median(r["tput"]["rps"] for r in runs), 2),
        "conc": runs[0]["tput"]["conc"],
        "vram_mib": round(max(vram)) if vram else None,
    }
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}: correct {out['correct']}, agree {out['agree']}, mean_dp {out['mean_dp']:.6f}")


if __name__ == "__main__":
    main()
