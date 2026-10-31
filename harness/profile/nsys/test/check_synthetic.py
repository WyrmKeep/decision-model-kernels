#!/usr/bin/env python3
"""Checks of `nsys_census.py analyze` and `overhead` on the synthetic data of make_synthetic.py (no GPU). Prints PASS/FAIL per check.
  A  legacy-{short,medium,long}, --include-warmup: pass/request/trace_total rows, kernels.tsv and mmq-launches.tsv equal the legacy
     census tables column by column (new columns appended after the legacy header; host-gap parts add up to pre_gap)
  B  tail-short: 3 of 12 requests missing from the client JSON -> their 9 passes are not analysed, other rows equal the legacy rows
  C  shift-short: client clock 3 ms off -> window assignment visibly fails (negative first-kernel offsets), --assign order recovers the legacy rows
  D  osrt-short: synthetic OSRT_API -> OS-wait split equals the injected wait calls
  E  stray-short: model-load-like H2D copy before the first pass and 3 stray kernels after the last pass are ignored
  F  overhead: paired differences against the injected ones; --dec and --passes tables
Usage: python3 check_synthetic.py [--work DIR] [--out DIR]
  --work  folder with the synthetic data of make_synthetic.py (default: <tmp>/nsys-synth, tmp = $TMPDIR or /tmp); full analysis outputs go to <work>/out
  --out   folder for the logs and small reference outputs (default: <work>/test-out)
The census tables trace-{passes,kernels,mmq-launches}.tsv come from the census data set (census_lib.data_file, $DMK_CENSUS_DATA)."""
import argparse
import csv
import importlib.util
import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "census"))
import census_lib  # noqa: E402
NC = HERE.parent / "nsys_census.py"
spec = importlib.util.spec_from_file_location("nsys_census", NC)
nc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nc)

FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def run(args, log):
    p = subprocess.run([sys.executable, str(NC)] + [str(a) for a in args], capture_output=True, text=True)
    Path(log).write_text(p.stdout + p.stderr)
    assert p.returncode == 0, (args, p.stdout[-2000:], p.stderr[-2000:])
    return p.stdout


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def md_row(md, run_name):
    for line in md.splitlines():
        if line.startswith("| %s |" % run_name):
            return [c.strip() for c in line.strip("|").split("|")]
    return None


def table_row(md, title_prefix, first_cell):
    """row whose first cell is first_cell inside the table that follows the line starting with title_prefix"""
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("**" + title_prefix):
            for l in lines[i + 2:]:
                if not l.startswith("|"):
                    break
                cells = [c.strip() for c in l.strip("|").split("|")]
                if cells[0] == first_cell:
                    return cells
    return None


# The legacy segmentation hands the API records that precede the first pass to pass 0; the whole-server code drops them (they
# belong to model load / startup). Expected, explained difference: exactly these 3 columns of the first pass, its request and the total.
KNOWN_KEYS = {("pass", "0", "0"), ("request", "0", ""), ("trace_total", "", "")}
KNOWN_COLS = {"n_sync_calls", "cpu_api_time_ms", "cpu_sync_api_ms"}
KNOWN_SEEN = []


def compare_rows(legacy_rows, new_rows, cols, key_new, key_leg):
    """compares the string values of `cols` of rows with equal key; returns (n_rows_compared, unexplained differences)"""
    leg = {key_leg(r): r for r in legacy_rows}
    diffs, n = [], 0
    for r in new_rows:
        k = key_new(r)
        if k not in leg:
            diffs.append(("missing in legacy", k))
            continue
        n += 1
        for c in cols:
            if r[c] != leg[k][c]:
                diffs.append((k, c, r[c], leg[k][c]))
    known = [d for d in diffs if len(d) == 4 and d[0] in KNOWN_KEYS and d[1] in KNOWN_COLS]
    KNOWN_SEEN.extend(known)
    return n, [d for d in diffs if d not in known]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "nsys-synth"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    syn = Path(args.work)
    out = Path(args.out) if args.out else syn / "test-out"
    out.mkdir(parents=True, exist_ok=True)
    big = syn / "out"          # full analysis outputs (several MB each) stay in the work directory
    big.mkdir(exist_ok=True)
    exp = json.load(open(syn / "expected.json"))
    LEG_PASSES = tsv(census_lib.data_file("trace-passes.tsv"))
    LEG_KERN = tsv(census_lib.data_file("trace-kernels.tsv"))
    LEG_MMQ = tsv(census_lib.data_file("trace-mmq-launches.tsv"))
    leg_header = list(LEG_PASSES[0].keys())
    cols_nokey = [c for c in leg_header if c != "trace"]

    # ---------------------------------------------------------------- A
    print("== A: legacy runs through `analyze` (whole-server code path)")
    oa = big / "A"
    oa.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "legacy-short,legacy-medium,legacy-long", "--out", oa, "--include-warmup"], out / "synthetic-A.log")
    new = tsv(oa / "passes.tsv")
    new_header = list(new[0].keys())
    check(new_header[:len(leg_header)] == leg_header, "A: passes.tsv starts with exactly the legacy trace-passes.tsv columns (%d)" % len(leg_header))
    check(new_header[len(leg_header):len(leg_header) + 5] == ["run", "segment", "client_idx", "kind", "case"], "A: run, segment, client_idx, kind, case follow the legacy columns")
    for name in ("short", "medium", "long"):
        lr = [r for r in LEG_PASSES if r["trace"] == name]
        nr = [r for r in new if r["run"] == "legacy-" + name]
        key = lambda r: (r["row_kind"], r["request_idx"], r["pass_idx_in_request"])
        n, diffs = compare_rows(lr, nr, cols_nokey, key, key)
        check(len(lr) == len(nr) and n == len(lr) and not diffs, "A: %s: %d rows (pass/request/trace_total) equal in all %d legacy columns (%d differences)" % (name, len(lr), len(cols_nokey), len(diffs)))
        if diffs:
            print("   first differences:", diffs[:5])
    kn = tsv(oa / "kernels.tsv")
    for name in ("short", "medium", "long"):
        lr = [r for r in LEG_KERN if r["trace"] == name]
        nr = [r for r in kn if r["trace"] == "legacy-" + name]
        kc = [c for c in LEG_KERN[0].keys() if c != "trace"]
        ok = len(lr) == len(nr) and all(all(a[c] == b[c] for c in kc) for a, b in zip(lr, nr))
        check(ok, "A: %s: kernels.tsv %d rows equal to trace-kernels.tsv" % (name, len(lr)))
    mm = tsv(oa / "mmq-launches.tsv")
    for name in ("short", "medium", "long"):
        lr = [r for r in LEG_MMQ if r["trace"] == name]
        nr = [r for r in mm if r["trace"] == "legacy-" + name]
        mc = [c for c in LEG_MMQ[0].keys() if c != "trace"]
        ok = len(lr) == len(nr) and all(all(a[c] == b[c] for c in mc) for a, b in zip(lr, nr))
        check(ok, "A: %s: mmq-launches.tsv %d rows equal to trace-mmq-launches.tsv in the legacy columns" % (name, len(lr)))
        ops = [r["op"] for r in nr]
        check(all(o and o != "?" for o in ops), "A: %s: every MMQ/MMVQ launch has a tensor op (%d launches, %d distinct ops)" % (name, len(ops), len(set(ops))))
        gm = [r["grid_matches"] for r in nr if r["grid_matches"] != ""]
        check(gm and all(g == "1" for g in gm), "A: %s: tile-exact grid check ok for %d launches" % (name, len(gm)))
    pr = [r for r in new if r["row_kind"] in ("pass", "request") and r["host_gap_drain_ms"] != ""]
    err = max(abs(sum(float(r[c]) for c in ("host_gap_drain_ms", "host_gap_cpu_ms", "host_gap_cuda_api_ms", "host_gap_first_launch_ms")) - float(r["pre_gap_ms"])) for r in pr if r["row_kind"] == "pass")
    check(err < 6e-6, "A: host-gap parts add up to pre_gap_ms on every pass row (max error %.2e ms over %d rows)" % (err, sum(1 for r in pr if r["row_kind"] == "pass")))
    rq = tsv(oa / "requests.tsv")
    err = max(abs(float(r["send_to_first_kernel_ms"]) + float(r["kernel_span_ms"]) + float(r["last_kernel_to_recv_ms"]) - float(r["client_wall_ms"])) for r in rq)
    check(err < 6e-6, "A: send->first kernel + kernel span + last kernel->recv = client wall on every request (max error %.2e ms, %d requests)" % (err, len(rq)))
    types = {}
    for r in new:
        if r["row_kind"] == "pass":
            types.setdefault(r["run"], {}).setdefault(r["pass_type"], 0)
            types[r["run"]][r["pass_type"]] += 1
    check(types["legacy-short"] == {"prefix-small": 12, "suffix": 24} and types["legacy-long"] == {"prefix-large": 4, "prefix-small": 4, "suffix": 12},
          "A: pass_type counts: short2q = 12 prefix-small + 24 suffix, long3q = 4 prefix-large + 4 prefix-small + 12 suffix")
    md = (oa / "summary.txt").read_text()
    check("\nnone\n" in md.split("7. Validation problems")[1], "A: summary.txt reports no validation problems")
    # default (warm-up excluded)
    od = big / "A-default"
    od.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "legacy-short", "--out", od], out / "synthetic-A-default.log")
    kd = tsv(od / "kernels.tsv")
    tot_inc = sum(int(r["n_launches"]) for r in kn if r["trace"] == "legacy-short")
    tot_def = sum(int(r["n_launches"]) for r in kd)
    wu = sum(int(r["n_kernels"]) for r in tsv(od / "passes.tsv") if r["row_kind"] == "pass" and r["phase"] == "warmup")
    check(tot_inc - tot_def == wu and wu > 0, "A: default run leaves the warm-up request out of kernels.tsv (%d kernels), passes.tsv still lists it" % wu)
    oep = big / "A-epoch"
    oep.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "legacy-short", "--out", oep, "--clock", "epoch"], out / "synthetic-A-epoch.log")
    pe, pm = tsv(oep / "passes.tsv"), tsv(od / "passes.tsv")
    check(len(pe) == len(pm) and all(a[c] == b[c] for a, b in zip(pe, pm) for c in leg_header if c not in ("start_ns",)) and
          all(a["window_assignment"] == "window" for a in pe if a["row_kind"] == "pass"),
          "A: --clock epoch (t_send_epoch_ns - utcEpochNs) gives the same pass assignment and rows as --clock mono (%d rows)" % len(pe))
    re_ = [r for r in tsv(oep / "requests.tsv")]
    check(all(abs(float(r["epoch_minus_mono_send_ms"]) - 0.02) < 1e-6 for r in re_), "A: the epoch clock differs from the monotonic clock by the injected 20 us on every request")

    # ---------------------------------------------------------------- B
    print("== B: tail-short (client knows requests 3..11 only)")
    ob = big / "B"
    ob.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "tail-short", "--out", ob], out / "synthetic-B.log")
    nb = tsv(ob / "passes.tsv")
    lr = [r for r in LEG_PASSES if r["trace"] == "short" and r["row_kind"] in ("pass", "request") and int(r["request_idx"]) >= 3]
    nr = [r for r in nb if r["row_kind"] in ("pass", "request")]
    skip = {"request_idx", "start_rel_ms", "trace"}
    first_only = {"pre_gap_ms", "idle_before_request_ms", "h2d_lead_ms", "gpu_busy_pct_of_span_incl_idle_before"}
    cols = [c for c in leg_header if c not in skip]
    key_new = lambda r: (r["row_kind"], str(int(r["request_idx"]) + 3), r["pass_idx_in_request"])
    key_leg = lambda r: (r["row_kind"], r["request_idx"], r["pass_idx_in_request"])
    n, diffs = compare_rows(lr, nr, cols, key_new, key_leg)
    # the first analysed request/pass lacks the "previous pass" columns: compare them only for the others
    diffs = [d for d in diffs if not (len(d) == 4 and d[1] in first_only and d[0][1] == "3")]
    check(n == len(lr) == len(nr) and not diffs, "B: %d pass/request rows for requests 3..11 equal the legacy rows (%d differences; previous-pass columns of the first analysed request excluded)" % (n, len(diffs)))
    if diffs:
        print("   first differences:", diffs[:5])
    first = [r for r in nb if r["row_kind"] == "pass" and r["pass_idx_in_request"] == "0"][0]
    check(first["pre_gap_ms"] == "" and first["host_gap_drain_ms"] == "", "B: the first analysed pass has no previous-pass gap")
    mdb = (ob / "summary.txt").read_text()
    row = md_row(mdb, "tail-short")
    check(row is not None and row[2:5] == ["36", "27", "9"], "B: Table 1: 36 valid passes, 27 analysed, 9 outside any request (%s)" % (row[2:5] if row else None))

    # ---------------------------------------------------------------- C
    print("== C: shift-short (client clock +3 ms)")
    oc = big / "C-window"
    oc.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "shift-short", "--out", oc], out / "synthetic-C-window.log")
    mdc = (oc / "summary.txt").read_text()
    row = md_row(mdc, "shift-short")
    check(row is not None and int(row[3]) < 36, "C: window assignment loses passes under the shifted clock (analysed %s of 36)" % (row[3] if row else None))
    r2 = table_row(mdc, "Table 2", "shift-short")
    r3 = table_row(mdc, "Table 3", "shift-short")
    check(r3 is not None and int(r3[7]) > 0, "C: Table 3 shows passes that only the order-based assignment finds (%s)" % (r3[7] if r3 else None))
    check("\nnone\n" not in mdc.split("7. Validation problems")[1], "C: validation problems are reported")
    oc2 = big / "C-order"
    oc2.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "shift-short", "--out", oc2, "--assign", "order", "--include-warmup"], out / "synthetic-C-order.log")
    nc_ = tsv(oc2 / "passes.tsv")
    lr = [r for r in LEG_PASSES if r["trace"] == "short"]
    nr = [r for r in nc_]
    key = lambda r: (r["row_kind"], r["request_idx"], r["pass_idx_in_request"])
    n, diffs = compare_rows(lr, nr, cols_nokey, key, key)
    check(n == len(lr) == len(nr) and not diffs, "C: --assign order reproduces all %d legacy rows (%d differences)" % (n, len(diffs)))
    r2 = table_row((oc2 / "summary.txt").read_text(), "Table 2", "shift-short")
    check(r2 is not None and float(r2[5]) < 0, "C: with the wrong clock the mapping check shows the negative first-kernel offset (min %s ms)" % (r2[5] if r2 else None))

    # ---------------------------------------------------------------- D
    print("== D: osrt-short (synthetic OSRT_API)")
    od_ = big / "D"
    od_.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "osrt-short", "--out", od_, "--include-warmup"], out / "synthetic-D.log")
    nd = tsv(od_ / "passes.tsv")
    lr = [r for r in LEG_PASSES if r["trace"] == "short"]
    key = lambda r: (r["row_kind"], r["request_idx"], r["pass_idx_in_request"])
    n, diffs = compare_rows(lr, [r for r in nd], cols_nokey, key, key)
    check(n == len(lr) and not diffs, "D: the OSRT table does not change any legacy column (%d rows, %d differences)" % (n, len(diffs)))
    bad = []
    n_with = 0
    for r in nd:
        if r["row_kind"] != "pass" or r["host_gap_drain_ms"] == "":
            continue
        e = exp["osrt"].get(r["pass_idx_global"])
        wait, n_os, cpu, busy = fnum(r["host_gap_cpu_os_wait_ms"]), r["host_gap_n_osrt_calls"], fnum(r["host_gap_cpu_ms"]), fnum(r["host_gap_cpu_busy_ms"])
        if e:
            n_with += 1
            if abs(wait - e["wait_ns"] / 1e6) > 2e-6 or n_os != str(e["n_calls"]) or abs(busy - (cpu - wait)) > 2e-6 or not r["host_gap_osrt_top"].startswith("pthread_cond_wait:"):
                bad.append((r["pass_idx_global"], wait, e["wait_ns"] / 1e6, n_os, r["host_gap_osrt_top"]))
        elif wait != 0 or n_os != "0":
            bad.append((r["pass_idx_global"], wait, n_os, "expected none"))
    check(n_with > 20 and not bad, "D: OS-wait time and call count equal the injected calls on %d passes with injected wait calls and 0 elsewhere (%d mismatches)" % (n_with, len(bad)))
    if bad:
        print("   ", bad[:5])
    os_rows = tsv(od_ / "osrt-request-start.tsv")
    check(any(r["thread"] == "decode" for r in os_rows) and any(r["thread"] == "t1" for r in os_rows) and any(r["class"] == "wait" for r in os_rows),
          "D: osrt-request-start.tsv lists decode-thread and other-thread calls around the first timed request (%d rows)" % len(os_rows))

    # ---------------------------------------------------------------- E
    print("== E: stray-short (H2D copy of 100 MB before the first pass, 3 kernels 200 ms after the last pass)")
    oe = big / "E"
    oe.mkdir(exist_ok=True)
    run(["analyze", "--results", syn, "--runs", "stray-short", "--out", oe, "--include-warmup"], out / "synthetic-E.log")
    ne = tsv(oe / "passes.tsv")
    n, diffs = compare_rows(lr, ne, cols_nokey, key, key)
    check(n == len(lr) and not diffs, "E: rows equal the legacy rows although the capture holds foreign copies and kernels (%d rows, %d differences)" % (n, len(diffs)))
    row = md_row((oe / "summary.txt").read_text(), "stray-short")
    check(row is not None and row[-1] == "3", "E: 3 stray kernels are cut off the last pass (Table 1: %s)" % (row[-1] if row else None))

    # ---------------------------------------------------------------- F
    print("== F: overhead")
    of = big / "F"
    of.mkdir(exist_ok=True)
    run(["overhead", "--results", syn, "--ctl", "syn-ctl", "--nsys", "legacy-short,legacy-medium,legacy-long", "--dec", syn / "syn-dec.client.json",
         "--passes", oa / "passes.tsv", "--out", of], out / "synthetic-F.log")
    ov = {(r["run"], r["kind"]): r for r in tsv(of / "overhead.tsv")}
    for name, kind in (("short", "short2q"), ("medium", "medium3q"), ("long", "long3q")):
        a = {r["ref"]: r for r in json.load(open(syn / ("legacy-%s.client.json" % name)))["requests"]}
        c = {r["ref"]: r for r in json.load(open(syn / "syn-ctl.client.json"))["requests"]}
        refs = [k for k, r in a.items() if r["phase"] == "timed"]
        d = [a[k]["wall_ms"] - c[k]["wall_ms"] for k in refs]
        cm = statistics.mean(c[k]["wall_ms"] for k in refs)
        r = ov[("legacy-" + name, kind)]
        ok = (int(r["n_pairs"]) == len(refs) and abs(float(r["delta_mean_ms"]) - statistics.mean(d)) < 1e-5 and abs(float(r["delta_pct_of_ctl_mean"]) - 100 * statistics.mean(d) / cm) < 1e-4
              and abs(float(r["delta_sd_ms"]) - statistics.stdev(d)) < 1e-5 and abs(float(r["ctl_mean_ms"]) - cm) < 1e-5)
        check(ok, "F: %s: %d pairs, delta mean %.4f ms (injected %.1f ms +- noise), %.2f %% of control, sd %.4f ms" % (kind, len(refs), statistics.mean(d), exp["inject_ms"][name], 100 * statistics.mean(d) / cm, statistics.stdev(d)))
        check(abs(statistics.mean(d) - exp["inject_ms"][name]) < 0.2, "F: %s: recovered difference within 0.2 ms of the injected %.1f ms" % (kind, exp["inject_ms"][name]))
    dv = {r["kind"]: r for r in tsv(of / "overhead-vs-dec.tsv")}
    check(all(abs(float(dv[k]["nsys_minus_dec_ms"]) - (exp["inject_ms"][n] + 0.5)) < 1e-3 for n, k in (("short", "short2q"), ("medium", "medium3q"), ("long", "long3q"))),
          "F: --dec: nsys minus decomposition client = injected + 0.5 ms for the three kinds (%s)" % ", ".join("%s %.3f" % (k, float(v["nsys_minus_dec_ms"])) for k, v in dv.items()))
    wh = {(r["run"], r["kind"]): r for r in tsv(of / "overhead-where.tsv")}
    reqs = [r for r in tsv(oa / "passes.tsv") if r["row_kind"] == "request" and r["phase"] != "warmup"]
    bad = []
    for (run_, kind), r in wh.items():
        rs = [q for q in reqs if q["run"] == run_]
        napi = statistics.mean(float(q["n_cudaLaunchKernel_api_calls"]) + float(q["n_sync_calls"]) + float(q["n_memcpy"]) for q in rs)
        host = statistics.mean(float(q["send_to_first_kernel_ms"]) + float(q["gap_inter_pass_ms"]) + float(q["last_kernel_to_recv_ms"]) for q in rs)
        d = float(r["delta_ms"])
        if abs(float(r["delta_us_per_cuda_api_call"]) - 1000 * d / napi) > 1e-4 or abs(float(r["delta_pct_of_host_time"]) - 100 * d / host) > 1e-3:
            bad.append((run_, r["delta_us_per_cuda_api_call"], 1000 * d / napi))
        wall = statistics.mean(float(q["client_wall_ms"]) for q in rs)
        parts = [statistics.mean(float(q[c]) for q in rs) for c in ("send_to_first_kernel_ms", "gpu_busy_ms", "gap_intra_pass_ms", "gap_inter_pass_ms", "last_kernel_to_recv_ms")]
        if abs(sum(parts) - wall) > 1e-4:
            bad.append((run_, "parts do not add up", sum(parts), wall))
    check(len(wh) == 3 and not bad, "F: --passes: Table 3 per-call figures and the wall-time parts recomputed independently (%d mismatches)" % len(bad))
    print("explained differences vs legacy rows (records before the first pass are dropped by the whole-server code): %d cells in columns %s of the first pass / its request / the trace total"
          % (len(KNOWN_SEEN), sorted({d[1] for d in KNOWN_SEEN})))
    import shutil
    for src, dst in ((oa / "summary.txt", "synthetic-A-summary.txt"), (oa / "requests.tsv", "synthetic-A-requests.tsv"),
                     (od_ / "summary.txt", "synthetic-D-summary.txt"), (of / "overhead.txt", "synthetic-F-overhead.txt"),
                     (of / "overhead.tsv", "synthetic-F-overhead.tsv")):
        shutil.copyfile(src, out / dst)
    print("\nsynthetic checks:", "ALL PASS" if not FAILS else "%d FAIL" % len(FAILS))
    (out / "synthetic-result.txt").write_text("ALL PASS\n" if not FAILS else "\n".join(FAILS) + "\n")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
