#!/usr/bin/env python3
"""Self-test of analyze_ncu.py (called by `analyze_ncu.py --self-test`; prints the results, returns 0 when everything passes).

1. parse_name on demangled names of the traces
2. synthetic reports (make_synthetic.py): aligned, base units, shifted by 3, misaligned, missing metrics, empty, three reports in one call;
   speed-of-light numbers compared with the hand computation in synthetic-expected.json
3. the sample reports shipped with the local ncu ($DMK_NCU; real .ncu-rep files through `ncu --import --csv --page raw`; they hold other kernels, so
   they test the parser, the units and the report-to-CSV step, not the llama classes): default units vs --print-units base must agree.
   Without DMK_NCU (or an ncu on PATH) this part prints SKIP and does not count as a failure.
All files of the test go to a temporary folder (TMPDIR is honored; $DMK_TEST_OUT keeps them in that folder). Nothing is written into the repository.
"""
import csv
import glob
import json
import os
import subprocess
import sys
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_ncu as an  # noqa: E402
import make_synthetic as ms  # noqa: E402
import ncu_cfg as cfg  # noqa: E402

TEST = None   # temporary folder of the run, set in main()
RES = []


def check(name, cond, detail=""):
    RES.append(("PASS" if cond else "FAIL", name, detail))


def close(a, b, rel=1e-6, ab=1e-9):
    return a is not None and b is not None and abs(a - b) <= max(ab, rel * abs(b))


def read_tsv(path):
    with open(path) as f:
        rows = list(csv.reader(f, delimiter="\t"))
    head = [h.split(" [")[0] for h in rows[0]]
    return [dict(zip(head, r)) for r in rows[1:]]


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def t_names():
    f, t = an.parse_name("void mul_mat_q<(ggml_type)8, (int)128, (bool)0, (ggml_prec)30, (bool)0>(const char *, const char *, int *)")
    check("parse_name mmq Q8_0 J=128", f == "mul_mat_q" and t[:3] == [8, 128, 0], f"{f} {t}")
    f, t = an.parse_name("void ggml_cuda_gated_delta_net_mma::gdn_single<(int)16, (int)128, (int)1>(const float *, float *)")
    check("parse_name namespaced gdn_single", f == "gdn_single" and t == [16, 128, 1], f"{f} {t}")
    f, t = an.parse_name("void mul_mat_vec_q<(ggml_type)2, (int)1, (bool)0, (bool)false>(const void *)")
    check("parse_name mmvq with bool false", f == "mul_mat_vec_q" and t == [2, 1, 0, 0], f"{f} {t}")
    f, t = an.parse_name("k_bin_bcast<&op_mul, float, float, float>(const float *)")
    check("parse_name template with a function pointer", f == "k_bin_bcast" and t[1:] == ["float", "float", "float"], f"{f} {t}")
    f, t = an.parse_name("rms_norm_f32")
    check("parse_name plain C name", f == "rms_norm_f32" and t == [], f"{f} {t}")
    check("unit factors", an.unit_factor("us") == 1e3 and an.unit_factor("Gbyte/s") == 1e9 and an.unit_factor("Mhz") == 1e6 and an.unit_factor("%") == 1.0
          and an.unit_factor("ms") == 1e6 and an.unit_factor("byte") == 1.0, "")
    check("number cells", an.num("1,234,567", "byte") == 1234567 and an.num("4.196352", "Mbyte") == 4196352 and an.num("", "us") is None and an.num("n/a") is None, "")


def analyze(files, out, ncu=None):
    return an.run_analysis(files, ncu or an.cfg.LOCAL_NCU, out, os.path.join(HERE, "plan"))


def t_synthetic():
    syn = os.path.join(TEST, "synthetic")
    subprocess.run([sys.executable, os.path.join(HERE, "make_synthetic.py"), "--out", syn], check=True, stdout=subprocess.DEVNULL)
    exp = json.load(open(os.path.join(syn, "synthetic-expected.json")))
    run = ms.RUN

    def path(v):
        return os.path.join(syn, v, run + ".csv")

    # ---- aligned
    r = analyze([path("aligned")], os.path.join(syn, "aligned", "analysis"))
    al = r["alignments"][run]
    check("aligned: status ALIGNED, 13 launches, shift 0", al["status"] == "ALIGNED" and al["n_obs"] == 13 and al["matched"] == 13 and al["shift"] == 0, str(al))
    sol = {s["class"]: s for s in r["sol"]}
    for cls, e in exp.items():
        s = sol.get(cls)
        if s is None:
            check(f"sol[{cls}] exists", False, "classes: " + "; ".join(sol))
            continue
        ok = s["bound"] == e["bound"] and close(s["floor_us"], e["floor_us"]) and close(s["pct_of_limit"], e["pct_of_limit"]) and close(s["gap_us"], e["gap_us"]) \
            and close(s["duration_us"], e["duration_us"])
        if "launches_per_pass" in e:
            ok = ok and s["launches_per_pass"] == e["launches_per_pass"]
        if "gap_ms_per_pass" in e:
            ok = ok and close(s["gap_ms_per_pass"], e["gap_ms_per_pass"])
        if "ms_per_pass" in e:
            ok = ok and close(s["ms_per_pass"], e["ms_per_pass"])
        if "useful_tops" in e:
            ok = ok and close(s["useful_tops"], e["useful_tops"])
        if "useful_tops_pct" in e:
            ok = ok and close(s["useful_tops_pct"], e["useful_tops_pct"])
        if "dram_gb_s" in e:
            ok = ok and close(s["dram_gb_s"], e["dram_gb_s"], 1e-5)
        check(f"sol[{cls}]: bound {e['bound']}, floor {e['floor_us']:.3f} us, {e['pct_of_limit']:.2f} % of limit", ok,
              f"got bound {s['bound']} floor {s['floor_us']:.3f} pct {s['pct_of_limit']:.2f} gap {s['gap_us']:.3f}; limits {s['floor_parts']}")
    ks = read_tsv(os.path.join(syn, "aligned", "analysis", "kernels.tsv"))
    k0 = ks[0]
    check("kernels.tsv: 13 rows, first is attn_qkv with its plan window label", len(ks) == 13 and k0["class"] == "mmq attn_qkv Q8_0 J=128 N=2048" and k0["align"] == "ALIGNED",
          f"{len(ks)} rows; {k0['class']}; window {k0['window']}")
    check("kernels.tsv: SM clock 1800 MHz and 3240 thousand cycles in 1.8 ms", close(fl(k0["sm_clock_mhz"]), 1800.0) and close(fl(k0["sm_cycles_k"]), 3240.0), f"{k0['sm_clock_mhz']} {k0['sm_cycles_k']}")
    check("kernels.tsv: stall shares 50 / 25 / 12.5 / 12.5 % (selected, not_selected excluded)",
          close(fl(k0["stall_long_scoreboard_pct"]), 50.0) and close(fl(k0["stall_short_scoreboard_pct"]), 25.0) and close(fl(k0["stall_barrier_pct"]), 12.5)
          and close(fl(k0["stall_mio_throttle_pct"]), 12.5), f"{k0['stall_top3']}")
    check("kernels.tsv: DRAM MB, GB/s and % of 422 GB/s of attn_qkv (tsv holds 3 decimals)", close(fl(k0["dram_read_mb"]), 28.0, 5e-4) and close(fl(k0["dram_write_mb"]), 67.108864, 5e-4)
          and close(fl(k0["dram_gb_s"]), 95.108864e6 / 1800e-6 / 1e9, 5e-4) and close(fl(k0["dram_pct_of_422"]), 100 * 95.108864e6 / 1800e-6 / 1e9 / 422, 5e-4),
          f"{k0['dram_read_mb']} {k0['dram_write_mb']} {k0['dram_gb_s']} {k0['dram_pct_of_422']}")
    check("kernels.tsv: nsys proxy duration of attn_qkv is 1720.42 us", close(fl(k0["nsys_proxy_us"]), 1720.42, 1e-3), k0["nsys_proxy_us"])
    cl = read_tsv(os.path.join(syn, "aligned", "analysis", "classes.tsv"))
    check("classes.tsv: one row per planned class", len(cl) == len({e["cls"] for e in json.load(open(os.path.join(HERE, "plan", run + ".json")))["expected"]}), f"{len(cl)} rows")
    md = open(os.path.join(syn, "aligned", "analysis", "sol.txt")).read()
    check("sol.txt: has the alignment, planned-vs-hit, SoL and coverage sections", all(h in md for h in ("## Alignment of the windows", "## Planned vs hit classes",
                                                                                                         "## Speed of light per class", "## Coverage of the passes")), "")
    check("sol.txt: planned-vs-hit has no DIFFERS line for the aligned run", "DIFFERS" not in md, "")

    # ---- base units give the same numbers
    rb = analyze([path("base")], os.path.join(syn, "base", "analysis"))
    sb = {s["class"]: s for s in rb["sol"]}
    same = all(close(sb[c]["duration_us"], sol[c]["duration_us"], 1e-4) and close(sb[c]["floor_us"], sol[c]["floor_us"], 1e-4) and close(sb[c]["dram_gb_s"], sol[c]["dram_gb_s"], 1e-3)
               for c in sol if c in sb)
    check("base units (thousands separators, ns / byte / hz) = auto units", same and len(sb) == len(sol), f"{len(sb)} vs {len(sol)} classes")

    # ---- shifted by 3
    rs = analyze([path("shifted")], os.path.join(syn, "shifted", "analysis"))
    al = rs["alignments"][run]
    check("shifted: status SHIFTED by +3 with the --shift hint (L0=-3)", al["status"].startswith("SHIFTED by +3") and "--shift L0=-3 " in al["status"]
          and al["shift"] == 3 and al["matched"] == 13, al["status"])
    planned = sum(1 for k in rs["kernels"] if k["align"] == "SHIFTED")
    check("shifted: 10 launches carry the plan class, the 3 after the window the kernel name", planned == 10, f"{planned} of {len(rs['kernels'])}")

    # ---- misaligned
    rm = analyze([path("misaligned")], os.path.join(syn, "misaligned", "analysis"))
    al = rm["alignments"][run]
    check("misaligned: status MISALIGNED, classes named from kernel names, no plan class", al["status"].startswith("MISALIGNED") and all(k["align"] == "unplanned" for k in rm["kernels"]), al["status"][:100])

    # ---- missing metrics
    rx = analyze([path("missing")], os.path.join(syn, "missing", "analysis"))
    kx = read_tsv(os.path.join(syn, "missing", "analysis", "kernels.tsv"))
    sx = {s["class"]: s for s in rx["sol"]}
    q = sx.get("mmq attn_qkv Q8_0 J=128 N=2048")
    check("missing metrics: no crash, 13 rows, empty cells, floors from the model, DRAM columns empty", len(kx) == 13 and kx[0]["dram_read_mb"] == "" and kx[0]["sm_clock_mhz"] == ""
          and q is not None and q["bound"] == "INT8" and close(q["floor_us"], exp["mmq attn_qkv Q8_0 J=128 N=2048"]["floor_us"]) and q["dram_gb_s"] is None, f"{len(kx)} rows")

    # ---- empty report and several reports
    re_ = analyze([path("empty")], os.path.join(syn, "empty", "analysis"))
    check("empty report: 0 kernels, status EMPTY, files written, sol.txt says so", re_["alignments"][run]["status"].startswith("EMPTY") and not re_["kernels"]
          and os.path.exists(os.path.join(syn, "empty", "analysis", "sol.txt")) and "no kernels" in open(os.path.join(syn, "empty", "analysis", "sol.txt")).read(), "")
    rr = analyze([path("aligned"), path("empty"), path("shifted"), path("missing")], os.path.join(syn, "several", "analysis"))
    runs = list(rr["alignments"])
    check("several reports with the same name: 4 runs kept apart, statuses ALIGNED / EMPTY / SHIFTED / ALIGNED", len(runs) == 4 and runs[1].endswith("#2")
          and [rr["alignments"][r]["status"].split()[0] for r in runs] == ["ALIGNED", "EMPTY", "SHIFTED", "ALIGNED"], str([(r, rr["alignments"][r]["status"][:20]) for r in runs]))
    check("several reports: kernels.tsv has 13 + 0 + 13 + 13 rows", len(rr["kernels"]) == 39, str(len(rr["kernels"])))


def t_samples():
    ncu = an.cfg.LOCAL_NCU
    if not ncu or not os.path.isdir(os.path.join(an.cfg.LOCAL_NCU_DIR, "extras", "samples")):
        print("SKIP: samples of the local ncu (set DMK_NCU to the ncu command line tool of Nsight Compute 2026.3.1)")
        return
    samples = os.path.join(an.cfg.LOCAL_NCU_DIR, "extras", "samples")
    reps = sorted(glob.glob(os.path.join(samples, "**", "*.ncu-rep"), recursive=True) + glob.glob(os.path.join(samples, "**", "*.ncu-repz"), recursive=True))
    check("sample reports found", len(reps) >= 1, f"{len(reps)} files under {samples}")
    if not reps:
        return
    out = os.path.join(TEST, "samples")
    r = analyze(reps, out, ncu)
    ks = r["kernels"]
    check("samples: every report gives a launch with duration, SM clock and 42 replay passes (--set full)", len(ks) == len(reps)
          and all(k["duration_us"] and 300 < k["sm_clock_mhz"] < 3000 and k["passes"] == 42 for k in ks), f"{len(ks)} launches from {len(reps)} reports")
    d = {k["run"]: k for k in ks}
    if "sobelDouble" in d:
        k = d["sobelDouble"]
        check("sobelDouble: 628.032 us, 1049.252 MHz, DRAM read 4.196352 MB (values read by hand from `ncu --import --page details`)", close(k["duration_us"], 628.032, 1e-6)
              and close(k["sm_clock_mhz"], 1049.252, 1e-5) and close(k["dram_read_mb"], 4.196352, 1e-6), f"{k['duration_us']} {k['sm_clock_mhz']} {k['dram_read_mb']}")
    # default units against --print-units base
    home = tempfile.mkdtemp(prefix="ncu-home-")
    bad = []
    for rep in reps:
        text_a = an.ncu_csv(ncu, rep)
        p = subprocess.run([ncu, "--import", rep, "--csv", "--page", "raw", "--print-units", "base"], capture_output=True, text=True, env=dict(os.environ, HOME=home))
        ka, kb = an.parse_csv(text_a), an.parse_csv(p.stdout)
        if len(ka) != len(kb) or not ka:
            bad.append(os.path.basename(rep) + ": count")
            continue
        for a, b in zip(ka, kb):
            for key, v in a["m"].items():
                w = b["m"].get(key)
                if v is None and w is None:
                    continue
                if v is None or w is None or abs(v - w) > max(2e-6 * abs(w), 0.006 + 1e-6):
                    bad.append(f"{os.path.basename(rep)} {key}: auto {v} base {w}")
    shutil.rmtree(home, ignore_errors=True)
    check("samples: default (auto-scaled) units = --print-units base for every metric the analysis reads", not bad, "; ".join(bad[:5]))
    # directory input
    d1 = os.path.dirname(reps[0])
    inp = an.collect_inputs([d1])
    check("directory input expands to its .ncu-rep / .ncu-repz files", all(i.endswith((".ncu-rep", ".ncu-repz")) for i in inp) and len(inp) >= 1, str(len(inp)))


def main():
    global TEST
    keep = os.environ.get("DMK_TEST_OUT")
    TEST = keep or tempfile.mkdtemp(prefix="ncu-selftest-")
    try:
        return run_tests()
    finally:
        if not keep:
            shutil.rmtree(TEST, ignore_errors=True)


def run_tests():
    t_names()
    t_synthetic()
    t_samples()
    lines = [f"{s}  {n}" + (f"  [{d}]" if d and s == "FAIL" else (f"  ({d})" if d else "")) for s, n, d in RES]
    nf = sum(1 for s, _, _ in RES if s == "FAIL")
    lines.append(f"{len(RES) - nf} passed, {nf} failed")
    os.makedirs(TEST, exist_ok=True)
    with open(os.path.join(TEST, "self-test.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if nf else 0


if __name__ == "__main__":
    sys.exit(main())
