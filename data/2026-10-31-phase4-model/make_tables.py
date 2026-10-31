#!/usr/bin/env python3
"""Phase 4 model: predicted time saved by each target, per request type.

Reads raw/pass_kernels.tsv, raw/requests.tsv and raw/targets.tsv.
Writes tables/floors.tsv, tables/savings.tsv and tables/ranked.tsv.
Python 3 standard library only. Run from any folder:

    python3 make_tables.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "tables")
TYPES = ["short1q", "short2q", "medium", "long"]


def read_tsv(name):
    with open(os.path.join(RAW, name), newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def params(text):
    out = {}
    for item in filter(None, text.split(";")):
        key, value = item.split("=")
        out[key] = float(value)
    return out


passes = {r["pass_type"]: r for r in read_tsv("pass_kernels.tsv")}
requests = {r["request_type"]: r for r in read_tsv("requests.tsv")}
targets = read_tsv("targets.tsv")


def f(row, col):
    return float(row[col])


def req_passes(t):
    return [passes[p] for p in requests[t]["passes"].split(",")]


def col_sum(t, col):
    return sum(f(p, col) for p in req_passes(t))


def mmq_by_size(t, large, mid, small):
    """Sum of MMQ ms times a factor that depends on the token count of the pass."""
    total = 0.0
    for p in req_passes(t):
        tok = p["tokens"]
        if tok in ("2048", "1025-2047"):
            total += f(p, "mmq_ms") * large
        elif tok in ("65-128", "129-256"):
            total += f(p, "mmq_ms") * mid
        else:
            total += f(p, "mmq_ms") * small
    return total


def saving(model, prm, t):
    r = requests[t]
    wall = f(r, "wall_ms")
    if model == "head":
        return int(r["output_passes"]) * (f(passes["suffix_le8"], "head_ms") - prm["head_after_ms"])
    if model == "mask":
        up = f(r, "upload_ms")
        return up + f(r, "mask_fill_ms") - up * prm["gpu_fraction"]
    if model == "concat":
        return col_sum(t, "concat_ms") * (1.0 - prm["kept_fraction"])
    if model == "gdn":
        gdn = sum(f(p, "gdn_ms") for p in req_passes(t) if p["gdn_kernel"] == "gdn_single")
        return gdn * (1.0 - 1.0 / prm["speedup"])
    if model == "measured":
        return wall * prm[t] / 100.0
    if model == "fusion":
        return (col_sum(t, "quantize_ms") + col_sum(t, "rms_norm_ms") + col_sum(t, "elementwise_ms")) * prm["saved_fraction"]
    if model == "mmq":
        return mmq_by_size(t, 1 - 1 / prm["speedup_large"], 1 - 1 / prm["speedup_mid"], 1 - 1 / prm["speedup_small"])
    if model == "mmq_tune":
        return mmq_by_size(t, prm["saved_large"], prm["saved_other"], prm["saved_other"])
    if model == "ragged":
        suffixes = [p for p in req_passes(t) if p["pass_type"].startswith("suffix")]
        if len(suffixes) < 2:
            return 0.0
        gap = prm["gap_ms"] if len(suffixes) == 2 else prm["gap3_ms"]
        old = sum(f(p, "all_ms") for p in suffixes) + gap * (len(suffixes) - 1)
        one = passes["single_129_256"]
        new = f(one, "all_ms") * prm[t + "_tokens"] / 158.0 + prm["extra_ms"]
        return old - new
    if model == "graphs":
        return f(r, "idle_in_passes_ms") * prm["saved_fraction"]
    if model == "fa":
        return col_sum(t, "fa_ms") * prm["saved_fraction"]
    if model == "mmq_after_g":
        g = next(x for x in targets if x["model"] == "mmq")
        gp = params(g["parameters"])
        left = mmq_by_size(t, 1 / gp["speedup_large"], 1 / gp["speedup_mid"], 1 / gp["speedup_small"])
        return left * (1.0 - 1.0 / prm["speedup"])
    if model == "build":
        return f(r, "build_ms") * prm["saved_fraction"]
    if model == "mega":
        return f(r, "idle_in_passes_ms") + col_sum(t, "all_ms") * prm["saved_fraction"]
    raise ValueError(model)


cases = {t: int(requests[t]["cases"]) for t in TYPES}
n_cases = sum(cases.values())
lat = sum(f(requests[t], "wall_ms") * cases[t] for t in TYPES) / n_cases

# Table 1: floors.
rows = []
for t in TYPES:
    r = requests[t]
    wall = f(r, "wall_ms")
    rows.append([t, r["cases"], f"{wall:.1f}", f"{col_sum(t, 'all_ms'):.1f}", r["weight_floor_ms"], r["roofline_ms"],
                 f"{wall / f(r, 'roofline_ms'):.2f}"])
write_tsv("floors.tsv", ["request_type", "cases", "wall_ms", "kernel_ms", "weight_floor_ms", "roofline_ms", "wall_over_roofline"], rows)

# Table 2: savings per target and request type.
rows, ranked = [], []
for x in targets:
    prm = params(x["parameters"])
    conf = float(x["confidence"])
    ms = {t: saving(x["model"], prm, t) for t in TYPES}
    pct = {t: 100.0 * ms[t] / f(requests[t], "wall_ms") for t in TYPES}
    lat_ms = sum(ms[t] * cases[t] for t in TYPES) / n_cases
    lat_pct = 100.0 * lat_ms / lat
    exp_pct = {t: pct[t] * conf for t in TYPES}
    row = [x["id"], x["target"], x["class"], x["effort"], f"{conf:.2f}"]
    for t in TYPES:
        row += [f"{ms[t]:.1f}", f"{pct[t]:.1f}"]
    row += [f"{lat_ms:.1f}", f"{lat_pct:.1f}", f"{max(exp_pct.values()):.1f}", f"{lat_pct * conf:.1f}"]
    rows.append(row)
    ranked.append((lat_pct * conf, row))
header = ["id", "target", "class", "effort", "confidence"]
for t in TYPES:
    header += [f"{t}_ms", f"{t}_pct"]
header += ["lat_ms", "lat_pct", "max_expected_pct", "expected_lat_pct"]
write_tsv("savings.tsv", header, rows)

# Table 3: the same rows, sorted by the expected gain of lat_ms (gain times confidence).
write_tsv("ranked.tsv", header, [r for _, r in sorted(ranked, key=lambda z: -z[0])])

# Table 4: simple sum of a set of targets that do not change the same time.
# H is left out because it changes the same MMQ time as G. N is left out because it comes after G.
SUM_IDS = ["A", "B", "C", "D", "E", "F", "G", "J"]
by_id = {r[0]: r for r in rows}
lat_col = header.index("lat_ms")
pred = sum(float(by_id[i][lat_col]) for i in SUM_IDS)
expd = sum(float(by_id[i][lat_col]) * float(by_id[i][4]) for i in SUM_IDS)
write_tsv("sum.tsv", ["targets", "lat_ms_now", "lat_ms_predicted", "lat_ms_predicted_pct", "lat_ms_expected", "lat_ms_expected_pct"],
          [["+".join(SUM_IDS), f"{lat:.1f}", f"{lat - pred:.1f}", f"{-100 * pred / lat:.1f}", f"{lat - expd:.1f}", f"{-100 * expd / lat:.1f}"]])
print(f"lat_ms of the model: {lat:.1f}")
print(f"wrote {OUT}/floors.tsv, savings.tsv, ranked.tsv, sum.tsv")
