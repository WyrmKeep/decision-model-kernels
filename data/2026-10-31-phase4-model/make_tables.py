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


def mmq_by_size(plist, large, mid, small):
    """Sum of MMQ ms of the passes times a factor that depends on the token count of the pass."""
    total = 0.0
    for p in plist:
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
        return mmq_by_size(req_passes(t), 1 - 1 / prm["speedup_large"], 1 - 1 / prm["speedup_mid"], 1 - 1 / prm["speedup_small"])
    if model == "mmq_tune":
        return mmq_by_size(req_passes(t), prm["saved_large"], prm["saved_other"], prm["saved_other"])
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
        left = mmq_by_size(req_passes(t), 1 / gp["speedup_large"], 1 / gp["speedup_mid"], 1 / gp["speedup_small"])
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

# Table 4: combined saving of a set of targets.
# The savings of the targets do not add, because some targets change the same kernel time.
# The model applies the targets of the set in this sequence:
# 1. J replaces the suffix passes by one pass. The new pass has the kernel times of single_129_256,
#    multiplied by its token count / 158, and one output head. J also removes the gaps between the
#    suffix passes and adds extra_ms (the "ragged" function above).
# 2. A, C, D, F and G then use the passes that remain. A uses the number of output passes that remain.
# 3. E is the measured saving, multiplied by the share of the GLU and quantization time that remains after J.
# 4. F uses the quantization, RMS norm and element-wise time that remains after E.
# 5. B changes only the copies and the host time of the mask. J does not change the total mask bytes,
#    because the mask bytes are proportional to the token count.
# H and N change the same MMQ time as G, so the set does not have them.
COMBO_IDS = ["A", "B", "C", "D", "E", "F", "G", "J"]
tp = {x["id"]: params(x["parameters"]) for x in targets}
tconf = {x["id"]: float(x["confidence"]) for x in targets}
single_save = {x["id"]: {t: saving(x["model"], params(x["parameters"]), t) for t in TYPES} for x in targets}


def merged_pass(tokens):
    """The one suffix pass of J: the columns of single_129_256 times tokens / 158, with one output head."""
    one = passes["single_129_256"]
    m = {c: str(f(one, c) * tokens / 158.0) for c in one if c.endswith("_ms")}
    m.update(pass_type="merged_suffix", tokens="129-256", gdn_kernel="gdn_single", head_ms=one["head_ms"])
    return m


def combined(t, active):
    """Time saved for one request of type t when all targets in `active` give their predicted gains."""
    r = requests[t]
    plist = req_passes(t)
    n_out = int(r["output_passes"])
    save = 0.0
    if "J" in active and sum(p["pass_type"].startswith("suffix") for p in plist) >= 2:
        save += saving("ragged", tp["J"], t)
        plist = [p for p in plist if not p["pass_type"].startswith("suffix")] + [merged_pass(tp["J"][t + "_tokens"])]
        n_out = 1

    def s(col):
        return sum(f(p, col) for p in plist)

    if "A" in active:
        save += n_out * (f(passes["suffix_le8"], "head_ms") - tp["A"]["head_after_ms"])
    if "B" in active:
        save += saving("mask", tp["B"], t)
    if "C" in active:
        save += s("concat_ms") * (1.0 - tp["C"]["kept_fraction"])
    if "D" in active:
        save += sum(f(p, "gdn_ms") for p in plist if p["gdn_kernel"] == "gdn_single") * (1.0 - 1.0 / tp["D"]["speedup"])
    e_ms = 0.0
    if "E" in active:
        share = (s("glu_ms") + s("quantize_ms")) / (col_sum(t, "glu_ms") + col_sum(t, "quantize_ms"))
        e_ms = saving("measured", tp["E"], t) * share
        save += e_ms
    if "F" in active:
        rest = s("quantize_ms") + s("rms_norm_ms") + s("elementwise_ms") - e_ms
        save += max(0.0, rest) * tp["F"]["saved_fraction"]
    if "G" in active:
        g = tp["G"]
        save += mmq_by_size(plist, 1 - 1 / g["speedup_large"], 1 - 1 / g["speedup_mid"], 1 - 1 / g["speedup_small"])
    return save


# Self-check: a set with one target gives the saving of Table 2 for that target.
for i in COMBO_IDS:
    for t in TYPES:
        assert abs(combined(t, {i}) - single_save[i][t]) < 1e-9, (i, t)

# Expected saving: the mean over all 2^8 outcomes. Each target passes with the probability of its
# confidence, independently of the other targets.
outcomes = []
for mask in range(1 << len(COMBO_IDS)):
    act = {COMBO_IDS[k] for k in range(len(COMBO_IDS)) if mask >> k & 1}
    w = 1.0
    for i in COMBO_IDS:
        w *= tconf[i] if i in act else 1.0 - tconf[i]
    outcomes.append((w, act))

full = {t: combined(t, set(COMBO_IDS)) for t in TYPES}
expd = {t: sum(w * combined(t, act) for w, act in outcomes) for t in TYPES}
simple = {t: sum(single_save[i][t] for i in COMBO_IDS) for t in TYPES}
rows = []
for scope in TYPES + ["lat_ms"]:
    if scope == "lat_ms":
        wall = lat
        fu, ex, si = (sum(d[t] * cases[t] for t in TYPES) / n_cases for d in (full, expd, simple))
    else:
        wall = f(requests[scope], "wall_ms")
        fu, ex, si = full[scope], expd[scope], simple[scope]
    rows.append([scope, "+".join(COMBO_IDS), f"{wall:.1f}", f"{si:.1f}", f"{fu:.1f}", f"{wall - fu:.1f}",
                 f"{-100 * fu / wall:.1f}", f"{ex:.1f}", f"{wall - ex:.1f}", f"{-100 * ex / wall:.1f}"])
write_tsv("sum.tsv", ["scope", "targets", "time_now_ms", "simple_sum_ms", "all_pass_saved_ms", "all_pass_time_ms",
                      "all_pass_pct", "expected_saved_ms", "expected_time_ms", "expected_pct"], rows)
print(f"lat_ms of the model: {lat:.1f}")
print(f"wrote {OUT}/floors.tsv, savings.tsv, ranked.tsv, sum.tsv")
