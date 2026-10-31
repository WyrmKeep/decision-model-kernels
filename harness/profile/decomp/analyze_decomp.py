#!/usr/bin/env python3
"""Decomposition of the request time of the decision server from client stamps and DMK_TIMING lines (phase 3, package 1).

Usage:
  python3 analyze_decomp.py <results dir> [--run p3-dec-timing] [--ref p3-dec-ref] [--out-dir DIR]
  python3 analyze_decomp.py --self-test [--test-dir DIR]        (synthetic data with known truth, see make_synthetic.py)

Reads <run>.client.json (client stamps, CLOCK_MONOTONIC), <run>.server.log (DMK_TIMING lines and the slot lines of the server)
and, for the instrumentation check, <ref>.client.json. python3 stdlib only; ../census/census_lib.py is imported read-only for
the floors of every pass (if it cannot be imported the floor columns stay empty). The census tables (census-cases.tsv, census-passes.tsv)
are found with census_lib.data_file() in the census data set ($DMK_CENSUS_DATA, default data/2026-10-30-shape-census of the repository).
Writes requests.tsv, passes.tsv, the table TSVs and summary.txt into the output directory and prints the tables.
What every column means: COLUMNS.tsv and NOTES.txt next to this file.
"""
import argparse
import bisect
import collections
import csv
import json
import math
import os
import re
import sys
import tempfile

sys.dont_write_bytecode = True      # no __pycache__ next to the scripts and in extra/ (rsynced to the host)

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.abspath(os.path.join(HERE, "..", "census"))    # census_lib.py
EARLIER_SHORT_MS = 36.7        # deep review: "a whole short prompt in one pass takes 36.7 ms" (untraced, build D)
TOL_US = 3.0                   # rounding of the logged stamps (t= has 6 decimals, durations are printed with %.0f)
PHASES = ("unique", "cycle", "pair1", "pair2")
KIND_ORDER = ("short1q", "short2q", "medium3q", "long3q")

# stacked decomposition, in time order: (column, what it is, where it comes from)
STACK = [
    ("http_in_ms", "client send -> server handler start", "server request t - total_us minus client t_send"),
    ("parse_ms", "JSON parse + parse_questions/state + prepare_state", "request.parse_us"),
    ("build_ms", "prompt render + tokenize + task grouping", "request.build_us"),
    ("sched_ms", "post_tasks -> first llama_decode start (queue wake-up, slot launch, batch build)", "decode.t - total_us of the first decode minus end of build"),
    ("first_setup_ms", "first llama_decode start -> first ubatch start (batch alloc, memory init_batch, output reserve)", "first ubatch t - total_us minus first decode start"),
    ("prepare_ms", "host graph preparation of all passes: memory apply, graph build, alloc, set inputs", "sum ubatch apply_us+build_us+alloc_us+inputs_us"),
    ("submit_ms", "host time of the compute submit (op dispatch / CUDA launches) of all passes", "sum ubatch submit_us"),
    ("between_ms", "host time between passes (end of ubatch i -> start of ubatch i+1) without the GPU waits", "stamps of ubatch/decode/sync lines"),
    ("post_submit_ms", "end of the last ubatch -> start of the blocking sync (decode tail, post_decode, logits read call)", "stamps"),
    ("gpu_wait_ms", "host blocked for the GPU (all syncs)", "sum sync.wait_us"),
    ("tail_server_ms", "end of the last sync -> results ready in the HTTP thread (send_decision, release, queue, wake-up)", "request t - format_us minus last sync t"),
    ("format_ms", "answer formatting (softmax of the answer tokens, JSON of the answers)", "request.format_us"),
    ("http_out_ms", "server request line -> response body read by the client (res->ok JSON dump, HTTP write, loopback, docker proxy, client read)", "client t_recv minus request t"),
]
STACK_COLS = [c for c, _, _ in STACK]
BETWEEN_DETAIL = [
    ("between_within_decode_ms", ("ub_end", "ub_start")),
    ("between_decode_post_ms", ("ub_end", "dec_end")),
    ("between_presync_ms", ("dec_end", "sync_begin")),
    ("between_loop_ms", ("dec_end", "dec_start"), ("sync_end", "dec_start"), ("sync_end", "sync_begin")),
    ("between_decode_pre_ms", ("dec_start", "ub_start")),
]


class AnalysisError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------- small utils
def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se(xs):
    xs = [x for x in xs if x is not None]
    return sd(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")


def stats(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return {"n": 0, "mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"n": len(xs), "mean": mean(xs), "p50": pct(xs, 50), "p95": pct(xs, 95), "min": min(xs), "max": max(xs)}


def fnum(x, nd=None):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if nd is None:
            nd = 1 if abs(x) >= 100 else 2 if abs(x) >= 10 else 3
        return f"{x:.{nd}f}"
    return str(x)


def tsvnum(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, bool):
        return "1" if x else "0"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


class Out:
    """Tables: printed as markdown, written as TSV, collected into summary.txt."""

    def __init__(self, outdir, quiet=False):
        self.dir = outdir
        self.quiet = quiet
        self.md = []
        self.files = []
        if outdir:
            os.makedirs(outdir, exist_ok=True)

    def text(self, s=""):
        self.md.append(s)
        if not self.quiet:
            print(s)

    def table(self, name, title, header, rows, source=None, note=None, show=True):
        if self.dir:
            path = os.path.join(self.dir, name + ".tsv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f, delimiter="\t", lineterminator="\n")
                w.writerow(header)
                for r in rows:
                    w.writerow([tsvnum(x) for x in r])
            self.files.append(path)
        self.text(f"\n### {title}\n")
        if note:
            self.text(note + "\n")
        if not show:
            self.text(f"({len(rows)} rows, not printed: see {name}.tsv)")
            if source:
                self.text(f"\nSource: {source}")
            return
        self.text("| " + " | ".join(header) + " |")
        self.text("|" + "|".join("---" for _ in header) + "|")
        for r in rows:
            self.text("| " + " | ".join(fnum(x) for x in r) + " |")
        if source:
            self.text(f"\nSource: {source}")

    def write_summary(self):
        if self.dir:
            with open(os.path.join(self.dir, "summary.txt"), "w") as f:
                f.write("\n".join(self.md) + "\n")


# ---------------------------------------------------------------------------------------------------------------- parsing
NUMRE = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
KVRE = re.compile(r"(\w+)=(\S+)")
DMKRE = re.compile(r"DMK_TIMING (\w+) ")
SLOTRE = re.compile(r"(\d+)\.(\d\d)\.(\d{3})\.(\d{3}) [A-Z] slot\s+(\w+):\s+id\s+(\d+) \| task (-?\d+) \| (.*?)\s*$")
NEEDED = {
    "ubatch": ("t", "n_tokens", "n_seqs", "n_seq_tokens", "n_outputs", "reuse", "apply_us", "build_us", "alloc_us", "inputs_us", "submit_us", "total_us"),
    "cuda_compute": ("t", "n_nodes", "host_us"),
    "decode": ("t", "n_tokens", "n_ubatches", "init_us", "total_us"),
    "sync": ("t", "passes", "tokens", "wait_us", "since_first_submit_us", "since_last_submit_us"),
    "request": ("t", "n_questions", "n_groups", "parse_us", "build_us", "wait_us", "format_us", "total_us"),
}


def conv(s):
    if NUMRE.match(s):
        return float(s) if ("." in s or "e" in s or "E" in s) else int(s)
    return s


def parse_log(path):
    """-> dict(events=[dmk events], slots=[slot events], n_lines, n_dmk, n_bad, head=[first lines])."""
    events, slots, head = [], [], []
    n_lines = n_bad = 0
    with open(path, errors="replace") as f:
        for pos, line in enumerate(f):
            n_lines += 1
            if len(head) < 20:
                head.append(line.rstrip("\n")[:220])
            marks = [m for m in DMKRE.finditer(line)]
            if marks:
                for i, m in enumerate(marks):
                    end = marks[i + 1].start() if i + 1 < len(marks) else len(line)
                    kind = m.group(1)
                    kv = {k: conv(v) for k, v in KVRE.findall(line[m.end():end])}
                    need = NEEDED.get(kind)
                    if need is None:
                        continue                      # a line kind this version does not know
                    if any(not isinstance(kv.get(k), (int, float)) for k in need):
                        n_bad += 1
                        continue
                    kv["kind"] = kind
                    kv["pos"] = pos
                    kv["end_us"] = round(kv["t"] * 1e6)
                    if kind in ("ubatch", "decode", "request"):
                        kv["start_us"] = kv["end_us"] - kv["total_us"]
                    if kind == "sync":
                        kv["start_us"] = kv["end_us"] - kv["wait_us"]
                    events.append(kv)
                continue
            m = SLOTRE.search(line)
            if m:
                tl = (((int(m.group(1)) * 60 + int(m.group(2))) * 1000 + int(m.group(3))) * 1000 + int(m.group(4)))
                func, slot, task, text = m.group(5), int(m.group(6)), int(m.group(7)), m.group(8)
                ev = {"pos": pos, "tlog_us": tl, "func": func, "slot": slot, "task": task, "text": text, "kind": "other"}
                if func.startswith("get_availab"):
                    ev["kind"] = "select"
                    if "selected slot by LRU" in text:
                        ev["how"] = "LRU"
                        mm = re.search(r"t_last = (-?\d+)", text)
                        ev["t_last_us"] = int(mm.group(1)) if mm else -1
                    elif "selected slot by LCP similarity" in text:
                        ev["how"] = "LCP"
                        mm = re.search(r"f_sim_best = ([\d.]+)", text)
                        ev["f_sim_best"] = float(mm.group(1)) if mm else None
                    elif "selected slot by id" in text:
                        ev["how"] = "id"
                elif func.startswith("launch_slot"):
                    mm = re.search(r"is_child = (\d)", text)
                    ev["kind"] = "launch"
                    ev["is_child"] = int(mm.group(1)) if mm else 0
                elif func == "release":
                    mm = re.search(r"n_tokens = (\d+)", text)
                    ev["kind"] = "release"
                    ev["n_tokens"] = int(mm.group(1)) if mm else None
                slots.append(ev)
    return {"events": events, "slots": slots, "n_lines": n_lines, "n_bad": n_bad, "head": head,
            "n_dmk": len(events)}


def load_client(path):
    with open(path) as f:
        j = json.load(f)
    reqs = sorted(j["requests"], key=lambda r: r["idx"])
    for r in reqs:
        r["t_send_us"] = r["t_send_mono_ns"] / 1e3
        r["t_recv_us"] = r["t_recv_mono_ns"] / 1e3
    return j, reqs


# ---------------------------------------------------------------------------------------------------------------- matching
def match_requests(creqs, sreqs):
    """Client requests <-> server `request` lines. Returns dict(mode, pairs [(ci, si)], in_ms, out_ms, ...).
    mode 'time': a server request matches when its end lies between the client send and receive stamps (and its start after the
    send stamp) - valid when >= 95 % of the client requests find their server request this way, or (log cut short, requests lost)
    when >= 95 % of the server request lines find their client request. Else mode 'order' (n-th request line = n-th client request):
    the two clocks do not agree."""
    ends = [s["end_us"] for s in sreqs]
    used = set()
    pairs = []
    for ci, c in enumerate(creqs):
        lo = bisect.bisect_left(ends, c["t_send_us"] - TOL_US)
        cand = []
        k = lo
        while k < len(sreqs) and ends[k] <= c["t_recv_us"] + TOL_US:
            if k not in used and sreqs[k]["start_us"] >= c["t_send_us"] - TOL_US:
                cand.append(k)
            k += 1
        if len(cand) == 1:
            used.add(cand[0])
            pairs.append((ci, cand[0]))
    frac = len(pairs) / len(creqs) if creqs else 0.0
    info = {"n_client": len(creqs), "n_server": len(sreqs), "n_time_matched": len(pairs), "time_fraction": frac}
    # offset estimate from the order pairing, for the report in both modes
    n = min(len(creqs), len(sreqs))
    if n:
        offs = [((creqs[i]["t_send_us"] + creqs[i]["t_recv_us"]) - (sreqs[i]["start_us"] + sreqs[i]["end_us"])) / 2e3 for i in range(n)]
        info["order_offset_ms_median"] = pct(offs, 50)
        info["order_offset_ms_p5"] = pct(offs, 5)
        info["order_offset_ms_p95"] = pct(offs, 95)
    frac_s = len(pairs) / len(sreqs) if sreqs else 0.0
    info["server_fraction"] = frac_s
    info["lost_lines"] = frac < 0.95 and frac_s >= 0.95 and len(pairs) >= 20
    if frac >= 0.95 or info["lost_lines"]:
        info["mode"] = "time"
        info["pairs"] = pairs
    else:
        info["mode"] = "order"
        info["pairs"] = [(i, i) for i in range(n)]
    return info


# ---------------------------------------------------------------------------------------------------------------- census
class Census:
    """Floors of every pass (census_lib) and the expected pass structure of every case (census-cases.tsv, variant ref)."""

    def __init__(self):
        self.lib = None
        self.model = None
        self.err = None
        self.data_file = None
        self.cache = {}
        self.cases = collections.defaultdict(dict)       # case -> slots string -> passes string
        self.tokens = {}                                  # case -> tokens_evaluated
        self.passes_rows = []
        try:
            sys.path.insert(0, CENSUS)
            import census_lib
            self.lib = census_lib
            self.data_file = census_lib.data_file
            self.model = census_lib.Model()
        except Exception as e:                            # census_lib is optional: floors stay empty
            self.err = f"{type(e).__name__}: {e}"
        finally:
            if CENSUS in sys.path:
                sys.path.remove(CENSUS)
        try:
            if self.data_file is None:
                raise OSError("census_lib could not be imported")
            with open(self.data_file("census-cases.tsv")) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    if r["variant"] == "ref":
                        self.cases[r["case"]][r["slots_Q0_Q1_Q2"]] = r["passes"]
                        self.tokens[r["case"]] = int(r["tokens_evaluated"])
            with open(self.data_file("census-passes.tsv")) as f:
                self.passes_rows = list(csv.DictReader(f, delimiter="\t"))
        except OSError as e:
            self.err = (self.err or "") + f" census tsv: {e}"

    def floor(self, n_tokens, n_seqs, n_seq_tokens, n_out):
        key = (n_tokens, n_seqs, n_seq_tokens, n_out)
        if key in self.cache:
            return self.cache[key]
        res = None
        if self.lib is not None:
            try:
                p = {"n_tokens": n_tokens, "n_out": n_out, "seqs": [(0, n_seq_tokens)] * n_seqs}
                c = self.lib.pass_costs(self.model, p)
                res = {"weights_ms": c["floor_weights_s"] * 1e3, "int8_ms": c["floor_int8_s"] * 1e3,
                       "max_ms": c["floor_pass_max_s"] * 1e3, "roofline_ms": c["floor_roofline_s"] * 1e3}
            except Exception as e:
                self.err = f"floor: {type(e).__name__}: {e}"
        self.cache[key] = res
        return res

    def expected(self, case, slots):
        """Passes string(s) that census-cases.tsv predicts for the case with these slots (list of strings)."""
        d = self.cases.get(case)
        if not d:
            return []
        if slots in d:
            return [d[slots]]
        return sorted(set(d.values()))


def observed_passes_string(ubs, n_questions):
    """P<n> prefix pass (no output), F<n> the one pass of a one-question request, S<seqs>x<tokens per seq> suffix passes."""
    parts = []
    for u in ubs:
        if u["n_outputs"] == 0:
            parts.append(f"P{u['n_tokens']}")
        elif n_questions == 1:
            parts.append(f"F{u['n_tokens']}")
        else:
            parts.append(f"S{u['n_seqs']}x{u['n_seq_tokens']}")
    return "+".join(parts)


def compare_passes(obs, exp):
    """-> (match, ref_delta_tokens). Structure and the suffix sizes must be equal; P/F pass sizes differ by the length of the
    Ref value in tokens (census: median 13, range 9-14), so they may differ by up to 6 tokens."""
    po, pe = obs.split("+") if obs else [], exp.split("+") if exp else []
    if len(po) != len(pe) or not po:
        return False, None
    delta = 0
    for a, b in zip(po, pe):
        if a[0] == "S" or b[0] == "S":
            if a != b:
                return False, None
        else:
            if a[0] != b[0]:
                return False, None
            d = int(a[1:]) - int(b[1:])
            if abs(d) > 6:
                return False, None
            if delta == 0 or (a[0] in "PF" and delta == 0):
                delta = d
    return True, delta


# ---------------------------------------------------------------------------------------------------------------- slots
def assign_slot_events(slot_events, req_pos):
    """Slot lines of request k = lines between the request lines k-1 and k (file order). A `release` line that sits in front of
    the first select/launch line of a request comes from the previous one (the log thread is asynchronous)."""
    n = len(req_pos)
    buckets = [[] for _ in range(n)]
    for ev in sorted(slot_events, key=lambda e: e["pos"]):
        k = bisect.bisect_left(req_pos, ev["pos"])
        if k < n:
            buckets[k].append(ev)
    for k in range(1, n):
        b = buckets[k]
        i = 0
        while i < len(b) and b[i]["kind"] in ("release", "other"):
            i += 1
        if i:
            buckets[k - 1].extend(b[:i])
            buckets[k] = b[i:]
    return buckets


def estimate_anchor(slot_events):
    """Offset (us) from the log time prefix (since process start) to CLOCK_MONOTONIC: the `t_last` of a "selected slot by LRU"
    line is ggml_time_us() (CLOCK_MONOTONIC) at the release of that slot, the release line has the log time of the same moment."""
    last_rel = {}
    offs = []
    for ev in sorted(slot_events, key=lambda e: e["pos"]):
        if ev["kind"] == "release":
            last_rel[ev["slot"]] = ev["tlog_us"]
        elif ev["kind"] == "select" and ev.get("how") == "LRU" and ev.get("t_last_us", -1) > 0 and ev["slot"] in last_rel:
            offs.append(ev["t_last_us"] - last_rel[ev["slot"]])
    if len(offs) < 5:
        return None
    med = pct(offs, 50)
    mad = pct([abs(o - med) for o in offs], 50)
    if mad > 2000:
        return None
    return {"offset_us": med, "n": len(offs), "mad_us": mad}


# ---------------------------------------------------------------------------------------------------------------- one request
def timeline(e_built, t_wait_end, ubs, decs, syns):
    """Walk the host timeline of a request from the end of `build` to the end of the wait. Returns (cats, detail, n_neg):
    cats in us by column, detail by (previous mark, mark) for the `between` pieces."""
    cats = collections.defaultdict(float)
    detail = collections.defaultdict(float)
    marks = []
    for j, d in enumerate(decs):
        marks += [(d["start_us"], 1, "dec_start", j), (d["end_us"], 4, "dec_end", j)]
    for i, u in enumerate(ubs):
        marks += [(u["start_us"], 2, "ub_start", i), (u["end_us"], 3, "ub_end", i)]
    for m, s in enumerate(syns):
        marks += [(s["start_us"], 5, "sync_begin", m), (s["end_us"], 6, "sync_end", m)]
    marks.sort(key=lambda x: (x[0], x[1]))
    n_neg = 0
    if not ubs:
        cats["unsplit"] = t_wait_end - e_built
        return cats, detail, n_neg
    first_start, last_end = ubs[0]["start_us"], ubs[-1]["end_us"]
    prev_t, prev_k = e_built, "built"
    for t, _, kind, _idx in marks:
        d = t - prev_t
        if d < -TOL_US:
            n_neg += 1
        if prev_k == "ub_start" and kind == "ub_end":
            cats["pass"] += d
        elif prev_k == "sync_begin" and kind == "sync_end":
            cats["gpu_wait_ms"] += d
        elif t <= first_start:
            cats["sched_ms" if prev_k == "built" else "first_setup_ms"] += d
        elif prev_t >= last_end:
            cats["post_submit_ms"] += d
        else:
            cats["between_ms"] += d
            detail[(prev_k, kind)] += d
        prev_t, prev_k = t, kind
    d = t_wait_end - prev_t
    if d < -TOL_US:
        n_neg += 1
    cats["tail_server_ms"] += d
    if prev_k != "sync_end":
        cats["_no_sync_tail"] = d
    return cats, detail, n_neg


def analyze_request(c, R, ev, slot_ev, floors, clock_ok, anchor):
    """One matched request: row dict (requests.tsv) and list of pass dicts (passes.tsv)."""
    row = {"idx": c["idx"], "kind": c["kind"], "case": c["case"], "phase": c["phase"], "pair": c["pair"], "wall_ms": c["wall_ms"],
           "input_tokens": c["input_tokens"], "flags": []}
    flags = row["flags"]
    ubs = sorted((e for e in ev if e["kind"] == "ubatch"), key=lambda e: (e["end_us"], e["pos"]))
    decs = sorted((e for e in ev if e["kind"] == "decode"), key=lambda e: (e["end_us"], e["pos"]))
    syns = sorted((e for e in ev if e["kind"] == "sync"), key=lambda e: (e["end_us"], e["pos"]))
    cudas = sorted((e for e in ev if e["kind"] == "cuda_compute"), key=lambda e: (e["end_us"], e["pos"]))
    start = R["start_us"]
    e_built = start + R["parse_us"] + R["build_us"]
    t_wait_end = R["end_us"] - R["format_us"]
    row["server_total_ms"] = R["total_us"] / 1e3
    row["http_total_ms"] = c["wall_ms"] - R["total_us"] / 1e3
    row["http_in_ms"] = (start - c["t_send_us"]) / 1e3 if clock_ok else None
    row["http_out_ms"] = (c["t_recv_us"] - R["end_us"]) / 1e3 if clock_ok else None
    row["parse_ms"] = R["parse_us"] / 1e3
    row["build_ms"] = R["build_us"] / 1e3
    row["format_ms"] = R["format_us"] / 1e3
    row["n_questions"], row["n_groups"] = R["n_questions"], R["n_groups"]
    if clock_ok and (row["http_in_ms"] < -TOL_US / 1e3 or row["http_out_ms"] < -TOL_US / 1e3):
        flags.append("stamp_order")

    # ubatch <-> decode, cuda_compute, sync group
    for u in ubs:
        u["decode"] = -1
        for j, d in enumerate(decs):
            if d["start_us"] - TOL_US <= u["start_us"] and u["end_us"] <= d["end_us"] + TOL_US:
                u["decode"] = j
                break
        u["group"] = -1
        for m, s in enumerate(syns):
            if s["end_us"] >= u["end_us"] - TOL_US:
                u["group"] = m
                break
        cs = [x for x in cudas if u["start_us"] - TOL_US <= x["end_us"] <= u["end_us"] + TOL_US]
        u["cuda_mode"] = "+".join(sorted({str(x.get("mode", "?")) for x in cs})) if cs else None
        u["cuda_host_us"] = sum(x["host_us"] for x in cs) if cs else None
        u["floor"] = floors.floor(u["n_tokens"], u["n_seqs"], u["n_seq_tokens"], u["n_outputs"]) if floors else None
    if not ubs:
        flags.append("no_ubatch")
    if ubs and not syns:
        flags.append("no_sync")
    if any(u["group"] < 0 for u in ubs):
        flags.append("pass_without_sync")
    for m, s in enumerate(syns):
        members = [u for u in ubs if u["group"] == m]
        s["members"] = members
        if len(members) != s["passes"]:
            flags.append("sync_passes_mismatch")
        if members and abs((s["end_us"] - members[0]["start_us"]) - s["since_first_submit_us"]) > 2 * TOL_US:
            flags.append("span_mismatch")
    for j, d in enumerate(decs):
        if sum(1 for u in ubs if u["decode"] == j) != d["n_ubatches"]:
            flags.append("decode_ubatches_mismatch")
            break

    cats, detail, n_neg = timeline(e_built, t_wait_end, ubs, decs, syns)
    if n_neg:
        flags.append("neg_gap")
    if "_no_sync_tail" in cats:
        flags.append("no_sync_tail")
    prep = sum(u["apply_us"] + u["build_us"] + u["alloc_us"] + u["inputs_us"] for u in ubs)
    subm = sum(u["submit_us"] for u in ubs)
    row["apply_ms"] = sum(u["apply_us"] for u in ubs) / 1e3
    row["graph_build_ms"] = sum(u["build_us"] for u in ubs) / 1e3
    row["alloc_ms"] = sum(u["alloc_us"] for u in ubs) / 1e3
    row["inputs_ms"] = sum(u["inputs_us"] for u in ubs) / 1e3
    row["prepare_ms"] = prep / 1e3
    row["submit_ms"] = subm / 1e3
    for col in ("sched_ms", "first_setup_ms", "between_ms", "post_submit_ms", "gpu_wait_ms", "tail_server_ms"):
        row[col] = cats.get(col, 0.0) / 1e3
    row["unsplit_ms"] = cats.get("unsplit", 0.0) / 1e3
    for name, *pairs in BETWEEN_DETAIL:
        row[name] = sum(detail.get(p, 0.0) for p in pairs) / 1e3
    stack = [row[c_] for c_ in STACK_COLS if row.get(c_) is not None]
    row["residual_ms"] = c["wall_ms"] - (sum(stack) + row["unsplit_ms"]) - (0.0 if clock_ok else row["http_total_ms"])
    if not clock_ok:
        row["http_in_ms"] = row["http_out_ms"] = None
    # tokens, structure, reuse
    row["n_decodes"], row["n_ubatches"], row["n_syncs"] = len(decs), len(ubs), len(syns)
    row["tokens_evaluated"] = sum(u["n_tokens"] for u in ubs)
    row["passes_obs"] = observed_passes_string(ubs, R["n_questions"])
    row["ubatches_per_decode"] = ",".join(str(d["n_ubatches"]) for d in decs)
    row["reuse_flags"] = "".join(str(int(u["reuse"])) for u in ubs)
    row["cuda_modes"] = ",".join(sorted({u["cuda_mode"] for u in ubs if u["cuda_mode"]}))
    row["cuda_host_ms"] = sum(u["cuda_host_us"] for u in ubs if u["cuda_host_us"] is not None) / 1e3 if any(u["cuda_host_us"] is not None for u in ubs) else None

    # sync groups: span, floors, exposed host time (bounds)
    span = exp_min = exp_max = f_sum = f_roof = f_w = sub_first = sub_grp = 0.0
    have_floor = bool(ubs) and all(u["floor"] for u in ubs)
    for s in syns:
        mem = s["members"]
        if not mem:
            continue
        sp = (s["end_us"] - mem[0]["start_us"]) / 1e3
        wait = s["wait_us"] / 1e3
        host_in = sp - wait
        pre = (mem[0]["apply_us"] + mem[0]["build_us"] + mem[0]["alloc_us"] + mem[0]["inputs_us"]) / 1e3
        sub_first += mem[0]["submit_us"] / 1e3
        sub_grp += sum(u["submit_us"] for u in mem) / 1e3
        fg = sum(u["floor"]["max_ms"] for u in mem) if have_floor and all(u["floor"] for u in mem) else 0.0
        span += sp
        exp_min += min(pre, host_in)
        exp_max += max(min(pre, host_in), min(host_in, sp - max(wait, fg)))
        s["span_ms"], s["floor_ms"], s["exp_min"] = sp, fg, pre
    row["span_ms"] = span if syns else None
    row["exposed_min_ms"] = exp_min if syns else None
    row["exposed_max_ms"] = exp_max if syns else None
    row["submit_first_ms"] = sub_first if syns else None
    row["submit_later_ms"] = sub_grp - sub_first if syns else None
    row["floor_weights_ms"] = sum(u["floor"]["weights_ms"] for u in ubs) if have_floor else None
    row["floor_max_ms"] = sum(u["floor"]["max_ms"] for u in ubs) if have_floor else None
    row["floor_int8_ms"] = sum(u["floor"]["int8_ms"] for u in ubs) if have_floor else None
    row["floor_roofline_ms"] = sum(u["floor"]["roofline_ms"] for u in ubs) if have_floor else None

    # slots
    launches = sorted((e for e in slot_ev if e["kind"] == "launch"), key=lambda e: e["task"])
    selects = [e for e in slot_ev if e["kind"] == "select"]
    rels = [e for e in slot_ev if e["kind"] == "release"]
    row["slots"] = ",".join(str(e["slot"]) for e in launches) if launches else None
    row["slot_select"] = selects[0].get("how") if selects else None
    row["f_sim_best"] = selects[0].get("f_sim_best") if selects else None
    by_task = {e["task"]: e for e in rels}
    row["release_tokens"] = ",".join(str(by_task[e["task"]]["n_tokens"]) if e["task"] in by_task else "?" for e in launches) if launches else None
    if not launches:
        flags.append("no_slot_lines")
    elif len(launches) != R["n_questions"]:
        flags.append("slot_task_count")
    if launches and rels and c.get("input_tokens") is not None and all(e["task"] in by_task for e in launches):
        if sum(by_task[e["task"]]["n_tokens"] for e in launches) != c["input_tokens"]:
            flags.append("input_tokens_mismatch")
    # census prediction
    row["census_expected"] = None
    row["census_match"] = None
    row["census_ref_delta_tokens"] = None
    if floors is not None and ubs and R["n_groups"] in (1, R["n_questions"]):
        key = "-" if R["n_questions"] == 1 else (row["slots"] or "")
        exp = floors.expected(c["case"], key)
        if exp:
            res = [compare_passes(row["passes_obs"], x) for x in exp]
            ok = [r for r in res if r[0]]
            row["census_expected"] = exp[0] if len(exp) == 1 else "|".join(exp[:2]) + ("|..." if len(exp) > 2 else "")
            row["census_match"] = bool(ok)
            row["census_ref_delta_tokens"] = ok[0][1] if ok else None
    # scheduling split (estimate, needs the anchor)
    row["sched_wake_est_ms"] = row["sched_launch_est_ms"] = None
    if anchor is not None and launches and (decs or ubs):
        t_launch = max(e["tlog_us"] for e in launches) + anchor["offset_us"]
        t_first = decs[0]["start_us"] if decs else ubs[0]["start_us"]
        w, l = (t_launch - e_built) / 1e3, (t_first - t_launch) / 1e3
        if w >= -0.05 and l >= -0.05:
            row["sched_wake_est_ms"], row["sched_launch_est_ms"] = max(w, 0.0), max(l, 0.0)
        else:
            flags.append("anchor_implausible")

    # passes
    passes = []
    prev_end = e_built
    sync_before = 0.0
    for i, u in enumerate(ubs):
        s_between = sum(s["wait_us"] for s in syns if prev_end - TOL_US <= s["start_us"] and s["end_us"] <= u["start_us"] + TOL_US) / 1e3 if i else 0.0
        g = u["group"]
        last_of_group = g >= 0 and syns[g]["members"] and syns[g]["members"][-1] is u
        s = syns[g] if last_of_group else None
        p = {"idx": c["idx"], "kind": c["kind"], "case": c["case"], "phase": c["phase"], "pass": i, "decode": u["decode"], "group": g,
             "n_tokens": u["n_tokens"], "n_seqs": u["n_seqs"], "n_seq_tokens": u["n_seq_tokens"], "n_outputs": u["n_outputs"],
             "n_nodes": u.get("n_nodes"), "reuse": u["reuse"], "apply_ms": u["apply_us"] / 1e3, "graph_build_ms": u["build_us"] / 1e3,
             "alloc_ms": u["alloc_us"] / 1e3, "inputs_ms": u["inputs_us"] / 1e3, "submit_ms": u["submit_us"] / 1e3,
             "total_ms": u["total_us"] / 1e3, "cuda_mode": u["cuda_mode"],
             "cuda_host_ms": u["cuda_host_us"] / 1e3 if u["cuda_host_us"] is not None else None,
             "start_after_request_start_ms": (u["start_us"] - start) / 1e3,
             "gap_before_ms": (u["start_us"] - prev_end) / 1e3 - s_between, "sync_between_ms": s_between,
             "sync_passes": s["passes"] if s else None, "sync_wait_ms": s["wait_us"] / 1e3 if s else None,
             "sync_span_ms": s["span_ms"] if s and "span_ms" in s else None,
             "sync_since_last_submit_ms": s["since_last_submit_us"] / 1e3 if s else None,
             "floor_weights_ms": u["floor"]["weights_ms"] if u["floor"] else None, "floor_int8_ms": u["floor"]["int8_ms"] if u["floor"] else None,
             "floor_max_ms": u["floor"]["max_ms"] if u["floor"] else None, "floor_roofline_ms": u["floor"]["roofline_ms"] if u["floor"] else None,
             "floor_group_ms": s["floor_ms"] if s and "floor_ms" in s else None}
        passes.append(p)
        prev_end = u["end_us"]
    row["valid"] = (not any(f in flags for f in ("no_ubatch", "no_sync", "neg_gap", "sync_passes_mismatch", "pass_without_sync", "stamp_order", "decode_ubatches_mismatch", "no_sync_tail"))
                    and abs(row["residual_ms"]) <= 0.05)
    return row, passes


# ---------------------------------------------------------------------------------------------------------------- run analysis
def analyze(results_dir, run, ref, out_dir, quiet=False):
    cpath = os.path.join(results_dir, run + ".client.json")
    spath = os.path.join(results_dir, run + ".server.log")
    for p in (cpath, spath):
        if not os.path.isfile(p):
            raise AnalysisError(f"missing input file {p}")
    cj, creqs = load_client(cpath)
    log = parse_log(spath)
    if log["n_dmk"] == 0:
        raise AnalysisError(
            f"no DMK_TIMING lines in {spath} ({log['n_lines']} lines). Was the run started with DMK_TIMING=1 "
            f"(DMK_DOCKER_EXTRA=\"-e DMK_TIMING=1\" on build:ablate-timing)? First lines of the log:\n  " + "\n  ".join(log["head"]))
    sreqs = sorted((e for e in log["events"] if e["kind"] == "request"), key=lambda e: (e["end_us"], e["pos"]))
    if not sreqs:
        raise AnalysisError(
            f"{log['n_dmk']} DMK_TIMING lines but no `request` line in {spath}: the server build does not print the request line "
            f"(server-context.cpp, commit aaf86b052) or no request completed. First lines:\n  " + "\n  ".join(log["head"]))
    out = Out(out_dir, quiet)
    res = {"out": out}
    out.text(f"# Decomposition of the request time: run `{run}`" + (f", reference `{ref}`" if ref else ""))
    out.text(f"\nclient file {cpath}: {len(creqs)} requests, {len(cj.get('errors', []))} errors, reconnects {cj['meta'].get('reconnects')}; "
             f"server log {spath}: {log['n_lines']} lines, {log['n_dmk']} DMK_TIMING lines ({log['n_bad']} unusable), {len(sreqs)} request lines, "
             f"{len(log['slots'])} slot lines")

    # ---- matching and clock check
    m = match_requests(creqs, sreqs)
    clock_ok = m["mode"] == "time"
    res["match"] = m
    in_ms = [(sreqs[si]["start_us"] - creqs[ci]["t_send_us"]) / 1e3 for ci, si in m["pairs"]]
    out_ms = [(creqs[ci]["t_recv_us"] - sreqs[si]["end_us"]) / 1e3 for ci, si in m["pairs"]]
    n_viol = sum(1 for a, b in zip(in_ms, out_ms) if a < -TOL_US / 1e3 or b < -TOL_US / 1e3)
    rows_clock = []
    for name, xs in (("client send -> server request start (http_in_ms)", in_ms), ("server request end -> client receive (http_out_ms)", out_ms)):
        s_ = stats(xs)
        rows_clock.append([name, s_["n"], s_["mean"], s_["p50"], s_["min"], s_["max"], sum(1 for x in xs if x < -TOL_US / 1e3)])
    out.table("clock-check", "Clock check: server request line against the client stamps (same CLOCK_MONOTONIC)",
              ["interval", "n", "mean_ms", "p50_ms", "min_ms", "max_ms", "n_negative"], rows_clock,
              source="client t_send/t_recv_mono_ns; server `request` t and total_us. Both intervals must be >= 0 for every request.")
    out.text(f"\nMatching: {m['n_client']} client requests, {m['n_server']} server request lines, {m['n_time_matched']} matched by time "
             f"({100 * m['time_fraction']:.1f} %) -> mode **{m['mode']}**." )
    if clock_ok:
        extra = m["n_server"] - m["n_time_matched"]
        out.text(f"Clock check passed ({'>= 95 % of the client requests' if not m['lost_lines'] else '>= 95 % of the server request lines'} contain exactly one match); "
                 f"{extra} server request line(s) without a client request, {m['n_client'] - m['n_time_matched']} client request(s) without a server line"
                 + (" (log cut short or lines lost: those requests are not analysed)." if m["lost_lines"] else "."))
    else:
        out.text(f"**CLOCK MISMATCH or lost lines**: fewer than 95 % of the requests satisfy client send <= server start and server end <= client receive. "
                 f"Falling back to order-only matching (n-th request line = n-th client request, first {len(m['pairs'])}); "
                 f"http_in_ms / http_out_ms are not available, only their sum http_total_ms = wall - server total. "
                 f"Median offset of the interval midpoints (client - server) from the order pairing: {m.get('order_offset_ms_median', float('nan')):.3f} ms "
                 f"(p5 {m.get('order_offset_ms_p5', float('nan')):.3f}, p95 {m.get('order_offset_ms_p95', float('nan')):.3f}). "
                 f"[INFERENCE] a constant offset means the container does not see CLOCK_MONOTONIC of the host kernel as the server does.")
        if m["n_client"] != m["n_server"]:
            out.text(f"WARNING: {m['n_client']} client requests but {m['n_server']} server request lines: the order pairing is unreliable.")

    # ---- events -> requests
    events = [e for e in log["events"] if e["kind"] != "request"]
    events.sort(key=lambda e: (e["end_us"], e["pos"]))
    ends = [r["end_us"] for r in sreqs]
    buckets = [[] for _ in sreqs]
    orphans = []
    for e in events:
        k = bisect.bisect_left(ends, e["end_us"])
        if k < len(sreqs) and e["end_us"] >= sreqs[k]["start_us"] - TOL_US:
            buckets[k].append(e)
        else:
            orphans.append(e)
    out.text(f"\nDMK_TIMING lines outside any request (server start-up warm-up, other endpoints): {len(orphans)} "
             f"({collections.Counter(e['kind'] for e in orphans).most_common()})")
    req_pos = [r["pos"] for r in sreqs]
    if req_pos != sorted(req_pos):
        out.text("WARNING: the request lines are not in time order in the file; slot lines may be assigned wrongly.")
    slot_buckets = assign_slot_events(log["slots"], sorted(req_pos))
    anchor = estimate_anchor(log["slots"])
    if anchor:
        out.text(f"Slot-line clock anchor from `t_last` of {anchor['n']} LRU selections: log time + {anchor['offset_us'] / 1e6:.6f} s = CLOCK_MONOTONIC "
                 f"(MAD {anchor['mad_us']:.0f} us). The scheduling split (sched_wake_est_ms / sched_launch_est_ms) uses it.")
    else:
        out.text("No usable clock anchor in the slot lines (fewer than 5 LRU selections with `t_last`): scheduling is not split.")
    floors = Census()
    if floors.lib is None:
        out.text(f"NOTE census_lib could not be imported ({floors.err}): floor columns are empty.")

    rows, passes = [], []
    unmatched_client = []
    matched_ci = {ci for ci, _ in m["pairs"]}
    for ci, c in enumerate(creqs):
        if ci not in matched_ci:
            unmatched_client.append(c["idx"])
    for ci, si in m["pairs"]:
        row, ps = analyze_request(creqs[ci], sreqs[si], buckets[si], slot_buckets[si] if si < len(slot_buckets) else [], floors, clock_ok, anchor)
        rows.append(row)
        passes.extend(ps)
    rows.sort(key=lambda r: r["idx"])
    res["rows"], res["passes"], res["clock_ok"], res["unmatched_client"] = rows, passes, clock_ok, unmatched_client
    res["orphans"] = len(orphans)
    res["floors"] = floors
    res["meta"] = cj["meta"]
    n_valid = sum(1 for r in rows if r["valid"])
    flag_count = collections.Counter(f for r in rows for f in set(r["flags"]))
    out.text(f"\nRequests analysed: {len(rows)} matched, {n_valid} valid for the tables (excluded: {len(rows) - n_valid}; flags {dict(flag_count)}); "
             f"client requests without a server request: {len(unmatched_client)} {unmatched_client[:10]}")
    anchor_bad = sum(1 for r in rows if "anchor_implausible" in r["flags"])
    if anchor_bad > 0.2 * max(len(rows), 1):
        for r in rows:
            r["sched_wake_est_ms"] = r["sched_launch_est_ms"] = None
        out.text(f"Scheduling split dropped: {anchor_bad} requests have an implausible anchor (launch line outside post_tasks..first decode).")
    resid = [abs(r["residual_ms"]) for r in rows if r["valid"]]
    if resid:
        out.text(f"Residual (wall minus the sum of the stacked components): max |residual| {max(resid):.4f} ms, mean |residual| {mean(resid):.4f} ms over valid requests.")
    gaps = sum(1 for r in rows if "neg_gap" in r["flags"])
    if gaps:
        out.text(f"WARNING: {gaps} requests have a negative interval between stamps (> {TOL_US} us): the meaning of a stamp is probably not as documented.")

    write_requests(out, rows, passes)
    tables(out, rows, passes, floors, cj, clock_ok, creqs, ref, results_dir, run)
    out.write_summary()
    return res


REQ_COLS = (["idx", "kind", "case", "phase", "pair", "wall_ms", "input_tokens", "server_total_ms", "http_total_ms"] + STACK_COLS +
            ["residual_ms", "unsplit_ms", "apply_ms", "graph_build_ms", "alloc_ms", "inputs_ms"] + [b[0] for b in BETWEEN_DETAIL] +
            ["sched_wake_est_ms", "sched_launch_est_ms", "n_questions", "n_groups", "n_decodes", "n_ubatches", "n_syncs", "tokens_evaluated",
             "passes_obs", "ubatches_per_decode", "reuse_flags", "cuda_modes", "cuda_host_ms", "span_ms", "floor_weights_ms", "floor_int8_ms",
             "floor_max_ms", "floor_roofline_ms", "exposed_min_ms", "exposed_max_ms", "submit_first_ms", "submit_later_ms", "slots", "slot_select", "f_sim_best", "release_tokens",
             "census_expected", "census_match", "census_ref_delta_tokens", "valid", "flags"])
PASS_COLS = ["idx", "kind", "case", "phase", "pass", "decode", "group", "n_tokens", "n_seqs", "n_seq_tokens", "n_outputs", "n_nodes", "reuse",
             "apply_ms", "graph_build_ms", "alloc_ms", "inputs_ms", "submit_ms", "total_ms", "cuda_mode", "cuda_host_ms",
             "start_after_request_start_ms", "gap_before_ms", "sync_between_ms", "sync_passes", "sync_wait_ms", "sync_span_ms",
             "sync_since_last_submit_ms", "floor_weights_ms", "floor_int8_ms", "floor_max_ms", "floor_roofline_ms", "floor_group_ms"]


def write_requests(out, rows, passes):
    if not out.dir:
        return
    for name, cols, data in (("requests", REQ_COLS, rows), ("passes", PASS_COLS, passes)):
        path = os.path.join(out.dir, name + ".tsv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(cols)
            for r in data:
                w.writerow([";".join(r[c]) if isinstance(r.get(c), list) else tsvnum(r.get(c)) for c in cols])
        out.files.append(path)


# ---------------------------------------------------------------------------------------------------------------- tables
def group(rows, kind, phase, only_valid=True):
    return [r for r in rows if r["kind"] == kind and r["phase"] == phase and (r["valid"] or not only_valid)]


def stack_cols(clock_ok):
    if clock_ok:
        return STACK_COLS
    return ["http_total_ms"] + [c for c in STACK_COLS if c not in ("http_in_ms", "http_out_ms")]


def tables(out, rows, passes, floors, cj, clock_ok, creqs, ref, results_dir, run):
    kinds = [k for k in KIND_ORDER if any(r["kind"] == k for r in rows)]
    # ---- 1 wall time per kind x phase (client)
    wrows = []
    for k in kinds:
        for ph in ("warmup",) + PHASES:
            cs = [c["wall_ms"] for c in creqs if c["kind"] == k and c["phase"] == ph]
            if not cs:
                continue
            s = stats(cs)
            sv = [r["server_total_ms"] for r in rows if r["kind"] == k and r["phase"] == ph]
            wrows.append([k, ph, s["n"], s["mean"], s["p50"], s["p95"], s["min"], s["max"], mean(sv)])
    out.table("summary-wall", "1. Client wall time per request type and phase (warmup rows are not used anywhere else)",
              ["kind", "phase", "n", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "server_total_mean_ms"], wrows,
              source=f"{run}.client.json wall_ms (client send stamp -> response body read); server_total = `request` total_us.")

    # ---- 2 stacked decomposition
    cols = stack_cols(clock_ok)
    srows = []
    for k in kinds:
        for ph in PHASES:
            g = group(rows, k, ph)
            if not g:
                continue
            srows.append([k, ph, len(g)] + [mean([r[c] for r in g]) for c in cols] + [mean([r["wall_ms"] for r in g]), mean([r["residual_ms"] for r in g]),
                                                                                         max(abs(r["residual_ms"]) for r in g)])
    out.table("stack", "2. Stacked decomposition of the client wall time (mean over the valid requests, ms; the columns add up to wall_ms)",
              ["kind", "phase", "n"] + cols + ["wall_ms", "residual_mean_ms", "residual_absmax_ms"], srows,
              source="requests.tsv; column definitions in COLUMNS.tsv and NOTES.txt (section 7) next to analyze_decomp.py. Interval definitions: see the table in the same NOTES.txt.")
    crow = []
    for k in kinds:
        for ph in PHASES:
            g = group(rows, k, ph)
            if not g:
                continue
            tot = mean([r["wall_ms"] for r in g])
            for c in cols + ["wall_ms", "residual_ms"]:
                xs = [r[c] for r in g]
                s = stats(xs)
                crow.append([k, ph, c, s["n"], s["mean"], s["p50"], s["p95"], 100 * s["mean"] / tot if tot else float("nan")])
    out.table("components", "3. Every component: mean, p50, p95 (ms) and share of the mean wall time",
              ["kind", "phase", "component", "n", "mean_ms", "p50_ms", "p95_ms", "mean_share_pct"], crow, source="requests.tsv", show=False)

    view_a(out, rows, floors, cj)
    view_b(out, rows, floors)
    view_c(out, creqs, ref, results_dir, kinds)
    view_d(out, rows, kinds, clock_ok)
    view_e(out, rows, kinds)


def census_floor_for(floors, case, request_type="short-1q"):
    for r in floors.passes_rows:
        if r["request_type"] == request_type and r["rep_case"] == case:
            return {"weights_ms": float(r["floor_weights_ms"]), "int8_ms": float(r["floor_int8_ms"]),
                    "max_ms": float(r["floor_pass_max_ms"]), "roofline_ms": float(r["floor_roofline_ms"]), "n_tokens": int(r["N_tokens"])}
    return None


def view_a(out, rows, floors, cj):
    g = group(rows, "short1q", "unique")
    out.text("\n## View a. short-1q (phase unique) against the weight floor and the earlier 36.7 ms")
    if not g:
        out.text("\nNo valid short1q/unique requests: skipped.")
        return
    case = cj["meta"].get("rep_case", {}).get("short1q", "check03")
    cf = census_floor_for(floors, case)
    if cf is None:
        obs = g[0]
        cf = {"weights_ms": mean([r["floor_weights_ms"] for r in g]), "int8_ms": mean([r["floor_int8_ms"] for r in g]),
              "max_ms": mean([r["floor_max_ms"] for r in g]), "roofline_ms": mean([r["floor_roofline_ms"] for r in g]), "n_tokens": None}
    fl = cf["weights_ms"]
    wall = mean([r["wall_ms"] for r in g])
    wall50 = pct([r["wall_ms"] for r in g], 50)
    # grouped waterfall: the spec's six groups, ms above the weight floor
    def m(c):
        return mean([r[c] for r in g])

    def p50(c):
        return pct([r[c] for r in g], 50)

    def msum(cs):
        return mean([sum(r[c] or 0.0 for c in cs) for r in g])

    def psum(cs):
        return pct([sum(r[c] or 0.0 for c in cs) for r in g], 50)

    http = ["http_in_ms", "http_out_ms"] if clock_ok_rows(g) else ["http_total_ms"]
    groups = [
        ("HTTP / client (send -> handler start, response -> client)", http),
        ("parse + build + tokenize (request.parse_us, build_us)", ["parse_ms", "build_ms"]),
        ("scheduling (post_tasks -> first decode start, decode setup)", ["sched_ms", "first_setup_ms"]),
        ("host graph build + submit (prepare, submit, between passes, post-submit)", ["prepare_ms", "submit_ms", "between_ms", "post_submit_ms"]),
        ("tail (server wake-up after the sync, softmax/format/JSON)", ["tail_server_ms", "format_ms"]),
    ]
    rowsw = [["weight floor of one pass with output (census-passes.tsv, " + case + ")", fl, fl, fl, 100 * fl / wall]]
    cum = fl
    for name, cs in groups:
        v = msum(cs)
        cum += v
        rowsw.append([name, v, psum(cs), cum, 100 * v / wall])
    wait = m("gpu_wait_ms")
    v = wait - fl
    cum += v
    rowsw.append(["GPU wait beyond the floor (gpu_wait_ms - floor; negative = part of the GPU time hides under the host submit)", v, p50("gpu_wait_ms") - fl, cum, 100 * v / wall])
    rowsw.append(["= client wall time", wall, wall50, cum, 100.0])
    rowsw.append(["residual (wall - stack)", m("residual_ms"), p50("residual_ms"), None, 100 * m("residual_ms") / wall])
    out.table("waterfall-short1q", "a1. Waterfall: where the short-1q request time goes, starting at the weight floor (mean over the valid requests)",
              ["step", "mean_ms", "p50_ms", "cumulative_mean_ms", "share_of_wall_pct"], rowsw,
              source=f"{len(g)} requests of {case} (phase unique); floor from the census table census-passes.tsv. cumulative: floor + the steps above it "
                     f"= wall. p50 of a sum is the p50 of that sum per request, so the p50 column does not add up.")
    det = []
    for c, what, src in STACK:
        if c in ("http_in_ms", "http_out_ms") and not clock_ok_rows(g):
            continue
        det.append([c, what, m(c), p50(c), pct([r[c] for r in g], 95), 100 * m(c) / wall])
    if not clock_ok_rows(g):
        det.insert(0, ["http_total_ms", "client wall minus the server request total (HTTP in + out, not separable without a common clock)", m("http_total_ms"),
                       p50("http_total_ms"), pct([r["http_total_ms"] for r in g], 95), 100 * m("http_total_ms") / wall])
    for c, what in (("apply_ms", "  of prepare: memory context apply"), ("graph_build_ms", "  of prepare: ggml graph build (0 if graph reused)"),
                    ("alloc_ms", "  of prepare: scheduler alloc (0 if reused)"), ("inputs_ms", "  of prepare: set_inputs")):
        det.append([c, what, m(c), p50(c), pct([r[c] for r in g], 95), 100 * m(c) / wall])
    if any(r["sched_wake_est_ms"] is not None for r in g):
        for c, what in (("sched_wake_est_ms", "  of sched [estimate, slot-line anchor]: post_tasks -> launch_slot_ (queue wake-up, get_available_slot)"),
                        ("sched_launch_est_ms", "  of sched [estimate]: launch_slot_ -> decode start (update_slots batch build)")):
            det.append([c, what, m(c), p50(c), pct([r[c] for r in g if r[c] is not None], 95), 100 * m(c) / wall])
    out.table("waterfall-short1q-detail", "a2. The same, every component", ["column", "what", "mean_ms", "p50_ms", "p95_ms", "share_of_wall_pct"], det, source="requests.tsv")
    span = [r["span_ms"] for r in g if r["span_ms"] is not None]
    refrows = [
        ["weight floor, pass with output (census-passes.tsv)", cf["weights_ms"]],
        ["int8 compute floor", cf["int8_ms"]],
        ["pass floor = max(weight, int8)", cf["max_ms"]],
        ["roofline floor (sum over kernels of max(bytes/BW, ops/peak))", cf["roofline_ms"]],
        ["earlier measurement 'a whole short prompt in one pass' (deep review, untraced)", EARLIER_SHORT_MS],
        ["this run: client wall, mean", wall],
        ["this run: client wall, p50", wall50],
        ["this run: GPU span (start of the first pass -> end of the sync), mean; upper bound of the GPU time of the pass", mean(span)],
        ["this run: GPU wait (blocked at the sync), mean; lower bound of the GPU time", wait],
        ["wall - weight floor", wall - fl],
        ["wall - 36.7 (this run against the earlier measurement)", wall - EARLIER_SHORT_MS],
        ["span - weight floor (GPU-side excess incl. launch exposure)", mean(span) - fl],
        ["wall - span (everything outside the GPU span: HTTP, parse, build, scheduling, tail)", wall - mean(span)],
        ["span / weight floor", mean(span) / fl],
        ["wall / weight floor", wall / fl],
    ]
    out.table("waterfall-short1q-ref", "a3. Reference numbers", ["quantity", "ms (or ratio)"], refrows,
              source="census-passes.tsv (rep case), requests.tsv. Observed N tokens: " + ", ".join(f"{k}x{v}" for k, v in
                      collections.Counter(r["passes_obs"] for r in g).most_common(3)) + (f"; census N = {cf['n_tokens']}" if cf.get("n_tokens") else ""))


def clock_ok_rows(g):
    return all(r["http_in_ms"] is not None for r in g)


def view_b(out, rows, floors):
    out.text("\n## View b. Warm path: the same request twice in a row (pair2 against pair1)")
    byp = collections.defaultdict(dict)
    for r in rows:
        if r["phase"] in ("pair1", "pair2") and r["pair"] is not None and r["valid"]:
            byp[(r["kind"], r["case"], r["pair"])][r["phase"]] = r
    keys = sorted({(k, c) for (k, c, _p), v in byp.items() if "pair1" in v and "pair2" in v}, key=lambda x: (KIND_ORDER.index(x[0]) if x[0] in KIND_ORDER else 9, x[1]))
    if not keys:
        out.text("\nNo complete pairs: skipped.")
        return
    cols = [c for c in stack_cols(all(r["http_in_ms"] is not None for r in rows if r["valid"])) if True]
    drows = []
    for k, c in keys:
        ps = [(v["pair1"], v["pair2"]) for (kk, cc, _p), v in byp.items() if kk == k and cc == c and "pair1" in v and "pair2" in v]
        row = [k, c, len(ps), mean([a["wall_ms"] for a, _ in ps]), mean([b["wall_ms"] for _, b in ps])]
        d = [b["wall_ms"] - a["wall_ms"] for a, b in ps]
        row += [mean(d), se(d)]
        for col in cols:
            dd = [(b[col] or 0.0) - (a[col] or 0.0) for a, b in ps]
            row.append(mean(dd))
        drows.append(row)
    out.table("warm-path", "b1. Pair 2 minus pair 1, mean over the pairs of the case (ms; negative = the second request is faster)",
              ["kind", "case", "n_pairs", "wall1_mean_ms", "wall2_mean_ms", "d_wall_ms", "d_wall_se_ms"] + ["d_" + c for c in cols], drows,
              source="requests.tsv; pairs = the same body (same Ref) sent twice back to back. d_<col> = mean(pair2 - pair1) of that component. "
                     "Differences below 2 x d_wall_se_ms are noise.")
    wrows = []
    for k, c in keys:
        ps = [(v["pair1"], v["pair2"]) for (kk, cc, _p), v in byp.items() if kk == k and cc == c and "pair1" in v and "pair2" in v]
        pred = floors.tokens.get(c)
        for lab, sel in (("pair1", [a for a, _ in ps]), ("pair2", [b for _, b in ps])):
            sl = collections.Counter(r["slot_select"] for r in sel)
            wrows.append([k, c, lab, len(sel), mean([r["tokens_evaluated"] for r in sel]), mean([r["n_ubatches"] for r in sel]), mean([r["n_decodes"] for r in sel]),
                          mean([int(r["reuse_flags"].count("1")) for r in sel]), pred,
                          sum(1 for r in sel if r["census_match"]), ",".join(f"{x}:{n}" for x, n in sl.most_common()),
                          collections.Counter(r["passes_obs"] for r in sel).most_common(1)[0][0]])
    out.table("warm-path-work", "b2. Did the server do less work the second time? (tokens of the ubatches, passes, graph reuse, slot choice)",
              ["kind", "case", "phase", "n", "tokens_evaluated_mean", "ubatches_mean", "decodes_mean", "graph_reuse_mean", "census_tokens_evaluated", "n_census_match",
               "slot_select", "passes_most_common"], wrows,
              source="requests.tsv; census_tokens_evaluated = census-cases.tsv, variant ref (full evaluation of the prompt, Ref line of median length). "
                     "tokens_evaluated below that number means the server reused cached tokens.")


def view_c(out, creqs, ref, results_dir, kinds):
    out.text("\n## View c. Instrumentation check: client wall time with DMK_TIMING against the reference run")
    if not ref:
        out.text("\nNo --ref given: skipped.")
        return
    rpath = os.path.join(results_dir, ref + ".client.json")
    if not os.path.isfile(rpath):
        out.text(f"\n{rpath} not found: skipped.")
        return
    rj, rreqs = load_client(rpath)
    rrows = []
    for k in kinds:
        for ph in PHASES + ("all timed",):
            sel = lambda reqs: [r for r in reqs if r["kind"] == k and (r["phase"] in PHASES if ph == "all timed" else r["phase"] == ph)]
            a, b = sel(creqs), sel(rreqs)
            if not a or not b:
                continue
            wa, wb = [r["wall_ms"] for r in a], [r["wall_ms"] for r in b]
            ma, mb = mean(wa), mean(wb)
            pa, pb = pct(wa, 50), pct(wb, 50)
            sed = math.sqrt(se(wa) ** 2 + se(wb) ** 2) if len(wa) > 1 and len(wb) > 1 else float("nan")
            # paired by request (same plan, same Ref): same idx in both runs
            ib = {r["idx"]: r for r in b}
            dp = [r["wall_ms"] - ib[r["idx"]]["wall_ms"] for r in a if r["idx"] in ib and ib[r["idx"]]["ref"] == r["ref"] and ib[r["idx"]]["case"] == r["case"]]
            pm, ps_ = (mean(dp), se(dp)) if len(dp) > 1 else (float("nan"), float("nan"))
            rrows.append([k, ph, len(wa), len(wb), ma, mb, ma - mb, 100 * (ma - mb) / mb, pa, pb, pa - pb, 100 * (pa - pb) / pb, sed,
                          "within noise" if sed == sed and abs(ma - mb) < 2 * sed else "DIFFERENT", len(dp), pm, ps_,
                          "within noise" if ps_ == ps_ and abs(pm) < 2 * ps_ else "DIFFERENT"])
    out.table("instrumentation", "c. Client wall time: timing build with DMK_TIMING=1 against the reference build (ms)",
              ["kind", "phase", "n_timing", "n_ref", "mean_timing_ms", "mean_ref_ms", "d_mean_ms", "d_mean_pct", "p50_timing_ms", "p50_ref_ms", "d_p50_ms", "d_p50_pct",
               "se_d_mean_ms", "verdict_unpaired_2se", "n_paired", "paired_d_mean_ms", "paired_se_ms", "verdict_paired_2se"], rrows,
              source=f"{rpath} and the run's client.json. d = timing - ref. se_d_mean = sqrt(se_timing^2 + se_ref^2); unpaired verdict: |d_mean| < 2 se. paired = same request "
                     f"(same idx, case and Ref) in both runs. Different builds (ablate-timing vs b12004-sm86): a difference is DMK_TIMING plus the extra host code of "
                     f"the ablate build; the two cannot be separated by this table alone.")


def exposure_stats(g):
    return {"min": mean([r["exposed_min_ms"] for r in g]), "max": mean([r["exposed_max_ms"] for r in g])}


def view_d(out, rows, kinds, clock_ok):
    out.text("\n## View d. Fraction of the wall time per request type")
    out.text("""
Definitions. **GPU wait** = time the host thread is blocked in the sync (sum of `sync.wait_us`): the GPU is certainly busy then.
**Host submit + prepare** = ubatch `apply+build+alloc+inputs+submit` (all host work of the passes). **Not overlapped**: the log has no GPU
timeline, so the part of the host work that runs while the GPU is idle can only be bounded. Inside the span of a sync group (start of
its first pass -> end of the sync) the host is busy for `span - wait`. Lower bound of the not overlapped part = the prepare phase
of the first pass of the group (the GPU has nothing queued yet: certain). Upper bound = `min(span - wait, span - max(wait, sum of the census pass floors))`
(the GPU needs at least its floor, so it can idle at most the rest) and never less than the lower bound. The real value lies between.
Accounting split of the submit column (`submit_first_pass_pct`, `submit_later_passes_pct`): the submit of the first pass of every sync group
runs while the GPU has nothing earlier to execute; the submit of later passes of the group can run under the GPU execution of the earlier pass.
That is a rule by pass order, not a measurement (a later pass is hidden only if the GPU is still busy with the earlier one).
""")
    cols = ["kind", "phase", "n", "wall_ms", "gpu_wait_pct", "host_prepare_submit_pct", "not_overlapped_min_pct", "not_overlapped_max_pct", "between_passes_pct",
            "post_submit_pct", "client_http_pct", "parse_build_pct", "scheduling_pct", "tail_pct", "submit_first_pass_pct", "submit_later_passes_pct", "sum_pct"]
    drows = []
    for k in kinds:
        for ph in PHASES:
            g = group(rows, k, ph)
            if not g:
                continue
            w = mean([r["wall_ms"] for r in g])

            def p(cs):
                return 100 * mean([sum((r[c] or 0.0) for c in cs) for r in g]) / w
            http = ["http_in_ms", "http_out_ms"] if clock_ok else ["http_total_ms"]
            vals = [p(["gpu_wait_ms"]), p(["prepare_ms", "submit_ms"]), 100 * mean([r["exposed_min_ms"] for r in g]) / w,
                    100 * mean([r["exposed_max_ms"] for r in g]) / w, p(["between_ms"]), p(["post_submit_ms"]), p(http), p(["parse_ms", "build_ms"]),
                    p(["sched_ms", "first_setup_ms"]), p(["tail_server_ms", "format_ms"])]
            total = vals[0] + vals[1] + vals[4] + vals[5] + vals[6] + vals[7] + vals[8] + vals[9]
            extra = [p(["submit_first_ms"]), p(["submit_later_ms"])]
            drows.append([k, ph, len(g), w] + vals + extra + [total])
    out.table("fractions", "d. Share of the wall time (mean over valid requests; not_overlapped_* and submit_first/later are parts of host_prepare_submit, not extra shares)",
              cols, drows, source="requests.tsv. sum_pct adds gpu_wait, host_prepare_submit, between, post_submit, client_http, parse_build, scheduling, tail (the stack, "
                                  "100 minus residual). not_overlapped_min/max are bounds of the part of the host time in the GPU spans (prepare, submit and between passes) that runs with an idle GPU. "
                                  "submit_first_pass_pct + submit_later_passes_pct = the submit part of host_prepare_submit_pct.")


def view_e(out, rows, kinds):
    out.text("\n## View e. Is a pass host-bound? Sum of the submit times against the GPU time")
    erows = []
    for k in kinds:
        for ph in ("unique", "cycle"):
            g = group(rows, k, ph)
            g = [r for r in g if r["span_ms"] is not None]
            if not g:
                continue
            sub, prep, wait, span = (mean([r[c] for r in g]) for c in ("submit_ms", "prepare_ms", "gpu_wait_ms", "span_ms"))
            fmax = mean([r["floor_max_ms"] for r in g])
            froof = mean([r["floor_roofline_ms"] for r in g])
            nub = mean([r["n_ubatches"] for r in g])
            hint = ("host-bound: the launch loop fills the span" if sub >= 0.9 * span else
                    "GPU-bound: the GPU finishes well after the last launch" if wait >= 0.5 * span else "mixed")
            erows.append([k, ph, len(g), nub, prep, sub, wait, span, fmax, froof, sub / span, wait / span, span / fmax if fmax else float("nan"),
                          sub / nub, hint])
    out.table("hostbound", "e. Sum of host submit against the GPU span per request (mean, ms)",
              ["kind", "phase", "n", "ubatches", "prepare_ms", "submit_ms", "gpu_wait_ms", "gpu_span_ms", "floor_max_ms", "floor_roofline_ms", "submit_over_span", "wait_over_span",
               "span_over_floor", "submit_per_pass_ms", "hint"], erows,
              source="requests.tsv. gpu_span = sum over sync groups of (sync end - start of its first pass): an upper bound of the GPU busy time; gpu_wait: a lower bound; floor_max = sum of "
                     "census pass floors max(weights, int8) (nominal 448 GB/s, so slightly optimistic); floor_roofline = sum of census_lib roofline floors. hint rule [INFERENCE]: "
                     "submit >= 0.9 span -> host-bound; wait >= 0.5 span -> GPU-bound; else mixed. The traces (nsys, not this table) showed ~1,300 launches in 6-8 ms of CPU and 12 ms of GPU for a "
                     "4-token pass.")


# ---------------------------------------------------------------------------------------------------------------- self test
def self_test(test_dir):
    """Synthetic data and the log go to test_dir; without one, to a temporary folder (TMPDIR) that is removed afterwards."""
    if test_dir is None:
        with tempfile.TemporaryDirectory(prefix="decomp-selftest-") as tmp:
            return _self_test_in(tmp, keep=False)
    return _self_test_in(test_dir, keep=True)


def _self_test_in(test_dir, keep):
    import io
    sys.path.insert(0, HERE)
    import make_synthetic as ms
    os.makedirs(test_dir, exist_ok=True)
    log_path = os.path.join(test_dir, "self-test.log")

    class Tee(io.TextIOBase):
        def __init__(self, f, other):
            self.f, self.other = f, other

        def write(self, s):
            self.f.write(s)
            return self.other.write(s)

        def flush(self):
            self.f.flush()
            self.other.flush()

    with open(log_path, "w") as lf:
        real = sys.stdout
        sys.stdout = Tee(lf, real)
        try:
            ok = _self_test(test_dir, ms)
        finally:
            sys.stdout = real
    if keep:
        print(f"self-test log: {log_path}")
    return 0 if ok else 1


def _self_test(test_dir, ms):
    fails = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # ------------------------------------------------------------------ 1. clock agrees
    d1 = os.path.join(test_dir, "synthetic")
    truth = ms.make(d1, offset_s=0.0, seed=1, anomalies=True)
    print(f"\n===== self-test 1: synthetic run, clocks agree ({d1}) =====")
    res = analyze(d1, "p3-dec-timing", "p3-dec-ref", os.path.join(test_dir, "out"))
    rows = {r["idx"]: r for r in res["rows"]}
    check(res["clock_ok"], "clock check passes when the stamps agree (mode time)")
    check(res["match"]["n_server"] == res["match"]["n_client"] + 1, "the extra server-only request line (curl) is left unmatched")
    comps = [c for c, _, _ in STACK] + ["residual_ms"]
    worst = collections.defaultdict(float)
    n_checked = 0
    bad_idx = set(truth["anomaly_requests"])
    for t in truth["requests"]:
        r = rows.get(t["idx"])
        if r is None:
            continue
        if t["idx"] in bad_idx:
            continue
        n_checked += 1
        for c in STACK_COLS:
            worst[c] = max(worst[c], abs(r[c] - t["components"][c]))
        worst["wall_ms"] = max(worst["wall_ms"], abs(r["wall_ms"] - t["components"]["wall_ms"]))
        for c in ("sched_wake_est_ms", "sched_launch_est_ms"):
            if r[c] is not None:
                worst[c] = max(worst[c], abs(r[c] - t["components"][c]))
    print("max |recovered - truth| per component over %d requests (ms): %s" % (n_checked, {k: round(v, 4) for k, v in worst.items()}))
    check(n_checked > 300, f"{n_checked} requests without anomaly were compared")
    check(all(v < 0.012 for v in worst.values()), "every component, the wall time and the scheduling split are recovered within 12 us")
    # structure and slots
    bad_slots = [t["idx"] for t in truth["requests"] if t["idx"] not in bad_idx and rows[t["idx"]]["slots"] != t["slots"]]
    check(not bad_slots, f"slot placement recovered for every request (wrong: {bad_slots[:5]})")
    bad_pass = [t["idx"] for t in truth["requests"] if t["idx"] not in bad_idx and rows[t["idx"]]["passes_obs"] != t["passes_obs"]]
    check(not bad_pass, f"pass structure recovered for every request (wrong: {bad_pass[:5]})")
    bad_mm = [t["idx"] for t in truth["requests"] if t["idx"] not in bad_idx and t["census_match_expected"] != rows[t["idx"]]["census_match"]]
    check(not bad_mm, f"census comparison as expected for every request (wrong: {bad_mm[:5]})")
    bad_sel = [t["idx"] for t in truth["requests"] if t["idx"] not in bad_idx and rows[t["idx"]]["slot_select"] != t["slot_select"]]
    check(not bad_sel, f"slot selection (LRU / LCP) recovered (wrong: {bad_sel[:5]})")
    # anomalies
    for name, idx in truth["anomalies"].items():
        r = rows.get(idx)
        want = truth["anomaly_flags"][name]
        check(r is not None and want in r["flags"] and not r["valid"], f"anomaly '{name}' (request {idx}) flagged {want} and excluded from the tables")
    ri = truth["race_idx"]
    check(ri is not None and ri not in bad_slots and rows[ri]["valid"] and not rows[ri]["flags"],
          f"a release line written after the request line (request {ri}) is still assigned to its request: slots right, no flags")
    check(res["orphans"] >= 3, f"start-up lines outside every request are not attributed ({res['orphans']} orphan lines)")
    # warm path
    wp = {}
    for r in res["rows"]:
        if r["valid"] and r["phase"] in ("pair1", "pair2"):
            wp.setdefault((r["kind"], r["phase"]), []).append(r["tokens_evaluated"])
    for k in KIND_ORDER:
        a, b = mean(wp[(k, "pair1")]), mean(wp[(k, "pair2")])
        if k in truth["warm_reuse_kinds"]:
            check(b < a, f"warm path: {k} pair2 evaluates fewer tokens than pair1 ({b:.1f} < {a:.1f})")
        else:
            check(abs(a - b) < 1e-9, f"warm path: {k} pair2 evaluates the same tokens as pair1 ({b:.1f} = {a:.1f})")
    # instrumentation
    ins = os.path.join(test_dir, "out", "instrumentation.tsv")
    rowsi = list(csv.DictReader(open(ins), delimiter="\t"))
    allt = [r for r in rowsi if r["phase"] == "all timed"]
    worst_d = max(abs(float(r["d_mean_ms"]) - truth["timing_overhead_ms"]) for r in allt)
    check(worst_d < 0.25, f"instrumentation overhead of {truth['timing_overhead_ms']} ms per request recovered within 0.25 ms in every kind (worst error {worst_d:.3f} ms)")
    # residual
    resid = max(abs(r["residual_ms"]) for r in res["rows"] if r["valid"])
    check(resid < 0.02, f"the stack adds up to the wall time: max |residual| {resid:.4f} ms")
    # views exist
    for f in ("requests.tsv", "passes.tsv", "summary.txt", "waterfall-short1q.tsv", "warm-path.tsv", "fractions.tsv", "hostbound.tsv", "clock-check.tsv"):
        check(os.path.isfile(os.path.join(test_dir, "out", f)), f"output file {f} written")

    # ------------------------------------------------------------------ 2. clock offset
    d2 = os.path.join(test_dir, "synthetic-offset")
    truth2 = ms.make(d2, offset_s=123.456789, seed=1, anomalies=False)
    print(f"\n===== self-test 2: server stamps shifted by +123.456789 s ({d2}) =====")
    res2 = analyze(d2, "p3-dec-timing", "p3-dec-ref", os.path.join(test_dir, "out-offset"))
    check(not res2["clock_ok"], "clock mismatch detected, order-only matching")
    off = res2["match"].get("order_offset_ms_median")
    check(off is not None and abs(off - (-123456.789)) < 5.0, f"offset estimate from the order pairing is -123456.8 ms (got {off:.1f})")
    rows2 = {r["idx"]: r for r in res2["rows"]}
    w2 = 0.0
    for t in truth2["requests"]:
        r = rows2[t["idx"]]
        for c in STACK_COLS:
            if c in ("http_in_ms", "http_out_ms"):
                check(r[c] is None, f"request {t['idx']}: {c} unavailable in fallback mode") if t["idx"] == 0 and c == "http_in_ms" else None
                continue
            w2 = max(w2, abs(r[c] - t["components"][c]))
        w2 = max(w2, abs(r["http_total_ms"] - (t["components"]["http_in_ms"] + t["components"]["http_out_ms"])))
    check(w2 < 0.012, f"fallback mode: server-side components and http_total recovered within 12 us (worst {w2:.4f} ms)")
    check(max(abs(r["residual_ms"]) for r in res2["rows"] if r["valid"]) < 0.02, "fallback mode: the stack adds up to the wall time")

    # ------------------------------------------------------------------ 3. no DMK lines
    d3 = os.path.join(test_dir, "synthetic-nodmk")
    ms.make(d3, offset_s=0.0, seed=1, anomalies=False, strip_dmk=True, n_per_kind=3)
    print(f"\n===== self-test 3: server log without DMK_TIMING lines ({d3}) =====")
    try:
        analyze(d3, "p3-dec-timing", None, os.path.join(test_dir, "out-nodmk"), quiet=True)
        check(False, "missing DMK_TIMING lines raise a clear error")
    except AnalysisError as e:
        print("AnalysisError:", str(e)[:600])
        check("no DMK_TIMING lines" in str(e) and "First lines of the log" in str(e), "missing DMK_TIMING lines raise a clear error that shows the first log lines")

    # ------------------------------------------------------------------ 4. log cut short (container removed early, lines lost)
    import shutil
    d4 = os.path.join(test_dir, "synthetic-cut")
    os.makedirs(d4, exist_ok=True)
    for fn in ("p3-dec-timing.client.json", "p3-dec-ref.client.json"):
        shutil.copy(os.path.join(d1, fn), d4)
    with open(os.path.join(d1, "p3-dec-timing.server.log"), "rb") as f:
        raw = f.read()
    cut = raw[:int(len(raw) * 0.8)]
    cut = cut[:cut.rfind(b"\n") - 30]                     # the last line is cut in the middle
    with open(os.path.join(d4, "p3-dec-timing.server.log"), "wb") as f:
        f.write(cut)
    print(f"\n===== self-test 4: server log cut after 80 % ({d4}) =====")
    res4 = analyze(d4, "p3-dec-timing", "p3-dec-ref", os.path.join(test_dir, "out-cut"), quiet=True)
    n4 = len(res4["rows"])
    check(res4["clock_ok"] and res4["match"]["lost_lines"], "cut log: the clocks agree for the server lines that exist (mode time, lost lines reported, no false clock mismatch)")
    check(200 < n4 < 348, f"cut log: {n4} of 348 requests analysed")
    t4 = {t["idx"]: t for t in truth["requests"]}
    w4 = max(abs(r[c] - t4[r["idx"]]["components"][c]) for r in res4["rows"] if r["valid"] and r["idx"] not in bad_idx for c in STACK_COLS)
    check(w4 < 0.012, f"cut log: the requests that are there are recovered within 12 us (worst {w4:.4f} ms)")
    print(f"\n{'ALL PASSED' if not fails else 'FAILED: ' + str(len(fails))} ({len(fails)} failures)")
    return not fails


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    global TOL_US
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results_dir", nargs="?")
    ap.add_argument("--run", default="p3-dec-timing")
    ap.add_argument("--ref", default="p3-dec-ref", help="reference run (no DMK_TIMING) for view c; '' = skip")
    ap.add_argument("--out-dir", help="default <results dir>/decomp-analysis")
    ap.add_argument("--tol-us", type=float, default=TOL_US, help="tolerance of the stamp comparisons in microseconds (default %(default)s: rounding of the logged stamps)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--test-dir", default=None, help="folder for the synthetic data and self-test.log of --self-test (kept; default: a temporary folder that is removed)")
    a = ap.parse_args()
    TOL_US = a.tol_us
    if a.self_test:
        return self_test(a.test_dir)
    if not a.results_dir:
        ap.error("results_dir is required (or --self-test)")
    out_dir = a.out_dir or os.path.join(a.results_dir, "decomp-analysis")
    try:
        analyze(a.results_dir, a.run, a.ref, out_dir)
    except AnalysisError as e:
        print("ERROR:", e, file=sys.stderr)
        return 2
    print(f"\nfiles written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
