"""Shared helpers of analyze_kvwin.py, analyze_childpass.py and make_synthetic.py (python 3 stdlib only).

* census_lib (read-only input, ../census/census_lib.py) is imported as `L`; its tables come from L.data_file(name) ($DMK_CENSUS_DATA).
* parse_server_log(): server log (docker logs of dmx-server) -> requests with their DMK_TIMING lines and slot lines.
* simulate(): pass structure of a request for a slot placement, in three models (all census_lib.simulate_request):
    "unified"  sequential = False = the census default (census_lib.SPLIT_SEQUENTIAL): what llama-memory-hybrid.cpp does with --kv-unified
               (production): split_equal(n_ubatch, !unified); every ubatch takes all sequences with unused tokens, in any seq id order
    "seq"      sequential = True: ubatches take only sequences with consecutive ids; what the source does with --no-kv-unified (n_stream > 1:
               unified = false -> sequential = true) and what the first census assumed for everything
    "fixed"    parent slot forced to the lowest id (all children join the parent's suffix batch), census default split
* window helpers: window seen / window needed for the flash-attention K/V window.
"""
import csv
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = os.path.abspath(os.path.join(HERE, "..", "census"))       # census_lib.py (code only; tables: L.data_file)
if CENSUS not in sys.path:
    sys.path.insert(0, CENSUS)
import census_lib as L  # noqa: E402  (read-only input)

N_UBATCH = L.N_UBATCH
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        _MODEL = L.Model()
    return _MODEL


def read_tsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def pad256(x):
    return (int(x) + 255) // 256 * 256


def num(x, default=None):
    """float or int from a TSV / log field; default when empty."""
    if x is None or x == "" or x == "None":
        return default
    try:
        v = float(x)
    except ValueError:
        return default
    return int(v) if v == int(v) and "." not in str(x) and "e" not in str(x).lower() else v


# ------------------------------------------------------------------------------------------ case plan (census)
_PLAN = None


def case_plan(variant="ref"):
    """case id -> {"shared": tokens, "suffix": [tokens per question], "n_questions", "size"} from the census table cases-plan.tsv."""
    global _PLAN
    if _PLAN is None:
        _PLAN = {}
        for r in read_tsv(L.data_file("cases-plan.tsv")):
            _PLAN.setdefault(r["variant"], {})[r["case"]] = {
                "shared": int(r["shared_prefix_tokens"]), "suffix": [int(x) for x in r["suffix_tokens_per_question"].split()],
                "n_questions": int(r["n_questions"]), "size": r["size"], "task": r["task"]}
    return _PLAN[variant]


def suffix_candidates(n_questions):
    """Distinct suffix tuples of the census plan (ref variant) with this number of questions."""
    out = []
    for c in case_plan().values():
        if c["n_questions"] == n_questions and tuple(c["suffix"]) not in out:
            out.append(tuple(c["suffix"]))
    return out


# ------------------------------------------------------------------------------------------ pass structures
def simulate(shared, suffix_lens, slots, mode="unified"):
    """Passes (census_lib pass dicts) of one request. See the module docstring for the modes."""
    if mode == "unified":
        return L.simulate_request(shared, list(suffix_lens), tuple(slots), sequential=False)
    if mode == "seq":
        return L.simulate_request(shared, list(suffix_lens), tuple(slots), sequential=True)
    if mode == "fixed":
        return L.simulate_request(shared, list(suffix_lens), tuple(range(len(suffix_lens))))
    raise ValueError(mode)


MODES = ("unified", "seq")


def structure(passes):
    """Census slot structure string: number of sequences of every pass, e.g. P1 S3 S2 S1."""
    return " ".join(("P" if p["kind"] == "prefix" else "F" if p["kind"].startswith("single") else "S") + str(len(p["seqs"])) for p in passes)


def describe(passes):
    parts = []
    for p in passes:
        if p["kind"] == "prefix":
            parts.append(f"P{p['n_tokens']}")
        elif p["kind"].startswith("single"):
            parts.append(f"F{p['n_tokens']}")
        else:
            parts.append(f"S{len(p['seqs'])}x{p['seqs'][0][1]}")
    return "+".join(parts)


def n_decodes(passes):
    """llama_decode calls of the request = distinct update_slots iterations that decode something."""
    return len({p["iteration"] for p in passes})


def ub_key(passes):
    """(n_tokens, n_seqs, n_outputs) per ubatch, the quantities the DMK_TIMING ubatch line shows."""
    return [(p["n_tokens"], len(p["seqs"]), p["n_out"]) for p in passes]


def pass_floors(passes):
    """(sum of max(weight, int8) floors, sum of serial roofline floors) in ms of a pass list (census pass_costs)."""
    m = model()
    mx = rf = 0.0
    for p in passes:
        c = L.pass_costs(m, p)
        mx += c["floor_pass_max_s"]
        rf += c["floor_roofline_s"]
    return mx * 1e3, rf * 1e3


# ------------------------------------------------------------------------------------------ K/V windows
SCAN_N = 1024       # fattn-common.cuh: the mask scan (flash_attn_mask_to_KV_max) runs only if Q->ne[1] >= 1024 or Q->ne[3] > 1


def eff_front(start, n):
    """Mean cells read per query row-tile of 16 rows by a pass of n tokens at position `start` when the sequence's own cells sit at the
    front of the pool and the kernel skips the masked tail (KV_max): each tile reads 256-cell blocks up to its last visible cell."""
    tiles = (n + 15) // 16
    blocks = [-(-(start + min(n, 16 * (j + 1))) // 256) for j in range(tiles)]
    return sum(blocks) / tiles * 256


def pass_windows(passes):
    """Per pass: the window each pass needs, from the cell positions of the request (census_lib semantics).
      needed_seq  = own cells of the longest sequence of the pass, padded to 256: pad256(max(start + n))   (census attn_cost n_kv)
      needed_pool = cells of the request that coexist in a perfectly compact unified pool, padded to 256:
                    prefix cells + the suffix cells of every sequence already placed (siblings' suffix cells too)
      eff_front   = mean number of cells a query tile reads when the own cells sit at the front of the pool and the tail is
                    skipped (KV_max, only used by the kernel when n_tokens >= 1024): for n_tokens < 1024 it is needed_seq
    """
    shared = sum(p["n_tokens"] for p in passes if p["kind"] == "prefix")
    done = {}          # question -> suffix tokens placed so far
    out = []
    for p in passes:
        n_tok = p["n_tokens"]
        seq_end = max(s + n for s, n in p["seqs"])
        needed_seq = pad256(seq_end)
        if p["kind"] == "prefix" or p["kind"].startswith("single"):
            pool = seq_end
        else:
            for (s, n), q in zip(p["seqs"], p["questions"]):
                done[q] = done.get(q, 0) + n
            pool = shared + sum(done.values())
        if n_tok >= SCAN_N and len(p["seqs"]) == 1:
            eff = eff_front(*p["seqs"][0])
        else:
            eff = needed_seq
        out.append({"needed_seq": needed_seq, "needed_pool": pad256(pool), "eff_front": eff})
    return out


# ------------------------------------------------------------------------------------------ server log
RE_DMK = re.compile(r"DMK_TIMING (ubatch|sync|decode|request|cuda_compute) (.*)$")
RE_KV = re.compile(r"(\w+)=([-+0-9.eE]+|\S+)")
REQUIRED = {"ubatch": ("t", "n_tokens", "n_seqs", "n_outputs", "total_us"), "decode": ("t", "n_tokens", "n_ubatches", "total_us"),
            "sync": ("t", "passes", "tokens", "wait_us", "since_first_submit_us"), "request": ("t", "n_questions", "total_us", "wait_us"), "cuda_compute": ("t",)}
RE_SLOT = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)\s+([IWED])\s+slot\s+(\S+):\s+id\s+(\d+)\s+\|\s+task\s+(-?\d+)\s+\|\s+(.*)$")
RE_LOAD = re.compile(r"load_model: initializing, n_slots = (\d+), n_ctx_slot = (\d+), kv_unified = '(\w+)'")


def _fields(s):
    d = {}
    for k, v in RE_KV.findall(s):
        try:
            d[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
        except ValueError:
            d[k] = v
    return d


def parse_server_log(path):
    """Parse a server log. Returns {"requests": [...], "slot_events": [...], "n_lines": n, "other_dmk": {kind: n}}.
    A request = the DMK_TIMING lines between the previous `request` line and its own (direct stderr writes keep their order)
    restricted to t >= t_end - total_us (drops the start-up warm-up decode); its slot events = the slot lines of the k-th
    group of n_questions `launch_slot_` lines (the async log thread keeps the order of the slot lines among themselves)."""
    events = []          # (kind, fields)
    slot_events = []
    incomplete = []      # (line number, kind, missing keys): a cut line (interleaved stderr) is skipped, not guessed
    n = 0
    load = None          # (n_slots, n_ctx_slot, kv_unified) of the load_model line
    with open(path, errors="replace") as f:
        for line in f:
            n += 1
            m = RE_DMK.search(line)
            if m:
                d = _fields(m.group(2))
                missing = [k for k in REQUIRED[m.group(1)] if k not in d]
                if missing:
                    incomplete.append((n, m.group(1), missing))
                    continue
                events.append((m.group(1), d))
                continue
            m = RE_LOAD.search(line)
            if m and load is None:
                load = (int(m.group(1)), int(m.group(2)), m.group(3) == "true")
                continue
            m = RE_SLOT.match(line)
            if m:
                mm, ss, ms, us = (int(x) for x in m.group(1, 2, 3, 4))
                slot_events.append({"t_log_s": mm * 60 + ss + ms / 1e3 + us / 1e6, "func": m.group(6), "slot": int(m.group(7)),
                                    "task": int(m.group(8)), "text": m.group(9).strip(), "line": n})
    requests = []
    cur = {"ubatches": [], "decodes": [], "syncs": [], "cuda": []}
    for kind, d in events:
        if kind == "ubatch":
            cur["ubatches"].append(d)
        elif kind == "decode":
            cur["decodes"].append(d)
        elif kind == "sync":
            cur["syncs"].append(d)
        elif kind == "cuda_compute":
            cur["cuda"].append(d)
        elif kind == "request":
            t_start = d["t"] - d["total_us"] / 1e6
            r = dict(d)
            r["t_start"] = t_start
            for key in ("ubatches", "decodes", "syncs", "cuda"):
                r[key] = [e for e in cur[key] if e["t"] >= t_start - 1e-6]
            r["k"] = len(requests)
            requests.append(r)
            cur = {"ubatches": [], "decodes": [], "syncs": [], "cuda": []}
    # slot events -> requests, by consuming n_questions launch lines per request
    launches = [i for i, e in enumerate(slot_events) if e["func"] == "launch_slot_"]
    pos = 0
    groups = []
    for r in requests:
        nq = int(r["n_questions"])
        if pos + nq > len(launches):
            groups.append(None)
            continue
        first = launches[pos]
        last = launches[pos + nq - 1]
        start = first
        while start > 0 and slot_events[start - 1]["func"] == "get_availabl" and (not groups or groups[-1] is None or start - 1 > groups[-1]["end"]):
            start -= 1
        groups.append({"start": start, "first": first, "last": last, "end": last})
        pos += nq
    for gi, g in enumerate(groups):
        if g is None:
            requests[gi]["slot_events"] = None
            continue
        nxt = next((h for h in groups[gi + 1:] if h is not None), None)
        stop = nxt["start"] if nxt else len(slot_events)
        evs = slot_events[g["start"]:stop]
        requests[gi]["slot_events"] = evs
        launches_r = [e for e in evs if e["func"] == "launch_slot_"]
        parent = [e for e in launches_r if "is_child = 0" in e["text"]]
        kids = sorted((e for e in launches_r if "is_child = 1" in e["text"]), key=lambda e: e["task"])
        requests[gi]["slots"] = tuple([e["slot"] for e in parent[:1]] + [e["slot"] for e in kids]) if parent else None
        sel = [e for e in evs if e["func"] == "get_availabl"]
        requests[gi]["select"] = ("LCP" if "LCP" in sel[0]["text"] else "LRU") if sel else None
        requests[gi]["release_tokens"] = {e["slot"]: int(m.group(1)) for e in evs if e["func"] == "release"
                                         for m in [re.search(r"n_tokens = (\d+)", e["text"])] if m}
    return {"requests": requests, "slot_events": slot_events, "n_lines": n, "n_launch_lines": len(launches), "incomplete": incomplete,
            "n_slots": load[0] if load else None, "n_ctx_slot": load[1] if load else None, "kv_unified": load[2] if load else None}


def request_kind(r):
    """Request type from the number of questions and the evaluated tokens (the client JSON `kind` is better when given)."""
    nq = int(r["n_questions"])
    tok = sum(int(u["n_tokens"]) for u in r["ubatches"])
    if nq == 1:
        return "short1q"
    if nq == 2:
        return "short2q"
    return "long3q" if tok > 2400 else "medium3q"


def obs_ubatches(r):
    return [(int(u["n_tokens"]), int(u["n_seqs"]), int(u["n_outputs"])) for u in r["ubatches"]]


def read_client_json(path):
    import json
    with open(path) as f:
        j = json.load(f)
    return j["requests"] if isinstance(j, dict) else j
