#!/usr/bin/env python3
"""Synthetic results directory for analyze_decomp.py (known truth). Nothing here is a measurement.

Usage: python3 make_synthetic.py <out dir> [--offset-s 0.0] [--seed 1] [--no-anomalies]
Writes <out dir>/p3-dec-timing.client.json (with `_truth`), p3-dec-timing.server.log and p3-dec-ref.client.json.

What is real: the print formats of the DMK_TIMING lines (src/dmk-timing.h, ggml-cuda.cu, server-context.cpp of the ablate build) and of the
slot lines (copied from a server log of the data set data/2026-10-30-window1), the request plan (decomp_client.plan_all, all four request types, the real
case list), the pass structure of 2- and 3-question requests (census_lib.simulate_request with the case token counts of census-cases.tsv and
a random slot placement), the floors of the passes (census_lib).
What is invented: every duration (host work per pass, GPU time per pass = census floor x 1.7..2.6 + 0.4 ms, HTTP and server overheads), chosen so
that a request lasts about as long as the earlier measurement. The components of every request are stored as the truth.
Anomalies (when enabled): a dropped sync line, a garbled ubatch line, a request line without any pass lines, a slot release line written
after its request line, unrelated text lines everywhere, a server request without a client request (curl), start-up warm-up lines.
"""
import argparse
import csv
import json
import os
import random
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "extra", "prof"))
sys.path.insert(0, HERE)
import common            # noqa: E402
import decomp_client     # noqa: E402
import analyze_decomp    # noqa: E402  (Census: floors and census-cases)

CASES = os.path.join(os.environ.get("DMK_BENCH_DIR") or os.path.join(REPO, "bench"), "cases.json")
BASE_US = 166_257_000_000_000        # a CLOCK_MONOTONIC value in us (46 h of uptime), like the t_last values of the real log
PROC_START_US = BASE_US - 8_900_000  # server process start (log time prefix is relative to it)
WARM_REUSE_KINDS = ("short2q",)      # kinds whose pair2 request finds the shared prefix cached (invented, to test the warm path views)
TIMING_OVERHEAD_MS = 0.12            # invented cost of DMK_TIMING for the reference run


def fmt_tlog(us):
    us = int(us)
    return "%d.%02d.%03d.%03d" % (us // 60_000_000, us // 1_000_000 % 60, us // 1000 % 1000, us % 1000)


class Gen:
    def __init__(self, seed, offset_us, census, anomalies):
        self.rnd = random.Random(seed)
        self.off = offset_us
        self.census = census
        self.anom = anomalies
        self.lines = []                  # (sort time us, seq, text)
        self.seq = 0
        self.ub_id = 0
        self.prev_shape = None
        self.slot_last_release = {0: -1, 1: -1, 2: -1}
        self.slot_busy_tokens = {0: 0, 1: 0, 2: 0}
        self.last_placement = {}         # (kind, ref) -> slots of the pair1 request
        self.T = BASE_US + 1_000_000     # client send time of the next request (us)

    def u(self, lo, hi):
        return int(round(self.rnd.uniform(lo, hi)))

    def emit(self, t_us, text, key_t=None):
        self.seq += 1
        self.lines.append((t_us if key_t is None else key_t, self.seq, text))

    # ---- line printers (formats of the real code)
    def dmk_t(self, t_us):
        return "%.6f" % ((t_us + self.off) / 1e6)

    def p_ubatch(self, t_end, n_tokens, n_seqs, n_seq_tokens, n_out, n_nodes, reuse, parts):
        self.ub_id += 1
        total = sum(parts)
        self.emit(t_end, "DMK_TIMING ubatch t=%s id=%d n_tokens=%d n_seqs=%d n_seq_tokens=%d n_outputs=%d n_nodes=%d reuse=%d apply_us=%.0f build_us=%.0f alloc_us=%.0f inputs_us=%.0f submit_us=%.0f total_us=%.0f"
                  % (self.dmk_t(t_end), self.ub_id, n_tokens, n_seqs, n_seq_tokens, n_out, n_nodes, reuse, *parts, total))

    def p_cuda(self, t, n_nodes, mode, host_us):
        self.emit(t, "DMK_TIMING cuda_compute t=%s n_nodes=%d mode=%s host_us=%.1f" % (self.dmk_t(t), n_nodes, mode, host_us))

    def p_decode(self, t, n_tokens, n_ub, init_us, total_us):
        self.emit(t, "DMK_TIMING decode t=%s n_tokens=%d n_ubatches=%d init_us=%.0f total_us=%.0f" % (self.dmk_t(t), n_tokens, n_ub, init_us, total_us))

    def p_sync(self, t, passes, tokens, wait, since_first, since_last):
        self.emit(t, "DMK_TIMING sync t=%s passes=%d tokens=%d wait_us=%.0f since_first_submit_us=%.0f since_last_submit_us=%.0f"
                  % (self.dmk_t(t), passes, tokens, wait, since_first, since_last))

    def p_request(self, t, nq, ng, parse, build, wait, fmt, total):
        self.emit(t, "DMK_TIMING request t=%s n_questions=%d n_groups=%d parse_us=%.0f build_us=%.0f wait_us=%.0f format_us=%.0f total_us=%.0f"
                  % (self.dmk_t(t), nq, ng, parse, build, wait, fmt, total))

    def slot_line(self, t_us, func, slot, task, text, key_t=None):
        self.emit(t_us, "%s I slot %12s: id %2d | task %d | %s" % (fmt_tlog(t_us - PROC_START_US), func, slot, task, text), key_t)

    def junk(self, t_us):
        self.emit(t_us, "%s I srv  update_slots: all slots are idle" % fmt_tlog(t_us - PROC_START_US))

    def gpu_time_us(self, n_tokens, n_seqs, n_seq_tokens, n_out):
        f = self.census.floor(n_tokens, n_seqs, n_seq_tokens, n_out)
        return f["max_ms"] * 1e3 * self.rnd.uniform(1.7, 2.6) + 400


def plan_requests(n_per_kind):
    cases = common.load_cases(CASES if os.path.exists(CASES) else common.CASES_PATH)
    kinds = list(common.KINDS)
    if n_per_kind:
        plans = decomp_client.plan_all(cases, kinds, n_per_kind, 1, 1, 1, 20261031)
    else:
        plans = decomp_client.plan_all(cases, kinds, 30, 2, 15, 3, 20261031)
    return cases, plans


def case_structure(census, case_id, n_q):
    """shared prefix and suffix lengths of a case from census-cases.tsv (variant ref)."""
    with open(census.data_file("census-cases.tsv")) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["variant"] == "ref" and r["case"] == case_id:
                if n_q == 1:
                    return 0, [int(r["sum_prompt_tokens"])]
                return int(r["shared_prefix_tokens"]), [int(x) for x in r["suffix_tokens_per_question"].split()]
    raise KeyError(case_id)


_struct_cache = {}


def make(outdir, offset_s=0.0, seed=1, anomalies=True, strip_dmk=False, n_per_kind=None):
    os.makedirs(outdir, exist_ok=True)
    sys.path.insert(0, analyze_decomp.CENSUS)
    import census_lib
    census = analyze_decomp.Census()
    cases, plans = plan_requests(n_per_kind)
    g = Gen(seed, int(round(offset_s * 1e6)), census, anomalies)
    rnd = g.rnd
    flat = [(i, s) for i, p in enumerate(plans) for s in p]
    n_total = len(flat)
    # which requests carry which anomaly (cycle-phase requests of fixed kinds, chosen by position)
    def nth(kind, phase, n):
        return [k for k, (_, s) in enumerate(flat) if s["kind"] == kind and s["phase"] == phase][n]
    pick = {}
    if anomalies and not n_per_kind:
        pick = {"drop_sync": nth("medium3q", "cycle", 3), "garbled_ubatch": nth("short2q", "cycle", 5),
                "no_pass_lines": nth("short1q", "cycle", 7), "release_after_request": nth("long3q", "cycle", 2)}
    extra_after = nth("short2q", "unique", 10) if pick else None
    flag_for = {"drop_sync": "no_sync", "garbled_ubatch": "sync_passes_mismatch", "no_pass_lines": "no_ubatch"}

    # start-up warm-up (outside every request): one decode of 2 tokens
    t0 = BASE_US - 3_000_000
    g.p_cuda(t0 + 900, 1296, "direct", 600.0)
    g.p_ubatch(t0 + 1000, 2, 1, 2, 1, 1296, 0, [30, 800, 300, 100, 700])
    g.p_decode(t0 + 1100, 2, 1, 60, 1180)
    g.p_sync(t0 + 1800, 1, 2, 650, 1780, 700)

    truth_reqs, client_reqs = [], []
    ref_reqs = []
    truth_extra = {}
    pair1_slots, pair1_delta = {}, {}
    last_kind = None
    ref_clock = BASE_US + 900_000_000 + 77_000_000   # arbitrary other time base for the reference run client stamps
    ref_T = ref_clock
    for n, (i_kind, step) in enumerate(flat):
        kind, case, phase, ref = step["kind"], step["case"], step["phase"], step["ref"]
        if last_kind is not None and kind != last_kind:
            g.T += 1_000_000
        last_kind = kind
        cid = case["id"]
        n_q = len(case["questions"])
        key = (cid, n_q)
        if key not in _struct_cache:
            _struct_cache[key] = case_structure(census, cid, n_q)
        shared, suffix = _struct_cache[key]
        is_pair2 = phase == "pair2"
        if is_pair2:                                            # same body, same Ref: same token count and the same slots
            delta, slots = pair1_delta[(kind, step["pair"])], pair1_slots[(kind, step["pair"])]
        else:
            delta = rnd.choice([-4, -3, -2, -1, 0, 0, 1])       # tokens of the Ref value differ from the census median
            slots = rnd.sample(range(3), n_q)
        if phase == "pair1":
            pair1_delta[(kind, step["pair"])] = delta
            pair1_slots[(kind, step["pair"])] = slots
        if n_q == 1:
            suffix = [suffix[0] + delta]
        else:
            shared += delta
        warm = is_pair2 and kind in WARM_REUSE_KINDS
        passes = census_lib.simulate_request(shared, suffix, slots)
        if warm:
            passes = [p for p in passes if p["kind"] != "prefix"]
            its = sorted({p["iteration"] for p in passes})
            for p in passes:
                p["iteration"] = its.index(p["iteration"])
        task_tokens = [suffix[0]] if n_q == 1 else [shared + s for s in suffix]
        sel = "LCP" if (is_pair2 or rnd.random() < 0.7) else "LRU"

        # ---------------------------------------------------------------- the request timeline (integer us)
        t_send = g.T
        http_in = g.u(100, 250)
        start = t_send + http_in
        parse = g.u(250, 400) + 30 * n_q
        build = 300 + int(1.1 * sum(task_tokens)) + g.u(0, 150)
        e_built = start + parse + build
        wake, l2d = g.u(100, 250), g.u(150, 400)
        t_launch = e_built + wake
        dec_start = t_launch + l2d
        # slot lines of the launch
        parent = slots[0]
        lru_slot_parent = parent
        if sel == "LCP":
            g.slot_line(t_launch - 15, "get_availabl", parent, -1, "selected slot by LCP similarity, f_sim_best = %.3f (> 0.100 thold), f_keep = %.3f"
                        % (1.0 if is_pair2 else rnd.uniform(0.15, 0.25), rnd.uniform(0.15, 0.25)))
        else:
            tl = g.slot_last_release[parent]
            g.slot_line(t_launch - 15, "get_availabl", parent, -1, "selected slot by LRU, n_tokens = %d, t_last = %d" % (0, (tl + g.off) if tl > 0 else -1))
        task0 = 1000 + 7 * n
        for qi in range(n_q - 1, 0, -1):
            g.slot_line(t_launch - 3 * qi, "launch_slot_", slots[qi], task0 + qi, "processing task, is_child = 1")
        g.slot_line(t_launch, "launch_slot_", parent, task0, "processing task, is_child = %d" % 0)

        # decodes
        by_it = {}
        for p in passes:
            by_it.setdefault(p["iteration"], []).append(p)
        t = dec_start
        comp = {"prepare": 0, "submit": 0, "between": 0, "post_submit": 0, "wait": 0}
        first_setup = None
        pending = []
        gpu_free = 0
        rem = list(suffix)
        rel_idx = []
        n_decodes = len(by_it)
        obs_ubs = []
        for di, it in enumerate(sorted(by_it)):
            dps = by_it[it]
            dec_pre = g.u(60, 140)
            if first_setup is None:
                first_setup = dec_pre
            else:
                comp["between"] += dec_pre
            t_dec_start = t
            t = t + dec_pre
            dec_tokens = sum(p["n_tokens"] for p in dps)
            for pi, p in enumerate(dps):
                nt, ns = p["n_tokens"], len(p["seqs"])
                nst = p["seqs"][0][1]
                shape = (nt, ns, nst, p["n_out"])
                reuse = 1 if shape == g.prev_shape else 0
                g.prev_shape = shape
                apply_ = g.u(30, 70)
                bld = 0 if reuse else int(g.u(900, 1300) + 0.25 * nt)
                alloc = 0 if reuse else g.u(250, 450)
                inp = int(g.u(150, 250) + 0.3 * nt)
                direct = nt > 1
                submit = g.u(5500, 7500) + 150 * (ns - 1) if direct else g.u(700, 1100)
                parts = [apply_, bld, alloc, inp, submit]
                ub_start = t
                t_end = ub_start + sum(parts)
                n_nodes = 1296 if nt > 8 else 1304
                g.p_cuda(t_end - 4, n_nodes, "direct" if direct else "graph_replay", submit - 15.0)
                g.p_ubatch(t_end, nt, ns, nst, p["n_out"], n_nodes, reuse, parts)
                comp["prepare"] += apply_ + bld + alloc + inp
                comp["submit"] += submit
                # GPU model
                launch_start = t_end - submit
                s_k = max(gpu_free, launch_start + 40)
                gpu_free = max(s_k + g.gpu_time_us(nt, ns, nst, p["n_out"]), t_end + 30)
                pending.append({"start": ub_start, "end": t_end, "tokens": nt})
                obs_ubs.append(p)
                t = t_end
                if pi < len(dps) - 1:
                    gap = g.u(20, 60) + (40 if p["n_out"] else 0)
                    comp["between"] += gap
                    t += gap
            decode_post = g.u(10, 40)
            t_dec_end = t + decode_post
            g.p_decode(t_dec_end, dec_tokens, len(dps), int(dec_pre * 0.6), t_dec_end - t_dec_start)
            t = t_dec_end
            has_out = any(p["n_out"] for p in dps)
            last = di == n_decodes - 1
            if has_out:
                pre_sync = g.u(10, 30)
                sb = t + pre_sync
                wait = max(5, int(gpu_free - sb))
                se = sb + wait
                g.p_sync(se, len(pending), sum(x["tokens"] for x in pending), wait, se - pending[0]["start"], se - pending[-1]["end"])
                comp["wait"] += wait
                if last:
                    comp["post_submit"] += decode_post + pre_sync
                else:
                    comp["between"] += decode_post + pre_sync
                pending = []
                gpu_free = 0
                # release lines (slot index order) of the questions that finish in this decode
                done_q = []
                for p in dps:
                    for k, q in enumerate(p["questions"]):
                        rem[q] -= p["seqs"][k][1]
                        if rem[q] == 0:
                            done_q.append(q)
                rel_t = se + 25
                for q in sorted(done_q, key=lambda q_: slots[q_]):
                    g.slot_last_release[slots[q]] = rel_t
                    g.slot_line(rel_t, "release", slots[q], task0 + q, "stop processing: n_tokens = %d, truncated = 0" % task_tokens[q])
                    rel_idx.append(len(g.lines) - 1)
                    rel_t += 8
                t = se
                if not last:
                    loop = g.u(100, 300)
                    comp["between"] += loop
                    t += loop
            else:
                loop = g.u(100, 300)
                comp["between"] += decode_post + loop
                t += loop
        tail = g.u(50, 150)
        t_wait_end = t + tail
        fmt = g.u(20, 60)
        t_req = t_wait_end + fmt
        http_out = g.u(80, 200)
        t_recv = t_req + http_out
        wait_us = t_wait_end - e_built
        total = t_req - start
        g.p_request(t_req, n_q, 1, parse, build, wait_us, fmt, total)
        if n % 9 == 0:
            g.junk(t_req + 5)
        g.T = t_recv + g.u(40, 120)
        # race: the last release line of this request is written after the request line (asynchronous log thread)
        if pick.get("release_after_request") == n:
            k, sq, x = g.lines[rel_idx[-1]]
            g.lines[rel_idx[-1]] = (t_req + 1, sq, x)
        # anomalies that edit the lines of this request
        truth_flag = None
        for name, idx in pick.items():
            if idx != n or name == "release_after_request":
                continue
            truth_flag = name
            if name == "drop_sync":
                g.lines = [(k, s, x) for k, s, x in g.lines if not (start <= k <= t_req and "DMK_TIMING sync" in x)]
            elif name == "garbled_ubatch":
                done = False
                out = []
                for k, s, x in reversed(g.lines):
                    if not done and "DMK_TIMING ubatch" in x and start <= k <= t_req:
                        x = x[:70]
                        done = True
                    out.append((k, s, x))
                g.lines = list(reversed(out))
            elif name == "no_pass_lines":
                g.lines = [(k, s, x) for k, s, x in g.lines if not (start <= k <= t_req and any(w in x for w in ("DMK_TIMING ubatch", "DMK_TIMING decode", "DMK_TIMING sync", "DMK_TIMING cuda_compute")))]
        census_expect = not warm
        components = {"http_in_ms": http_in / 1e3, "parse_ms": parse / 1e3, "build_ms": build / 1e3, "sched_ms": (wake + l2d) / 1e3,
                      "first_setup_ms": first_setup / 1e3, "prepare_ms": comp["prepare"] / 1e3, "submit_ms": comp["submit"] / 1e3,
                      "between_ms": comp["between"] / 1e3, "post_submit_ms": comp["post_submit"] / 1e3, "gpu_wait_ms": comp["wait"] / 1e3,
                      "tail_server_ms": tail / 1e3, "format_ms": fmt / 1e3, "http_out_ms": http_out / 1e3,
                      "wall_ms": (t_recv - t_send) / 1e3, "sched_wake_est_ms": wake / 1e3, "sched_launch_est_ms": l2d / 1e3}
        passes_obs = "+".join(("P%d" % p["n_tokens"]) if p["n_out"] == 0 else (("F%d" % p["n_tokens"]) if n_q == 1 else "S%dx%d" % (len(p["seqs"]), p["seqs"][0][1]))
                              for p in passes)
        truth_reqs.append({"idx": n, "kind": kind, "case": cid, "phase": phase, "components": components, "slots": ",".join(map(str, slots)),
                           "passes_obs": passes_obs, "census_match_expected": census_expect, "slot_select": sel, "ref_delta_tokens": delta,
                           "flag": truth_flag, "input_tokens": sum(task_tokens)})
        client_reqs.append({"idx": n, "kind": kind, "case": cid, "phase": phase, "pair": step["pair"], "ref": ref,
                            "t_send_mono_ns": t_send * 1000, "t_header_mono_ns": (t_recv - 50) * 1000,
                            "t_recv_mono_ns": t_recv * 1000, "t_send_epoch_ns": 1_793_400_000_000_000_000 + (t_send - BASE_US) * 1000,
                            "wall_ms": (t_recv - t_send) / 1e3, "input_tokens": sum(task_tokens)})
        if n == extra_after:
            # a request from somebody else (curl): one request line, no pass lines, between two client requests
            s0 = g.T + 500
            g.p_request(s0 + 1100, 1, 1, 300, 200, 400, 100, 1100)
            g.T = s0 + 1100 + 600
    # ---------------------------------------------------------------- finish the lines of the log
    g.lines.sort(key=lambda x: (x[0], x[1]))
    # unrelated text lines at random places (not splitting a line)
    texts = [x for _, _, x in g.lines]
    if not strip_dmk:
        n_junk = 40 if anomalies else 5
        for _ in range(n_junk):
            pos = rnd.randrange(1, len(texts))
            texts.insert(pos, rnd.choice(["srv  update_slots: all slots are idle", "srv  log_server_r: request: POST /health 127.0.0.1 200",
                                          "%s I srv  update_slots: all slots are idle" % fmt_tlog(rnd.randrange(1, 10 ** 9))]))
    else:
        texts = [x for x in texts if "DMK_TIMING" not in x]
    header = ["ERROR: driverInitFileInfo 578 result=11ERROR: init 664 result=11ERROR: init 250 result=11warn: LLAMA_ARG_HOST environment variable is set, but will be overwritten by command line argument --host",
              "0.00.387.436 I srv  llama_server: initializing ...", "0.07.554.165 I srv    load_model: initializing, n_slots = 3, n_ctx_slot = 16384, kv_unified = 'true'",
              "0.07.560.253 I srv  server_on_li: listening on http://0.0.0.0:8080"]
    with open(os.path.join(outdir, "p3-dec-timing.server.log"), "w") as f:
        f.write("\n".join(header + texts) + "\n")

    anomaly_idx = {name: idx for name, idx in pick.items() if name in flag_for}
    anomaly_flags = {name: flag_for[name] for name in anomaly_idx}
    truth = {"requests": truth_reqs, "anomalies": anomaly_idx, "anomaly_flags": anomaly_flags,
             "anomaly_requests": list(anomaly_idx.values()), "race_idx": pick.get("release_after_request"),
             "extra_server_request_after_idx": extra_after, "warm_reuse_kinds": list(WARM_REUSE_KINDS), "timing_overhead_ms": TIMING_OVERHEAD_MS,
             "offset_s": offset_s, "seed": seed, "note": "synthetic data: every duration is invented, see make_synthetic.py"}
    meta = {"argv": ["decomp_client.py", "--out", "/runs/p3-dec-timing.client.json"], "started_epoch_ns": client_reqs[0]["t_send_epoch_ns"], "kinds": list(common.KINDS),
            "reps": 30, "cycle_rounds": 2, "pairs": 15, "warmup": 3, "seed": 20261031, "port": 18080, "reconnects": 1 if anomalies else 0,
            "n_requests": len(client_reqs), "n_errors": 0, "rep_case": common.REP_CASE, "synthetic": True}
    common.write_json(os.path.join(outdir, "p3-dec-timing.client.json"), {"meta": meta, "requests": client_reqs, "errors": [], "_truth": truth})
    # reference run: same requests, DMK_TIMING costs a little, other time base
    rr = []
    shift = (BASE_US + 1_200_000_000_000 - g.T) * 1000
    for c in client_reqs:
        w = max(0.5, c["wall_ms"] - TIMING_OVERHEAD_MS + rnd.gauss(0, 0.04))
        t_send = c["t_send_mono_ns"] + shift
        rr.append(dict(c, t_send_mono_ns=t_send, t_header_mono_ns=t_send + int(w * 1e6) - 30000, t_recv_mono_ns=t_send + int(w * 1e6), wall_ms=w,
                       t_send_epoch_ns=c["t_send_epoch_ns"] + 600_000_000_000))
    meta_ref = dict(meta, argv=["decomp_client.py", "--out", "/runs/p3-dec-ref.client.json"], reconnects=0)
    common.write_json(os.path.join(outdir, "p3-dec-ref.client.json"),
                      {"meta": meta_ref, "requests": rr, "errors": [], "_truth": {"timing_overhead_ms": TIMING_OVERHEAD_MS, "note": "synthetic"}})
    return truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--offset-s", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-anomalies", action="store_true")
    a = ap.parse_args()
    t = make(a.outdir, a.offset_s, a.seed, not a.no_anomalies)
    print("wrote", a.outdir, len(t["requests"]), "requests, anomalies:", t["anomalies"])


if __name__ == "__main__":
    main()
