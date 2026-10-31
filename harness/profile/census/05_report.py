#!/usr/bin/env python3
"""Step 5: write census-report.txt from the TSVs of steps 1-4 (python3 stdlib only). Every table of the report is generated here.

Run:  python3 05_report.py [--tables DIR] [--out DIR]

Inputs : the tables of 03_census.py and 04_crosscheck.py in --tables (default: tables/ of the census data set); the inputs of the
         census (hparams.tsv, tensors.tsv, cases-plan.tsv, token-check.tsv, trace-passes.tsv) are found with census_lib.data_file().
Output : census-report.txt in --out (default: reports/ of the census data set). It is plain text; its tables are written with | bars.
"""
import argparse
import csv
import os
import statistics as st
from collections import Counter, OrderedDict, defaultdict

import census_lib as L

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("--tables", default=os.path.join(L.data_dir(), "tables"), help="folder with the tables of steps 3 and 4 (default: tables/ of the census data set)")
_ap.add_argument("--out", default=os.path.join(L.data_dir(), "reports"), help="output folder (default: reports/ of the census data set)")
_args = _ap.parse_args()
TABLES, OUT = _args.tables, _args.out
os.makedirs(OUT, exist_ok=True)
M = L.Model()


def read_tsv(name):
    path = os.path.join(TABLES, name)
    if not os.path.exists(path):
        path = L.data_file(name)
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def f(x, nd=1):
    x = float(x)
    return f"{x:,.{nd}f}"


def i(x):
    return f"{int(round(float(x))):,}"


def table(headers, rows, source):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")
    out.append("Source: " + source)
    out.append("")
    return "\n".join(out)


hp = {r["parameter"]: r for r in read_tsv("hparams.tsv")}
tensors = read_tsv("tensors.tsv")
plans = read_tsv("cases-plan.tsv")
tcheck = read_tsv("token-check.tsv")
summ = read_tsv("census-summary.tsv")
cpass = read_tsv("census-passes.tsv")
chead = read_tsv("census-head.tsv")
ccases = read_tsv("census-cases.tsv")
xsum = read_tsv("crosscheck-summary.tsv")
xreq = read_tsv("crosscheck-requests.tsv")
xpass = read_tsv("crosscheck-passes.tsv")
xkern = read_tsv("crosscheck-kernels.tsv")
tpass = read_tsv("trace-passes.tsv")
xs = {r["check"]: r for r in xsum}

S = []
w = S.append

# ------------------------------------------------------------------------------------------------ header
w("# Shape and launch census of the decision model (phase 3)\n")
w("What every pass of every request type of the benchmark computes (matmul shapes, CUDA paths, FLOPs, bytes, roofline floors) and what the three "
  "old Nsight Systems traces show (launch counts, gaps), with a cross-check of the two. Model: H2O Lightning 4B mix8 on llama.cpp b12004 "
  "(commit 3c809f4), RTX 3060 Ti (sm_86, 38 SMs). Nothing here runs on the GPU: the scripts run on the CPU (python3 stdlib, `uv run` only for step 2) "
  "and read the header of the model file, the tokenizer and the three old captures.\n")
w("All tables: units are in the column headers, one quantity per column, the source is under the table. Values that no source or trace shows are "
  "marked [INFERENCE]. `make_tables.sh` of the data set writes this report from the tables in `raw/`.\n")

# ------------------------------------------------------------------------------------------------ 1 key results
sr = {(r["group"], r["slot_structure"], r["variant"]): r for r in summ if r["kind"] == "request_type"}
alls = "all (uniform over slot arrangements)"


# ------------------------------------------------------------------------------------------------ numbers used in the text
def pass_class(r):
    if r["pass_kind"].startswith("prefix"):
        return "prefix, N<200" if int(r["N_tokens"]) < 200 else "prefix, N>=200"
    return "suffix, N<=8" if int(r["N_tokens"]) <= 8 else "suffix, N>8"


cl14 = OrderedDict((k, []) for k in ["prefix, N<200", "prefix, N>=200", "suffix, N<=8", "suffix, N>8"])
for r in xpass:
    cl14[pass_class(r)].append(r)


def med(v, c):
    return st.median(float(x[c]) for x in v)


ratio_max = [med(v, "busy_over_pass_max_floor") for v in cl14.values()]
ratio_roof = [med(v, "busy_over_roofline_floor") for v in cl14.values()]
# achieved rates of the matmul kernels, median over the passes of the class: model int8 work of the MMQ launches / observed MMQ kernel time (incl. fix-up);
# model weight bytes of the MMVQ launches / observed MMVQ kernel time
mmq_tops = [st.median(float(x["model_mmq_gflop"]) / float(x["observed_mmq_kernel_ms"]) for x in v) for k, v in cl14.items() if k != "suffix, N<=8"]
mmvq_gbs = [st.median(float(x["model_mmvq_weight_mb"]) / float(x["observed_mmvq_kernel_ms"]) for x in v) for k, v in cl14.items() if k == "suffix, N<=8"]
pc_out = L.pass_costs(M, {"n_tokens": 51, "n_out": 1, "seqs": [(0, 51)]})
pc_pre = L.pass_costs(M, {"n_tokens": 51, "n_out": 0, "seqs": [(0, 51)]})
_g = defaultdict(list)
for r in tpass:
    if r["row_kind"] == "pass":
        _g[(r["trace"], r["pass_kind"], int(r["n_mmvq"]) > 10)].append(float(r["gap_share_of_wall_pct"]))
_gm = {k: st.median(v) for k, v in _g.items()}
_pb = [v for k, v in _gm.items() if k[1].startswith("prefix") and k[0] != "short"]
_sf = [v for k, v in _gm.items() if k[1] == "suffix" and not (k[0] == "long" and k[2])]  # the long MMVQ-path class holds two lazy module loads (printed by trace_census.py)
gap_prefix_big = (min(_pb), max(_pb))
gap_suffix = (min(_sf), max(_sf))
_oth = [int(r["n_kernels"]) - sum(int(r[c]) for c in ("n_mmq", "n_mmq_fixup", "n_mmvq", "n_quantize", "n_flash_attn", "n_flash_aux", "n_gdn")) for r in tpass if r["row_kind"] == "pass" and r["pass_kind"] == "suffix" and int(r["n_mmvq"]) < 10]
other_n = st.median(_oth)


def sm(g, col, struct=alls, var="ref"):
    return float(sr[(g, struct, var)][col])


w("## 1 Key results\n")
n_ok = sum(1 for r in xsum if r["all_passed"] == "yes")
w(f"- **Cross-check**: the analytic model predicts the launch count of every kernel class per pass in all {xs['matmul count (MMQ) per pass']['total']} traced passes and, for the "
  f"{xs['launch keys (type, tile, need_check, grid.x, fix-up) identical as multiset, passes with launch records']['total']} passes that have a launch listing, "
  f"every one of the 9,119 matmul launches (weight type, tile width, need_check, grid.x, fix-up kernel) exactly. {n_ok} of {len(xsum)} checks pass without exception; "
  f"the exception is the flash-attention K/V window (section 9, item 5).")
w(f"- **Passes per request** (ref variant, with the `Ref:` line): short one-question {sm('short-1q','n_passes_mean'):.0f}, short two-question {sm('short-2q','n_passes_mean'):.0f}, "
  f"medium {sm('medium-3q','n_passes_mean'):.0f}, long {sm('long-3q','n_passes_mean'):.0f}. The question suffixes are *not* always batched by length: which sequences share a pass depends on "
  f"the slots of the parent and the children (section 3.2); the pass count is the same for every slot placement, the pass sizes are not.")
w(f"- **Every pass streams almost the same weights**: {pc_out['wbytes'] / 1e9:.2f} GB for a pass with the output head, {pc_pre['wbytes'] / 1e9:.2f} GB without "
  f"(prefix passes), {pc_out['floor_weights_s'] * 1e3:.2f} ms and {pc_pre['floor_weights_s'] * 1e3:.2f} ms at 448 GB/s. A request is therefore {f(sm('short-1q','floor_weights_ms_mean'),1)} / {f(sm('short-2q','floor_weights_ms_mean'),1)} / "
  f"{f(sm('medium-3q','floor_weights_ms_mean'),1)} / {f(sm('long-3q','floor_weights_ms_mean'),1)} ms of weight reads (short-1q / short-2q / medium / long); the int8 compute floor is "
  f"{f(sm('short-1q','floor_int8_ms_mean'),1)} / {f(sm('short-2q','floor_int8_ms_mean'),1)} / {f(sm('medium-3q','floor_int8_ms_mean'),1)} / {f(sm('long-3q','floor_int8_ms_mean'),1)} ms at 144.8 TOPS "
  f"(medium and long requests are bound by int8 work, short requests by weight reads).")
w("- **MMQ is the only prompt matmul path** for Q4_0 and Q8_0 once a pass has more than 8 tokens; a pass of at most 8 tokens (4 tokens of one sequence, 2 x 4 tokens of two) runs *all* matmuls as "
  "MMVQ. cuBLAS is never used. 245 MMQ launches per pass, plus 3 MMVQ launches (fused FFN of the last layer on the output rows, output head) when the pass has an output.")
w(f"- **The output head is computed for {i(M.n_vocab)} rows per output; the API reads 2-8 of them** (section 6.3): {f(sm('short-1q','head_gb_read_mean'),2)} GB (one head read) per one-question "
  f"request, {f(sm('medium-3q','head_gb_read_mean'),2)} GB per medium or long request (three head reads).")
w(f"- **Traces vs floor** (nsys overhead not removed): the GPU-busy time of a pass is {min(ratio_max):.1f}x (pass of at most 8 tokens) to {max(ratio_max):.1f}x (prefix passes) the larger of the weight and int8 floors and "
  f"{min(ratio_roof):.1f}x to {max(ratio_roof):.1f}x the serial per-kernel roofline floor (section 8.2); the MMQ kernels reach {min(mmq_tops):.0f}-{max(mmq_tops):.0f} int8 TOPS "
  f"({100 * min(mmq_tops) / 144.8:.0f}-{100 * max(mmq_tops) / 144.8:.0f} % of 144.8; {mmq_tops[2]:.0f} TOPS in the suffix passes, {mmq_tops[1]:.0f} TOPS in the large prefix passes) and the all-MMVQ pass reads weights at {mmvq_gbs[0]:.0f} GB/s ({100 * mmvq_gbs[0] / 448:.0f} % of 448) (section 8.3). "
  f"In-pass launch gaps are {gap_prefix_big[0]:.2f}-{gap_prefix_big[1]:.2f} % of a long or medium prefix pass and {gap_suffix[0]:.1f}-{gap_suffix[1]:.1f} % of a suffix pass (section 7).")
w("")

# ------------------------------------------------------------------------------------------------ 2 hyperparameters
w("## 2 Model and weights\n")
want = ["arch", "model_name", "gguf_file_bytes", "n_tensors", "n_layer (executed)", "n_layers_recurrent (Gated DeltaNet)", "n_layers_attention", "layer_pattern", "n_embd", "n_ff", "n_head", "n_head_kv",
        "head_dim_k", "head_dim_v", "gqa_ratio", "attn_q_rows (query + output gate)", "attn_kv_rows (each of K, V)", "attn_o_in", "rope_dimension_count", "rope_sections", "rope_freq_base", "rms_norm_eps",
        "context_length_train", "ssm_conv_kernel", "ssm_inner_size (d_inner)", "ssm_state_size = GDN head_k_dim", "ssm_group_count = GDN num_k_heads", "ssm_time_step_rank = GDN num_v_heads",
        "GDN head_v_dim", "GDN conv_channels", "GDN qkv rows (attn_qkv ne1)", "GDN recurrent state per layer per sequence", "GDN conv state per layer per sequence", "n_vocab", "output head",
        "decision.n_labels", "tokenizer", "tensor bytes, type Q4_0", "tensor bytes, type Q8_0", "tensor bytes, type F32", "tensor bytes in the 32 layers", "tensor bytes token_embd (= LM head)",
        "matmul weight elements in the 32 layers"]
rows = []
for k in want:
    r = hp[k]
    v = r["value"]
    if v.lstrip("-").isdigit():
        v = f"{int(v):,}"
    rows.append([k, v, r["unit"] or "-", r["source"]])
w("### 2.1 Hyperparameters (Table 1)\n")
w(table(["parameter", "value", "unit", "GGUF key / derivation"], rows,
        "`tensors.tsv`, `hparams.tsv`, `layers.tsv` from `01_gguf_header.py` (GGUF header of the mix8 model file, first 16 MB read; "
        "the file was not modified)."))

groups = OrderedDict()
for r in tensors:
    if r["role"].endswith("norm.weight") or r["role"] in ("ssm_a", "ssm_dt.bias", "ssm_conv1d.weight") or r["ggml_type"] == "F32":
        key = ("F32 vectors / conv (all)", "F32", "-", "-")
    else:
        key = (r["role"], r["ggml_type"], r["ne0_k_in"], r["ne1_m_out"])
    groups.setdefault(key, []).append(r)
total_b = sum(int(r["bytes"]) for r in tensors)
rows = []
for (role, ty, k, m), v in groups.items():
    b = sum(int(r["bytes"]) for r in v)
    n_per = int(v[0]["bytes"]) / 1e6 if role != "F32 vectors / conv (all)" else b / len(v) / 1e6
    path = "MMQ (N>8) / MMVQ (N<=8)" if role != "F32 vectors / conv (all)" else "element-wise / conv kernels"
    if role == "token_embd.weight":
        path = "get_rows (input) and MMVQ (output head, N<=3)"
    rows.append([role, ty, k, m, len(v), f(n_per, 3), f(b / 1e6, 1), f(100 * b / total_b, 1), path])
w("### 2.2 Weight tensors (Table 2)\n")
w(table(["tensor", "ggml type", "K (input dim)", "M (output rows)", "tensors in the file", "bytes per tensor (MB)", "bytes total (MB)", "share of tensor bytes (%)", "CUDA path"], rows,
        "`tensors.tsv` (426 tensors; sizes = 18 B per 32 Q4_0 elements, 34 B per 32 Q8_0 elements). The 24 recurrent layers have attn_qkv, attn_gate, ssm_beta, ssm_alpha, ssm_out; the 8 attention layers have attn_q (query + gate), attn_k, attn_v, attn_output; "
        "all 32 layers have ffn_up, ffn_gate, ffn_down (Q4_0). The NextN/MTP layer (block 32) has no tensors in this file."))

# ------------------------------------------------------------------------------------------------ 3 requests and passes
w("## 3 Requests and passes\n")
w("### 3.1 Token counts (Table 3)\n")
by_task = defaultdict(list)
for r in plans:
    by_task[(r["variant"], r["task"])].append(r)
rows = []
order = ["check", "commit", "triage", "style", "syslog-medium", "syslog-long"]
for t in order:
    rf = by_task[("ref", t)]
    rw = by_task[("raw", t)]
    sh = [int(r["shared_prefix_tokens"]) for r in rf]
    sf = defaultdict(list)
    for r in rf:
        for j, s in enumerate(r["suffix_tokens_per_question"].split()):
            sf[j].append(int(s))
    suff = " / ".join(f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}" for _, v in sorted(sf.items()))
    spt = [int(r["sum_prompt_tokens"]) for r in rf]
    rows.append([t, rf[0]["size"], len(rf), rf[0]["n_questions"], f"{min(sh)}-{max(sh)}" if max(sh) else "0 (no sharing)", suff,
                 f"{st.fmean(int(r['sum_prompt_tokens']) for r in rw):,.1f}", f"{st.fmean(spt):,.1f}", f"{min(spt)}-{max(spt)}"])
w(table(["task", "size", "cases", "questions per request", "shared prefix, ref variant (tokens)", "suffix per question, ref variant (tokens)",
         "sum of prompt tokens, raw variant, mean (tokens)", "sum of prompt tokens, ref variant, mean (tokens)", "sum of prompt tokens, ref variant, min-max (tokens)"], rows,
        "`token-counts.tsv`, `cases-plan.tsv` from `02_tokens.py` (systemone template rendered as the server does, tokenised with the b12004 `llama-tokenize` and the GGUF vocabulary). "
        "The raw variant matches `quality.input_tokens_mean` of a benchmark run (`data/2026-10-30-window1/raw/p1-bench-b1-S.json`) exactly: " +
        ", ".join(f"{r['size']} {r['mean_sum_prompt_tokens_raw_here']} vs {r['mean_input_tokens_production_run']}" for r in tcheck) +
        ". The ref variant adds the `Ref: <10 hex>\\n` line (about 13 tokens per prompt) that the timed benchmark sends. Prefix = shared tokens of the question prompts of one request; suffix = tokens after the prefix; "
        "one-question requests have no sharing (the whole prompt is the suffix)."))

w("### 3.2 How a request becomes passes\n")
w("The server (`tools/server/server-context.cpp`, `server-decision.cpp`, `src/llama-batch.cpp` `split_equal`) turns a request into passes by these rules (all read from the b12004 source, then confirmed by 20 traced requests):\n")
w("1. A request with one question has no sharing: one pass with the whole prompt (split at 2048 tokens), output on its last token.")
w("2. A request with 2-3 questions is a parent task (question 0) plus child tasks that run in other slots (`-np 3`). The parent evaluates the shared prefix alone in ubatches of at most 2048 tokens (no outputs; the last layer's FFN and the output head are empty and are skipped).")
w("3. When the prefix is done the parent starts the children. In the same `update_slots()` loop the children whose slot index is *above* the parent's are visited after it, so their suffix is in the same `llama_decode` batch as the parent's suffix; "
  "children with a *lower* slot index were visited before and follow in the next batch (one extra iteration).")
w("4. Every batch is split by `split_equal(n_ubatch, sequential)`; the hybrid memory passes `sequential = !unified` (`src/llama-memory-hybrid.cpp` lines 83-90), and production runs `--kv-unified`, so `sequential = false`: "
  "a ubatch takes the first sequence with unused tokens and then every other sequence with unused tokens, each with min(remaining) tokens (so a sequence in slot 0 and one in slot 2 share a pass); a sequence that finishes in a ubatch gets an output row. "
  "[Correction: the first version of this census used `sequential = true`, the setting of the pure recurrent memory, which only lets consecutive seq ids share a pass. The 20 traced requests are reproduced by both settings; they differ only for a two-question request in slots (0, 2). "
  "`census_lib.SPLIT_SEQUENTIAL` selects the setting; the DMK_TIMING `ubatch` lines (n_seqs) of a request with the child in a higher slot than the parent decide it on the running server.]")
w("5. Slot placement is decided by the server (least recently used among idle slots), not by the benchmark. All six placements of 2 or 3 questions on 3 slots are enumerated by `census_lib.arrangements()`.\n")

# structure table
ex = {"short-2q": (81, [120, 51]), "medium-3q": (1288, [43, 47, 73])}
xr_struct = Counter()
for r in xreq:
    xr_struct[(r["trace"], r["observed_passes (tokens/outputs)"].replace("/0", "").replace("/1", ""))] += 1
rows = []
for name, (sh, suf) in ex.items():
    seen = OrderedDict()
    for a in L.arrangements(len(suf)):
        ps = L.simulate_request(sh, suf, a)
        sig = " + ".join((f"prefix {p['n_tokens']}" if p["kind"] == "prefix" else f"{len(p['seqs'])} seq x {p['seqs'][0][1]}") for p in ps)
        seen.setdefault(sig, []).append(a)
    for sig, arrs in seen.items():
        ps = L.simulate_request(sh, suf, arrs[0])
        n_vq = sum(1 for p in ps if p["n_tokens"] <= 8)
        traced = [r for r in xreq if ",".join(map(str, arrs[0])) in r["slot_arrangements_matching_Q0,Q1,Q2"].split(" | ")]
        rows.append([name, f"{sh} + {suf}", sig, " ".join("(" + ",".join(map(str, a)) + ")" for a in arrs), len(arrs), n_vq, len(traced)])
w("**Table 4: pass structures of a request for every slot placement** (example: prefix 81, suffixes 120/51 for two questions; prefix 1,288, suffixes 43/47/73 for three; a prefix above 2,048 tokens adds one more prefix pass)\n")
w(table(["request type", "prefix + suffix tokens", "passes (tokens = sequences x tokens per sequence)", "slots of (Q0, Q1[, Q2]) that give it", "placements of 6", "passes with N <= 8 (all-MMVQ)", "traced requests with this structure"], rows,
        "`census_lib.simulate_request()` (rules above) over `census_lib.arrangements()`; traced count = requests of `crosscheck-requests.tsv` whose observed passes are reproduced by that placement "
        "(the three captured request types: short two-question = triage, medium, long)."))
w("In the traces the 12 short requests all ran the two suffixes as separate one-sequence passes (120, 51), never 2 x 51 + 69; with `sequential = false` this means the child's slot was below the parent's in all 12 (placements (1,0), (2,0), (2,1): the child runs one decode batch after the parent's suffix, the case of upstream PR #30779); the 8 three-question requests show three structures (3 x 43 + 2 x 4 + 26 twice, 43 + 2 x 47 + 26 three times, 2 x 43 + 4 + 73 three times); "
  "the parent slot rotated 1, 2, 0, 1 in the long trace and 2, 0, 1, 2 in the medium trace [INFERENCE: least-recently-used rotation; the arrangement of a production request is a function of the previous requests]. "
  "`crosscheck-requests.tsv` lists the matching placements of each traced request.\n")

# ------------------------------------------------------------------------------------------------ 4 shapes and kernels
w("## 4 Matmul shapes and the kernels they hit\n")
w("Rules (`ggml/src/ggml-cuda/ggml-cuda.cu` `ggml_cuda_mul_mat`, `mmq.cu/cuh`, `mmvq.cu`, `mmq-config-ampere.cuh`; every one reproduced by the traces): "
  "Q4_0/Q8_0 weights with N <= 8 columns use `mul_mat_vec_q` (MMVQ), otherwise `mul_mat_q` (MMQ, stream-k, 128-row tiles, 256 threads, tile width J chosen as the smallest of {8,16,24,32,40,48,64,80,96,112,128} "
  "(fallback tiles for M % 128 != 0: {8,16,32,64,128}) that gives the fewest column tiles); stream-k grid.x = number of tiles if the tile efficiency is >= 90 %, else min(38, work units) with a fix-up kernel when tiles % grid.x != 0. "
  "Every matmul has its own activation quantize kernel (`quantize_mmq_q8_1` or `quantize_q8_1`); there is no gate/up fusion for MMQ with Q4_0, so up and gate are two launches. "
  "MMVQ with N = 1 fuses gate+up+SwiGLU and down+residual add (`mul_mat_vec_q<Q4_0,1,fusion>`).\n")

NS = [1, 4, 8, 26, 43, 51, 86, 129, 1288, 2048]
OPS = [("attn_qkv", "Q8_0", 8192, 2560, "24 recurrent"), ("attn_gate (z)", "Q8_0", 4096, 2560, "24 recurrent"), ("ssm_beta", "Q8_0", 32, 2560, "24 recurrent"),
       ("ssm_alpha", "Q8_0", 32, 2560, "24 recurrent"), ("ssm_out", "Q8_0", 2560, 4096, "24 recurrent"), ("attn_q (+gate)", "Q8_0", 8192, 2560, "8 attention"),
       ("attn_k", "Q8_0", 1024, 2560, "8 attention"), ("attn_v", "Q8_0", 1024, 2560, "8 attention"), ("attn_output", "Q8_0", 2560, 4096, "8 attention"),
       ("ffn_up", "Q4_0", 9216, 2560, "32 (last: output rows only)"), ("ffn_gate", "Q4_0", 9216, 2560, "32 (last: output rows only)"), ("ffn_down", "Q4_0", 2560, 9216, "32 (last: output rows only)"),
       ("output head (token_embd)", "Q8_0", M.n_vocab, 2560, "1 (output rows only)")]
rows = []
for name, ty, m, k, lay in OPS:
    cells = []
    for n in NS:
        if name.startswith("output head") and n > 3:
            cells.append("-")
            continue
        o = L.matmul(name, 0, ty, m, n, k)
        if o["path"] == "MMQ":
            cells.append(f"MMQ J={o['J']} g={o['grid_x']}" + ("+fix" if o["fixup"] else ""))
        else:
            cells.append(f"MMVQ n={n} g={o['grid_x']}" + (" small_k" if o["small_k"] else ""))
    rows.append([name, lay, ty, f"{m:,}", f"{k:,}"] + cells)
w("**Table 5: kernel hit by every matmul shape for the N values of the benchmark passes** (cell = path, tile width J of `mul_mat_q<type,J,...>`, grid.x `g`, `+fix` = stream-k fix-up kernel follows; MMVQ cell = ncols_dst `n`, grid.x `g`; "
  "N = 1 is the last layer's FFN and the head of a pass with one output; the head has at most 3 outputs)\n")
w(table(["op", "layers", "weight type", "M (rows)", "K"] + [f"N={n}" for n in NS], rows,
        "`census_lib.matmul()` (tile and grid prediction); all 9,119 launches of `trace-mmq-launches.tsv` are reproduced with the same type, J, need_check, grid.x and fix-up (`crosscheck-passes.tsv`). "
        "ssm_beta and ssm_alpha (M = 32) always use the fallback tiles (`need_check = true`)."))

w("**Table 6: the other kernels of a pass**\n")
rows = [
    ["quantize", "one per matmul launch, immediately before it (`prev_quant_distance` = 1 in all 9,119 launches)", "`quantize_mmq_q8_1` (MMQ) / `quantize_q8_1` (MMVQ)", f"{xs['quantize launches per pass']['passed']}/{xs['quantize launches per pass']['total']} passes"],
    ["Gated DeltaNet (24 layers)", "tokens per sequence >= 64: tensor-core chunk kernel; fewer: recurrent kernel", "`gdn_single<16,128,1>` grid 32 x 1 x 1 (x sequences) / `gated_delta_net_cuda<128,false,false,4>` grid 32 x S x 8", f"{xs['Gated DeltaNet kernel and grid (sequences) over the whole trace']['passed']}/{xs['Gated DeltaNet kernel and grid (sequences) over the whole trace']['total']} kernel/grid classes"],
    ["flash attention (8 layers)", "`flash_attn_ext_f16<256,256,ncols1,4>`, ncols1 = 2/4/8/16 for N <= 2/4/8/more (gqa 4, F16 K/V, MMA path of cc 8.6)", "+ one stream-k fix-up kernel per layer", f"{xs['flash-attention kernel template (ncols1) over the whole trace']['passed']}/{xs['flash-attention kernel template (ncols1) over the whole trace']['total']} kernel classes"],
    ["state gather / write-back", "`k_get_rows_float_vec` gathers the S states of the pass (grid.x = S), `cpy` writes them back; 2.1 MB state per layer and sequence", "get_rows + cpy per layer", "grid.x = S in all passes (`trace-kernels.tsv`)"],
    [f"other (about {other_n:.0f} launches)", "rms_norm, conv1d (`ssm_conv*`), silu/sigmoid gating, elementwise, rope, set_rows (KV cache)", "see `trace-kernels.tsv`", "counts per pass in `trace-passes.tsv`"],
]
w(table(["kernel group", "rule", "kernel", "validated"], rows,
        "`ggml/src/ggml-cuda/gated-delta-net-mma.cu` (`select_gdn_mma_path`), `fattn.cu`, `llama-graph.cpp` `build_rs`; counts from `crosscheck-summary.tsv` and `crosscheck-kernels.tsv`."))

# ------------------------------------------------------------------------------------------------ 5 per pass type
w("## 5 Census per pass type\n")
seen = OrderedDict()
for r in cpass:
    key = (r["pass_kind"], r["N_tokens"], r["n_seqs"], r["n_outputs"])
    seen.setdefault(key, r)
rows = []
for (kind, n, s, o), r in sorted(seen.items(), key=lambda kv: (kv[0][0] != "prefix", int(kv[0][1]))):
    rows.append([kind, i(n), s, o, i(r["n_mmq"]), i(r["n_mmvq"]), i(r["n_mmq_fixup_pred"]), r["gdn_kernel"].replace("<128,false,false,4>", ""), r["fa_ncols1"], f(r["matmul_gflop"], 1), f(r["padded_gops"], 1),
                 f(r["weight_mb"], 0), f(r["floor_weights_ms"], 2), f(r["floor_int8_ms"], 2), f(r["floor_roofline_ms"], 1)])
w("**Table 7: every distinct pass of the representative requests** (the median case of each task, every slot structure)\n")
w(table(["pass kind", "N (tokens)", "sequences", "output rows", "MMQ launches", "MMVQ launches", "MMQ fix-up kernels (pred.)", "GDN kernel", "FA ncols1", "matmul work (GFLOP)", "padded tile work (GOP)",
         "weight bytes read (MB)", "weight floor (ms)", "int8 floor (ms)", "serial roofline floor (ms)"], rows,
        "`census-passes.tsv` from `03_census.py` (`census_lib.pass_costs`). Weight floor = weight bytes / 448 GB/s; int8 floor = matmul FLOPs / 144.8 TOPS; serial roofline floor = sum over kernels of max(bytes / 448 GB/s, ops / peak) with all "
        "activation traffic assumed to come from DRAM (an upper bound for small N, where tensors fit the 4 MiB L2), FP32 peak 18.1 TFLOPS for the GDN kernels, FP16-tensor peak = int8 / 4 for flash attention [INFERENCE: GeForce GA10x FP16 with FP32 accumulate]. "
        "Prefix passes have no outputs: no head, no last-layer FFN. Full per-op rows: `census.tsv`; per-launch order: `census-sequence.tsv`."))

# ------------------------------------------------------------------------------------------------ 6 per request
w("## 6 Census per request type\n")
w("### 6.1 Per request (Table 8)\n")
rows = []
for kind, labels in (("request_type", ["short-1q", "short-2q", "medium-3q", "long-3q"]),):
    for g in labels:
        structs = [r for r in summ if r["kind"] == kind and r["group"] == g and r["variant"] == "ref"]
        for r in structs:
            rows.append([g, r["slot_structure"], i(r["n_case_arrangements"]), f(r["n_passes_mean"], 0), f(r["sum_prompt_tokens_mean"], 0), f(r["tokens_evaluated_mean"], 0),
                         f(r["n_mmq_mean"], 0), f(r["n_mmvq_mean"], 0), f(r["matmul_gflop_mean"], 0), f(r["weight_gb_read_mean"], 2), f(r["floor_weights_ms_mean"], 1), f(r["floor_int8_ms_mean"], 1),
                         f(r["floor_pass_max_ms_mean"], 1), f(r["floor_roofline_ms_mean"], 1)])
w(table(["request type", "slot structure (sequences per pass; P = prefix)", "cases x slot placements", "passes", "prompt tokens counted by the API (tokens)", "tokens evaluated (tokens)", "MMQ launches", "MMVQ launches",
         "matmul work (GFLOP)", "weight bytes read (GB)", "weight floor (ms)", "int8 floor (ms)", "max(weight, int8) per pass, summed (ms)", "serial roofline floor (ms)"], rows,
        "`census-summary.tsv` (ref variant; means over the cases of the type and over the slot placements of the structure; the first row of a type is the mean over all six placements, weights of the placements unknown). "
        "The raw variant (no `Ref:` line) and the per-task rows (check, commit, triage, style) are in the same file. Rooflines as in Table 7."))
w(f"Sharing the prefix saves tokens, not passes: the three-question requests evaluate {i(sm('medium-3q','tokens_evaluated_mean'))} (medium) and {i(sm('long-3q','tokens_evaluated_mean'))} (long) tokens instead of the "
  f"{i(sm('medium-3q','sum_prompt_tokens_mean'))} and {i(sm('long-3q','sum_prompt_tokens_mean'))} that the API counts, but need {sm('medium-3q','n_passes_mean'):.0f} and {sm('long-3q','n_passes_mean'):.0f} passes of the whole model.\n")

w("### 6.2 Where the FLOPs go (Table 9)\n")
rows = []
for g in ["short-1q", "short-2q", "medium-3q", "long-3q"]:
    r = sr[(g, alls, "ref")]
    mm, fa, gd = float(r["matmul_gflop_mean"]), float(r["fa_gflop_mean"]), float(r["gdn_gflop_mean"])
    rows.append([g, f(mm, 0), f(fa, 0), f(gd, 0), f(100 * mm / (mm + fa + gd), 1), f(float(r["state_traffic_gb_mean"]), 2), f(float(r["other_activation_traffic_gb_mean"]), 1)])
w(table(["request type", "matmul (GFLOP)", "flash attention, causal (GFLOP)", "Gated DeltaNet (GFLOP)", "matmul share of FLOPs (%)", "recurrent-state traffic, read+write (GB)", "activation traffic of quantize, matmul output and element-wise ops (GB)"], rows,
        "`census-summary.tsv` (ref variant, mean over all slot placements). Attention FLOPs = 4 x 256 x 16 heads x visible keys per layer (causal); the K/V window of the model is the request's own cells padded to 256 (the traces show wider windows, section 9 item 5). "
        "Activation traffic is a per-op estimate (`census_lib.side_ops`, [INFERENCE], not traced)."))

w("### 6.3 Output head: rows computed vs rows needed (Table 10)\n")
rows = [[r["request_type"], r["task"], r["n_questions"], r["passes_with_head"], i(r["head_rows_computed_per_request"]), r["head_rows_needed_per_request"], f(r["head_weight_mb_read_per_request"], 0),
         f(r["head_weight_mb_needed_per_request"], 3), f(r["head_time_at_448GBps_ms"], 2)] for r in chead]
w(table(["request type", "task", "questions", "passes with the head", "head rows computed per request (rows)", "head rows the API reads per request (rows)", "head weight bytes read (MB)", "head weight bytes needed (MB)", "head weight read time at 448 GB/s (ms)"], rows,
        "`census-head.tsv`. Computed rows = one output row per question x 248,320 vocabulary rows (MMVQ, N = 1; confirmed by the trace: `mul_mat_vec_q<Q8_0,1>` grid.x 248,320 and a 993,280 B logits copy per output). "
        "Needed rows = the answer tokens of the question options (2 for an on/off question, 4-6 for a choice question; `decision.labels` has 255 entries in total), "
        "read from `token-counts.tsv` `n_options`; the read-out happens on the host from the full logits row."))

# ------------------------------------------------------------------------------------------------ 7 trace census
w("## 7 Trace census (three Nsight Systems captures)\n")
w("`trace-passes.tsv`, `trace-kernels.tsv`, `trace-mmq-launches.tsv` were exported by `trace_export.sh` + `trace_census.py` from the three old Nsight Systems captures `short`, `medium` and `long` (the README of the data set names them; `trace_census.py` prints the validation of the pass segmentation). "
  "Captured requests: 12 short (triage00-11), 4 medium, 4 long; one stream; no CUDA graph launched in any captured pass.\n")
cls = OrderedDict()
for r in tpass:
    if r["row_kind"] != "pass":
        continue
    kind = r["pass_kind"] if r["pass_kind"] != "suffix" else ("suffix, MMVQ path (N<=8)" if int(r["n_mmvq"]) > 10 else "suffix, MMQ path")
    cls.setdefault((r["trace"], kind), []).append(r)
rows = []
for (t, kind), v in cls.items():
    g = lambda c: st.median(float(x[c]) for x in v)
    rows.append([t, kind, len(v), f"{min(int(x['n_tokens']) for x in v)}-{max(int(x['n_tokens']) for x in v)}", i(g("n_kernels")), i(g("n_mmq")), i(g("n_mmq_fixup")), i(g("n_mmvq")), i(g("n_quantize")), i(g("n_flash_attn") + g("n_flash_aux")), i(g("n_gdn")),
                 f(g("wall_ms"), 1), f(g("gpu_busy_ms"), 1), f(g("gap_total_ms"), 2), f(g("gap_share_of_wall_pct"), 2), i(g("gap_n_over_5us")), i(g("gap_n_over_20us"))])
w("**Table 11: launches and in-pass gaps per pass class (median over the passes of the class)**\n")
w(table(["trace", "pass class", "passes", "N (tokens)", "kernels per pass", "main MMQ", "stream-k fix-up", "MMVQ", "quantize", "flash attention (main + fix-up)", "GDN", "pass wall, first kernel start to last end (ms)",
         "GPU busy (ms)", "gap total in the pass (ms)", "gap share of wall (%)", "gaps > 5 us per pass", "gaps > 20 us per pass"], rows,
        "`trace-passes.tsv` (`row_kind = pass`; gap = start of kernel i+1 minus end of kernel i inside one pass, one stream). The long MMVQ-path row contains two one-time lazy module loads of 11.4 and 12.6 ms (printed by `trace_census.py`)."))
rows = []
for r in tpass:
    if r["row_kind"] == "trace_total":
        rows.append([r["trace"], r["n_passes"], i(r["n_kernels"]), f(r["wall_ms"], 1), f(r["gpu_busy_ms"], 1), f(r["gap_total_ms"], 1), f(r["gap_intra_pass_ms"], 1), f(r["gap_inter_pass_ms"], 1),
                     f"{i(r['gap_n_lt1us'])} / {f(r['gap_t_lt1us_ms'], 1)}", f"{i(r['gap_n_1_5us'])} / {f(r['gap_t_1_5us_ms'], 1)}", f"{i(r['gap_n_5_20us'])} / {f(r['gap_t_5_20us_ms'], 1)}",
                     f"{i(r['gap_n_20_100us'])} / {f(r['gap_t_20_100us_ms'], 2)}", f"{i(r['gap_n_gt100us'])} / {f(r['gap_t_gt100us_ms'], 1)}"])
w("**Table 12: gap totals per trace** (count / total ms for the histogram columns)\n")
w(table(["trace", "passes", "kernels", "first kernel start to last kernel end (ms)", "GPU busy (ms)", "gap total (ms)", "of which inside passes (ms)", "of which between passes (ms)", "in-pass gaps < 1 us (count / ms)",
         "1-5 us (count / ms)", "5-20 us (count / ms)", "20-100 us (count / ms)", "> 100 us (count / ms)"], rows,
        "`trace-passes.tsv` (`row_kind = trace_total`). Between-pass gaps are idle time of the GPU between two `llama_decode` calls (input copies, logits copy, sampling, HTTP); the > 100 us in-pass gaps of the long trace are one-time lazy module loads."))

# ------------------------------------------------------------------------------------------------ 8 cross-check
w("## 8 Cross-check: analytic census against the traces\n")
w("### 8.1 What agrees (Table 13)\n")
rows = [[r["check"], r["passed"], r["total"], r["all_passed"]] for r in xsum]
w(table(["check", "passed", "compared", "all passed"], rows,
        "`crosscheck-summary.tsv` from `04_crosscheck.py` (`crosscheck-requests.tsv`, `crosscheck-passes.tsv`, `crosscheck-keys.tsv`, `crosscheck-kernels.tsv`). Matmul launches are compared as multisets per pass "
        "because ggml executes the graph depth-first (the attention_gate matmul of a recurrent layer runs after the delta-net, not next to attn_qkv). The launch listing of `trace-mmq-launches.tsv` covers the first pass of each distinct (trace, pass kind, N): 37 of the 72 passes; "
        "equal-shape passes have identical grids (checked in `trace_census.py`)."))

w("### 8.2 Observed pass time against the floors (Table 14)\n")
cl = OrderedDict()
for r in xpass:
    k = r["pass_kind"] if r["pass_kind"] != "suffix" else ("suffix, N<=8" if int(r["N_tokens"]) <= 8 else "suffix, N>8")
    if r["pass_kind"].startswith("prefix") and int(r["N_tokens"]) < 200:
        k = "prefix, N<200"
    elif r["pass_kind"].startswith("prefix"):
        k = "prefix, N>=200"
    cl.setdefault(k, []).append(r)
rows = []
for k in ["prefix, N<200", "prefix, N>=200", "suffix, N<=8", "suffix, N>8"]:
    v = cl.get(k)
    if not v:
        continue
    g = lambda c: st.median(float(x[c]) for x in v)
    rows.append([k, len(v), f"{min(int(x['N_tokens']) for x in v)}-{max(int(x['N_tokens']) for x in v)}", f(g("observed_gpu_busy_ms"), 1), f(g("model_floor_weights_ms"), 2), f(g("model_floor_int8_ms"), 2),
                 f(g("model_floor_roofline_ms"), 1), f(g("busy_over_pass_max_floor"), 2), f(g("busy_over_roofline_floor"), 2)])
w(table(["pass class (all traces)", "passes", "N (tokens)", "observed GPU busy under nsys (ms)", "weight floor (ms)", "int8 floor (ms)", "serial roofline floor (ms)", "busy / max(weight, int8)", "busy / roofline"], rows,
        "`crosscheck-passes.tsv` (medians; model passes chosen to reproduce each traced request). The trace includes nsys overhead (an earlier estimate is about 3 % for long and 25 % for short requests; not verified here). "
        "Per-pass rows: `crosscheck-passes.tsv`."))

w("### 8.3 Where the observed time goes and the rate of the matmul kernels (Table 15)\n")
rows = []
for k, v in cl14.items():
    def mg(c, v=v):
        return st.median(float(x[c]) for x in v)
    other_t = st.median(float(x["observed_gpu_busy_ms"]) - sum(float(x[c]) for c in ("observed_mmq_kernel_ms", "observed_mmvq_kernel_ms", "observed_quantize_kernel_ms", "observed_flash_kernel_ms", "observed_gdn_kernel_ms")) for x in v)
    tops = st.median(float(x["model_mmq_gflop"]) / float(x["observed_mmq_kernel_ms"]) for x in v) if mg("model_mmq_gflop") > 0 else None
    gbs = st.median(float(x["model_mmvq_weight_mb"]) / float(x["observed_mmvq_kernel_ms"]) for x in v) if mg("model_mmvq_weight_mb") > 0 else None
    rows.append([k, len(v), f(mg("observed_gpu_busy_ms"), 1), f(mg("observed_mmq_kernel_ms"), 2), f(mg("observed_mmvq_kernel_ms"), 2), f(mg("observed_quantize_kernel_ms"), 2), f(mg("observed_flash_kernel_ms"), 2),
                 f(mg("observed_gdn_kernel_ms"), 2), f(other_t, 2), f(mg("model_mmq_gflop"), 0), f(tops, 1) if tops else "-", f(100 * tops / 144.8, 0) if tops else "-",
                 f(mg("model_mmvq_weight_mb"), 0), f(gbs, 0) if gbs else "-", f(100 * gbs / 448, 0) if gbs else "-"])
w(table(["pass class (all traces)", "passes", "GPU busy (ms)", "MMQ + stream-k fix-up kernels (ms)", "MMVQ kernels (ms)", "quantize kernels (ms)", "flash attention incl. mask + fix-up (ms)", "Gated DeltaNet kernels (ms)",
         "all other kernels (ms)", "model int8 work of the MMQ launches (GFLOP)", "MMQ rate (int8 TOPS)", "MMQ rate (% of 144.8 TOPS)", "model weight bytes of the MMVQ launches (MB)", "MMVQ weight rate (GB/s)", "MMVQ weight rate (% of 448 GB/s)"], rows,
        "`crosscheck-passes.tsv` (medians over the passes of the class; kernel times from `trace-passes.tsv` category sums; rate = model work or bytes / observed kernel time of the category, so the MMQ rate counts useful FLOPs, not padded tile work, "
        "and includes the stream-k fix-up kernels; in the N > 8 suffix class the 3 MMVQ launches are the fused last-layer FFN and the 675 MB output head). Under nsys; an earlier estimate of the MMQ rate was 52 TOPS (36 %): the same quantity."))

# ------------------------------------------------------------------------------------------------ 9 discrepancies
kv_same = xs["flash-attention K/V window (padded n_kv) per pass"]
n_kv_total = int(kv_same["total"])
n_kv_equal = int(kv_same["passed"])
kvmax = max(int(r["n_kv_padded_obs"]) / int(r["n_kv_window_model"]) for r in xpass if r["n_kv_window_model"])
w("## 9 Discrepancies and corrections\n")
w("Each item: what the first version of the model (or an earlier estimate) assumed, what the source or trace shows, and what the census now does.\n")
w("1. **Passes are not formed by distinct suffix lengths alone.** First model: all active sequences with the same remaining length share a pass (3 x 43, then 2 x 4, then 1 x 26). Traces: that holds for 2 of 8 three-question requests; "
  "the others ran 43 + 2 x 47 + 26 or 2 x 43 + 4 + 73, and all 12 two-question requests ran 120 and 51 as separate passes. Cause: slot order in `update_slots()` (a child below the parent follows in the next batch; section 3.2) and `split_equal(sequential = false)` of the hybrid memory (the first version of the census assumed `sequential = true`; see 3.2 rule 4). "
  "Census: `simulate_request()` over all slot placements; the pass count per request is the same in every placement, the N of the passes differs, and the all-MMVQ pass (N <= 8) exists only in the structure 3 x 43 + 2 x 4 + 26.")
w("2. **The last layer's FFN at one output row is two fused MMVQ launches, not three.** First model: up, gate, down. Trace: `mul_mat_vec_q<Q4_0,1,fusion>` grid 9,216 (gate+up+SwiGLU) and grid 2,560 (down + residual add); "
  "a pass with an output therefore has 245 MMQ + 3 MMVQ (fused up/gate, fused down, head) = 248 matmul launches, and 248 MMVQ in an N <= 8 pass. Fixed in `pass_matmuls()`.")
w("3. **A second Gated DeltaNet kernel.** First model: only `gated_delta_net_cuda`. Trace: `gdn_single<16,128,1>` (tensor-core chunk kernel, grid 32 x S x 1) in every pass with >= 64 tokens per sequence (all prefix passes, 73- and 120-token suffix passes); "
  "the 4/26/43/51-token passes (also 2 x 43 tokens) use the recurrent kernel. Fixed in `gdn_cost()`.")
w("4. **Graph order is not creation order.** The census-sequence order (qwen35.cpp node order) does not match the launch order in the trace (for example ssm_beta/ssm_alpha follow attn_qkv, attn_gate follows the delta-net). "
  "The cross-check compares multisets per pass; `census-sequence.tsv` is a list of the matmuls of a pass, not a schedule.")
w(f"5. **The flash-attention K/V window is wider than the request's own cells.** First model: n_kv = the highest position of the request, padded to 256. Trace (derived from the mask copy): equal in {n_kv_equal} of {n_kv_total} passes, "
  f"larger in the rest by up to {kvmax:.1f}x (for example 5,376 cells for a 2,048-token prefix pass); never smaller. [INFERENCE] cause: cells of idle slots stay in the unified cache. Not corrected in the model (needs the slot history). "
  "Effect: the attention FLOPs and the attention mask (N x n_kv x 2 B per pass, host to device) are larger than modelled for those passes; whether the flash-attention kernel time grows with the window was not measured.")
w("6. **Recurrent-state copies.** The model assumes the gather of S states plus a write-back per layer (get_rows grid.x = S in all passes, confirmed). The three 86-token suffix passes (2 sequences x 43 tokens, structure 2 + 1 + 1) "
  "additionally contain 48 device-to-device copies of 2.2 MB (about 1 ms per pass, column `d2d_gpu_in_pass_ms` of `trace-passes.tsv`); [INFERENCE] the third sequence's recurrent cell lies between the two used cells and is copied as well. Not in the model.")
w("7. **Kernel counts per pass.** An earlier estimate was about 1,280 launches per pass; the traces show 1,065-1,296 (1,166 for 2,048-token prefix passes; 1,248 or 1,296 for suffix passes with the MMQ path; 1,065 for N <= 8). 245 MMQ launches per pass is exact. "
  "The census counts only matmul, quantize, flash-attention and GDN launches (the rest is in `trace-kernels.tsv`).")
w("8. **Main MMQ grid.x is not always 38.** It is the tile count when the tile efficiency is >= 90 % (for example 1,024 and 72) and 38 or 10 with a fix-up kernel otherwise; the model predicts every observed value (Table 5, `crosscheck-keys.tsv` is empty).")
w("9. **Token counts**: the sum of the question prompts per request (what the API counts) matches the production run exactly for the raw variant (Table 3). The traced short requests use a random `Ref:` value, so their prefix length varies by a few tokens; "
  "the cross-check takes the observed prefix length and the plan's suffix lengths.")
w("10. **Not verifiable here**: the nsys overhead (3 % long, 25 % short, earlier estimate), the frequency of the slot placements in production, and the cause of the wider K/V window (item 5). No untraced measurement was made.\n")

# ------------------------------------------------------------------------------------------------ 10 limits / gpu / files
w("## 10 Limits, GPU needs, files\n")
w("**What needs the GPU**: nothing. All steps (`run_all.sh`) run on the CPU: steps 1 and 2 need the model file and the tokenizer, the trace step needs the three captures, steps 3 to 5 need only the data set. "
  "For phase 3 the model gives measurement targets that do need the GPU (not run here): the time per kernel class at the N values of Table 7, the pass time for N <= 8 against the 7.44 ms weight floor, and the slot placement distribution of a production-like request stream.\n")
w("**Assumptions of the analytic columns**: FLOPs are 2 x M x N x K per matmul (padded tile work is in `padded_gops`); weight bytes are the file sizes (34 / 18 bytes per 32 elements); floors use the nominal 448 GB/s and 144.8 int8 TOPS (1,861 MHz); "
  "the FP16 and FP32 peaks and the element-wise traffic per layer (`census_lib.side_ops`) are [INFERENCE]; the K/V window is the request's own cells (item 5).\n")
w("**Files** (the code is in `harness/profile/census/`; the inputs and tables are in the data set `data/2026-10-30-shape-census/`):\n")
rows = [
    ["`run_all.sh`", "lists every step with its inputs and runs them (steps 1-5 and the trace step)"],
    ["`01_gguf_header.py`", "-> `tensors.tsv` (426 tensors), `hparams.tsv`, `layers.tsv` (reads the header of the GGUF file)"],
    ["`02_tokens.py`", "-> `token-counts.tsv`, `cases-plan.tsv`, `token-check.tsv`, `ref-line-tokens.tsv` (reads the template, the labels and the vocabulary of the GGUF file)"],
    ["`census_lib.py`", "the analytic model: MMQ / MMVQ launch prediction, `pass_matmuls`, `gdn_cost`, `attn_cost`, `simulate_request`, `pass_costs`"],
    ["`03_census.py`", "-> `census.tsv`, `census-sequence.tsv`, `census-passes.tsv`, `request-floors.tsv`, `census-cases.tsv` (48 cases x raw/ref x slot placements), `census-summary.tsv`, `census-head.tsv`"],
    ["`trace_export.sh`, `trace_census.py`", "-> `trace-kernels.tsv`, `trace-passes.tsv`, `trace-mmq-launches.tsv` (reads the three old captures; the captures are not in the data set)"],
    ["`04_crosscheck.py`", "-> `crosscheck-requests.tsv`, `crosscheck-passes.tsv`, `crosscheck-keys.tsv`, `crosscheck-kernels.tsv`, `crosscheck-summary.tsv`"],
    ["`05_report.py`", "-> this report (every table is generated from the TSVs)"],
]
w(table(["file", "content"], rows, "`harness/profile/census/`."))
open(os.path.join(OUT, "census-report.txt"), "w").write("\n".join(S) + "\n")
print("wrote", os.path.join(OUT, "census-report.txt"), len("\n".join(S)), "bytes")
