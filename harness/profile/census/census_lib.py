"""Analytic model of one llama.cpp b12004 model pass (one llama_decode ubatch) of H2O Lightning 4B mix8 on sm_86.

Everything here is derived from the b12004 sources (paths in the comments) and from the GGUF tensor list
(tensors.tsv, hparams.tsv). Nothing is measured. Used by 03_census.py (tables) and 04_crosscheck.py (vs the traces).
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
# The data set with the inputs and tables of the census. Set DMK_CENSUS_DATA to use another folder.
DEFAULT_DATA = os.path.join(REPO, "data", "2026-10-30-shape-census")


def data_dir():
    """The folder of the census data set: $DMK_CENSUS_DATA, else data/2026-10-30-shape-census of this repository."""
    return os.environ.get("DMK_CENSUS_DATA") or DEFAULT_DATA


def data_file(name):
    """Path of a file of the census data set, found by its name.
    The search is: <data>/tables/<name>, <data>/raw/*/<name>, <data>/<name>. The last case is a flat folder of TSV files."""
    import glob
    root = data_dir()
    for pattern in (os.path.join(root, "tables", name), os.path.join(root, "raw", "*", name), os.path.join(root, name)):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"census file {name} not found in {root} (tables/, raw/*/ or the folder itself); set DMK_CENSUS_DATA to the data set folder")

# ------------------------------------------------------------------ hardware / limits (nominal values of the project)
NSM = 38                       # RTX 3060 Ti (GA104), ggml_cuda_info().devices[].nsm
SMPBO = 101376                 # sharedMemPerBlockOptin of sm_86 (99 KiB), used by ggml_cuda_mmq_get_J_best  [INFERENCE: CUDA limit for cc 8.6]
BW_GBPS = 448.0                # nominal DRAM bandwidth
INT8_TOPS = 144.8              # dense int8 tensor peak at 1,861 MHz (nominal; the card-limits data set has the measured rates)
FP16_ACC32_TFLOPS = INT8_TOPS / 4.0   # [INFERENCE] GeForce GA10x: FP16 tensor with FP32 accumulate = 1/2 of FP16-acc = 1/4 of int8 per clock; used only for the flash attention floor, see the census report
FP32_TFLOPS = 2 * 128 * NSM * 1.861e9 / 1e12   # CUDA cores, FMA = 2 flops, 128 lanes/SM/clk at 1,861 MHz
N_UBATCH = 2048

# ------------------------------------------------------------------ ggml quant types
BLOCK = {"Q4_0": (32, 18), "Q8_0": (32, 34), "F32": (1, 4), "F16": (1, 2)}   # (elements per block, bytes per block)


def weight_bytes(wtype, m, k):
    e, b = BLOCK[wtype]
    return m * k // e * b


# ------------------------------------------------------------------ model
def load_hparams():
    d = {}
    with open(data_file("hparams.tsv")) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            d[row["parameter"]] = row["value"]
    return d


def load_tensors():
    with open(data_file("tensors.tsv")) as f:
        return list(csv.DictReader(f, delimiter="\t"))


class Model:
    def __init__(self):
        hp = load_hparams()
        self.n_layer = int(hp["n_layer (executed)"])
        self.pattern = hp["layer_pattern"]
        self.n_embd = int(hp["n_embd"])
        self.n_ff = int(hp["n_ff"])
        self.n_head = int(hp["n_head"])
        self.n_head_kv = int(hp["n_head_kv"])
        self.head_dim = int(hp["head_dim_k"])
        self.n_vocab = int(hp["n_vocab"])
        self.d_inner = int(hp["ssm_inner_size (d_inner)"])
        self.gdn_heads_v = int(hp["ssm_time_step_rank = GDN num_v_heads"])
        self.gdn_heads_k = int(hp["ssm_group_count = GDN num_k_heads"])
        self.gdn_dim = int(hp["ssm_state_size = GDN head_k_dim"])
        self.conv_channels = int(hp["GDN conv_channels"])
        self.conv_k = int(hp["ssm_conv_kernel"])
        self.state_bytes = int(hp["GDN recurrent state per layer per sequence"])
        self.conv_state_bytes = int(hp["GDN conv state per layer per sequence"])
        self.tensors = {t["name"]: t for t in load_tensors()}
        self.tok_embd_bytes = int(self.tensors["token_embd.weight"]["bytes"])

    def wtype(self, name):
        return self.tensors[name]["ggml_type"]

    def tensor_bytes(self, name):
        return int(self.tensors[name]["bytes"])


# ------------------------------------------------------------------ dispatch (ggml/src/ggml-cuda/ggml-cuda.cu ggml_cuda_mul_mat)
MMVQ_MAX_BATCH = 8   # mmvq.cuh MMVQ_MAX_BATCH_SIZE; ggml_cuda_should_use_mmvq(type, cc=860, ne11) = ne11 <= 8 for Q4_0 / Q8_0 on sm_86


def path_for(wtype, n):
    """Quantized weight, F32 activations, sm_86: mul_mat_vec_f/mul_mat_f do not take quantized types ->
    MMVQ for ne11 <= 8, else MMQ (turing_mma_available); cuBLAS is never reached for Q4_0/Q8_0."""
    assert wtype in ("Q4_0", "Q8_0")
    return "MMVQ" if n <= MMVQ_MAX_BATCH else "MMQ"


# ------------------------------------------------------------------ MMQ launch (mmq.cu / mmq.cuh / mmq-config-ampere.cuh)
MMQ_I = 128
MMQ_NTHREADS = 256
MMQ_K_VRAM = 256
J_NOFALLBACK = (8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128)   # CASE() lines of mmq-config-ampere.cuh, fallback=false (Q4_0, Q8_0)
J_FALLBACK = (8, 16, 32, 64, 128)                                # fallback=true (nrows_x % 128 != 0)


def mmq_shared_bytes(J):
    stride = 2 * 32 + 2 * 32 // 8 + 4          # ggml_cuda_mmq_get_sram_stride(GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0) = 76 (32-bit words)
    nbs_ids = J * 4
    nbs_x = MMQ_I * stride * 4
    nbs_y = J * 144                              # sizeof(block_q8_1_mmq) = 144
    pad = MMQ_NTHREADS * 4
    return nbs_ids + nbs_x + (nbs_y + pad - 1) // pad * pad


def mmq_j_best(ncols, fallback):
    """ggml_cuda_mmq_get_J_best: the smallest tile width J (step 8, configured for the type) with the fewest column tiles
    that fits into shared memory."""
    best, ntiles_best = 0, 1 << 30
    for J in range(8, 129, 8):
        if ntiles_best <= 1:
            break
        if J not in (J_FALLBACK if fallback else J_NOFALLBACK):
            continue
        if mmq_shared_bytes(J) > SMPBO:
            continue
        nt = (ncols + J - 1) // J
        if nt < ntiles_best:
            best, ntiles_best = J, nt
    return best


def mmq_launch(wtype, m, n, k):
    """Host-side launch parameters of ggml_cuda_mul_mat_q for dst = [m, n], src0 = [k, m] (stream-k, Ampere config)."""
    fallback = m % 128 != 0
    J = mmq_j_best(n, fallback)
    nty = (m + MMQ_I - 1) // MMQ_I
    ntx = (n + J - 1) // J
    ntiles = ntx * nty
    waves = (ntiles + NSM - 1) // NSM
    eff = 100 * ntiles // (NSM * waves)
    nwork = ntiles * ((k + MMQ_K_VRAM - 1) // MMQ_K_VRAM)
    grid_x = ntiles if eff >= 90 else min(NSM, nwork)
    fixup = ntiles % grid_x != 0
    return {
        "J": J, "fallback": int(fallback), "tiles_m": nty, "tiles_n": ntx, "tiles": ntiles, "tile_eff_pct": eff,
        "grid_x": grid_x, "fixup": int(fixup), "block": "32x8x1",
        "kernel": f"mul_mat_q<{wtype},J={J},fallback={int(fallback)}>",
        "padded_ops": 2 * (nty * MMQ_I) * (ntx * J) * k,
    }


def mmvq_launch(wtype, m, n, k):
    """calc_nwarps / calc_rows_per_block / should_use_small_k of mmvq.cu, MMVQ_PARAMETERS_GENERIC (sm_86 is neither
    Volta/Turing-only nor GB10) for Q4_0 / Q8_0 (vdr = 2; qi = 4 for Q4_0, 8 for Q8_0)."""
    nwarps = 4 if n <= 4 else 2
    rpb = 1 if n == 1 else 2
    qi = 4 if wtype == "Q4_0" else 8
    blocks_per_iter_1warp = 2 * 32 // qi
    small_k = int(n == 1 and nwarps > 1 and (k // 32) < nwarps * blocks_per_iter_1warp)
    if small_k:
        rpb = nwarps
    return {"grid_x": (m + rpb - 1) // rpb, "block": f"32x{nwarps}x1", "kernel": f"mul_mat_vec_q<{wtype},ncols={n}>",
            "small_k": small_k, "rows_per_block": rpb}


def matmul(name, layer, wtype, m, n, k):
    """One mul_mat node: weight [k, m] (type wtype) x activations [k, n] (F32) -> [m, n]."""
    path = path_for(wtype, n)
    d = {
        "layer": layer, "op": name, "wtype": wtype, "M": m, "N": n, "K": k, "path": path,
        "flops": 2 * m * n * k, "weight_bytes": weight_bytes(wtype, m, k),
        "act_in_f32_bytes": n * k * 4,
        "act_q8_bytes": n * k * 9 // 8,
        "act_out_f32_bytes": n * m * 4,
    }
    d.update(mmq_launch(wtype, m, n, k) if path == "MMQ" else mmvq_launch(wtype, m, n, k))
    if path == "MMVQ":
        d.update({"J": "", "fallback": "", "tiles_m": "", "tiles_n": "", "tiles": "", "tile_eff_pct": "", "fixup": 0, "padded_ops": 2 * m * n * k})
    return d


# ------------------------------------------------------------------ one pass
def pass_matmuls(model, n_tokens, n_out):
    """Ordered list of mul_mat nodes of one ubatch (graph build order of src/models/qwen35.cpp).
    n_tokens = total tokens of the ubatch (= n_seqs * n_seq_tokens), n_out = rows with logits.
    Layer 31 (last): the FFN and everything after the attention residual run on the n_out output rows only
    (crop_before_nextn: embeddings_nextn is off in the server), and nothing runs when n_out == 0."""
    ops = []
    ne, nff, hd = model.n_embd, model.n_ff, model.head_dim
    for il in range(model.n_layer):
        recurrent = model.pattern[il] == "R"
        pre = f"blk.{il}."
        if recurrent:
            ops.append(matmul("attn_qkv", il, model.wtype(pre + "attn_qkv.weight"), 2 * model.gdn_heads_k * model.gdn_dim + model.d_inner, n_tokens, ne))
            ops.append(matmul("attn_gate(z)", il, model.wtype(pre + "attn_gate.weight"), model.d_inner, n_tokens, ne))
            ops.append(matmul("ssm_beta", il, model.wtype(pre + "ssm_beta.weight"), model.gdn_heads_v, n_tokens, ne))
            ops.append(matmul("ssm_alpha", il, model.wtype(pre + "ssm_alpha.weight"), model.gdn_heads_v, n_tokens, ne))
            ops.append(matmul("ssm_out", il, model.wtype(pre + "ssm_out.weight"), ne, n_tokens, model.d_inner))
        else:
            ops.append(matmul("attn_q(+gate)", il, model.wtype(pre + "attn_q.weight"), 2 * model.n_head * hd, n_tokens, ne))
            ops.append(matmul("attn_k", il, model.wtype(pre + "attn_k.weight"), model.n_head_kv * hd, n_tokens, ne))
            ops.append(matmul("attn_v", il, model.wtype(pre + "attn_v.weight"), model.n_head_kv * hd, n_tokens, ne))
            ops.append(matmul("attn_output", il, model.wtype(pre + "attn_output.weight"), ne, n_tokens, model.n_head * hd))
        n_ffn = n_out if il == model.n_layer - 1 else n_tokens
        if n_ffn == 1:
            # ggml-cuda.cu fuses MUL_MAT + MUL_MAT + GLU (gate, up) and MUL_MAT + ADD (down + residual) into
            # mul_mat_vec_q<type, 1, fusion> when ncols_dst == 1 (ggml_cuda_should_fuse_mul_mat_vec_q): two launches, not three
            up = matmul("ffn_up+gate+swiglu (fused)", il, model.wtype(pre + "ffn_up.weight"), nff, 1, ne)
            for key in ("flops", "weight_bytes", "act_out_f32_bytes"):
                up[key] *= 2
            up["fused"] = 1
            ops.append(up)
            down = matmul("ffn_down+add (fused)", il, model.wtype(pre + "ffn_down.weight"), ne, 1, nff)
            down["fused"] = 1
            ops.append(down)
        elif n_ffn > 0:
            ops.append(matmul("ffn_up", il, model.wtype(pre + "ffn_up.weight"), nff, n_ffn, ne))
            ops.append(matmul("ffn_gate", il, model.wtype(pre + "ffn_gate.weight"), nff, n_ffn, ne))
            ops.append(matmul("ffn_down", il, model.wtype(pre + "ffn_down.weight"), ne, n_ffn, nff))
    if n_out > 0:
        ops.append(matmul("output_head(token_embd)", -1, model.wtype("token_embd.weight"), model.n_vocab, n_out, ne))
    return ops


def gdn_cost(model, n_tokens, n_seqs):
    """Fused gated_delta_net op, one per recurrent layer (cparams.fused_gdn_ch/ar = true, llama-context.cpp), and the
    state copies of the hybrid memory around it (llm_graph_context::build_rs): ggml_get_rows gathers the n_seqs states of the
    ubatch (read + write; the traces show grid.x = n_seqs, no extra states), the op reads + writes the state of its n_seqs
    sequences, and ggml_cuda_cpy writes the new states back to the cache (read + write).
    Kernel (ggml-cuda/gated-delta-net-mma.cu select_gdn_mma_path, cc 8.6): tokens per sequence >= 64 -> tensor-core chunk kernel
    gdn_single<16,128,1> (grid H*n_seqs blocks of 32x8 threads); fewer -> gated_delta_net_cuda<128,...> (grid H x n_seqs x 8)."""
    hv, d = model.gdn_heads_v, model.gdn_dim
    st = model.state_bytes
    flops = 3 * 2 * d * d * hv * n_tokens                 # S^T k, S += k (x) delta, S^T q  per head and token (MACs x 2)
    state_rw = st * n_seqs * 2                            # read + write of the recurrent state by the op itself
    io = n_tokens * ((2 * model.gdn_heads_k * d) + model.d_inner) * 4 + n_tokens * model.d_inner * 4 + 2 * n_tokens * hv * 4
    copies = st * n_seqs * 4                              # get_rows (r+w) + cpy to the cache (r+w)
    mma = n_tokens // n_seqs >= 64
    if mma:
        kernel, grid = "gdn_single<16,128,1>", (hv * n_seqs, 1, 1)
    else:
        kernel, grid = "gated_delta_net_cuda<128,false,false,4>", (hv, n_seqs, 8)
    return {"flops": flops, "bytes": state_rw + io, "state_copy_bytes": copies, "blocks": grid[0] * grid[1] * grid[2],
            "kernel": kernel, "grid": grid, "mma": mma}


def attn_cost(model, seqs):
    """Flash attention of one full-attention layer. seqs = [(start_pos, n_tok)] of the ubatch.
    Useful (causal) FLOPs = 4 * head_dim * n_head * sum(visible keys). K/V window n_kv = max(start+n) padded to 256
    [assumption: only this request's cells are in the unified cache; stale cells of idle slots would widen it]."""
    keys = 0
    n_kv = 0
    n_tok = 0
    for start, n in seqs:
        keys += n * start + n * (n + 1) // 2
        n_kv = max(n_kv, start + n)
        n_tok += n
    n_kv_pad = (n_kv + 255) // 256 * 256
    hd, nh, nkv = model.head_dim, model.n_head, model.n_head_kv
    ncols2 = 4 if nh // nkv > 2 else (2 if nh // nkv > 1 else 1)      # fattn.cu switch_ncols2 (gqa_ratio 4 -> 4)
    ncols1 = 2 if n_tok <= 8 // ncols2 else 4 if n_tok <= 16 // ncols2 else 8 if n_tok <= 32 // ncols2 else 16
    return {
        "flops": 4 * hd * nh * keys,
        "kv_bytes": n_kv_pad * 2 * nkv * hd * 2,          # K + V read once, F16
        "kv_write_bytes": n_tok * 2 * nkv * hd * 2,
        "n_kv": n_kv_pad, "n_tok": n_tok, "ncols1": ncols1, "ncols2": ncols2,
        "kernel": f"flash_attn_ext_f16<{hd},{hd},{ncols1},{ncols2}> (mma)",
        "q_tiles": (n_tok + ncols1 - 1) // ncols1,
    }


# activation traffic estimate of the other graph ops, bytes per token and layer (F32 = 4 B), read + write.
# Derived from src/models/qwen35.cpp; element-wise ops are assumed to read their inputs and write their output once.
def side_ops(model):
    ne, nff, di = model.n_embd, model.n_ff, model.d_inner
    qkv = 2 * model.gdn_heads_k * model.gdn_dim + di
    cc = model.conv_channels
    f = 4
    rows = []

    def op(layer_kind, name, rd_elems, wr_elems, kernel):
        rows.append((layer_kind, name, (rd_elems + wr_elems) * f, kernel))

    # recurrent layer
    op("R", "attn_norm (rms_norm*weight)", ne, ne, "rms_norm_f32 (+mul)")
    op("R", "quantize act x4 (qkv, z, beta, alpha)", 4 * ne, 4 * ne * 9 // 32, "quantize_mmq_q8_1")
    op("R", "matmul out qkv+z+beta+alpha", 0, qkv + di + 64, "(written by mul_mat_q)")
    op("R", "conv concat (conv_state | qkv^T)", cc, cc, "concat_non_cont")
    op("R", "ssm_conv", cc, cc, "ssm_conv")
    op("R", "silu(conv out)", cc, cc, "unary silu")
    op("R", "l2_norm q, k", 2 * model.gdn_heads_k * model.gdn_dim, 2 * model.gdn_heads_k * model.gdn_dim, "l2_norm")
    op("R", "gated_delta_net (token streams only; state traffic is per sequence)", 2 * model.gdn_heads_k * model.gdn_dim + di + 2 * model.gdn_heads_v, di, "gated_delta_net")
    op("R", "gated norm: rms_norm(out)*w, silu(z), mul", 3 * di, 3 * di, "rms_norm + silu + mul")
    op("R", "quantize act (ssm_out)", di, di * 9 // 32, "quantize_mmq_q8_1")
    op("R", "ssm_out out + residual add", 3 * ne, ne, "mul_mat_q out; add")
    # attention layer
    op("A", "attn_norm (rms_norm*weight)", ne, ne, "rms_norm_f32 (+mul)")
    op("A", "quantize act x3 (q, k, v)", 3 * ne, 3 * ne * 9 // 32, "quantize_mmq_q8_1")
    op("A", "matmul out q(+gate), k, v", 0, 2 * model.n_head * model.head_dim + 2 * model.n_head_kv * model.head_dim, "(written by mul_mat_q)")
    op("A", "q/k rms_norm + mrope (q: 2 x, k: 2 x)", 2 * (model.n_head + model.n_head_kv) * model.head_dim, 2 * (model.n_head + model.n_head_kv) * model.head_dim, "rms_norm; rope_multi")
    op("A", "k,v -> KV cache (F16 cpy)", 2 * model.n_head_kv * model.head_dim, 2 * model.n_head_kv * model.head_dim // 2, "cpy_f32_f16")
    op("A", "gate cont + sigmoid + mul", 3 * model.n_head * model.head_dim, 2 * model.n_head * model.head_dim, "cont; sigmoid; mul")
    op("A", "quantize act (attn_output) + residual add", model.n_head * model.head_dim + 2 * ne, model.n_head * model.head_dim * 9 // 32 + ne, "quantize_mmq_q8_1; add")
    # ffn (all layers)
    op("RA", "post_attention_norm (rms_norm*weight)", ne, ne, "rms_norm_f32 (+mul)")
    op("RA", "quantize act x2 (up, gate)", 2 * ne, 2 * ne * 9 // 32, "quantize_mmq_q8_1")
    op("RA", "matmul out up, gate", 0, 2 * nff, "(written by mul_mat_q)")
    op("RA", "swiglu(gate, up)", 2 * nff, nff, "swiglu_f32")
    op("RA", "quantize act (down)", nff, nff * 9 // 32, "quantize_mmq_q8_1")
    op("RA", "down out + residual add", 2 * ne, ne, "mul_mat_q out; add")
    return rows


def side_class(name):
    return "quant" if name.startswith("quantize") else "mmout" if name.startswith("matmul out") else "other"


def side_bytes_per_token(model, cls="other"):
    """(recurrent layer, attention layer, ffn) bytes per token of the side ops of one class ('other' = everything that is
    not a quantize kernel and not the write of a matmul output; those two are counted per matmul)."""
    r = a = ffn = 0
    for kind, name, b, _ in side_ops(model):
        if side_class(name) != cls:
            continue
        if kind == "R":
            r += b
        elif kind == "A":
            a += b
        else:
            ffn += b
    return r, a, ffn


def side_bytes(model, n_tokens, n_out):
    """Activation traffic (bytes) of the non-matmul, non-quantize ops of a pass, see side_ops()."""
    r, a, ffn = side_bytes_per_token(model)
    n_r = model.pattern.count("R")
    n_a = model.pattern.count("A")
    return n_tokens * (n_r * r + n_a * a + (model.n_layer - 1) * ffn) + n_out * ffn


def split_batch(n):
    """llama_decode splits a batch of one sequence into ubatches of at most n_ubatch tokens (split_equal)."""
    out = []
    while n > 0:
        t = min(n, N_UBATCH)
        out.append(t)
        n -= t
    return out


# llama-memory-hybrid.cpp (b12004, lines 83-90): the hybrid memory splits every batch with split_equal(n_ubatch, !unified) where
# unified = (the attention cache has one stream) = --kv-unified. Production runs --kv-unified, so sequential = false.
# (The pure recurrent memory, llama-memory-recurrent.cpp, uses sequential = true; the first version of this census took that one.
# The 20 traced requests are reproduced by both settings, they differ only for a two-question request in slots (0, 2).)
SPLIT_SEQUENTIAL = False


def split_equal(entries, sequential=None, n_ubatch=N_UBATCH):
    """llama_batch_allocr::split_equal(n_ubatch, sequential) (src/llama-batch.cpp).
    entries = [{"slot", "pos", "n", "q"}] in batch order (the server fills the batch by slot index, so increasing seq id).
    Each ubatch takes the first sequence with unused tokens and then every other sequence with unused tokens (sequential = false)
    or only sequences whose id is the previous id + 1 (sequential = true), with t = min(remaining) tokens each
    (capped by n_ubatch // n_seqs); a sequence finishing in the ubatch has an output. None = SPLIT_SEQUENTIAL."""
    if sequential is None:
        sequential = SPLIT_SEQUENTIAL
    rem = {e["slot"]: e["n"] for e in entries}
    pos = {e["slot"]: e["pos"] for e in entries}
    qid = {e["slot"]: e["q"] for e in entries}
    order = [e["slot"] for e in entries]
    ubatches = []
    while any(rem[s] > 0 for s in order):
        cur, last = [], -1
        for s in order:
            if rem[s] > 0 and (not cur or not sequential or s == last + 1):
                cur.append(s)
                last = s
        t = min(min(rem[s] for s in cur), n_ubatch // len(cur))
        ub = []
        for s in cur:
            ub.append({"slot": s, "q": qid[s], "start": pos[s], "n": t, "finishes": rem[s] == t})
            rem[s] -= t
            pos[s] += t
        ubatches.append(ub)
    return ubatches


def _pass(kind, ub, iteration):
    return {"kind": kind, "seqs": [(u["start"], u["n"]) for u in ub], "slots": [u["slot"] for u in ub], "questions": [u["q"] for u in ub],
            "n_tokens": sum(u["n"] for u in ub), "n_out": sum(1 for u in ub if u["finishes"]), "iteration": iteration}


def simulate_request(shared, suffix_lens, slots, sequential=None):
    """Passes (ubatches, in decode order) of one request as the b12004 server issues them.
    shared      : tokens of the shared prefix (0 for a one-question request: no sharing, the whole prompt is the suffix)
    suffix_lens : suffix length of every question (the full prompt length if shared == 0); question 0 = the parent task
    slots       : slot (= seq id) of every question. Placement rules (tools/server/server-context.cpp):
      * the parent evaluates the shared prefix alone, in ubatches of at most 2048 tokens, without outputs;
      * when the visit of the parent slot in update_slots() finds the prefix done, it starts all children and the children
        with a slot index above the parent are visited later in the same loop, so their suffix is in the same llama_decode
        batch as the parent's suffix; children with a lower slot index were already skipped and follow in the next batch;
      * every batch is split by split_equal(sequential); sequential=None takes SPLIT_SEQUENTIAL (False, see above)."""
    passes = []
    if shared == 0:
        assert len(suffix_lens) == 1
        pos = 0
        chunks = split_batch(suffix_lens[0])
        for i, t in enumerate(chunks):
            passes.append({"kind": "single" if len(chunks) == 1 else "single-chunk", "seqs": [(pos, t)], "slots": [slots[0]], "questions": [0],
                           "n_tokens": t, "n_out": 1 if i == len(chunks) - 1 else 0, "iteration": i})
            pos += t
        return passes
    it = 0
    pos = 0
    for t in split_batch(shared):
        passes.append({"kind": "prefix", "seqs": [(pos, t)], "slots": [slots[0]], "questions": [0], "n_tokens": t, "n_out": 0, "iteration": it})
        pos += t
        it += 1
    p = slots[0]
    first = [i for i in range(len(suffix_lens)) if i == 0 or slots[i] > p]
    later = [i for i in range(1, len(suffix_lens)) if slots[i] < p]
    for group in (first, later):
        if not group:
            continue
        entries = sorted(({"slot": slots[i], "pos": shared, "n": suffix_lens[i], "q": i} for i in group), key=lambda e: e["slot"])
        for ub in split_equal(entries, sequential):
            passes.append(_pass("suffix", ub, it))
        it += 1
    return passes


def arrangements(n_questions, n_slots=3):
    """All assignments of the questions to distinct slots (the parent is question 0)."""
    import itertools
    return list(itertools.permutations(range(n_slots), n_questions))


# ------------------------------------------------------------------ everything about one pass
_cache = {}
_BW = BW_GBPS * 1e9
_INT8 = INT8_TOPS * 1e12
_FP16 = FP16_ACC32_TFLOPS * 1e12
_FP32 = FP32_TFLOPS * 1e12


def pass_costs(model, p):
    """Launches, FLOPs, bytes and roofline floors of one pass dict from simulate_request() (cached by shape).
    floor_weights_s  = weight bytes / 448 GB/s;   floor_int8_s = useful matmul ops / 144.8 TOPS;
    floor_pass_max_s = max of the two;            floor_roofline_s = sum over kernels of max(bytes / BW, ops / peak) with all
    activation traffic from DRAM (matmul: weights + quantised input + output; quantize kernels; GDN; flash attention; element-wise)."""
    key = (p["n_tokens"], p["n_out"], tuple(p["seqs"]))
    if key in _cache:
        return _cache[key]
    n, n_out = p["n_tokens"], p["n_out"]
    n_seqs = len(p["seqs"])
    mms = pass_matmuls(model, n, n_out)
    n_r, n_a = model.pattern.count("R"), model.pattern.count("A")
    gdn = gdn_cost(model, n, n_seqs)
    att = attn_cost(model, p["seqs"])

    flops = sum(o["flops"] for o in mms)
    padded = sum(o["padded_ops"] for o in mms)
    wbytes = sum(o["weight_bytes"] for o in mms)
    head_bytes = sum(o["weight_bytes"] for o in mms if o["layer"] == -1)
    quant_bytes = sum(o["act_in_f32_bytes"] + o["act_q8_bytes"] for o in mms)
    mmout_bytes = sum(o["act_out_f32_bytes"] for o in mms)
    side = side_bytes(model, n, n_out)
    state_bytes = n_r * (gdn["bytes"] + gdn["state_copy_bytes"])
    kv_bytes = n_a * (att["kv_bytes"] + att["kv_write_bytes"] + n * model.n_head * model.head_dim * 4 * 2)   # + Q read, O write
    fa_flops = n_a * att["flops"]
    gdn_flops = n_r * gdn["flops"]

    t = 0.0
    for o in mms:
        t += max(o["flops"] / _INT8, (o["weight_bytes"] + o["act_q8_bytes"] + o["act_out_f32_bytes"]) / _BW)
        t += (o["act_in_f32_bytes"] + o["act_q8_bytes"]) / _BW
    t += n_r * max(gdn["flops"] / _FP32, (gdn["bytes"] + gdn["state_copy_bytes"]) / _BW)
    t += n_a * max(att["flops"] / _FP16, (att["kv_bytes"] + att["kv_write_bytes"]) / _BW)
    t += side / _BW
    n_mmq = sum(1 for o in mms if o["path"] == "MMQ")
    res = {
        "mms": mms, "gdn": gdn, "att": att, "n_seqs": n_seqs,
        "n_matmul": len(mms), "n_mmq": n_mmq, "n_mmvq": len(mms) - n_mmq,
        "n_mmq_fixup": sum(1 for o in mms if o["path"] == "MMQ" and o["fixup"]),
        "flops": flops, "padded": padded, "wbytes": wbytes, "head_bytes": head_bytes,
        "quant_bytes": quant_bytes, "mmout_bytes": mmout_bytes, "side_bytes": side,
        "state_bytes": state_bytes, "kv_bytes": kv_bytes, "fa_flops": fa_flops, "gdn_flops": gdn_flops,
        "floor_weights_s": wbytes / _BW, "floor_int8_s": flops / _INT8,
        "floor_int8_padded_s": padded / _INT8, "floor_roofline_s": t,
        "floor_pass_max_s": max(wbytes / _BW, flops / _INT8),
    }
    _cache[key] = res
    return res
