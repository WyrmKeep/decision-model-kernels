#!/usr/bin/env python3
"""Step 1: tensor list + hyperparameters of the mix8 GGUF, from its header only.

Run with:  uv run --with numpy --with pyyaml --with tqdm --with requests python 01_gguf_header.py \
               --gguf MODEL.gguf --gguf-py <llama.cpp>/gguf-py [--out DIR]

Reads (read-only):
  --gguf FILE   a local GGUF file. Only its first N bytes are read (N = 16, 32, 64 or 128 MB: the smallest that parses).
                The model header (metadata incl. the 248,320-token vocabulary, then the 426 tensor infos) is ~11 MB; the
                tensor data (3.3 GB) is never read. A sparse temporary file of the full size is made from the head so that
                gguf-py (GGUFReader, memmap) can open it; it is removed at the end. The temporary folder is $TMPDIR.
                A file that holds only a header and then a sparse hole of the full size works as well.
  --gguf-py DIR the gguf-py directory of llama.cpp (default: $GGUF_PY; if that is not set, an installed `gguf` package is used).
Writes into --out (default: the current folder):
  tensors.tsv     one row per tensor: name, layer, role, ggml shape (ne0 = K = input dim, ne1 = M = output rows), type, bytes
  hparams.tsv     model hyperparameters used by the census (GGUF key or derived, with the source)
  layers.tsv      layer pattern (recurrent / attention) and per-layer weight bytes
"""
import argparse
import os
import shutil
import sys
import tempfile

HEAD_SIZES = (16_000_000, 32_000_000, 64_000_000, 128_000_000)


def import_gguf(gguf_py):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    try:
        from gguf import GGUFReader
    except ImportError:
        raise SystemExit("cannot import gguf: set --gguf-py (or GGUF_PY) to the gguf-py directory of llama.cpp b12004")
    return GGUFReader


def build_sparse(src_path, sparse_path, nbytes, total):
    """Copy the first nbytes of the model into a sparse file of `total` bytes."""
    with open(sparse_path, "wb") as f:
        f.truncate(total)
    with open(src_path, "rb") as src, open(sparse_path, "r+b") as dst:
        dst.write(src.read(nbytes))


def open_reader(GGUFReader, gguf_path, tmpdir):
    total = os.path.getsize(gguf_path)
    sparse = os.path.join(tmpdir, "head-sparse.gguf")
    for nbytes in HEAD_SIZES:
        build_sparse(gguf_path, sparse, nbytes, total)
        try:
            r = GGUFReader(sparse)
            n = sum(int(t.n_bytes) for t in r.tensors)  # touches every tensor info
            if r.data_offset <= nbytes and r.data_offset + n <= total + 4096 and len(r.tensors) == int(r.fields["GGUF.tensor_count"].contents()):
                print(f"header parsed from the first {nbytes} bytes; data_offset={r.data_offset}; file size={total}")
                return r, total, nbytes
        except Exception as e:  # truncated header: try more bytes
            print(f"{nbytes} bytes not enough: {type(e).__name__}: {e}")
    raise SystemExit("header did not parse with 128 MB")


def f(r, key):
    return r.fields[key].contents()


def role_of(name):
    if name.startswith("blk."):
        return name.split(".", 2)[2]
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf", required=True, help="local GGUF file (only its first 16-128 MB are read)")
    ap.add_argument("--gguf-py", default=os.environ.get("GGUF_PY"), help="the gguf-py directory of llama.cpp (default: $GGUF_PY, else an installed gguf package)")
    ap.add_argument("--out", default=".", help="output folder for tensors.tsv, hparams.tsv, layers.tsv (default: the current folder)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    GGUFReader = import_gguf(args.gguf_py)

    tmpdir = tempfile.mkdtemp(prefix="census-gguf-")
    try:
        r, total, nbytes_fetched = open_reader(GGUFReader, args.gguf, tmpdir)
        write_tables(r, total, nbytes_fetched, args.out)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def write_tables(r, total, nbytes_fetched, out_dir):
    tensors = []
    for t in r.tensors:
        ne = [int(x) for x in t.shape]
        tensors.append({
            "name": t.name,
            "layer": int(t.name.split(".")[1]) if t.name.startswith("blk.") else -1,
            "role": role_of(t.name),
            "ne": ne,
            "type": t.tensor_type.name,
            "bytes": int(t.n_bytes),
            "elements": int(t.n_elements),
            "offset": int(t.data_offset),
        })

    with open(os.path.join(out_dir, "tensors.tsv"), "w") as o:
        o.write("name\tlayer\trole\tne0_k_in\tne1_m_out\tne2\tne3\tggml_type\telements\tbytes\tdata_offset_b\n")
        for t in tensors:
            ne = t["ne"] + [1] * (4 - len(t["ne"]))
            o.write(f"{t['name']}\t{t['layer']}\t{t['role']}\t{ne[0]}\t{ne[1]}\t{ne[2]}\t{ne[3]}\t{t['type']}\t{t['elements']}\t{t['bytes']}\t{t['offset']}\n")

    # ---- hyperparameters (GGUF keys, then values derived the way src/models/qwen35.cpp derives them)
    arch = f(r, "general.architecture")
    p = arch + "."
    n_layer_all = int(f(r, p + "block_count"))
    n_nextn = int(f(r, p + "nextn_predict_layers"))
    n_layer = n_layer_all - n_nextn
    recur = [bool(x) for x in f(r, p + "attention.recurrent_layers")]
    d_inner = int(f(r, p + "ssm.inner_size"))
    d_state = int(f(r, p + "ssm.state_size"))
    n_group = int(f(r, p + "ssm.group_count"))
    dt_rank = int(f(r, p + "ssm.time_step_rank"))
    d_conv = int(f(r, p + "ssm.conv_kernel"))
    n_embd = int(f(r, p + "embedding_length"))
    n_head = int(f(r, p + "attention.head_count"))
    n_head_kv = int(f(r, p + "attention.head_count_kv"))
    head_k = int(f(r, p + "attention.key_length"))
    head_v = int(f(r, p + "attention.value_length"))
    n_vocab = len(r.fields["tokenizer.ggml.tokens"].data)
    conv_channels = d_inner + 2 * n_group * d_state
    labels = f(r, p + "decision.labels")

    rows = [
        ("arch", arch, "general.architecture", ""),
        ("model_name", f(r, "general.name"), "general.name", ""),
        ("gguf_file_bytes", total, "size of the GGUF file", "B"),
        ("gguf_data_offset", r.data_offset, "gguf-py", "B"),
        ("head_bytes_fetched", nbytes_fetched, "first N bytes of the GGUF file (smallest N in 16/32/64/128 MB that parses)", "B"),
        ("n_tensors", len(tensors), "GGUF.tensor_count", ""),
        ("n_layer_all (block_count)", n_layer_all, p + "block_count", ""),
        ("n_layer_nextn", n_nextn, p + "nextn_predict_layers", ""),
        ("n_layer (executed)", n_layer, "block_count - nextn_predict_layers (qwen35.cpp: the NextN/MTP layer has no tensors in this file and is not executed)", ""),
        ("n_embd", n_embd, p + "embedding_length", ""),
        ("n_ff", int(f(r, p + "feed_forward_length")), p + "feed_forward_length", ""),
        ("n_head", n_head, p + "attention.head_count", ""),
        ("n_head_kv", n_head_kv, p + "attention.head_count_kv", ""),
        ("head_dim_k", head_k, p + "attention.key_length", ""),
        ("head_dim_v", head_v, p + "attention.value_length", ""),
        ("gqa_ratio", n_head // n_head_kv, "n_head / n_head_kv", ""),
        ("attn_q_rows (query + output gate)", 2 * n_head * head_k, "wq ne1 = 2 * n_head * head_dim", ""),
        ("attn_kv_rows (each of K, V)", n_head_kv * head_k, "n_head_kv * head_dim", ""),
        ("attn_o_in", n_head * head_v, "n_head * head_dim_v", ""),
        ("rope_dimension_count", int(f(r, p + "rope.dimension_count")), p + "rope.dimension_count", ""),
        ("rope_sections", " ".join(str(x) for x in f(r, p + "rope.dimension_sections")), p + "rope.dimension_sections (M-RoPE)", ""),
        ("rope_freq_base", f(r, p + "rope.freq_base"), p + "rope.freq_base", ""),
        ("rms_norm_eps", f(r, p + "attention.layer_norm_rms_epsilon"), p + "attention.layer_norm_rms_epsilon", ""),
        ("context_length_train", int(f(r, p + "context_length")), p + "context_length", ""),
        ("full_attention_interval", int(f(r, p + "full_attention_interval")), p + "full_attention_interval", ""),
        ("n_layers_recurrent (Gated DeltaNet)", sum(recur[:n_layer]), "attention.recurrent_layers[:n_layer]", ""),
        ("n_layers_attention", n_layer - sum(recur[:n_layer]), "n_layer - recurrent", ""),
        ("layer_pattern", "".join("R" if x else "A" for x in recur[:n_layer]), "attention.recurrent_layers (R = Gated DeltaNet, A = full attention)", ""),
        ("ssm_conv_kernel", d_conv, p + "ssm.conv_kernel", ""),
        ("ssm_inner_size (d_inner)", d_inner, p + "ssm.inner_size", ""),
        ("ssm_state_size = GDN head_k_dim", d_state, p + "ssm.state_size", ""),
        ("ssm_group_count = GDN num_k_heads", n_group, p + "ssm.group_count", ""),
        ("ssm_time_step_rank = GDN num_v_heads", dt_rank, p + "ssm.time_step_rank", ""),
        ("GDN head_v_dim", d_inner // dt_rank, "d_inner / num_v_heads", ""),
        ("GDN conv_channels", conv_channels, "d_inner + 2 * n_group * d_state", ""),
        ("GDN qkv rows (attn_qkv ne1)", 2 * n_group * d_state + d_inner, "2 * n_group * d_state + d_inner", ""),
        ("GDN recurrent state per layer per sequence", (d_inner // dt_rank) * (d_inner // dt_rank) * dt_rank * 4, "head_v_dim^2 * num_v_heads * 4 B (F32)", "B"),
        ("GDN conv state per layer per sequence", (d_conv - 1) * conv_channels * 4, "(d_conv-1) * conv_channels * 4 B (F32)", "B"),
        ("n_vocab", n_vocab, "len(tokenizer.ggml.tokens)", ""),
        ("output head", "tied to token_embd" if not any(t["name"] == "output.weight" for t in tensors) else "output.weight", "tensor list (no output.weight in the file => LM head = token_embd, Q8_0)", ""),
        ("decision.type", f(r, p + "decision.type"), p + "decision.type", ""),
        ("decision.n_labels", len(labels), p + "decision.labels", ""),
        ("decision.label_first_5", "|".join(labels[:5]), p + "decision.labels", ""),
        ("tokenizer", f(r, "tokenizer.ggml.model") + "/" + f(r, "tokenizer.ggml.pre"), "tokenizer.ggml.model / .pre", ""),
        ("general.file_type", int(f(r, "general.file_type")), "general.file_type (2 = MOSTLY_Q4_0, mixed per tensor)", ""),
    ]
    by_type = {}
    for t in tensors:
        by_type.setdefault(t["type"], [0, 0])
        by_type[t["type"]][0] += t["bytes"]
        by_type[t["type"]][1] += t["elements"]
    for k, (b, e) in sorted(by_type.items()):
        rows.append((f"tensor bytes, type {k}", b, "sum over tensors.tsv", "B"))
        rows.append((f"tensor elements, type {k}", e, "sum over tensors.tsv", ""))
    layer_bytes = sum(t["bytes"] for t in tensors if t["layer"] >= 0)
    rows.append(("tensor bytes in the 32 layers", layer_bytes, "sum over tensors.tsv (layer >= 0)", "B"))
    rows.append(("tensor bytes token_embd (= LM head)", sum(t["bytes"] for t in tensors if t["name"] == "token_embd.weight"), "tensors.tsv", "B"))
    rows.append(("matmul weight elements in the 32 layers", sum(t["elements"] for t in tensors if t["layer"] >= 0 and t["type"] in ("Q4_0", "Q8_0")), "sum over Q4_0/Q8_0 tensors of the layers", ""))
    with open(os.path.join(out_dir, "hparams.tsv"), "w") as o:
        o.write("parameter\tvalue\tsource\tunit\n")
        for k, v, s, u in rows:
            o.write(f"{k}\t{v}\t{s}\t{u}\n")

    with open(os.path.join(out_dir, "layers.tsv"), "w") as o:
        o.write("layer\tkind\tweight_bytes\tq4_0_bytes\tq8_0_bytes\tf32_bytes\n")
        for il in range(n_layer):
            ts = [t for t in tensors if t["layer"] == il]
            q4 = sum(t["bytes"] for t in ts if t["type"] == "Q4_0")
            q8 = sum(t["bytes"] for t in ts if t["type"] == "Q8_0")
            f32 = sum(t["bytes"] for t in ts if t["type"] == "F32")
            o.write(f"{il}\t{'recurrent' if recur[il] else 'attention'}\t{q4+q8+f32}\t{q4}\t{q8}\t{f32}\n")
    print("wrote tensors.tsv hparams.tsv layers.tsv in", out_dir, "; tensors:", len(tensors), "layer weight bytes:", layer_bytes)


if __name__ == "__main__":
    main()
