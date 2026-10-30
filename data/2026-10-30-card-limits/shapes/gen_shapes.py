#!/usr/bin/env python3
"""Derive the cuBLASLt benchmark shape list (M N K ...) from the weight matrices of a GGUF file.

Only the GGUF header is read. Two ways to give it:
  --gguf PATH          a local .gguf file, or just its first tens of MB (the script pads a copy to the real size)
  --ssh HOST:PATH      fetch the first --head-bytes of PATH over ssh (`ssh HOST head -c N PATH`), read-only

Needs numpy and the gguf package from the gguf-py directory of llama.cpp (set --gguf-py or the environment variable
GGUF_PY to that directory; the data set used b12004):
  uv run --with numpy --with pyyaml --with tqdm python gen_shapes.py --gguf-py <llama.cpp>/gguf-py \\
      --gguf <model>.gguf --origin-label 'the GGUF header of the "mix8" file' -o qwen35-4b-mix8.tsv

Convention (GEMM of a weight matrix with N tokens): a GGUF tensor with shape [ne0, ne1] has ne0 = K (input features,
contiguous) and ne1 = M (output features). Matrices with the same (M, K) are merged into one row; `label` joins
their names, `count` is how many such matrices one forward pass multiplies (layer count), `wtype` the quantization type.
A model without output.weight ties the output head to token_embd.weight: it is listed once as output_head, as a GEMM
(the embedding lookup itself is a gather, not a GEMM). Two square reference GEMMs (count 0) are appended for the
large-matrix int8 peak of cuBLASLt itself.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import OrderedDict


def fetch_header(host_path, head_bytes, tmpdir):
    host, _, path = host_path.partition(":")
    if not host or not path:
        sys.exit("--ssh needs HOST:PATH")
    size = int(subprocess.check_output(["ssh", host, "stat", "-c", "%s", path]).decode().strip())
    out = os.path.join(tmpdir, "header.gguf")
    with open(out, "wb") as f:
        subprocess.run(["ssh", host, "head", "-c", str(head_bytes), path], stdout=f, check=True)
    if os.path.getsize(out) < min(size, head_bytes):
        sys.exit("short read from ssh")
    return out, size


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gguf")
    src.add_argument("--ssh")
    ap.add_argument("--size", type=int, help="real file size in bytes when --gguf is a truncated header")
    ap.add_argument("--head-bytes", type=int, default=64_000_000)
    ap.add_argument("--gguf-py", default=os.environ.get("GGUF_PY", "gguf-py"), help="the gguf-py directory of llama.cpp (default: $GGUF_PY or ./gguf-py)")
    ap.add_argument("--origin-label", help="text for the second comment line of the output (default: the file name of the GGUF file)")
    ap.add_argument("--n", default="8,16,32,64,128,512,2048", help="token counts (default 8,16,32,64,128,512,2048)")
    ap.add_argument("--no-ref", action="store_true", help="do not append the square reference GEMMs")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.gguf_py)
    from gguf import GGUFReader  # noqa: E402

    ns = [int(x) for x in args.n.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        if args.ssh:
            path, size = fetch_header(args.ssh, args.head_bytes, tmp)
            origin = os.path.basename(args.ssh.partition(":")[2])
        else:
            path = os.path.join(tmp, "header.gguf")
            with open(args.gguf, "rb") as s, open(path, "wb") as d:
                d.write(s.read(args.head_bytes))
            size = args.size or os.path.getsize(args.gguf)
            origin = os.path.basename(args.gguf)
        os.truncate(path, max(size, os.path.getsize(path)))  # sparse padding: GGUFReader maps tensor data by offset
        r = GGUFReader(path)

        def field(name, default="?"):
            f = r.fields.get(name)
            if f is None:
                return default
            try:
                return str(f.contents())
            except Exception:
                return default

        arch = field("general.architecture")
        name = field("general.name")
        tensors = list(r.tensors)
        has_output = any(t.name == "output.weight" for t in tensors)

        groups = OrderedDict()  # (M, K, wtype) -> {"names": [...], "count": n}
        for t in tensors:
            shape = [int(x) for x in t.shape]
            if len(shape) != 2 or t.tensor_type.name in ("F32", "F64"):
                continue
            if t.name == "token_embd.weight":
                if has_output:
                    continue
                label = "output_head"
            else:
                label = re.sub(r"^blk\.\d+\.", "", t.name)
                label = re.sub(r"\.weight$", "", label)
            k, m = shape[0], shape[1]
            g = groups.setdefault((m, k, t.tensor_type.name), {"names": [], "count": 0})
            if label not in g["names"]:
                g["names"].append(label)
            g["count"] += 1

    rows = []
    for (m, k, wt), g in sorted(groups.items(), key=lambda kv: (kv[0][2] != "Q4_0", -kv[0][0] * kv[0][1])):
        for n in ns:
            rows.append((m, n, k, "+".join(g["names"]), g["count"], wt))
    if not args.no_ref:
        for s in (4096, 8192):
            rows.append((s, s, s, "ref_gemm", 0, "-"))

    with open(args.out, "w") as f:
        f.write("# cuBLASLt int8 benchmark shapes: M = weight rows (output features), N = tokens, K = input features\n")
        f.write("# derived from %s (%s, arch %s, file %d bytes, %d tensors) by shapes/gen_shapes.py\n" % (args.origin_label or origin, name, arch, size, len(tensors)))
        f.write("# count = matrices of this shape multiplied per forward pass; count 0 = reference GEMM, not part of the model\n")
        f.write("#M\tN\tK\tlabel\tcount\twtype\n")
        for row in rows:
            f.write("%d\t%d\t%d\t%s\t%d\t%s\n" % row)
    print("%d shapes written to %s (%d matrix classes x %d token counts%s)" % (len(rows), args.out, len(groups), len(ns), "" if args.no_ref else " + 2 reference"))
    for (m, k, wt), g in groups.items():
        print("  M=%-7d K=%-5d %-5s x%-3d %s" % (m, k, wt, g["count"], "+".join(g["names"])))


if __name__ == "__main__":
    main()
