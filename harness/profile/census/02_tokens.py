#!/usr/bin/env python3
"""Step 2: token counts per case, as the b12004 server forms the prompts of POST /v1/systemone.

Run with:  uv run --with jinja2 --with numpy --with pyyaml --with tqdm --with requests python 02_tokens.py \
               --tokenize <llama.cpp b12004>/llama-tokenize --gguf MODEL.gguf --gguf-py <llama.cpp>/gguf-py [--out DIR]
(CPU only. Needs the b12004 build of llama-tokenize and a GGUF file of the mix8 model; a file that holds the header and then a sparse
hole of the full size is enough, because only the vocabulary and the metadata are read.)

How the server forms a prompt (tools/server/server-decision.cpp at b12004, type H2O_LIGHTNING):
  * post_systemone(): one task per question, in the order of the "questions" JSON object (insertion order).
  * parse_questions(): choice -> options in criteria order; noul -> options [true, false] (noul_true_first), a
    description only if criteria is given; labels = decision.labels[i] (" A", " B", ...), trimmed by the template.
  * prepare_state(): a string state stays a string (cut only above 32000 tokens), special-token spellings are quoted.
  * render_h2o() + the GGUF "systemone" Jinja template -> the prompt text; common_tokenize(text, add_special=false,
    parse_special=true) -> tokens.
  * server_decision_group_tasks(): groups of n_slots(3) consecutive tasks; the parent = first task; the shared prefix
    = common token prefix of the group, at most size-1 of every task; the parent evaluates the shared prefix once, the
    children copy its memory and evaluate only their remaining tokens.
  * usage.input_tokens of the response = sum of the full prompt lengths of all questions (no sharing).
Reproduction: the template is rendered with Python Jinja2 (not the llama.cpp jinja runtime); the template only uses
{{ }}, for, if/elif/else, set-block and the trim filter, so the outputs are identical (checked below against the
input_tokens_mean per size of a benchmark run, which the server returns; see token-check.tsv).
Tokenization: build b12004 llama-tokenize, CPU only, vocabulary from the GGUF header.
The template ("tokenizer.chat_template.systemone"), the 255 decision labels ("<arch>.decision.labels") and the texts of the control
tokens (for the quoting) are read from the GGUF file given with --gguf.

Inputs (arguments):
  --cases FILE      the benchmark cases (default: bench/cases.json of this repository)
  --tokenize PATH   the llama-tokenize program of the b12004 build (default: $DMK_TOKENIZE)
  --gguf FILE       the GGUF file for the vocabulary and the metadata
  --gguf-py DIR     the gguf-py directory of llama.cpp (default: $GGUF_PY; else an installed gguf package)
  --runs FILE       the JSON of one benchmark run with quality.input_tokens_mean (default: the first run of the stock image
                    in data/2026-10-30-window1/raw)
  --prompts DIR     keep the rendered prompt files in this folder (default: a temporary folder that is removed at the end)
Outputs into --out (default: the current folder): token-counts.tsv (one row per case x variant x question),
cases-plan.tsv (one row per case x variant), token-check.tsv (comparison with the benchmark run), ref-line-tokens.tsv
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile

import jinja2

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_CASES = os.path.join(REPO, "bench", "cases.json")
DEFAULT_RUNS = os.path.join(REPO, "data", "2026-10-30-window1", "raw", "p1-bench-b1-S.json")
N_SLOTS = 3
H2O_MAX_STATE_TOKENS = 32000

# set by main()
TOKENIZE = None
GGUF = None
labels = None
template = None
QUOTED = None
QUOTED_FIRST = None


# ---------------------------------------------------------------- model metadata (from the GGUF file)
def import_gguf(gguf_py):
    if gguf_py:
        sys.path.insert(0, gguf_py)
    try:
        from gguf import GGUFReader
    except ImportError:
        raise SystemExit("cannot import gguf: set --gguf-py (or GGUF_PY) to the gguf-py directory of llama.cpp b12004")
    return GGUFReader


def load_model_text(gguf_path, gguf_py):
    """-> (systemone template source, decision labels, quoted spellings) read from the GGUF metadata."""
    GGUFReader = import_gguf(gguf_py)
    r = GGUFReader(gguf_path)
    arch = r.fields["general.architecture"].contents()
    template_src = r.fields["tokenizer.chat_template.systemone"].contents()
    lab = r.fields[arch + ".decision.labels"].contents()
    # Texts of the CONTROL / USER_DEFINED tokens (init_quoted_spellings(CONTROL|USER_DEFINED, 1)).
    toks = r.fields["tokenizer.ggml.tokens"]
    types = r.fields["tokenizer.ggml.token_type"]
    out = []
    for i, ti in enumerate(types.data):
        if int(types.parts[ti][0]) in (3, 4):  # 3 = CONTROL, 4 = USER_DEFINED in gguf token types
            out.append(bytes(toks.parts[toks.data[i]]).decode("utf-8", "replace"))
    out = sorted((s for s in out if len(s) >= 1), key=lambda s: -len(s))  # stable sort, longest first
    return template_src, lab, out


# ---------------------------------------------------------------- special-token quoting (h2o_quote)
def h2o_quote(text):
    out, i = [], 0
    while i < len(text):
        if text[i] not in QUOTED_FIRST:
            out.append(text[i]); i += 1; continue
        for sp in QUOTED:
            if text.startswith(sp, i):
                out.append(sp.replace("<", "\\u003c").replace(">", "\\u003e")); i += len(sp)
                break
        else:
            out.append(text[i]); i += 1
    return "".join(out)


PY_SPACE = set(chr(c) for c in list(range(0x09, 0x0e)) + list(range(0x1c, 0x21)) + [0x85, 0xa0, 0x1680, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000] + list(range(0x2000, 0x200b)))


def py_lstrip(s):
    i = 0
    while i < len(s) and s[i] in PY_SPACE:
        i += 1
    return s[i:]


def py_rstrip(s):
    j = len(s)
    while j > 0 and s[j - 1] in PY_SPACE:
        j -= 1
    return s[:j]


def decision_dumps(val, indent, item_sep, key_sep, ensure_ascii):
    return json.dumps(val, indent=indent, separators=(item_sep.strip() if indent is not None else item_sep, key_sep), ensure_ascii=ensure_ascii)


def parse_question(qid, q):
    """-> dict(id, type, instructions, options=[(key, description|None)])  (parse_questions for h2o-lightning)"""
    qtype = q.get("type", "choice")
    crit = q.get("criteria")
    opts = []
    if qtype == "choice":
        for k, d in crit.items():
            opts.append((k, d))
    elif qtype == "score":
        for i, d in enumerate(crit):
            opts.append((str(i), d))
    elif qtype == "noul":
        for key in ("false", "true"):
            d = None
            if isinstance(crit, dict):
                if key in crit:
                    d = crit[key]
                else:
                    alias = "yes" if key == "true" else "no"
                    d = crit.get(alias)
            opts.append((key, d))
        opts[0], opts[1] = opts[1], opts[0]  # noul_true_first
    else:
        raise ValueError(qtype)
    return {"id": qid, "type": qtype, "instructions": q.get("instructions", ""), "options": opts}


def render(state_prepared, q):
    ins = q["instructions"]
    ins = ins if isinstance(ins, str) else decision_dumps(ins, 1, ",", ": ", True)
    ins = py_lstrip(ins)
    options = []
    n = len(q["options"])
    for i, (key, desc) in enumerate(q["options"]):
        d = None
        if q["type"] == "score" and desc is None:
            d = "null"
        elif desc is not None:
            d = h2o_quote(desc if isinstance(desc, str) else decision_dumps(desc, None, ", ", ": ", False))
        key_q = h2o_quote(key)
        if i + 1 == n:
            is_key = q["type"] == "choice" and d is None
            if is_key or isinstance(d, str):
                text = key_q if is_key else d
                stripped = py_rstrip(text)
                if stripped != text:
                    d = " " if stripped == "" else stripped
        options.append({"label": labels[i], "key": key_q, "description": d})
    inp = {"type": q["type"], "state": state_prepared, "instructions": h2o_quote(ins), "options": options, "images": []}
    return template.render(**inp)


def prepare_state(state):
    if not isinstance(state, str):
        state = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
    assert len(state) <= H2O_MAX_STATE_TOKENS, "state cut not modelled"  # bytes <= tokens bound
    return h2o_quote(state)


def tokenize_file(path):
    r = subprocess.run([TOKENIZE, "-m", GGUF, "--file", path, "--no-escape", "--ids", "--log-disable"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return json.loads(r.stdout.decode().strip().splitlines()[-1])


def tag(*parts):
    return hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:10]


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    global TOKENIZE, GGUF, labels, template, QUOTED, QUOTED_FIRST
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", default=DEFAULT_CASES, help="benchmark cases JSON (default: bench/cases.json of the repository)")
    ap.add_argument("--tokenize", default=os.environ.get("DMK_TOKENIZE"), help="path of the b12004 llama-tokenize (default: $DMK_TOKENIZE)")
    ap.add_argument("--gguf", required=True, help="GGUF file for the vocabulary and the metadata")
    ap.add_argument("--gguf-py", default=os.environ.get("GGUF_PY"), help="the gguf-py directory of llama.cpp (default: $GGUF_PY, else an installed gguf package)")
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="JSON of one benchmark run with quality.input_tokens_mean")
    ap.add_argument("--prompts", default=None, help="keep the rendered prompts in this folder (default: a temporary folder)")
    ap.add_argument("--out", default=".", help="output folder (default: the current folder)")
    args = ap.parse_args()
    if not args.tokenize:
        raise SystemExit("set --tokenize (or DMK_TOKENIZE) to the llama-tokenize program of the b12004 build")
    TOKENIZE = os.path.abspath(args.tokenize)
    GGUF = os.path.abspath(args.gguf)
    os.makedirs(args.out, exist_ok=True)

    template_src, labels, QUOTED = load_model_text(GGUF, args.gguf_py)
    QUOTED_FIRST = "".join(sorted({s[0] for s in QUOTED}))
    template = jinja2.Environment(autoescape=False, keep_trailing_newline=True).from_string(template_src)

    prompts_dir = args.prompts or tempfile.mkdtemp(prefix="census-prompts-")
    os.makedirs(prompts_dir, exist_ok=True)
    try:
        run(args, prompts_dir)
    finally:
        if not args.prompts:
            shutil.rmtree(prompts_dir, ignore_errors=True)


def run(args, prompts_dir):
    out_dir = args.out
    cases = json.load(open(args.cases))
    jobs = []  # (variant, case, qid, path)
    prompts = {}
    for c in cases:
        for variant in ("raw", "ref"):
            state = c["state"] if variant == "raw" else f"Ref: {tag('seq', 0, c['id'])}\n{c['state']}"
            st = prepare_state(state)
            for qid, q in c["questions"].items():
                pq = parse_question(qid, q)
                text = render(st, pq)
                path = os.path.join(prompts_dir, f"{c['id']}.{variant}.{qid}.txt")
                with open(path, "w", newline="") as f:
                    f.write(text)
                prompts[(variant, c["id"], qid)] = (pq, path, text)
                jobs.append((variant, c["id"], qid, path))
    with cf.ThreadPoolExecutor(16) as ex:
        toks = dict(zip([(j[0], j[1], j[2]) for j in jobs], ex.map(lambda j: tokenize_file(j[3]), jobs)))

    # Ref-line token cost: 100 sample lines "Ref: <uuid4 hex[:10]>\n" (what bench.py / prof_client.py prepend)
    import random
    rnd = random.Random(1)
    ref_counts = []
    samples = ["".join(rnd.choice("0123456789abcdef") for _ in range(10)) for _ in range(100)]

    def ref_tokens(s):
        p = os.path.join(prompts_dir, f"ref-{s}.txt")
        open(p, "w").write(f"Ref: {s}\n")
        return len(tokenize_file(p))
    with cf.ThreadPoolExecutor(16) as ex:
        ref_counts = list(ex.map(ref_tokens, samples))
    with open(os.path.join(out_dir, "ref-line-tokens.tsv"), "w") as o:
        o.write("stat\ttokens\n")
        o.write(f"min\t{min(ref_counts)}\nmedian\t{statistics.median(ref_counts)}\nmean\t{statistics.fmean(ref_counts):.2f}\nmax\t{max(ref_counts)}\nsamples\t{len(ref_counts)}\n")

    rows, plans = [], []
    for variant in ("raw", "ref"):
        for c in cases:
            qids = list(c["questions"])
            tl = [toks[(variant, c["id"], q)] for q in qids]
            n_q = len(qids)
            assert n_q <= N_SLOTS, "census models one group of at most n_slots questions"
            if n_q >= 2:
                shared = len(tl[0]) - 1
                for t in tl[1:]:
                    shared = min(shared, common_prefix(tl[0], t), len(t) - 1)
                if shared == 0:
                    shared = 0
            else:
                shared = 0
            suff = [len(t) - shared for t in tl]
            for qid, t in zip(qids, tl):
                q = c["questions"][qid]
                pq = prompts[(variant, c["id"], qid)][0]
                rows.append([variant, c["id"], c["task"], c["size"], n_q, qid, pq["type"], len(pq["options"]), len(t), shared, len(t) - shared])
            plans.append([variant, c["id"], c["task"], c["size"], n_q, sum(len(t) for t in tl), shared, " ".join(str(s) for s in suff),
                          len(set(suff)), min(len(t) for t in tl)])

    with open(os.path.join(out_dir, "token-counts.tsv"), "w") as o:
        o.write("variant\tcase\ttask\tsize\tn_questions\tquestion\tqtype\tn_options\tprompt_tokens\tshared_prefix_tokens\tsuffix_tokens\n")
        for r_ in rows:
            o.write("\t".join(map(str, r_)) + "\n")
    with open(os.path.join(out_dir, "cases-plan.tsv"), "w") as o:
        o.write("variant\tcase\ttask\tsize\tn_questions\tsum_prompt_tokens\tshared_prefix_tokens\tsuffix_tokens_per_question\tn_distinct_suffix_lengths\tmin_prompt_tokens\n")
        for p_ in plans:
            o.write("\t".join(map(str, p_)) + "\n")

    # check against the benchmark run (usage.input_tokens, quality phase = raw state)
    runs = json.load(open(args.runs))
    want = runs["quality"]["input_tokens_mean"]
    with open(os.path.join(out_dir, "token-check.tsv"), "w") as o:
        o.write("size\tn_cases\tmean_sum_prompt_tokens_raw_here\tmean_input_tokens_production_run\tdiff_tokens\n")
        for size in ("short", "medium", "long"):
            vals = [p[5] for p in plans if p[0] == "raw" and p[3] == size]
            mean = statistics.fmean(vals)
            o.write(f"{size}\t{len(vals)}\t{mean:.4f}\t{want[size]:.4f}\t{mean - want[size]:+.4f}\n")
    print(open(os.path.join(out_dir, "token-check.tsv")).read())


if __name__ == "__main__":
    main()
