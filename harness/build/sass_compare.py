#!/usr/bin/env python3
"""Compare the sm_86 SASS of two libggml-cuda.so builds.

Usage: sass_compare.py <dir-A> <dir-B> <outdir> [name-regex ...]

<dir-A> and <dir-B> hold the cubins of one build each. Extract them with
    cuobjdump -arch sm_86 -xelf all <path to libggml-cuda.so>
in an empty folder. The files are named libggml-cuda.<N>.sm_86.cubin. Both
folders must have the same set of N. The script needs cuobjdump in PATH. The
builder image of build.sh has it, and so has python3.

The script matches functions by name after the normalization in norm_name().
nvcc puts a hash that is different in each build into the names of internal
functions (_INTERNAL_<hash>_...), and it puts the architecture list into the
CUB namespace. The script removes addresses and encoding comments from the SASS
text before it compares the text. It also compares the raw 128-bit instruction
encodings, which hold the scheduling control bits that the text hides. A name
that exists in more than one translation unit is compared as a multiset of all
its bodies. The register, stack, shared, local and constant figures come from
`cuobjdump -res-usage`. For a name with more than one body, the last body counts.

Output in <outdir>: sass-compare.json (summary and differences) and key-kernels.tsv
(one row for each name that matches a name-regex). A name-regex that matches a
name with different SASS also gives a diff-<hash>.txt file.
The script prints the summary on standard output. The two builds have the same
SASS when names_sass_different, names_encoding_different, names_only_A,
names_only_B and names_resusage_different are all 0.
"""
import collections, glob, hashlib, json, os, re, subprocess, sys
from multiprocessing import Pool

if len(sys.argv) < 4:
    sys.exit(__doc__)
A, B, OUT = sys.argv[1:4]
KEYS = sys.argv[4:]


def norm_name(n):
    n = re.sub(r'_INTERNAL_[0-9a-f]{8}_', '_INTERNAL_X_', n)
    n = re.sub(r'\d+(CUB|THRUST)_\d+(_[0-9]+)+_NS', r'\1_NS', n)   # length prefix first: _ZN3cub50CUB_..._NS -> _ZN3cubCUB_NS
    return n


COMMENT = re.compile(r'/\*.*?\*/')
INSTR = re.compile(r'^\s+/\*[0-9a-f]{4,}\*/')


def cubins(d):
    """Return {N: path} for the files libggml-cuda.<N>.sm_86.cubin in folder d."""
    found = {}
    for p in glob.glob(os.path.join(d, 'libggml-cuda.*.sm_86.cubin')):
        m = re.search(r'libggml-cuda\.(\d+)\.sm_86\.cubin$', p)
        if m:
            found[int(m.group(1))] = p
    return found


def dump(path):
    """Return ({function: normalized SASS lines}, {function: encoding words}) for one cubin."""
    txt = subprocess.run(['cuobjdump', '-sass', path], capture_output=True, text=True, check=True).stdout
    funcs, enc, cur = {}, {}, None
    for line in txt.splitlines():
        m = re.match(r'\s*Function : (\S+)', line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            enc[cur] = []
            continue
        if cur is None:
            continue
        # every instruction is two 64-bit words printed as /* 0x... */ comments; the second word holds the scheduling control bits
        enc[cur].extend(re.findall(r'/\* (0x[0-9a-f]{16}) \*/', line))
        if INSTR.match(line):
            s = COMMENT.sub('', line).strip()
            s = re.sub(r'_INTERNAL_[0-9a-f]{8}_', '_INTERNAL_X_', s)
            s = re.sub(r'\d+(CUB|THRUST)_\d+(_[0-9]+)+_NS', r'\1_NS', s)
            funcs[cur].append(s)
    return funcs, enc


def resusage(path):
    """Return {normalized function name: {REG: .., STACK: .., SHARED: .., LOCAL: .., CONSTANT[0]: .., ...}}."""
    txt = subprocess.run(['cuobjdump', '-res-usage', path], capture_output=True, text=True, check=True).stdout
    d, cur = {}, None
    for line in txt.splitlines():
        rest = line.strip()
        if rest.startswith('Function '):
            cur = norm_name(rest[9:-1])
        elif cur and rest.startswith('REG:'):
            d[cur] = dict(kv.split(':', 1) for kv in rest.split())
    return d


def work(args):
    side, i, p = args
    res = {}
    funcs, enc = dump(p)
    for name, lines in funcs.items():
        res[norm_name(name)] = (name, hashlib.sha256('\n'.join(lines).encode()).hexdigest(), len(lines), lines if KEYS and any(re.search(k, name) for k in KEYS) else None,
                                hashlib.sha256(' '.join(enc[name]).encode()).hexdigest())
    return side, i, res, resusage(p)


if __name__ == '__main__':
    ca, cb = cubins(A), cubins(B)
    if not ca or set(ca) != set(cb):
        sys.exit('sass_compare.py: both folders must have the same, not empty set of libggml-cuda.<N>.sm_86.cubin files')
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, i, c[i]) for s, c in (('A', ca), ('B', cb)) for i in sorted(c)]
    with Pool(8) as p:
        results = p.map(work, jobs)
    # per side: normalized name -> list of (cubin index, original name, sass hash, instr count, lines, encoding hash)
    side = {'A': collections.defaultdict(list), 'B': collections.defaultdict(list)}
    res = {'A': {}, 'B': {}}
    for s, i, r, ru in results:
        for n, (orig, hsh, cnt, lines, eh) in r.items():
            side[s][n].append((i, orig, hsh, cnt, lines, eh))
        res[s].update(ru)
    ra, rb = res['A'], res['B']
    names = sorted(set(side['A']) | set(side['B']))
    onlyA = [n for n in names if n not in side['B']]
    onlyB = [n for n in names if n not in side['A']]
    both = [n for n in names if n in side['A'] and n in side['B']]
    # a name may occur in several translation units; compare the multiset of SASS hashes
    same, diff, encsame = [], [], []
    for n in both:
        ha = sorted(x[2] for x in side['A'][n])
        hb = sorted(x[2] for x in side['B'][n])
        (same if ha == hb else diff).append(n)
        if sorted(x[5] for x in side['A'][n]) == sorted(x[5] for x in side['B'][n]):
            encsame.append(n)
    inst = lambda s: sum(x[3] for v in side[s].values() for x in v)
    summary = {
        'cubins_per_side': len(ca),
        'function_bodies_A': sum(len(v) for v in side['A'].values()), 'function_bodies_B': sum(len(v) for v in side['B'].values()),
        'distinct_normalized_names_A': len(side['A']), 'distinct_normalized_names_B': len(side['B']),
        'names_in_both': len(both), 'names_only_A': len(onlyA), 'names_only_B': len(onlyB),
        'names_sass_identical': len(same), 'names_sass_different': len(diff),
        'names_encoding_identical_incl_control_bits': len(encsame), 'names_encoding_different': len(both) - len(encsame),
        'instr_total_A': inst('A'), 'instr_total_B': inst('B'),
        'names_resusage_different': sum(1 for n in both if ra.get(n) != rb.get(n)),
    }
    print(json.dumps(summary, indent=1))
    rows = []
    for n in diff:
        a, b = side['A'][n][0], side['B'][n][0]
        rows.append({'name': a[1], 'bodiesA': len(side['A'][n]), 'bodiesB': len(side['B'][n]), 'instrA': sorted(x[3] for x in side['A'][n]), 'instrB': sorted(x[3] for x in side['B'][n]), 'resA': ra.get(n), 'resB': rb.get(n)})
    json.dump({'summary': summary, 'different': rows, 'onlyA': onlyA[:50], 'onlyB': onlyB[:50]}, open(os.path.join(OUT, 'sass-compare.json'), 'w'), indent=1)
    # key kernels table (TSV): one row for each normalized name that matches a name-regex
    import difflib
    with open(os.path.join(OUT, 'key-kernels.tsv'), 'w') as f:
        f.write('name\tsass_identical\tencoding_identical\tinstrA\tinstrB\tregA\tregB\tsharedA\tsharedB\tstackA\tstackB\tlocalA\tlocalB\n')
        for n in both:
            a = side['A'][n][0]
            if not (KEYS and any(re.search(k, a[1]) for k in KEYS)):
                continue
            b = side['B'][n][0]
            ok = n in same
            r1, r2 = ra.get(n, {}), rb.get(n, {})
            f.write('\t'.join(str(x) for x in [a[1], int(ok), int(n in encsame), a[3], b[3], r1.get('REG'), r2.get('REG'), r1.get('SHARED'), r2.get('SHARED'), r1.get('STACK'), r2.get('STACK'), r1.get('LOCAL'), r2.get('LOCAL')]) + '\n')
            if not ok and a[4] and b[4]:
                with open(os.path.join(OUT, 'diff-%s.txt' % hashlib.sha1(n.encode()).hexdigest()[:10]), 'w') as g:
                    g.write(n + '\n' + '\n'.join(difflib.unified_diff(a[4], b[4], 'A', 'B', lineterm='', n=2)))
