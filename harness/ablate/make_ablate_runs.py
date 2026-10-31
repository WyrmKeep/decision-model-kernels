#!/usr/bin/env python3
"""Make the run files of the timing-only ablation build. Python standard library only.

  <arms dir>/<arm>.env   docker --env-file of a server arm
                         (DMK_DOCKER_EXTRA=--env-file=<work dir>/builds/<build>/dist/abl/<arm>.env)
  <arms dir>/<arm>.sh    wrapper of a tool arm (DMK_WRAP=/opt/dmk/abl/<arm>.sh, for llama bench)
  <arms dir>/SHA256SUMS
  <out>/sched/*.txt      schedule files for harness/run_schedule.sh

Usage:
  make_ablate_runs.py --out DIR [--remote-dir DIR] [--build NAME] [--ref-build NAME] [--model FILE]
                      [--arms-dir DIR] [--passes N] [--blocks-core N] [--llb-reps N] [--seed N]

--remote-dir is the work folder on the GPU host (DMK_REMOTE_DIR of the env file). The default is the
environment variable DMK_REMOTE_DIR. The script stops if neither is set. The files are the same for the same
arguments. The dict ARMS below defines the arms: name -> environment variables.
"""
import argparse
import hashlib
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, help="folder for the schedule files (written to <out>/sched)")
ap.add_argument("--remote-dir", default=os.environ.get("DMK_REMOTE_DIR"),
                help="work folder on the GPU host (default: environment variable DMK_REMOTE_DIR)")
ap.add_argument("--build", default="ablate-timing", help="name of the ablation build under <remote-dir>/builds")
ap.add_argument("--ref-build", default="b12004-sm86", help="name of the baseline build, the arm 'ref'")
ap.add_argument("--model", default="h2o-lightning-4b-rq-Q4_0-mix8.gguf", help="model file name in the folder /models")
ap.add_argument("--arms-dir", default=os.path.join(here, "arms"), help="folder for the .env and .sh files")
ap.add_argument("--passes", type=int, default=5, help="timed passes of ablate_bench.py in stage 1 and 2")
ap.add_argument("--blocks-core", type=int, default=2, help="blocks of the stage 2 schedule")
ap.add_argument("--llb-reps", type=int, default=5)
ap.add_argument("--seed", type=int, default=20261101)
args = ap.parse_args()
if not args.remote_dir:
    sys.exit("set --remote-dir or the environment variable DMK_REMOTE_DIR")

remote_abl = f"{args.remote_dir}/builds/{args.build}/dist/abl"

# name: (env, tier)   tier: ctl (controls), core (stage 1 and 2), fine (stage 1 only)
ARMS = {
    # controls
    "base":       ({}, "ctl"),                                    # this build, no variable set: must equal b12004
    "ref":        ({}, "ctl"),                                    # the baseline build itself (no abl files): partner of "base" in the same schedule
    "timing":     ({"DMK_TIMING": "1"}, "ctl"),                   # per-pass host timing lines in the server log
    "all":        ({"DMK_ABLATE": "all"}, "ctl"),                 # every node skipped: host launch floor (graph structure kept)
    "all_timing": ({"DMK_ABLATE": "all", "DMK_TIMING": "1"}, "ctl"),
    # the big ones (core)
    "mmq":        ({"DMK_ABLATE": "mmq"}, "core"),                # mul_mat_q kernels, activation quantize kept
    "mmvq":       ({"DMK_ABLATE": "mmvq"}, "core"),
    "quantize":   ({"DMK_ABLATE": "quantize"}, "core"),           # quantize_mmq_q8_1 + quantize_q8_1
    "matmul":     ({"DMK_ABLATE": "matmul"}, "core"),             # mmq + mmvq + mmvf + mmf + cublas, quantize kept
    "matmul_q":   ({"DMK_ABLATE": "matmul,quantize"}, "core"),    # all matmul work incl. quantize
    "head":       ({"DMK_ABLATE": "head"}, "core"),               # output head only (248,320 rows), quantize included
    "fattn":      ({"DMK_ABLATE": "fattn"}, "core"),
    "gdn":        ({"DMK_ABLATE": "gdn"}, "core"),                # GATED_DELTA_NET recurrence
    "glu":        ({"DMK_ABLATE": "glu"}, "core"),                # standalone GLU + silu(z)*x / sigmoid(g)*x gated activations
    "norm":       ({"DMK_ABLATE": "norm"}, "core"),               # rms_norm (+mul +add +scale, fused)
    "ssm_conv":   ({"DMK_ABLATE": "ssm_conv"}, "core"),
    "elementwise": ({"DMK_ABLATE": "elementwise"}, "core"),
    "copies":     ({"DMK_ABLATE": "copies"}, "core"),             # cpy + set_rows + concat + get_rows
    # finer splits (stage 1 only)
    "l2norm":     ({"DMK_ABLATE": "l2norm"}, "fine"),
    "cpy":        ({"DMK_ABLATE": "cpy"}, "fine"),
    "set_rows":   ({"DMK_ABLATE": "set_rows"}, "fine"),
    "concat":     ({"DMK_ABLATE": "concat"}, "fine"),
    "get_rows":   ({"DMK_ABLATE": "get_rows"}, "fine"),
    "rope":       ({"DMK_ABLATE": "rope"}, "fine"),
    "mmq_quantize": ({"DMK_ABLATE": "mmq,quantize"}, "fine"),
    "gdn_conv_l2": ({"DMK_ABLATE": "gdn,ssm_conv,l2norm"}, "fine"),  # the whole recurrent-layer core except projections
    # other baselines of the same build, existing b12004 switches (not ablation): what fusion and CUDA graphs are worth now
    "nofusion":   ({"GGML_CUDA_DISABLE_FUSION": "1"}, "fine"),
    "nofusion_glu": ({"GGML_CUDA_DISABLE_FUSION": "1", "DMK_ABLATE": "glu"}, "fine"),  # unfused: SwiGLU, gated activations as separate ops, skipped
    "nographs":   ({"GGML_CUDA_DISABLE_GRAPHS": "1"}, "fine"),
}
# operation-level sync runs (DMK_OPS table), slow, run separately
OPS_ARMS = {
    "sync":     {"DMK_ABLATE_SYNC": "1"},
    "sync_all": {"DMK_ABLATE": "all", "DMK_ABLATE_SYNC": "1"},
}
# llama bench arms: controls and core only
LLB_ARMS = [a for a, (_, t) in ARMS.items() if t in ("ctl", "core")]


def write(path, text, mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if mode:
        os.chmod(path, mode)


envs = {**{a: e for a, (e, _) in ARMS.items() if a != "ref"}, **OPS_ARMS}
for arm, env in envs.items():
    body = "".join(f"{k}={v}\n" for k, v in env.items())
    write(f"{args.arms_dir}/{arm}.env", f"# DMK ablation arm {arm}\n" + body)
    exports = "".join(f"export {k}={v}\n" for k, v in env.items())
    write(f"{args.arms_dir}/{arm}.sh", f"#!/bin/sh\n# DMK ablation arm {arm}: set the variables, run the command\n{exports}exec \"$@\"\n", 0o755)
sums = []
for fn in sorted(os.listdir(args.arms_dir)):
    if fn != "SHA256SUMS":
        sums.append(f"{hashlib.sha256(open(f'{args.arms_dir}/{fn}', 'rb').read()).hexdigest()}  {fn}\n")
write(f"{args.arms_dir}/SHA256SUMS", "".join(sums))


def client_line(name, arm, passes, linger=0):
    cmd = (f"/extra/ablate/ablate_bench.py --cases /bench/cases.json --size all --passes {passes} "
           f"--out /runs/{{name}}.json" + (f" --linger-s {linger}" if linger else ""))
    if arm == "ref":
        return f"client {name} build:{args.ref_build} {cmd}"
    return f"DMK_DOCKER_EXTRA=--env-file={remote_abl}/{arm}.env client {name} build:{args.build} {cmd}"


def tool_line(name, arm, sizes="8,32,160,1100,3200", reps=None):
    cmd = f"bench -m /models/{args.model} -ngl 999 -fa auto -b 2048 -ub 2048 -p {sizes} -n 0 -r {reps or args.llb_reps} -o json"
    if arm == "ref":
        return f"tool {name} build:{args.ref_build} {cmd}"
    return f"DMK_WRAP=/opt/dmk/abl/{arm}.sh tool {name} build:{args.build} {cmd}"


def blocked(prefix, arms, blocks, seed, mk):
    rng = random.Random(seed)
    lines = []
    for b in range(1, blocks + 1):
        order = list(arms)
        rng.shuffle(order)
        for arm in order:
            lines.append(mk(f"{prefix}-b{b}-{arm}", arm))
    return lines


def emit(fn, header, lines):
    write(f"{args.out}/sched/{fn}", "".join(f"# {h}\n" for h in header) + "\n".join(lines) + "\n")
    print(f"{fn}: {len(lines)} runs")


def stage1_lines():
    """Every arm once, shuffled. The reference build (ref) also runs first, in the middle and last, so drift shows."""
    rng = random.Random(args.seed)
    order = [a for a in all_arms if a != "ref"]
    rng.shuffle(order)
    mid = len(order) // 2
    lines = [client_line("abl-s1-b1-ref-start", "ref", args.passes)]
    for i, arm in enumerate(order):
        if i == mid:
            lines.append(client_line("abl-s1-b1-ref-mid", "ref", args.passes))
        lines.append(client_line(f"abl-s1-b1-{arm}", arm, args.passes))
    lines.append(client_line("abl-s1-b1-ref-end", "ref", args.passes))
    return lines


all_arms = list(ARMS)
core_arms = [a for a, (_, t) in ARMS.items() if t in ("ctl", "core")]
gen = f"make_ablate_runs.py --build {args.build} --passes {args.passes} --blocks-core {args.blocks_core} --seed {args.seed}"
emit("abl-s1.txt", [gen, "stage 1: every arm once, shuffled; ref (build b12004-sm86) also first, in the middle and last, to show drift; requests: all 48 cases"],
     stage1_lines())
emit("abl-s2.txt", [gen, f"stage 2: controls and core arms, {args.blocks_core} more blocks (new shuffles)"],
     blocked("abl-s2", core_arms, args.blocks_core, args.seed + 1, lambda n, a: client_line(n, a, args.passes)))
emit("abl-ops.txt", [gen, "per-op wall times (DMK_ABLATE_SYNC=1): sync alone, and sync with every node skipped (overhead floor)"],
     [client_line(f"abl-ops-{a}", a, 2, linger=3) for a in OPS_ARMS])
emit("abl-llb.txt", [gen, "llama bench, prompt sizes of a short parent / suffix, medium and long request; same flags as production"],
     blocked("abl-llb", LLB_ARMS, 1, args.seed + 2, tool_line))
emit("abl-smoke.txt", [gen, "pipeline check before the long schedules: env-file and wrapper reach the process, 'all' runs through, DMK_OPS dump arrives"],
     [client_line("abl-smoke-base", "base", 1),
      client_line("abl-smoke-all", "all", 1),
      client_line("abl-smoke-all_timing", "all_timing", 1),
      client_line("abl-smoke-sync", "sync", 1, linger=3),
      tool_line("abl-smoke-llb-mmq", "mmq", sizes="8,160", reps=1)])
