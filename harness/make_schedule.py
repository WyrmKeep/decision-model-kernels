#!/usr/bin/env python3
"""Write a blocked, shuffled A/B schedule for run_schedule.sh.

Each block holds one run of each arm in a random order (fixed seed), so slow drift
of the host affects all arms equally. Run names: <prefix>-b<block>-<tag>.

Example:
  make_schedule.py --prefix p1-bench --blocks 6 --seed 1 \
      --arm S=stock --arm D=build:b12004-sm86 \
      --kind bench --args "--write-reference {name}"
  make_schedule.py --prefix p1-llb --blocks 4 --seed 2 --arm S=stock --arm D=build:b12004-sm86 \
      --kind tool --args "bench -m /models/{model} -ngl 999 -fa 1 -b 2048 -ub 2048 -p 64,256,3000 -n 0 -r 5 -o json"
"""
import argparse
import random

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True)
ap.add_argument("--blocks", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arm", action="append", required=True, help="TAG=ARM, e.g. S=stock")
ap.add_argument("--kind", choices=["bench", "tool"], required=True)
ap.add_argument("--args", default="", help="arguments; {name} and {model} are filled in")
ap.add_argument("--model", default="h2o-lightning-4b-rq-Q4_0-mix8.gguf")
a = ap.parse_args()

arms = [x.split("=", 1) for x in a.arm]
rng = random.Random(a.seed)
print(f"# make_schedule.py --prefix {a.prefix} --blocks {a.blocks} --seed {a.seed} "
      f"{' '.join('--arm ' + x for x in a.arm)} --kind {a.kind}")
for b in range(1, a.blocks + 1):
    order = arms[:]
    rng.shuffle(order)
    for tag, arm in order:
        name = f"{a.prefix}-b{b}-{tag}"
        args = a.args.format(name=name, model=a.model)
        print(f"{a.kind} {name} {arm} {args}".rstrip())
