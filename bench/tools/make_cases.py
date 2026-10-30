#!/usr/bin/env python3
"""Build cases.json, the benchmark cases.

The file has 48 cases with 100 labeled decisions:

- 32 short cases, read from dectest/cases.json (written by hand);
- 8 medium and 8 long synthetic syslog cases, made here with a fixed seed.

The label of each syslog case is known by construction: the script puts a
known fault block (or none) into a log of routine lines.

Usage:
    python3 tools/make_cases.py            # write cases.json next to this tool
    python3 tools/make_cases.py --check    # fail if cases.json differs

The output is part of the fixed harness. Do not change the seed, the line
templates or the order of the random draws. Any change moves the cases and
makes older results incomparable.
"""
import argparse
import json
import pathlib
import random
import sys

BENCH = pathlib.Path(__file__).resolve().parent.parent
SEED = 20261030
HOST = "Homelab"  # generic host name in every syslog line
rng = random.Random(SEED)


def wire(questions):
    """Convert the smoke-test question types to the wire types of the server."""
    out = {}
    for qid, q in questions.items():
        q = dict(q)
        if q["type"] == "bool":
            q["type"] = "noul"
        out[qid] = q
    return out


def ts(index):
    seconds = 14 * 3600 + 30 * 60 + index * 20 + rng.randrange(20)
    return f"Oct 30 {seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def hexid(n=12):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def pid():
    return rng.randrange(1000, 60000)


# Addresses use the documentation ranges of RFC 5737 only. The tokenizer of the
# model splits numbers into single digits, so the number of digits sets the
# token count. Each address gets the digit count of the first (private) version
# of the benchmark, so each case keeps its token count.
DOC_PREFIXES = (("192.0.2", 5), ("203.0.113", 7), ("198.51.100", 8))
# No IPv4 documentation address has 12 digits. For that case the address is a
# documentation IPv6 address (2001:db8::/32) that has the same token count (15
# tokens) as a 12-digit IPv4 address with three dots.
DOC_IPV6 = (
    "2001:db8:c325:c5::9", "2001:db8:d47e:ea::aba", "2001:db8:acac:bd::5ff",
    "2001:db8:d9ee:d2::ee6", "2001:db8:7dc:66::acd", "2001:db8:2ada:af::3dd",
    "2001:db8:fadf:e5::8b0", "2001:db8:b6ff:dd::ab0",
)


def doc_addr(a, b, c):
    """Documentation address with 3 + len(a) + len(b) + len(c) digits."""
    want = 3 + len(str(a)) + len(str(b)) + len(str(c))
    fits = [(p, n) for p, n in DOC_PREFIXES if 1 <= want - n <= 3]
    if not fits:
        return DOC_IPV6[c % len(DOC_IPV6)]
    prefix, n = fits[b % len(fits)]
    last = (1 + c % 9, 10 + c % 90, 100 + c % 155)[want - n - 1]
    return f"{prefix}.{last}"


def sshd_accepted():
    # Keep the order of the draws: pid, three address numbers, port, key hash.
    p = pid()
    a, b, c = rng.randrange(64, 128), rng.randrange(256), rng.randrange(256)
    port = rng.randrange(30000, 65000)
    key = hexid(43)
    return (f"{HOST} sshd-session[{p}]: Accepted publickey for root from {doc_addr(a, b, c)} "
            f"port {port} ssh2: ED25519 SHA256:{key}")


BENIGN = [
    lambda: f"{HOST} crond[{pid()}]: (root) CMD (/usr/local/emhttp/plugins/dynamix/scripts/monitor &> /dev/null)",
    lambda: f"{HOST} emhttpd: read SMART /dev/sd{rng.choice('bcdefg')}",
    lambda: f"{HOST} emhttpd: spinning down /dev/sd{rng.choice('bcdefg')}",
    lambda: f"{HOST} kernel: docker0: port {rng.randrange(1, 40)}(veth{hexid(7)}) entered forwarding state",
    lambda: f"{HOST} kernel: br-{hexid(12)}: port {rng.randrange(1, 9)}(veth{hexid(7)}) entered blocking state",
    lambda: f"{HOST} kernel: eth0: renamed from veth{hexid(7)}",
    lambda: f'{HOST} dockerd[{pid()}]: time="2026-10-30T14:{rng.randrange(60):02d}:{rng.randrange(60):02d}.{rng.randrange(10**6):06d}Z" level=info msg="ignoring event" container={hexid(64)} module=libcontainerd namespace=moby topic=/tasks/delete type="*events.TaskDelete"',
    sshd_accepted,
    lambda: f"{HOST} tailscaled[{pid()}]: magicsock: derp-{rng.choice([1, 2, 4, 10])} connected; connGen=1",
    lambda: f"{HOST} tailscaled[{pid()}]: wgengine: Reconfig: configuring router",
    lambda: f"{HOST} smbd[{pid()}]: [2026/10/30 14:{rng.randrange(60):02d}:{rng.randrange(60):02d}.{rng.randrange(10**6):06d},  0] ../../source3/smbd/server.c:1746(main)",
    lambda: f"{HOST} webGUI: Successful login user root from 203.0.113.{rng.randrange(2, 250)}",
    lambda: f"{HOST} rc.docker: {rng.choice(['sonarr', 'radarr', 'tdarr', 'immich', 'backrest', 'gitea', 'plex'])}: started succesfully!",
    lambda: f"{HOST} kernel: mdcmd ({rng.randrange(30, 900)}): spindown {rng.randrange(1, 6)}",
    lambda: f"{HOST} nginx: 2026/10/30 14:{rng.randrange(60):02d}:{rng.randrange(60):02d} [info] {pid()}#{pid()}: *{rng.randrange(10**5)} client closed connection while waiting for request, client: 127.0.0.1, server: 127.0.0.1:80",
    lambda: f"{HOST} backrest[{pid()}]: task completed: snapshot main-local, {rng.randrange(20, 400)} files changed, {rng.randrange(1, 900)} MiB added",
    lambda: f"{HOST} kernel: veth{hexid(7)}: entered promiscuous mode",
    lambda: f"{HOST} ntpd[{pid()}]: kernel reports TIME_ERROR: 0x41: Clock Unsynchronized",
]

ANOMALY = {
    "storage": lambda: [
        f"{HOST} kernel: ata5.00: exception Emask 0x0 SAct 0x{hexid(6)} SErr 0x0 action 0x0",
        f"{HOST} kernel: ata5.00: failed command: READ FPDMA QUEUED",
        f"{HOST} kernel: I/O error, dev sdc, sector {rng.randrange(10**9)} op 0x0:(READ) flags 0x80700 phys_seg 1 prio class 0",
        f"{HOST} kernel: md: disk2 read error, sector={rng.randrange(10**9)}",
    ],
    "memory": lambda: [
        f"{HOST} kernel: {rng.choice(['python3', 'node', 'java', 'ffmpeg'])} invoked oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0, oom_score_adj=0",
        f"{HOST} kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=docker-{hexid(64)}.scope,mems_allowed=0",
        f"{HOST} kernel: Memory cgroup out of memory: Killed process {pid()} (python3) total-vm:{rng.randrange(10**6, 10**7)}kB, anon-rss:{rng.randrange(10**6, 4 * 10**6)}kB, file-rss:0kB",
    ],
    "network": lambda: [
        f"{HOST} kernel: igb 0000:06:00.0 eth1: igb: eth1 NIC Link is Down",
        f"{HOST} kernel: bond0: (slave eth0): link status definitely down, disabling slave",
        f"{HOST} tailscaled[{pid()}]: control: dial: dial tcp: lookup controlplane.tailscale.com: network is unreachable",
        f"{HOST} kernel: bond0: now running without any active interface!",
    ],
}

LONG_QUESTIONS = {
    "disk_error": {
        "type": "noul",
        "instructions": "Does this log show a disk, filesystem or I/O error?",
    },
    "oom_kill": {
        "type": "noul",
        "instructions": "Did the kernel kill a process because the system or a container ran out of memory?",
    },
    "worst": {
        "type": "choice",
        "instructions": "Which kind of problem is the most serious one in this log?",
        "criteria": {
            "storage": "Disk, filesystem or I/O errors",
            "memory": "Out-of-memory conditions or killed processes",
            "network": "Network links or connectivity going down",
            "none": "No problem; only routine activity",
        },
    },
}


def log_case(idx, size, kind):
    n = {"medium": 20, "long": 58}[size]
    lines = [rng.choice(BENIGN)() for _ in range(n)]
    if kind != "none":
        block = ANOMALY[kind]()
        at = rng.randrange(2, n - 2)
        lines[at:at] = block
    stamped = [f"{ts(i)} {line}" for i, line in enumerate(lines)]
    return {
        "id": f"{size}{idx:02d}-{kind}",
        "task": f"syslog-{size}",
        "state": f"Excerpt of the Unraid syslog on {HOST}:\n" + "\n".join(stamped),
        "questions": LONG_QUESTIONS,
        "expect": {
            "disk_error": kind == "storage",
            "oom_kill": kind == "memory",
            "worst": kind,
        },
    }


def build():
    short = json.loads((BENCH / "dectest" / "cases.json").read_text())
    out = []
    for c in short:
        out.append({"id": c["id"], "task": c["task"], "size": "short",
                    "state": c["state"], "questions": wire(c["questions"]),
                    "expect": c["expect"]})
    kinds = ["storage", "memory", "network", "none"]
    for size in ("medium", "long"):
        for i in range(8):
            c = log_case(i, size, kinds[i % 4])
            c["size"] = size
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(BENCH / "cases.json"))
    ap.add_argument("--check", action="store_true", help="compare with the existing file, write nothing")
    args = ap.parse_args()
    text = json.dumps(build(), indent=1) + "\n"
    out = pathlib.Path(args.out)
    if args.check:
        if not out.exists() or out.read_text() != text:
            sys.exit(f"{out} differs from the generated cases")
        print(f"{out} matches the generator")
        return
    out.write_text(text)
    cases = json.loads(text)
    print(len(cases), "cases,", sum(len(c["questions"]) for c in cases), "decisions")


if __name__ == "__main__":
    main()
