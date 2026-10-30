#!/usr/bin/env python3
"""A tiny fake of the Gitea push-mirror API, for the gate tests.

It serves two endpoints on 127.0.0.1 with basic authentication:
  GET  /api/v1/repos/<owner>/<name>/push_mirrors
  POST /api/v1/repos/<owner>/<name>/push_mirrors-sync

A sync request starts a background job. The mode decides what the job does.

Modes
  ok        push heads and tags from the Gitea bare repo to the GitHub bare repo, then set last_update
  nopush    set last_update but push nothing (the heads then differ)
  error     set last_update and last_error, push nothing
  nomove    never change last_update
  nomirror  report an empty mirror list

The server logs one line a request: "<method> <path> <auth ok|auth bad>".
It never logs a password. It writes its port to --port-file when it is ready.
"""
import argparse
import base64
import datetime
import http.server
import json
import subprocess
import threading
import time

parser = argparse.ArgumentParser()
parser.add_argument("--port-file", required=True)
parser.add_argument("--log", required=True)
parser.add_argument("--user", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--repo", required=True)
parser.add_argument("--gitea-git", required=True)
parser.add_argument("--github-git", required=True)
parser.add_argument("--mode", default="ok")
parser.add_argument("--delay", type=float, default=0.6)
args = parser.parse_args()

state = {"last_update": "2026-01-01T00:00:00Z", "last_error": ""}
lock = threading.Lock()
expected = "Basic " + base64.b64encode(f"{args.user}:{args.password}".encode()).decode()
base = f"/api/v1/repos/{args.repo}"


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def job():
    time.sleep(args.delay)
    if args.mode == "nomove":
        return
    if args.mode == "ok":
        subprocess.run(
            ["git", f"--git-dir={args.gitea_git}", "push", "--quiet", "--prune", args.github_git,
             "+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*"],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with lock:
        state["last_update"] = now()
        state["last_error"] = "push failed: remote rejected" if args.mode == "error" else ""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        pass

    def record(self, ok):
        with open(args.log, "a") as fh:
            fh.write(f"{self.command} {self.path} {'auth ok' if ok else 'auth bad'}\n")

    def reply(self, code, body=b""):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authed(self):
        ok = self.headers.get("Authorization") == expected
        self.record(ok)
        if not ok:
            self.reply(401, b'{"message":"bad credentials"}')
        return ok

    def do_GET(self):
        if not self.authed():
            return
        if self.path == f"{base}/push_mirrors":
            if args.mode == "nomirror":
                self.reply(200, b"[]")
                return
            with lock:
                body = json.dumps([{
                    "repo_name": args.repo.split("/")[1], "remote_name": "remote_mirror_test",
                    "remote_address": "https://github.example.invalid/owner/repo.git",
                    "interval": "8h0m0s", "sync_on_commit": False,
                    "last_update": state["last_update"], "last_error": state["last_error"],
                }]).encode()
            self.reply(200, body)
        else:
            self.reply(404, b"{}")

    def do_POST(self):
        if not self.authed():
            return
        if self.path == f"{base}/push_mirrors-sync":
            threading.Thread(target=job, daemon=True).start()
            self.reply(200, b"{}")
        else:
            self.reply(404, b"{}")


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(args.port_file, "w") as fh:
    fh.write(str(server.server_address[1]))
server.serve_forever()
