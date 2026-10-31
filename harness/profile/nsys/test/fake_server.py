#!/usr/bin/env python3
"""Fake /v1/systemone server for testing nsys_client.py without a GPU: answers after a delay proportional to the request
size, reports usage.input_tokens = len(state)//4, logs every Ref to --log (one JSON line per request). Stops after --max-requests."""
import argparse
import http.server
import json
import sys
import threading
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--log", required=True)
    ap.add_argument("--max-requests", type=int, default=10 ** 9)
    ap.add_argument("--ms-per-ktoken", type=float, default=1.0)
    args = ap.parse_args()
    logf = open(args.log, "w")
    count = [0]

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers["Content-Length"])
            j = json.loads(self.rfile.read(n))
            state = j["state"]
            ref = state.split("\n", 1)[0]
            ntok = len(state) // 4
            time.sleep(0.0005 + args.ms_per_ktoken * ntok / 1e6)
            out = json.dumps({"answers": ["x"] * len(j["questions"]), "usage": {"input_tokens": ntok}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            logf.write(json.dumps({"ref": ref, "n_questions": len(j["questions"]), "input_tokens": ntok}) + "\n")
            logf.flush()
            count[0] += 1
            if count[0] >= args.max_requests:
                threading.Thread(target=srv.shutdown, daemon=True).start()

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print("fake server on", args.port, flush=True)
    srv.serve_forever()
    logf.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
