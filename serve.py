#!/usr/bin/env python3
"""serve.py — static file server + /submit POST endpoint for response capture.

GETs are served like `python3 -m http.server`.

POST /submit takes a full participant-state snapshot and writes it to
./responses/<bucket>/<participant_id>.json (overwriting). Each call is a
complete checkpoint — if a participant closes the tab mid-session, the
latest snapshot on disk has every pair they finished.

Bucket selection:
  - is_test == true                                  → responses/test/
  - study == "convergence_3dgs_interactive"          → responses/convergence/
  - study == "cross_method_side_by_side"             → responses/cross_method/
  - anything else                                    → responses/other/

Body shape:
  {
    "participant_id": "<initials>-YYYYMMDD-HHMMSS",  (matches PID_RE below)
    "initials":       "<verbatim>",
    "is_test":        true/false,
    "study":          "<study key>",
    ...
  }

Usage:
  cd /home/ubuntu/repos/interactive-3d-human-eval
  python3 serve.py 8080
"""
import json
import os
import re
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESPONSES_DIR = Path(__file__).parent / "responses"

# Strict whitelist for participant_id — used as a filename, so reject anything
# that isn't lowercase alphanum + dash. Prevents path traversal and weird FS chars.
PID_RE = re.compile(r"^[a-z0-9][a-z0-9\-]{1,60}$")

# Map study key → on-disk bucket. is_test always wins regardless of study.
STUDY_BUCKETS = {
    "convergence_3dgs_interactive": "convergence",
    "cross_method_side_by_side":    "cross_method",
}


def select_bucket(data: dict) -> str:
    if data.get("is_test"):
        return "test"
    study = data.get("study")
    return STUDY_BUCKETS.get(study, "other")


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # keep the default style but prepend timestamp ourselves for clarity
        sys.stderr.write(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} - {fmt % args}\n")

    def do_POST(self):
        if self.path != "/submit":
            self.send_error(404, "Not Found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            data = json.loads(body or b"null")
        except json.JSONDecodeError:
            self.send_error(400, "Bad JSON")
            return
        if not isinstance(data, dict):
            self.send_error(400, "expected JSON object")
            return
        pid = data.get("participant_id", "")
        if not isinstance(pid, str) or not PID_RE.match(pid):
            self.send_error(400, "invalid participant_id")
            return

        bucket = select_bucket(data)
        bucket_dir = RESPONSES_DIR / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        out  = bucket_dir / f"{pid}.json"
        # Atomic write: tmp file then rename so a crash mid-write can't corrupt the snapshot.
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, out)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True, "saved": out.name, "bucket": bucket}).encode())

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"serving on :{port}, responses -> {RESPONSES_DIR}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
