"""Browse loadgen dry-run logdirs in a TensorBoard-style web page.

Serves index.html plus a small JSON API over a tree of logdirs, each the output
of loadgen/dry_run.sh (run_info.txt, arrivals.jsonl, loadgen.csv, metrics.csv).
Any directory under --logdir holding a run_info.txt is a run, so sweep dirs with
nested runs are listed too. Files are re-read whenever they change, so a run
that is still in progress shows up live. Standard library only.

Usage:
    python vistool/server.py                       # ~/scratch/logs on :6006
    python vistool/server.py --logdir ~/scratch/logs --port 6006 --host 0.0.0.0
    nohup python vistool/server.py > ~/scratch/vistool.log 2>&1 &
"""

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
# Unix timestamps are kept to the ms, durations to the us, ratios to 1e-4.
TURN_TIME_COLUMNS = ["t_send", "t_end"]
TURN_LATENCY_COLUMNS = [
    "turn_e2e_latency",
    "turn_queuing_latency",
    "turn_predicted_queuing_latency",
    "turn_prefill_latency",
    "turn_decode_latency",
    "turn_toolcall_latency",
    "server_e2e_latency",
]
TURN_RATIO_COLUMNS = ["kvc_hit_ratio"]
TURN_COUNT_COLUMNS = ["prompt_tokens", "cached_tokens"]
# pin_decisions.csv (queuing_predictor sidecar, --pin-mode cost): kept as rows.
PIN_TEXT_COLUMNS = ["instance", "rid", "decision", "reason", "unpin_status"]
PIN_NUMBER_COLUMNS = [
    "t_unix", "replica_id", "turn_id", "toolcall_s", "path_tokens", "shared_tokens",
    "unique_tokens", "reuse_tokens", "next_input_len", "num_running", "num_waiting",
    "kt_s", "delta_mean_finish_s", "pin_mean_finish_s", "nopin_mean_finish_s",
    "pin_sum_wait_s", "nopin_sum_wait_s", "pin_r_wait_s", "nopin_r_wait_s",
    "nopin_r_prefix", "nopin_reprefill_tokens", "nopin_reprefill_s", "num_requests",
    "num_truncated", "snapshot_age_s", "decide_ms", "handler_ms",
]
GAUGE_COLUMNS = [
    "num_queue_reqs",
    "num_running_reqs",
    "token_usage",
    "full_token_usage",
    "mamba_usage",
    "pinned_token_usage",
    "pinned_mamba_usage",
    "num_retracted_requests_total",
    "pin_refused_over_budget",
    "pin_refused_nothing_cached",
    "pin_refused_already_finished",
    "pin_force_released_admission",
    "pin_force_released_decode",
    "pin_force_released_mamba_headroom",
]
RUN_FILES = ["run_info.txt", "arrivals.jsonl", "loadgen.csv", "metrics.csv", "pin_decisions.csv"]
# A run without an "end" line whose files are all older than this was killed.
STALE_AFTER_S = 600
# Runs nest at most this deep under --logdir (e.g. sweep_x/r12_y).
MAX_RUN_DEPTH = 3
GZIP_MIN_BYTES = 2048

# corpus path -> (mtime, {session_id: (number of turns, first turn_index)})
_corpus_cache: dict = {}


def _log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _float(s, ndigits: int):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return round(v, ndigits) if math.isfinite(v) else None


def _read_csv(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        # A row being written right now can be truncated; drop short rows.
        return [r for r in csv.DictReader(f) if None not in r.values()]


def _read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                # Blank, or the last line still being written.
                continue
    return rows


def _read_run_info(logdir: str) -> dict:
    info = {}
    path = os.path.join(logdir, "run_info.txt")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                key, sep, value = line.partition(":")
                if sep:
                    info[key.strip()] = value.strip()
    return info


def _corpus_turn_counts(path: str) -> dict:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    cached = _corpus_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    counts = {}
    for s in _read_jsonl(path):
        turns = s["turns"]
        counts[s["session_id"]] = (len(turns), turns[0]["turn_index"] if turns else 0)
    _corpus_cache[path] = (mtime, counts)
    return counts


def _resolve_corpus(logdir: str, info: dict):
    corpus = info.get("corpus", "")
    candidates = [os.path.join(logdir, "corpus_synthetic.jsonl")]
    if corpus and corpus != "synthetic":
        candidates.insert(0, os.path.expanduser(corpus))
    return next((p for p in candidates if os.path.exists(p)), None)


def _find_run_dirs(root: str) -> list:
    found = []

    def walk(path: str, depth: int):
        try:
            entries = os.listdir(path)
        except OSError:
            return
        if "run_info.txt" in entries:
            found.append(path)
            return
        if depth < MAX_RUN_DEPTH:
            for e in entries:
                sub = os.path.join(path, e)
                if os.path.isdir(sub):
                    walk(sub, depth + 1)

    for e in os.listdir(root):
        sub = os.path.join(root, e)
        if os.path.isdir(sub):
            walk(sub, 1)
    return found


def _run_summary(*, root: str, logdir: str) -> dict:
    files = set(os.listdir(logdir))
    info = _read_run_info(logdir)
    mtime = max(
        [os.path.getmtime(logdir)]
        + [os.path.getmtime(os.path.join(logdir, f)) for f in files]
    )
    if "end" in info:
        status = "finished"
    elif time.time() - mtime > STALE_AFTER_S:
        status = "aborted"
    else:
        status = "running"
    return {
        "name": os.path.relpath(logdir, root),
        "mtime": mtime,
        "status": status,
        "has_loadgen": "loadgen.csv" in files,
        "has_metrics": "metrics.csv" in files,
        "start": info.get("start", ""),
    }


def list_runs(root: str) -> list:
    runs = [_run_summary(root=root, logdir=d) for d in _find_run_dirs(root)]
    runs.sort(key=lambda r: (r["start"], r["name"]), reverse=True)
    return runs


def resolve_run(*, root: str, name: str):
    """Map a run name from the client to its logdir, refusing anything outside root."""
    if not name:
        return None
    logdir = os.path.realpath(os.path.join(root, name))
    if not logdir.startswith(root + os.sep):
        return None
    if not os.path.isfile(os.path.join(logdir, "run_info.txt")):
        return None
    return logdir


def run_etag(logdir: str) -> str:
    info = _read_run_info(logdir)
    paths = [os.path.join(logdir, f) for f in RUN_FILES]
    paths.append(_resolve_corpus(logdir, info) or "")
    parts = []
    for p in paths:
        try:
            st = os.stat(p)
            parts.append(f"{p}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{p}:-")
    return '"' + hashlib.sha1("|".join(parts).encode()).hexdigest()[:20] + '"'


def _load_turns(logdir: str) -> dict:
    """loadgen.csv as columns: {name: [value per turn]}."""
    cols = {
        "replica_id": [],
        "session_id": [],
        "turn_id": [],
        "status": [],
        "error": [],
        **{c: [] for c in TURN_TIME_COLUMNS + TURN_LATENCY_COLUMNS + TURN_RATIO_COLUMNS},
        **{c: [] for c in TURN_COUNT_COLUMNS},
    }
    for r in _read_csv(os.path.join(logdir, "loadgen.csv")):
        try:
            replica_id, turn_id = int(r["replica_id"]), int(r["turn_id"])
        except ValueError:
            continue
        cols["replica_id"].append(replica_id)
        cols["session_id"].append(r["session_id"])
        cols["turn_id"].append(turn_id)
        cols["status"].append(r["status"])
        cols["error"].append(r.get("error") or "")
        for c in TURN_TIME_COLUMNS:
            cols[c].append(_float(r.get(c), 3))
        for c in TURN_LATENCY_COLUMNS:
            cols[c].append(_float(r.get(c), 6))
        for c in TURN_RATIO_COLUMNS:
            cols[c].append(_float(r.get(c), 4))
        for c in TURN_COUNT_COLUMNS:
            cols[c].append(_float(r.get(c), 0))
    return cols


def _load_pins(logdir: str) -> list:
    """pin_decisions.csv as one dict per decision; empty when the run had none."""
    rows = []
    for r in _read_csv(os.path.join(logdir, "pin_decisions.csv")):
        row = {c: r.get(c) or None for c in PIN_TEXT_COLUMNS}
        row.update({c: _float(r.get(c), 4) for c in PIN_NUMBER_COLUMNS})
        rows.append(row)
    return rows


def _load_sessions(*, arrivals: list, turns: dict, turn_counts: dict) -> list:
    def session(replica_id, session_id, arrival_offset):
        total, first = turn_counts.get(session_id, (None, None))
        return {
            "replica_id": replica_id,
            "session_id": session_id,
            "arrival_offset": arrival_offset,
            "total_turns": total,
            "first_turn": first,
        }

    # Sessions are keyed by replica: the same corpus session can be replayed twice.
    sessions = {
        a["replica_id"]: session(a["replica_id"], a["session_id"], a.get("arrival_offset"))
        for a in arrivals
    }
    for replica_id, session_id in zip(turns["replica_id"], turns["session_id"]):
        if replica_id not in sessions:
            sessions[replica_id] = session(replica_id, session_id, None)
    return sorted(sessions.values(), key=lambda s: s["replica_id"])


def _load_metrics(logdir: str) -> dict:
    metrics = {}
    for r in _read_csv(os.path.join(logdir, "metrics.csv")):
        t = _float(r.get("t"), 3)
        if r.get("error") or t is None:
            continue
        series = metrics.setdefault(r["endpoint"], {"t": [], **{c: [] for c in GAUGE_COLUMNS}})
        series["t"].append(t)
        for c in GAUGE_COLUMNS:
            # Older metrics.csv files lack the newer gauges; those stay None.
            series[c].append(_float(r.get(c), 4))
    return metrics


def load_run(logdir: str) -> dict:
    info = _read_run_info(logdir)
    corpus_path = _resolve_corpus(logdir, info)
    turn_counts = _corpus_turn_counts(corpus_path) if corpus_path else {}
    turns = _load_turns(logdir)
    sessions = _load_sessions(
        arrivals=_read_jsonl(os.path.join(logdir, "arrivals.jsonl")),
        turns=turns,
        turn_counts=turn_counts,
    )
    del turns["session_id"]  # carried by sessions, keyed by replica_id
    return {
        "info": info,
        "corpus": corpus_path,
        "sessions": sessions,
        "turns": turns,
        "metrics": _load_metrics(logdir),
        "pins": _load_pins(logdir),
    }


class Handler(BaseHTTPRequestHandler):
    root = ""

    def log_message(self, fmt, *args):
        pass

    def _route(self):
        """Return (status, body, content type, extra headers)."""
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return 200, f.read(), "text/html; charset=utf-8", {}
        if url.path == "/api/runs":
            body = json.dumps({"root": self.root, "runs": list_runs(self.root)})
            return 200, body.encode(), "application/json", {}
        if url.path == "/api/run":
            name = parse_qs(url.query).get("name", [""])[0]
            logdir = resolve_run(root=self.root, name=name)
            if logdir is None:
                return 404, json.dumps({"error": f"no such run: {name}"}).encode(), "application/json", {}
            etag = run_etag(logdir)
            if self.headers.get("If-None-Match") == etag:
                return 304, b"", "application/json", {"ETag": etag}
            body = json.dumps(load_run(logdir), separators=(",", ":"))
            return 200, body.encode(), "application/json", {"ETag": etag}
        return 404, b"not found", "text/plain", {}

    def do_GET(self):
        t0 = time.perf_counter()
        try:
            code, body, ctype, headers = self._route()
        except Exception as e:
            _log(f"ERROR {self.path}\n{traceback.format_exc()}")
            code, ctype, headers = 500, "application/json", {}
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
        raw_len = len(body)
        if len(body) >= GZIP_MIN_BYTES and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, compresslevel=5)
            headers["Content-Encoding"] = "gzip"
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            for k, v in headers.items():
                self.send_header(k, v)
            if code != 304:
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if code != 304:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError) as e:
            _log(f"{code} {self.path} client went away: {type(e).__name__}")
            return
        dt_ms = (time.perf_counter() - t0) * 1e3
        _log(f"{code} {self.path} {raw_len}B->{len(body)}B {dt_ms:.0f}ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logdir", default="~/scratch/logs")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
    args = parser.parse_args()

    Handler.root = os.path.realpath(os.path.expanduser(args.logdir))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    _log(f"serving {Handler.root} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
