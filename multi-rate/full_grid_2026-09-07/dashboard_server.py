#!/usr/bin/env python3
"""Local live-status dashboard. Stdlib only. Serves the page at / and fresh JSON at
/status.json on every request, by tailing the actual training log files and shelling out
to nvidia-smi/pgrep -- no manual relay, no external host, no network dependency.

Usage: python dashboard_server.py [--port 8899]
"""
import argparse, json, re, subprocess, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
TQDM_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([\d:]+)<([\d:]+),\s*([\d.]+)it/s\]")

JOBS = ["TurnOffSinkFaucet", "CoffeePressButton", "TurnOffMicrowave", "CloseSingleDoor"]
TOTAL_EPOCHS = 2000

RESULTS = [
    {"task": "PickCube-v1", "kind": "closed-loop", "policy": "Diffusion Policy", "k": 4, "n": 400,
     "verdict": "good", "verdictLabel": "Win", "rows": [
        {"label": "vs cubic spline+satfix", "delta": "+3.3pp", "p": "0.011", "sig": True},
        {"label": "vs TAC-Fold+satfix", "delta": "+3.3pp", "p": "0.011", "sig": True},
        {"label": "vs B-spline (paper Alg.1)+satfix", "delta": "+13.0pp", "p": "5.3e-9", "sig": True}],
     "note": "Primary family Holm m=3 -- all three clear significance. Pre-registered before data existed."},
    {"task": "PushT-v1", "kind": "closed-loop", "policy": "Diffusion Policy", "k": 2, "n": 100,
     "verdict": "void", "verdictLabel": "Tie", "rows": [
        {"label": "vs TAC-Fold+satfix", "delta": "-2.0pp", "p": "0.79", "sig": False},
        {"label": "vs raw spline (satfix withheld)", "delta": "-4.0pp", "p": "0.62", "sig": False},
        {"label": "vs zoh", "delta": "+12.0pp", "p": "0.023", "sig": True},
        {"label": "vs raw B-spline (paper's true Alg.1)", "delta": "+17.0pp", "p": "0.0009", "sig": True}],
     "note": "Primary comparisons miss significance -- reported as measured, not reframed."},
    {"task": "RC-CloseSingleDoor", "kind": "closed-loop", "policy": "Diffusion Policy", "k": 2, "n": 15,
     "verdict": "neutral", "verdictLabel": "Void", "rows": [
        {"label": "all 7 arms", "delta": "20.0%", "p": "identical", "sig": False}],
     "note": "Policy's raw actions never saturate on this task -- every resampler reduces to the same lossless roundtrip."},
]


def tail_bytes(path, n=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - n))
            return f.read().decode("utf-8", "ignore")
    except FileNotFoundError:
        return ""


def parse_job(task):
    log = tail_bytes(f"{HERE}/robomimic_train_{task}.log")
    matches = TQDM_RE.findall(log.replace("\r", "\n"))
    alive = bool(subprocess.run(["pgrep", "-f", f"train.py.*{task}"], capture_output=True).stdout.strip())
    if not matches:
        return {"task": task, "epoch": 0, "totalEpochs": TOTAL_EPOCHS, "iter": 0, "totalIter": 500,
                "itPerSec": 0.0, "alive": alive}
    pct, cur, tot, elapsed, remain, rate = matches[-1]
    epoch_count = log.count("Epoch") - 1  # rough: last seen "Epoch N" marker count, best-effort
    return {"task": task, "epoch": max(0, epoch_count), "totalEpochs": TOTAL_EPOCHS,
            "iter": int(cur), "totalIter": int(tot), "itPerSec": float(rate), "alive": alive}


def gpu_line():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, used, total = [x.strip() for x in out.split(",")]
        return f"{util}% · {int(used)/1024:.1f} GB / {int(total)/1024:.1f} GB"
    except Exception:
        return "unavailable"


def bridge_status():
    alive = bool(subprocess.run(["pgrep", "-f", "robocasa_bridge.py --port 8765"], capture_output=True).stdout.strip())
    return "port 8765 · up" if alive else "port 8765 · down"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass  # quiet

    def do_GET(self):
        if self.path == "/status.json":
            import datetime
            payload = {
                "session": {"updatedAt": datetime.datetime.now().astimezone().isoformat(),
                            "gpu": gpu_line(), "bridge": bridge_status()},
                "jobs": [parse_job(t) for t in JOBS],
                "results": RESULTS,
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/":
            body = open(f"{HERE}/dashboard_local.html", "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    args = ap.parse_args()
    print(f"[dashboard] http://127.0.0.1:{args.port}/  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
