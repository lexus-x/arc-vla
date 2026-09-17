"""Master Multi-Rate Evaluation Runner for Flow Matching and DeepONet on Blackwell.

Runs matched-budget 8,300-step evaluations across 10 Hz, 20 Hz, and 40 Hz
for Flow Matching and DeepONet (raw/unmodified, native, and resampled arms).
Logs progress continuously and outputs structured JSON scoreboards.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Blackwell experiment directory
CAMPAIGN_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN_DIR)

import evaluate_multirate_honest as E

# Register flow8300 checkpoint
FLOW8300_CKPT = "/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300"
if os.path.exists(FLOW8300_CKPT):
    E.MODELS["flow8300"] = ("flow", FLOW8300_CKPT, None, None)
    E.ARMS["flow8300_native_20env"] = ("flow8300", 20, None, None)
    E.ARMS["flow8300_naive_40env"] = ("flow8300", 40, None, None)
    E.ARMS["flow8300_naive_10env"] = ("flow8300", 10, None, None)
    E.ARMS["flow8300_cadmag_spline_40env"] = ("flow8300", 40, None, "spline+magscale+cadence")
    E.ARMS["flow8300_mag_spline_10env"] = ("flow8300", 10, None, "spline+magscale")

# Arm evaluation suite
TARGET_ARMS = [
    # 1. Flow Matching @ 10 Hz, 20 Hz, 40 Hz
    "flow8300_native_20env",
    "flow8300_naive_10env",
    "flow8300_naive_40env",
    "flow8300_mag_spline_10env",
    "flow8300_cadmag_spline_40env",
    # 2. DeepONet (til - vanilla no Fourier) @ 10 Hz, 20 Hz, 40 Hz
    "til_native_20env",
    "til_naive_40env",
    "til_folding_40env",
    "til_zoh_40env",
    "til_cadmag_spline_40env",
    # 3. DeepONet (asrc - with Fourier) @ 10 Hz, 20 Hz, 40 Hz
    "asrc_native_20env",
    "asrc_folding_10env",
    "asrc_mag_folding_10env",
    "asrc_naive_40env",
    "asrc_cadmag_folding_40env",
    "asrc_cadmag_spline_40env",
]

OUT_DIR = os.path.join(CAMPAIGN_DIR, "results_matched8300_master_eval")
os.makedirs(OUT_DIR, exist_ok=True)
STATUS_FILE = os.path.join(OUT_DIR, "eval_status.json")


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(os.path.join(OUT_DIR, "master_eval.log"), "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def main():
    log("=================================================================")
    log("Starting Blackwell High-Throughput Multi-Rate Evaluation Matrix")
    log(f"Target Arms ({len(TARGET_ARMS)}): {TARGET_ARMS}")
    log(f"Output Directory: {OUT_DIR}")
    log("=================================================================")

    status_records = {}
    if os.path.exists(STATUS_FILE):
        try:
            status_records = json.loads(Path(STATUS_FILE).read_text(encoding="utf-8"))
        except Exception:
            pass

    for idx, arm in enumerate(TARGET_ARMS):
        if arm not in E.ARMS:
            log(f"WARNING: Arm '{arm}' not found in ARMS registry. Skipping.")
            continue

        arm_out_dir = os.path.join(OUT_DIR, f"arm_{arm}")
        summary_file = os.path.join(arm_out_dir, "summary.json")

        if arm in status_records and status_records[arm].get("status") == "COMPLETE" and os.path.exists(summary_file):
            log(f"[{idx+1}/{len(TARGET_ARMS)}] Arm '{arm}' already COMPLETE ({status_records[arm].get('success_rate')}). Skipping.")
            continue

        log(f"-----------------------------------------------------------------")
        log(f"[{idx+1}/{len(TARGET_ARMS)}] Launching Evaluation for Arm: '{arm}' (n=50/task)")
        start_time = time.time()
        
        status_records[arm] = {
            "status": "RUNNING",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        Path(STATUS_FILE).write_text(json.dumps(status_records, indent=2), encoding="utf-8")

        try:
            # Configure and run evaluation
            sys.argv = [
                "evaluate_multirate_honest.py",
                "--arms", arm,
                "--trials_per_task", "5",  # n=50 rollouts total across 10 tasks
                "--suite", "libero_spatial",
                "--out", arm_out_dir,
            ]
            E.main()

            elapsed = time.time() - start_time
            # Read output summary
            sr_str = "UNKNOWN"
            if os.path.exists(summary_file):
                try:
                    s_data = json.loads(Path(summary_file).read_text(encoding="utf-8"))
                    sr_str = f"{s_data.get('success_rate', 'N/A')}"
                except Exception:
                    pass

            log(f"[{idx+1}/{len(TARGET_ARMS)}] Arm '{arm}' FINISHED in {elapsed:.1f}s | Success Rate: {sr_str}")
            status_records[arm] = {
                "status": "COMPLETE",
                "success_rate": sr_str,
                "elapsed_s": round(elapsed, 1),
                "finish_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            log(f"ERROR on Arm '{arm}': {e}")
            status_records[arm] = {
                "status": "ERROR",
                "error": str(e),
                "finish_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

        Path(STATUS_FILE).write_text(json.dumps(status_records, indent=2), encoding="utf-8")

    log("=================================================================")
    log("All Evaluations in Matrix Completed!")
    log("=================================================================")


if __name__ == "__main__":
    main()
