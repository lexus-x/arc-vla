"""High-Throughput 8-Worker Parallel Multi-Rate Evaluation Matrix on Blackwell.

Launches 8 parallel worker processes across target evaluation arms
for Flow Matching, DeepONet-v2, and ASRC across 10 Hz, 20 Hz, and 40 Hz on LIBERO-Spatial.
Max throughput on RTX PRO 6000 (98 GB VRAM).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CAMPAIGN_DIR = "/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
PYTHON_BIN = "/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

OUT_DIR = os.path.join(CAMPAIGN_DIR, "results_matched8300_parallel_eval")
os.makedirs(OUT_DIR, exist_ok=True)
STATUS_FILE = os.path.join(OUT_DIR, "eval_status.json")

# Target evaluation arms (Comprehensive Matrix)
TARGET_ARMS = [
    # 1. Flow Matching (8,300 steps)
    "flow8300_native_20env",
    "flow8300_naive_10env",
    "flow8300_naive_40env",
    "flow8300_mag_spline_10env",
    "flow8300_cadmag_spline_40env",
    # 2. Vanilla DeepONet (til_s0 - 8,300 steps)
    "til_native_20env",
    "til_naive_40env",
    "til_folding_40env",
    "til_zoh_40env",
    "til_cadmag_spline_40env",
    # 3. Canonical DeepONet-v2 (v2_spatial - CrossAttnPool + Fourier)
    "v2_spatial_native_20env",
    "v2_spatial_naive_10env",
    "v2_spatial_naive_40env",
    "v2_spatial_mag_spline_10env",
    "v2_spatial_cadmag_spline_40env",
    # 4. Continuous DeepONet (asrc_s0 - 8,300 steps)
    "asrc_native_20env",
    "asrc_folding_10env",
    "asrc_mag_folding_10env",
    "asrc_naive_40env",
    "asrc_cadmag_folding_40env",
    "asrc_cadmag_spline_40env",
]


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{ts}] {msg}"
    print(formatted, flush=True)
    with open(os.path.join(OUT_DIR, "parallel_eval.log"), "a", encoding="utf-8") as f:
        f.write(formatted + "\n")


def run_single_arm(arm_name: str) -> dict:
    arm_out = os.path.join(OUT_DIR, f"arm_{arm_name}")
    os.makedirs(arm_out, exist_ok=True)
    log_file = os.path.join(arm_out, "eval.log")
    summary_file = os.path.join(arm_out, "multirate_honest.json")

    cmd = [
        PYTHON_BIN, "-u",
        "-c", f"""
import sys, os
sys.path.insert(0, '{CAMPAIGN_DIR}')
import evaluate_multirate_honest as E

FLOW8300_CKPT = '/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300'
if os.path.exists(FLOW8300_CKPT):
    E.MODELS['flow8300'] = ('flow', FLOW8300_CKPT, None, None)
    E.ARMS['flow8300_native_20env'] = ('flow8300', 20, None, None)
    E.ARMS['flow8300_naive_40env'] = ('flow8300', 40, None, None)
    E.ARMS['flow8300_naive_10env'] = ('flow8300', 10, None, None)
    E.ARMS['flow8300_cadmag_spline_40env'] = ('flow8300', 40, None, 'spline+magscale+cadence')
    E.ARMS['flow8300_mag_spline_10env'] = ('flow8300', 10, None, 'spline+magscale')

# v2_spatial arms
E.ARMS['v2_spatial_naive_40env'] = ('v2_spatial', 40, None, None)
E.ARMS['v2_spatial_naive_10env'] = ('v2_spatial', 10, None, None)
E.ARMS['v2_spatial_mag_spline_10env'] = ('v2_spatial', 10, None, 'spline+magscale')

sys.argv = ['evaluate_multirate_honest.py', '--arms', '{arm_name}', '--trials_per_task', '5', '--suite', 'libero_spatial', '--out', '{arm_out}']
E.main()
"""
    ]

    t0 = time.time()
    log(f"-> STARTING Worker for Arm: '{arm_name}'")
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            cwd=CAMPAIGN_DIR,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl"},
        )

    elapsed = time.time() - t0
    success_rate = "UNKNOWN"
    total_trials = 0
    total_successes = 0

    if os.path.exists(summary_file):
        try:
            data = json.loads(Path(summary_file).read_text(encoding="utf-8"))
            if arm_name in data:
                entry = data[arm_name]
                total_trials = entry.get("trials", 0)
                total_successes = entry.get("successes", 0)
                sr_pct = (total_successes / total_trials * 100) if total_trials > 0 else 0
                success_rate = f"{sr_pct:.1f}% ({total_successes}/{total_trials})"
        except Exception:
            pass

    status = "COMPLETE" if proc.returncode == 0 else f"ERROR (code {proc.returncode})"
    log(f"<- FINISHED Arm: '{arm_name}' [{status}] in {elapsed:.1f}s | Result: {success_rate}")

    return {
        "arm": arm_name,
        "status": status,
        "success_rate": success_rate,
        "successes": total_successes,
        "trials": total_trials,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    max_workers = 8
    log("=================================================================")
    log(f"Blackwell 8-Worker Parallel Evaluation Matrix (Flow, DeepONet-v2, ASRC)")
    log(f"Total Arms: {len(TARGET_ARMS)} | Concurrent Workers: {max_workers}")
    log(f"Output Directory: {OUT_DIR}")
    log("=================================================================")

    start_matrix_time = time.time()
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_arm = {executor.submit(run_single_arm, arm): arm for arm in TARGET_ARMS}
        for future in as_completed(future_to_arm):
            arm = future_to_arm[future]
            try:
                res = future.result()
                results[arm] = res
                Path(STATUS_FILE).write_text(json.dumps(results, indent=2), encoding="utf-8")
            except Exception as e:
                log(f"EXCEPTION in future for arm '{arm}': {e}")
                results[arm] = {"arm": arm, "status": "EXCEPTION", "error": str(e)}
                Path(STATUS_FILE).write_text(json.dumps(results, indent=2), encoding="utf-8")

    total_time = time.time() - start_matrix_time
    log("=================================================================")
    log(f"ALL {len(TARGET_ARMS)} ARMS COMPLETED PARALLEL EVALUATION IN {total_time/60:.1f} MINUTES!")
    log("=================================================================")


if __name__ == "__main__":
    main()
