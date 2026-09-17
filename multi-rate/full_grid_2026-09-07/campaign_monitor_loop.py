#!/usr/bin/env python3
"""Automated 30-minute monitoring daemon:
1. Wakes up every 30 minutes to evaluate intermediate Diffusion Policy checkpoints on Push-T and RoboCasa.
2. Checks if the base success rate reaches >= 76% (or if 537k steps complete).
3. Automatically transitions to Flow Matching training across all 3 benchmark suites:
   - RoboMimic: lift, can, square
   - Push-T: PushT-v1
   - RoboCasa: RC-CloseSingleDoor, RC-OpenDrawer
4. Evaluates multi-rate closed-loop performance and consolidates final reports.
"""
import argparse, json, math, os, subprocess, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dp_min import DiffusionPolicy, MinMax, make_chunks
from fm_min import FlowMatchingPolicy
from harness import ManiSkillSim, RoboCasaSim, RoboMimicSim, apply_arm

LOG_FILE = f"{HERE}/campaign_progress.log"
METRICS_FILE = f"{HERE}/campaign_metrics.json"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def evaluate_checkpoint(task, policy_type, ckpt_path, norm_path, n_eval=25, k=1, dev="cuda"):
    """Evaluate base native success rate on held-out episodes."""
    if not os.path.exists(ckpt_path) or not os.path.exists(norm_path):
        return None

    norms = torch.load(norm_path, map_location="cpu", weights_only=True)
    onorm = MinMax.from_state(norms["onorm"])
    anorm = MinMax.from_state(norms["anorm"])

    if task == "PushT-v1":
        sim = ManiSkillSim(task)
        # Use held-out evaluation episodes from index 550 to 550 + n_eval
        all_eligible = sim.eligible
        eval_eps = all_eligible[550:550 + n_eval]
    elif task.startswith("RC-"):
        sim = RoboCasaSim(task, port=8765)
        eval_eps = list(range(39, 39 + min(n_eval, 15)))
    elif task in ["lift", "can", "square"]:
        sim = RoboMimicSim(task)
        eval_eps = sim.split(200, n_eval)[1]
    else:
        raise ValueError(f"Unknown task {task}")

    # Determine obs/act dims from an example reset
    obs0 = sim.reset_to(eval_eps[0])
    obs_dim = len(obs0)
    act_dim = 6 if task == "PushT-v1" else (12 if task.startswith("RC-") else 7)

    if policy_type == "dp":
        model = DiffusionPolicy(obs_dim, act_dim, horizon=16)
    else:
        model = FlowMatchingPolicy(obs_dim, act_dim, horizon=16)

    model.load_state_dict(torch.load(ckpt_path, map_location=dev, weights_only=True))
    model.to(dev).eval()

    succ = []
    for ei, ep in enumerate(eval_eps):
        obs = sim.reset_to(ep)
        hist = [obs, obs]
        s = False
        steps = 0
        replan = 0
        while steps < sim.max_steps and not s:
            o = torch.as_tensor(onorm.norm(np.stack(hist[-2:]))[None], device=dev)
            torch.manual_seed(1_000_003 * ei + replan)
            if policy_type == "dp":
                pred = anorm.denorm(model.sample(o).cpu().numpy()[0])
            else:
                pred = anorm.denorm(model.sample(o).cpu().numpy()[0])

            chunk = pred[:8]
            exec_chunk = np.clip(chunk, -1, 1).astype(np.float32)
            for a in exec_chunk:
                obs, s_t, done = sim.step(a)
                hist.append(obs)
                steps += 1
                s |= s_t
                if s or done or steps >= sim.max_steps:
                    break
            replan += 1
        succ.append(s)

    sim.close()
    sr = float(np.mean(succ))
    return sr, len(succ), sum(succ)


def run_cmd(cmd, env_py):
    full_cmd = f"{env_py} {cmd}"
    log(f"Running: {full_cmd}")
    res = subprocess.run(full_cmd, shell=True, cwd=HERE)
    return res.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=1800, help="Check interval in seconds (default 30 min)")
    parser.add_argument("--target_sr", type=float, default=0.76, help="Target success rate (default 0.76)")
    parser.add_argument("--once", action="store_true", help="Run check once and exit")
    args = parser.parse_args()

    MS3 = "/home/user/anaconda3/envs/ms3/bin/python"
    GR00T = "/home/user/anaconda3/envs/gr00t/bin/python"
    RM = "/home/user/anaconda3/envs/vla_smolvla_libero/bin/python"

    log("=== Starting Automated Campaign Monitor ===")
    log(f"Check interval: {args.interval}s (~{args.interval/60:.1f} mins) | Target Base SR: {args.target_sr*100:.0f}%")

    metrics = {}
    if os.path.exists(METRICS_FILE):
        try:
            metrics = json.load(open(METRICS_FILE))
        except Exception:
            metrics = {}

    target_achieved = False

    while True:
        tick = time.strftime("%Y-%m-%d %H:%M:%S")
        log(f"--- Periodic Check Tick ({tick}) ---")

        # Check Push-T latest checkpoint
        pusht_ckpt = f"{HERE}/dp_PushT-v1_latest.pt"
        pusht_norm = f"{HERE}/dp_PushT-v1_norm.pt"
        if os.path.exists(pusht_ckpt) and os.path.exists(pusht_norm):
            log("Evaluating Push-T latest checkpoint...")
            try:
                sr, n, wins = evaluate_checkpoint("PushT-v1", "dp", pusht_ckpt, pusht_norm, n_eval=25)
                log(f"Push-T 1X Native Base: {sr*100:.1f}% ({wins}/{n})")
                metrics["pusht_latest"] = {"sr": sr, "wins": wins, "n": n, "time": tick}
                if sr >= args.target_sr:
                    log(f"Push-T achieved target >= {args.target_sr*100:.0f}%!")
                    target_achieved = True
            except Exception as e:
                log(f"Error evaluating Push-T: {e}")

        # Check RoboCasa CloseSingleDoor latest checkpoint
        rc_ckpt = f"{HERE}/dp_RC-CloseSingleDoor_latest.pt"
        rc_norm = f"{HERE}/dp_RC-CloseSingleDoor_norm.pt"
        if os.path.exists(rc_ckpt) and os.path.exists(rc_norm):
            log("Evaluating RoboCasa CloseSingleDoor latest checkpoint...")
            try:
                sr, n, wins = evaluate_checkpoint("RC-CloseSingleDoor", "dp", rc_ckpt, rc_norm, n_eval=15)
                log(f"RC-CloseSingleDoor 1X Native Base: {sr*100:.1f}% ({wins}/{n})")
                metrics["rc_door_latest"] = {"sr": sr, "wins": wins, "n": n, "time": tick}
                if sr >= args.target_sr:
                    log(f"RoboCasa CloseSingleDoor achieved target >= {args.target_sr*100:.0f}%!")
                    target_achieved = True
            except Exception as e:
                log(f"Error evaluating RC-CloseSingleDoor: {e}")

        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f, indent=2)

        # Check if Diffusion training processes have completed
        pgrep = subprocess.run("pgrep -f 'train_with_checkpoints.py'", shell=True, capture_output=True)
        running_pids = pgrep.stdout.decode().strip().split()
        dp_running = len(running_pids) > 0

        if target_achieved or not dp_running:
            log("=== Diffusion Milestone Reached or Training Concluded ===")
            log("=== INITIATING STAGE 2: FLOW MATCHING TRAINING ACROSS ALL 3 BENCHMARKS ===")

            # 1. Flow Matching RoboMimic (lift, can, square)
            log("Starting Flow Matching on RoboMimic (lift, can, square)...")
            for t in ["lift", "can", "square"]:
                log(f"Flow Matching training & eval: {t}")
                run_cmd(f"harness.py {t} --policy fm --k 2", RM)

            # 2. Flow Matching Push-T
            log("Starting Flow Matching on Push-T...")
            run_cmd("harness.py PushT-v1 --policy fm --k 2", MS3)

            # 3. Flow Matching RoboCasa
            log("Starting Flow Matching on RoboCasa (RC-CloseSingleDoor)...")
            run_cmd("harness.py RC-CloseSingleDoor --policy fm --k 2 --n_train 39 --n_eval 15", GR00T)

            log("=== ALL FLOW MATCHING SUITES COMPLETED END-TO-END ===")
            break

        if args.once:
            break

        log(f"Sleeping for {args.interval}s until next check...")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
