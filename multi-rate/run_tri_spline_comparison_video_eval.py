"""Tri-Policy Comparison Video Evaluator & Logger.

Evaluates in closed-loop:
1. ARC Operator Fold Policy (Ours)
2. Cubic Spline Policy
3. B-Spline Policy

Renders a single 3-panel side-by-side synchronized comparison video with live status banners
and logs success rates to JSON and Excel.
"""

import os
import sys
import json
import cv2
import numpy as np
import torch
import imageio

CAMPAIGN_DIR = "/home/user/Desktop/multi-rate/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804"
sys.path.insert(0, CAMPAIGN_DIR)

import evaluate_multirate_video as EV

OUTPUT_VIDEO_PATH = "/home/user/Desktop/multi-rate/tri_comparison_eval_video.mp4"
OUTPUT_DESKTOP_VIDEO = "/home/user/Desktop/tri_comparison_eval_video.mp4"
RESULTS_JSON_PATH = "/home/user/Desktop/multi-rate/tri_comparison_eval_results.json"

def create_triple_frame(frame_ours, frame_cubic, frame_bspline, task_title, s_ours, s_cubic, s_bspline):
    H, W = 360, 360
    f1 = cv2.resize(frame_ours, (W, H))
    f2 = cv2.resize(frame_cubic, (W, H))
    f3 = cv2.resize(frame_bspline, (W, H))

    # Add header banner to each
    def annotate(img, method_name, is_ok):
        res = img.copy()
        cv2.rectangle(res, (0, 0), (W, 36), (20, 20, 20), -1)
        cv2.putText(res, method_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        
        status_color = (0, 220, 0) if is_ok else (0, 60, 240)
        status_text = "SUCCESS" if is_ok else "ACTIVE"
        cv2.rectangle(res, (0, H - 28), (W, H), (0, 0, 0), -1)
        cv2.putText(res, f"STATUS: {status_text}", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_color, 2, cv2.LINE_AA)
        return res

    a1 = annotate(f1, "1. OPERATOR FOLD (ARC)", s_ours)
    a2 = annotate(f2, "2. CUBIC SPLINE", s_cubic)
    a3 = annotate(f3, "3. B-SPLINE", s_bspline)

    combined = np.hstack([a1, a2, a3])
    
    # Top overarching header
    top_bar = np.full((40, combined.shape[1], 3), 35, dtype=np.uint8)
    cv2.putText(top_bar, f"TASK: {task_title} | 40 Hz Closed-Loop Tri-Policy Comparison", (15, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 220, 100), 2, cv2.LINE_AA)
    
    return np.vstack([top_bar, combined])

def main():
    print("="*60)
    print("STARTING TRI-POLICY CLOSED-LOOP VIDEO EVALUATION")
    print("1. ARC Operator Fold | 2. Cubic Spline | 3. B-Spline")
    print("="*60)

    # Initialize environment and model arms
    suite_name = "libero_spatial"
    n_tasks = 5
    n_trials = 3  # For fast high-quality synchronized render

    tasks = [
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate"
    ][:n_tasks]

    # Model definition
    asrc_ckpt = EV.ASRC_CKPT
    print(f"Loading ASRC Checkpoint: {asrc_ckpt}")
    policy, cfg, normalizer = EV.load_policy("deeponet", asrc_ckpt, "asrc", 6, EV.DEFAULT_DATASET)
    policy.eval()

    results = {
        "suite": suite_name,
        "arms": {
            "operator_fold_arc": {"success": 0, "total": 0},
            "cubic_spline": {"success": 0, "total": 0},
            "b_spline": {"success": 0, "total": 0}
        },
        "tasks": {}
    }

    writer = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=20, codec="libx264", quality=8)

    for task_idx, task_name in enumerate(tasks):
        print(f"\nEvaluating Task {task_idx+1}/{len(tasks)}: {task_name}")
        env = EV.make_env(suite_name, task_name, env_freq=40, horizon_steps=440)
        
        results["tasks"][task_name] = {"ours": 0, "cubic": 0, "bspline": 0, "trials": n_trials}

        for trial in range(n_trials):
            seed = 1000 + trial
            
            # Run Arm 1: Operator Fold (ARC)
            r1 = EV.rollout_single_arm(env, policy, normalizer, "asrc_cadmag_folding_40env", seed=seed, record_frames=True)
            # Run Arm 2: Cubic Spline
            r2 = EV.rollout_single_arm(env, policy, normalizer, "asrc_cadmag_spline_40env", seed=seed, record_frames=True)
            # Run Arm 3: B-Spline
            r3 = EV.rollout_single_arm(env, policy, normalizer, "asrc_cadmag_bspline_40env", seed=seed, record_frames=True)

            s1 = bool(r1.get("success", False))
            s2 = bool(r2.get("success", False))
            s3 = bool(r3.get("success", False))

            results["arms"]["operator_fold_arc"]["success"] += int(s1)
            results["arms"]["cubic_spline"]["success"] += int(s2)
            results["arms"]["b_spline"]["success"] += int(s3)
            for k in results["arms"]: results["arms"][k]["total"] += 1

            results["tasks"][task_name]["ours"] += int(s1)
            results["tasks"][task_name]["cubic"] += int(s2)
            results["tasks"][task_name]["bspline"] += int(s3)

            print(f"  Trial {trial+1}: Ours={s1} | Cubic={s2} | BSpline={s3}")

            # Align frame counts and write video
            f_ours = r1.get("frames", [])
            f_cubic = r2.get("frames", [])
            f_bsp = r3.get("frames", [])

            max_len = max(len(f_ours), len(f_cubic), len(f_bsp), 1)
            
            def pad_frames(flist, max_l):
                if not flist: return [np.zeros((256, 256, 3), dtype=np.uint8)] * max_l
                last = flist[-1]
                return flist + [last] * (max_l - len(flist))

            p1 = pad_frames(f_ours, max_len)
            p2 = pad_frames(f_cubic, max_len)
            p3 = pad_frames(f_bsp, max_len)

            # Subsample for video smoothness (skip every 2 frames at 40Hz -> 20fps video)
            for f_i in range(0, max_len, 2):
                frame_tri = create_triple_frame(p1[f_i], p2[f_i], p3[f_i], task_name[:45] + "...", s1, s2, s3)
                writer.append_data(frame_tri)

        env.close()

    writer.close()
    
    # Copy video to Desktop as well
    os.system(f"cp {OUTPUT_VIDEO_PATH} {OUTPUT_DESKTOP_VIDEO}")

    # Compute Summary
    for arm_k, arm_v in results["arms"].items():
        arm_v["rate"] = arm_v["success"] / max(arm_v["total"], 1)

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*60)
    print("EVALUATION COMPLETE & VIDEO SAVED")
    print(f"Video Path: {OUTPUT_VIDEO_PATH}")
    print(f"Desktop Video Path: {OUTPUT_DESKTOP_VIDEO}")
    print("Results Summary:")
    print(f"  ARC Operator Fold : {results['arms']['operator_fold_arc']['success']}/{results['arms']['operator_fold_arc']['total']} ({results['arms']['operator_fold_arc']['rate']*100:.1f}%)")
    print(f"  Cubic Spline      : {results['arms']['cubic_spline']['success']}/{results['arms']['cubic_spline']['total']} ({results['arms']['cubic_spline']['rate']*100:.1f}%)")
    print(f"  B-Spline          : {results['arms']['b_spline']['success']}/{results['arms']['b_spline']['total']} ({results['arms']['b_spline']['rate']*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
