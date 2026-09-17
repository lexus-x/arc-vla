"""
Sweep 2x, 4x, 8x decimation across TAC-Fold, Cubic Spline, and B-Spline on Push-T
Saves results incrementally to sweep_2x4x8x_results.json
"""
import sys, os, time, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_pusht_hub_resamplers as eval_hub
import argparse

def main():
    ks = [2, 4, 8]
    arms = "tac_fold,spline,bspline_eps_raw"
    n_episodes = 50
    batch_size = 10
    start_seed = 1000
    
    print(f"=== Starting 2x, 4x, 8x Evaluation Sweep ===")
    print(f"Arms: {arms}")
    print(f"Decimation ratios (k): {ks}")
    print(f"Episodes per arm: {n_episodes} (batch_size={batch_size})")
    print(f"Estimated time per k: ~4 mins | Total ETA: ~12-14 mins\n")
    
    all_results = {}
    out_summary_file = os.path.join(HERE, "sweep_2x4x8x_results.json")
    
    start_time = time.time()
    for idx, k in enumerate(ks, 1):
        print(f"\n=======================================================")
        print(f"[{idx}/{len(ks)}] Running Decimation k={k} ({k}x)")
        print(f"=======================================================")
        t0 = time.time()
        
        args = argparse.Namespace(
            policy_dir=os.path.join(HERE, "hub_diffusion_pusht"),
            k=k,
            arms=arms,
            n_episodes=n_episodes,
            batch_size=batch_size,
            start_seed=start_seed,
            policy_seed=20260916,
            suffix=f"sweep_k{k}",
            out_file=os.path.join(HERE, f"result_sweep_k{k}.json")
        )
        
        eval_hub.run_eval(args)
        
        # Load and store summary
        with open(args.out_file, "r") as f:
            k_res = json.load(f)
        all_results[f"k{k}"] = {
            arm: {
                "avg_max_reward": data["avg_max_reward"],
                "pc_success": data["pc_success"],
                "n_success": data["n_success"],
                "n_episodes": data["n_episodes"],
                "wall_s": data["wall_s"],
            }
            for arm, data in k_res["arms"].items()
        }
        
        # Save incremental master results
        with open(out_summary_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[Completed k={k} in {time.time() - t0:.1f}s — Saved to {out_summary_file}]")

    total_wall = time.time() - start_time
    print(f"\n=======================================================")
    print(f"ALL SWEEPS COMPLETE in {total_wall/60:.1f} minutes!")
    print(f"Master results saved to: {out_summary_file}")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
