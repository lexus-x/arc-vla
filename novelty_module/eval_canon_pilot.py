"""
eval_canon_pilot.py — Paired single-seed pilot: training-free test-time robustness
modules on LIBERO-Plus, reusing the lab's certified harness (evaluate_plus.py +
libero_plus_wrapper.py, v2 bundle).

Arms (frozen flow8300_s0 checkpoint; identical tasks + init states across arms):
  base             — plain rollout (harness convention: replan every 5 steps)
  canon            — Track A: illumination canonicalization before pre-processor
  consensus        — Track B: N noise-perturbed synchronized policy clones, mean action
  canon+consensus  — both

Categories: the four visual families (Light Conditions, Background Textures,
Objects Layout, Camera Viewpoints). Paired design: same task + env.reset(seed=0)
across arms — per the vault's eval-harness-traps doc (~10-21% rerun noise floor,
only paired comparisons count at pilot n).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

V2_DIR = "/home/user/Desktop/multi-rate/DeepONet_and_Novel_VLA/From_Blackwell/Ayush PH test/DeepONet PH/v2"
sys.path.insert(0, V2_DIR)
sys.path.insert(0, "/home/user/Desktop/novelty_module")

import evaluate_plus as EP  # noqa: E402  (imports libero_plus_wrapper, isolates LIBERO-Plus)
from libero_plus_wrapper import LiberoPlusEnv, list_perturbed_tasks, CATEGORIES  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
import torch  # noqa: E402
from vis_canon import canonicalize_pin, noise_pin, self_test  # noqa: E402

DEV = "cuda"
PILOT_CATEGORIES = ["Light Conditions", "Background Textures",
                    "Objects Layout", "Camera Viewpoints"]


@torch.no_grad()
def rollout_arm(policy, clones, pre, post, env, task_description, max_steps,
                use_canon=False, n_consensus=1, noise_sigma=0.02, gen=None):
    """One episode. Mirrors EP.rollout, plus optional canonicalization/consensus.

    Consensus: synchronized copies of the same policy (index 0 IS the base
    policy object) each queried with an independently noised copy of the input;
    post-processed actions averaged. Queues advance in lockstep because every
    clone is queried at the same env steps with the same replan interval.
    """
    policies = [policy] + clones
    for p in policies:
        p.reset()
    obs = env.reset(seed=0)

    def make_pin(i):
        pin = EP.plus_obs_to_policy_input(obs, task_description)
        if use_canon:
            pin = canonicalize_pin(pin)
        if i > 0:
            pin = noise_pin(pin, noise_sigma, gen)
        return pin

    def step_arms():
        acts = []
        for p, pin in zip(policies, [pre(make_pin(i)) for i in range(n_consensus)]):
            pin_d = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in pin.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = p.select_action(pin_d)
            acts.append(post(a).to("cpu").float().reshape(-1))
        return torch.stack(acts).mean(dim=0).clamp(-1.0, 1.0).numpy()

    for _ in range(max_steps):
        a = step_arms()
        obs, reward, done, info = env.step(a)
        if env.check_success():
            return True
        if done:
            break
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/media/user/C2FE578FFE577A9D/vla_matched/flow8300_s0/checkpoints/8300")
    ap.add_argument("--head", default="flow", choices=["flow", "deeponet"],
                    help="policy head type; deeponet requires a run_config.json next to the ckpt run dir")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--n_per_cat", type=int, default=6)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--consensus_n", type=int, default=4)
    ap.add_argument("--noise_sigma", type=float, default=0.02)
    ap.add_argument("--out", default="/home/user/Desktop/novelty_module/results/pilot_results.json")
    ap.add_argument("--smoke", action="store_true", help="2 tasks x 40 steps, base+canon only")
    args = ap.parse_args()

    self_test()
    n_per_cat = 2 if args.smoke else args.n_per_cat
    max_steps = 40 if args.smoke else args.max_steps
    arms = ["base", "canon"] if args.smoke else \
        ["base", "canon", "consensus", "canon+consensus"]

    # DeepONet checkpoints are guarded by training provenance: the loader reads
    # run_config.json next to the run dir and refuses to load unless eval env vars
    # match the trained flags. Set them from the run config automatically.
    if args.head == "deeponet":
        import os as _osc, json as _jsonc
        run_dir = Path(args.ckpt).parent.parent  # <run>/checkpoints/<step>/ -> <run>
        rc_path = run_dir / "run_config.json"
        if not rc_path.exists():
            raise SystemExit(f"[FATAL] --head deeponet requires {rc_path} (training provenance)")
        trained = _jsonc.load(open(rc_path))["args"]
        _osc.environ["DEEPONET_HEAD"] = trained.get("deeponet_head", "deeponet")
        # Older checkpoints predate these flags; their training default was False.
        _osc.environ["DEEPONET_TRUNK_BANDLIMIT"] = "1" if trained.get("trunk_bandlimit", False) else "0"
        _osc.environ["DEEPONET_POOL_CHANNEL_NORM"] = "1" if trained.get("pool_channel_norm", False) else "0"
        _osc.environ["DEEPONET_P"] = str(trained.get("deeponet_p", 256))
        _osc.environ["DEEPONET_BLOCKS"] = str(trained.get("deeponet_blocks", 3))
        _osc.environ["DEEPONET_QUERIES"] = str(trained.get("deeponet_queries", 8))
        _osc.environ["DEEPONET_FOURIER"] = str(trained.get("deeponet_fourier", 16))
        print(f"[pilot] deeponet provenance: head={_osc.environ['DEEPONET_HEAD']} "
              f"bandlimit={_osc.environ['DEEPONET_TRUNK_BANDLIMIT']} "
              f"pool_norm={_osc.environ['DEEPONET_POOL_CHANNEL_NORM']}", flush=True)

    stats = LeRobotDatasetMetadata(f"lerobot/{args.suite}_image").stats
    policy, pre, post = EP.load_policy(args.head, args.ckpt, stats, args.replan)
    print("[pilot] policy loaded", flush=True)

    n_clones = (args.consensus_n - 1) if any("consensus" in a for a in arms) else 0
    clones = [copy.deepcopy(policy).to(DEV).eval() for _ in range(n_clones)] if n_clones else []
    gen = torch.Generator()  # CPU generator (noise drawn on CPU, moved to device)
    gen.manual_seed(1234)

    bench, all_tasks = list_perturbed_tasks(args.suite)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["_config"] = {"ckpt": args.ckpt, "head": args.head, "n_per_cat": n_per_cat,
                          "max_steps": max_steps, "replan": args.replan,
                          "consensus_n": args.consensus_n, "noise_sigma": args.noise_sigma,
                          "categories": PILOT_CATEGORIES,
                          "task_sample_seed": 42, "rollout_reset_seed": 0}
    results.setdefault("arms", {})
    out_path.write_text(json.dumps(results, indent=2))

    for cat in PILOT_CATEGORIES:
        cat_tasks = [t for t in all_tasks if t.get("category") == cat]
        if not cat_tasks:
            raise SystemExit(f"[FATAL] no tasks classified under {cat!r}; available: {CATEGORIES}")
        tasks = EP.stratified_sample(cat_tasks, n_per_cat, seed=42)
        print(f"[pilot] {cat}: {len(tasks)} tasks (idx {[t['index'] for t in tasks]})", flush=True)

        for arm in arms:
            ares = results["arms"].setdefault(arm, {}).setdefault(cat, {"per_task": {}})
            for t in tasks:
                key = str(t["index"])
                if key in ares["per_task"]:
                    continue
                env = LiberoPlusEnv(bench, t["index"], img_size=256, control_freq=20)
                t0 = time.time()
                succ = rollout_arm(policy, clones, pre, post, env,
                                   env.task_description, max_steps,
                                   use_canon=("canon" in arm),
                                   n_consensus=args.consensus_n if "consensus" in arm else 1,
                                   noise_sigma=args.noise_sigma, gen=gen)
                env.close()
                ares["per_task"][key] = {"name": t.get("name", ""),
                                         "difficulty": t.get("difficulty_level"),
                                         "success": bool(succ),
                                         "sec": round(time.time() - t0, 1)}
                vals = [v["success"] for v in ares["per_task"].values()]
                ares["average"] = float(sum(vals) / len(vals))
                out_path.write_text(json.dumps(results, indent=2))
                print(f"[pilot/{arm}/{cat}] idx{t['index']} d{t.get('difficulty_level')}: "
                      f"{'OK' if succ else 'x'} ({ares['per_task'][key]['sec']}s)", flush=True)

    print("[pilot] DONE", flush=True)
    for arm in arms:
        cells = [f"{c.split()[0]}={results['arms'][arm][c]['average']:.2f}"
                 for c in PILOT_CATEGORIES if c in results["arms"][arm]]
        print(f"  {arm:18s} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
