"""Task-5 failure diagnostic.

Why does "pick up the black bowl ON THE RAMEKIN" score 8-12% when the BDDL-identical
stacked configuration "...ON THE COOKIE BOX" scores 96-100%?

Mirrors evaluate_libero_standard._rollout EXACTLY (same policy, preprocessor, seed,
MAX_STEPS, early-return on is_success) and only ADDS logging. Behaviour is not
changed -- init_state_id is recorded, not pinned, so this reproduces the scored run.

Diagnosis, from where the gripper ends up and what moved:
  A. ends near a NON-target object      -> referent selection / language grounding
  B. reaches target, no z-lift          -> precision grasp off a narrow support
  C. never gets near the target         -> approach failure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_libero_standard as E


def _sim(env):
    """Locate the robosuite sim. Re-fetched every call, never cached: LiberoEnv
    self-resets on termination and frees the old MjSim, so a held reference dies."""
    for path in (("_env",), ("_env", "env"), ("env",), ()):
        obj = env
        try:
            for attr in path:
                obj = getattr(obj, attr)
            sim = getattr(obj, "sim", None)
            if sim is not None and getattr(sim, "data", None) is not None:
                return sim
        except AttributeError:
            continue
    raise RuntimeError("could not locate a live robosuite sim on env")


def _body_names(sim):
    model = sim.model
    id2name = getattr(model, "body_id2name", None) or model._body_id2name
    names = []
    for i in range(model.nbody):
        try:
            n = id2name(i)
        except Exception:
            n = None
        if n and n != "world" and not n.startswith(("robot0", "gripper0")):
            names.append(n)
    return names


def _bodies(env):
    sim = _sim(env)
    out = {}
    for name in _body_names(sim):
        try:
            out[name] = [float(v) for v in sim.data.get_body_xpos(name)]
        except Exception:
            continue
    return out


def _eef(env):
    return [float(v) for v in _sim(env).data.get_site_xpos("gripper0_grip_site")]


def rollout(policy, preprocessor, postprocessor, env, task_description, seed):
    """Byte-identical to E._rollout, plus logging."""
    policy.reset()
    observation, _ = env.reset(seed=seed)
    start = _bodies(env)
    end = start
    traj = []
    success, step = False, 0
    for step in range(1, E.MAX_STEPS + 1):
        batch = preprocessor(E._policy_input(observation, task_description))
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        act = postprocessor(action).to("cpu").float().numpy().reshape(-1)
        # snapshot BEFORE stepping: env.step may self-reset on termination and
        # invalidate the scene we are trying to describe.
        pre_eef, pre_bodies = _eef(env), _bodies(env)
        observation, _, terminated, truncated, info = env.step(act)
        traj.append({"t": step, "eef": pre_eef, "grip": float(act[6])})
        end = pre_bodies
        if info.get("is_success", False):
            success = True
            break
        if terminated or truncated:
            break
    return {
        "seed": seed, "success": success, "steps": step,
        "final_eef": traj[-1]["eef"] if traj else None,
        "objects_start": start, "objects_end": end,
        "z_lift": {k: end[k][2] - start[k][2] for k in start if k in end},
        "min_dist_to_object": {
            k: float(min(np.linalg.norm(np.array(p["eef"]) - np.array(start[k])) for p in traj))
            for k in start
        } if traj else {},
        "trajectory": traj[::5],  # every 5th step keeps the file small
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="NAME=flow|deeponet=CKPT")
    ap.add_argument("--tasks", default="3,5")
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()

    assert (E.SUITE, E.MAX_STEPS, E.REPLAN, E.CONTROL_FREQ) == ("libero_spatial", 220, 1, 20), \
        "protocol drift vs the scored evaluator"
    E._assert_native_runtime()

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(E.DATASET).stats
    name, head, ckpt = a.model.split("=", 2)
    policy, (pre, post) = E.load_policy(head, E.resolve_latest(ckpt), stats)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in [int(t) for t in a.tasks.split(",")]:
        env = E._make_env(task_id)
        desc = env.task_description
        for trial in range(a.trials):
            init_id = getattr(env, "init_state_id", None)
            r = rollout(policy, pre, post, env, desc, seed=1000 + trial)
            r.update(task_id=task_id, task=desc, trial=trial, init_state_id_before=init_id)
            records.append(r)
            print(f"[task{task_id}] trial{trial:02d} init{init_id}: "
                  f"{'OK' if r['success'] else 'x'} steps={r['steps']}", flush=True)
        env.close()

    (out / "probe_task5.json").write_text(json.dumps(records))
    print(f"\nwrote {len(records)} episodes -> {out/'probe_task5.json'}\n")

    for tid in sorted({r["task_id"] for r in records}):
        rs = [r for r in records if r["task_id"] == tid]
        sr = float(np.mean([r["success"] for r in rs]))
        print(f"=== task{tid}  SR={sr:.1%}  ({rs[0]['task']})")
        fails = [r for r in rs if not r["success"]]
        if not fails:
            continue
        lifts = {k: float(np.median([r["z_lift"].get(k, 0.0) for r in fails])) for k in fails[0]["z_lift"]}
        near = {k: float(np.median([r["min_dist_to_object"].get(k, 9.9) for r in fails]))
                for k in fails[0]["min_dist_to_object"]}
        top_lift = sorted(lifts.items(), key=lambda kv: -abs(kv[1]))[:4]
        closest = sorted(near.items(), key=lambda kv: kv[1])[:4]
        print(f"    FAIL n={len(fails)}  median steps={np.median([r['steps'] for r in fails]):.0f}"
              f"/{E.MAX_STEPS}  timeouts={sum(r['steps'] >= E.MAX_STEPS for r in fails)}")
        print(f"    largest |z-lift| : {[(k, round(v,4)) for k,v in top_lift]}")
        print(f"    gripper got closest to: {[(k, round(v,4)) for k,v in closest]}")


if __name__ == "__main__":
    main()
