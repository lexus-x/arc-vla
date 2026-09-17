"""Does grasp-height error scale with control rate? (the height<->rate unification test)

OSC ZOH plant:  x_{k+1} = x_k + a*u_k + b*xdot_k,  a = 1-(1+h)e^-h,  h = w*T,  T = 1/f.
Displacement per commanded action GROWS as the rate falls. So a policy carrying a learned
descent depth descends DEEPER at 10 Hz and SHALLOWER at 40 Hz.

PRE-REGISTERED PREDICTION (written before running):
  task 5 (bowl +44mm, flow descends too LOW at 20Hz) -> 40 Hz should IMPROVE it
  task 3 (bowl +8mm,  flow is correct at 20Hz)       -> 10 and 40 Hz should BOTH degrade
  and median bowl z-displacement should get MORE negative as rate falls.

If task 5 does not improve at 40 Hz, the unification is dead and I say so.

Episode wall-clock is held constant (11 s) by scaling MAX_STEPS with the rate, so the
comparison is across control rates, not across time budgets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import evaluate_libero_standard as E
import lerobot.envs.libero as L

BASE_RATE, BASE_STEPS = 20, 220
_ORIG_ENV_CLS = L.OffScreenRenderEnv


def set_rate(freq: int):
    """Inject control_freq into the robosuite env LiberoEnv builds, and scale the horizon."""
    class _Patched(_ORIG_ENV_CLS):
        def __init__(self, **kw):
            kw.setdefault("control_freq", freq)
            super().__init__(**kw)
    L.OffScreenRenderEnv = _Patched
    E.MAX_STEPS = round(BASE_STEPS * freq / BASE_RATE)


def _sim(env):
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
    raise RuntimeError("no live sim")


def _bowl_z(env, name="akita_black_bowl_1_main"):
    try:
        return float(_sim(env).data.get_body_xpos(name)[2])
    except Exception:
        return float("nan")


def rollout(policy, pre, post, env, desc, seed):
    policy.reset()
    obs, _ = env.reset(seed=seed)
    z0 = _bowl_z(env)
    z_last, success, step = z0, False, 0
    for step in range(1, E.MAX_STEPS + 1):
        batch = pre(E._policy_input(obs, desc))
        batch = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            action = policy.select_action(batch)
        act = post(action).to("cpu").float().numpy().reshape(-1)
        z_last = _bowl_z(env)          # snapshot BEFORE stepping (env self-resets on term)
        obs, _, term, trunc, info = env.step(act)
        if info.get("is_success", False):
            success = True
            break
        if term or trunc:
            break
    return {"seed": seed, "success": success, "steps": step, "z0": z0,
            "z_end": z_last, "dz": z_last - z0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rates", default="10,20,40")
    ap.add_argument("--tasks", default="3,5")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--force-steps", type=int, default=0,
                    help="override horizon; isolates step budget from control rate")
    a = ap.parse_args()

    E._assert_native_runtime()
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    stats = LeRobotDatasetMetadata(E.DATASET).stats
    name, head, ckpt = a.model.split("=", 2)
    policy, (pre, post) = E.load_policy(head, E.resolve_latest(ckpt), stats)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rate in [int(r) for r in a.rates.split(",")]:
        set_rate(rate)
        if a.force_steps:
            E.MAX_STEPS = a.force_steps
        for task_id in [int(t) for t in a.tasks.split(",")]:
            env = E._make_env(task_id)
            got = env._env.env.control_freq
            assert got == rate, f"control_freq not applied: wanted {rate}, env says {got}"
            desc = env.task_description
            for trial in range(a.trials):
                env.init_state_id = trial          # pin: defeat the lerobot drift defect
                r = rollout(policy, pre, post, env, desc, seed=1000 + trial)
                r.update(rate=rate, task_id=task_id, trial=trial)
                rows.append(r)
                print(f"[{rate}Hz task{task_id}] t{trial:02d} "
                      f"{'OK' if r['success'] else 'x'} steps={r['steps']}/{E.MAX_STEPS} "
                      f"dz={r['dz']:+.4f}", flush=True)
            env.close()
    (out / f"rate_{name}.json").write_text(json.dumps(rows))

    print("\n=== SR by rate x task ===")
    for task_id in sorted({r["task_id"] for r in rows}):
        line = []
        for rate in sorted({r["rate"] for r in rows}):
            s = [r for r in rows if r["rate"] == rate and r["task_id"] == task_id]
            fails = [r["dz"] for r in s if not r["success"] and not np.isnan(r["dz"])]
            line.append(f"{rate}Hz: SR={np.mean([r['success'] for r in s]):5.1%} "
                        f"dz_fail={np.median(fails) if fails else float('nan'):+.4f}")
        print(f"  task{task_id}  " + "   ".join(line))


if __name__ == "__main__":
    main()
