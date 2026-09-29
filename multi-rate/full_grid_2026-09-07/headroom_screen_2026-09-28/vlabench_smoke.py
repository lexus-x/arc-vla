"""Cheapest possible VLABench check: load_env + reset + a few random steps, no policy involved.
Run before spending GPU time on the SmolVLA checkpoint. Exits nonzero on any failure."""
import os, sys, traceback

os.environ.setdefault("MUJOCO_GL", "egl")
TASKS = ["select_fruit", "select_toy", "select_book", "select_painting", "select_drink",
         "select_chemistry_tube", "select_poker", "add_condiment", "insert_flower", "select_mahjong"]

import VLABench.robots  # noqa: F401  (registers robot configs)
import VLABench.tasks  # noqa: F401  (registers task configs)
from VLABench.envs import load_env

ok, bad = [], []
for t in TASKS:
    try:
        env = load_env(task=t, robot="franka", random_init=False)
        ts = env.reset()
        obs0 = env.get_observation()
        for _ in range(3):
            env.step(env.action_spec().minimum * 0)  # zero action, just exercise the physics loop
        obs1 = env.get_observation()
        print(f"[{t}] OK  action_spec={env.action_spec().shape}  obs keys={list(obs0.keys())[:4]}")
        ok.append(t)
    except Exception as e:
        print(f"[{t}] FAIL  {type(e).__name__}: {e}")
        traceback.print_exc()
        bad.append(t)

print(f"\n{len(ok)}/{len(TASKS)} tasks load+reset+step OK. Failed: {bad}")
sys.exit(1 if bad else 0)
