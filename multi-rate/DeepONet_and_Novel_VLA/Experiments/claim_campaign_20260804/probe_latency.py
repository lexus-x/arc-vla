"""Detection LATENCY for the absorbing zero-action state -- does an escape wrapper have runway?

probe_absorbing_sweep.py established WHAT the failure is (8/8 of classifiable failures) and
bounded the prize at ~26 pp. It did NOT establish WHEN the collapse becomes detectable, and that
number alone decides whether any escape mechanism is buildable:

    runway = MAX_STEPS - first_step_at_which_the_40-step_window_sum_drops_below_threshold

Same rollout loop as probe_absorbing_sweep.py, byte-for-byte, with one change: the full per-step
command trace C is retained instead of being collapsed to cmd_late. Everything downstream is
computed offline from the saved traces, so thresholds can be re-derived without new GPU.

PRE-REGISTERED KILL RULES (fixed before the run):
  K1  median runway < 40 steps (2 s) -> a wrapper cannot act in time. The idea dies here.
  K2  >20% of SUCCESSES also trip the detector -> the signal is not specific; a detector would
      sabotage rollouts that were going to succeed. The idea dies here.
  K3  threshold is per-task, data-derived (min cmd_late among that task's OWN successes).
      No global constant. Tasks with no successes are excluded, not guessed.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
os.environ.setdefault('DEEPONET_HEAD', 'asrc'); os.environ.setdefault('DEEPONET_FOURIER', '6')
os.environ.setdefault('DEEPONET_STATE_HISTORY_STEPS', '8'); os.environ.setdefault('DEEPONET_P', '256')
sys.path.insert(0, '/home/user/DeepONet_and_Novel_VLA/Experiments/claim_campaign_20260804')
import torch
import evaluate_multirate_honest as honest, evaluate_height_screen as screen
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

honest.patch_control_freq(20)
screen.SUITE = 'libero_spatial'; screen.MAX_STEPS = 220; screen.REPLAN = 10; screen.CONTROL_FREQ = 20
policy, (pre, post) = screen.load_policy('deeponet', honest.MODELS['asrc'][1],
                                         LeRobotDatasetMetadata(screen.DATASET).stats)
TRIALS = 5
out = {}
for task in range(10):
    rows = []
    for trial in range(TRIALS):
        env = screen._make_env(task); env.num_steps_wait = honest.SETTLE_STEPS_20HZ
        env.init_state_id = trial; assert env.init_state_id == trial
        obs, _ = env.reset(seed=1000 + trial); policy.reset()
        C = []; ok = False
        for step in range(1, 221):
            batch = pre(screen._policy_input(obs, env.task_description))
            with torch.no_grad():
                act = policy.select_action(batch)
            act = post(act)
            an = np.asarray(act.squeeze(0).float().cpu() if torch.is_tensor(act) else act, float)
            obs, _, term, _, info = env.step(an)
            C.append(float(np.abs(an[:3]).sum()))
            if info.get('is_success'):
                ok = True; break
            if term:
                break
        env.close()
        rows.append(dict(ok=ok, steps=len(C), cmd=C))
        print(f"  task {task} trial {trial}: ok={ok} steps={len(C)}", flush=True)
    out[str(task)] = rows

json.dump(out, open('absorbing_latency.json', 'w'))
print('wrote absorbing_latency.json')

# ---- offline analysis on the traces just collected ----
W = 40
print("\ntask  thr      failures: first_trip_step -> runway     successes tripping")
for t, rows in out.items():
    S = [r for r in rows if r['ok']]; F = [r for r in rows if not r['ok']]
    if not S:
        print(f"  {t}: no in-task control, excluded"); continue
    def win(c):
        return [sum(c[max(0, i - W + 1):i + 1]) for i in range(len(c))]
    thr = min(sum(r['cmd'][-W:]) for r in S)
    def first_trip(c):
        w = win(c)
        for i in range(W, len(w)):
            if w[i] < thr:
                return i
        return None
    ftr = [(first_trip(r['cmd']), r['steps']) for r in F]
    runways = [s - f for f, s in ftr if f is not None]
    strip = sum(1 for r in S if first_trip(r['cmd']) is not None)
    print(f"  {t}  thr={thr:7.2f}  {[(f, s - f if f else None) for f, s in ftr]}  "
          f"succ_tripping={strip}/{len(S)}")
